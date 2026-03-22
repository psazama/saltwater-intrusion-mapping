"""FarSeg semantic segmentation model for coastal land cover classification.

This module provides:

- :class:`FarSegDataset` - a :class:`~swmaps.models.dataset.SegDataset`
  subclass with FarSeg-specific image and mask loading.
- :class:`FarSegModel` - a wrapper around TorchGeo's FarSeg architecture
  with support for multi-spectral input and dynamic class adjustment.

Reference: https://arxiv.org/pdf/2011.09766
"""

import numpy as np
import rasterio
import torch
from torchgeo.models import FarSeg

from swmaps.models.base import BaseSegModel
from swmaps.models.dataset import SegDataset


class FarSegDataset(SegDataset):
    """Dataset class for FarSeg training and inference.

    Extends :class:`~swmaps.models.dataset.SegDataset` with FarSeg-specific
    image normalisation - uint16 images are divided by 65535, uint8 by 255.
    NoData pixels (all-zero across bands) are masked out in the label with
    the ignore index ``255``.

    Args:
        data_pairs: List of ``(image_path, mask_path, sat_id)`` tuples.
        label_map: Optional dict mapping raw mask values to class indices.
        target_size: Square crop size in pixels.
        transform: Optional Albumentations transform.
        inference_only: When ``True`` returns a dummy mask instead of
            reading a mask file.
    """
    def __init__(
        self,
        data_pairs,
        label_map=None,
        target_size=512,
        transform=None,
        inference_only=False,
    ):
        super().__init__(data_pairs, label_map, target_size, transform, inference_only)

    def __getitem__(self, idx):
        """Return the ``idx``-th sample as ``(image_tensor, mask_tensor, sat_id)``.

        Args:
            idx: Sample index.

        Returns:
            tuple: Float32 image tensor ``(C, H, W)``, int64 mask tensor
            ``(H, W)``, and integer satellite ID.
        """
        img_path, mask_path, sat_id = self.data_pairs[idx]

        # Load Image
        with rasterio.open(img_path) as src:
            img = src.read().transpose(1, 2, 0).astype("float32")
            nodata_mask = np.all(img == 0, axis=-1)
            if img.max() > 1.0:
                img /= 65535.0 if src.dtypes[0] == "uint16" else 255.0

        # Load Mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype("int64")

        mask[nodata_mask] = 255

        # Apply label mapping via vectorized lookup
        if self.lookup is not None:
            # Ensure mask values don't exceed lookup bounds
            mask = np.clip(mask, 0, 255)
            mask = self.lookup[mask]

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img_tensor = augmented["image"]
            mask_tensor = augmented["mask"].long()
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1))
            mask_tensor = torch.from_numpy(mask).long()

        return img_tensor, mask_tensor, sat_id


class FarSegModel(BaseSegModel):
    """FarSeg (Foreground-Aware Relation Network) segmentation model.

    Wraps TorchGeo's FarSeg implementation with:

    - Support for arbitrary input channel counts via first-layer replacement.
    - Dynamic class adjustment when a ``label_map`` remaps the number of
      output classes at training time.
    - Delegation to :meth:`~swmaps.models.base.BaseSegModel.train_core` for
      the shared training loop.

    Reference: https://arxiv.org/pdf/2011.09766

    Args:
        backbone: ResNet backbone variant. Defaults to ``"resnet50"``.
        num_classes: Number of output segmentation classes. Defaults to ``16``.
        backbone_pretrained: Whether to load ImageNet-pretrained backbone
            weights. Defaults to ``True``.
        in_channels: Number of input spectral bands. When not ``3``, the
            first convolution layer is replaced to accept the correct number
            of channels. Defaults to ``6``.
    """
    def __init__(
        self,
        backbone="resnet50",
        num_classes=16,
        backbone_pretrained=True,
        in_channels=6,
    ):
        super().__init__(num_classes)
        self.model = FarSeg(
            backbone=backbone,
            classes=num_classes,
            backbone_pretrained=backbone_pretrained,
        )

        self.backbone = backbone

        # If input channels != 3, we must replace the first conv layer
        if in_channels != 3:
            # For ResNet backbones, the first layer is usually self.model.backbone.conv1
            old_conv = self.model.backbone.conv1
            self.model.backbone.conv1 = torch.nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

    def forward(self, x, sat_ids=None):
        """Run a forward pass through the FarSeg model.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.
            sat_ids: Satellite ID tensor - accepted for API consistency with
                Panopticon but not used by FarSeg.

        Returns:
            torch.Tensor: Logit tensor of shape ``(B, num_classes, H, W)``.
        """
        return self.model(x)

    def train_model(
        self,
        data_pairs,
        out_dir,
        label_map=None,
        val_pairs=None,
        epochs=10,
        batch_size=64,
        loss_type="ce",
        lr=5e-5,
        lr_patience=5,
        stopping_patience=15,
        target_size=512,
        **kwargs,
    ):
        """Configure and launch FarSeg training.

        Dynamically adjusts the number of output classes based on *label_map*
        before delegating to :meth:`~swmaps.models.base.BaseSegModel.train_core`.

        Args:
            data_pairs: List of ``(image_path, mask_path, sat_id)`` tuples.
            out_dir: Directory for checkpoints and training logs.
            label_map: Optional dict mapping raw mask values to class indices.
                When provided the model head is rebuilt to match the number
                of unique output classes.
            val_pairs: Optional validation pairs.
            epochs: Maximum training epochs. Defaults to ``10``.
            batch_size: Samples per mini-batch. Defaults to ``64``.
            loss_type: One of ``"ce"``, ``"focal"``, ``"dice"``, ``"hybrid"``.
            lr: Initial Adam learning rate. Defaults to ``5e-5``.
            lr_patience: Epochs without improvement before LR reduction.
            stopping_patience: Epochs without improvement before early stop.
            target_size: Square crop size in pixels. Defaults to ``512``.
            **kwargs: Forwarded to :meth:`~swmaps.models.base.BaseSegModel.train_core`.
        """

        # 1. DYNAMIC CLASS ADJUSTMENT
        if label_map:
            unique_superclasses = set(label_map.values())
            new_num_classes = max(unique_superclasses) + 1

            if new_num_classes != self.num_classes:
                print(
                    f"[*] Reconfiguring model from {self.num_classes} to {new_num_classes} classes."
                )

                # Capture current input channels before re-initializing
                old_in_channels = self.model.backbone.conv1.in_channels

                # Rebuild FarSeg internal model
                self.num_classes = new_num_classes
                self.model = FarSeg(
                    backbone=self.backbone,
                    classes=self.num_classes,
                    backbone_pretrained=True,
                )

                # Restore in_channels if they were modified (e.g., for 4-band or multi-spectral)
                if old_in_channels != 3:
                    old_conv = self.model.backbone.conv1
                    self.model.backbone.conv1 = torch.nn.Conv2d(
                        old_in_channels,
                        old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        padding=old_conv.padding,
                        bias=False,
                    )

        self.meta = {
            "model_type": "farseg",
            "model_kwargs": {
                "num_classes": self.num_classes,
                "backbone": (
                    self.backbone if hasattr(self, "backbone") else "resnet50"
                ),
                # Add any other FarSeg specific init args here
            },
        }

        dataset_class = FarSegDataset

        self.train_core(
            data_pairs,
            out_dir,
            label_map,
            val_pairs,
            epochs,
            batch_size,
            loss_type,
            lr,
            lr_patience,
            stopping_patience,
            target_size,
            dataset_class,
            **kwargs,
        )
