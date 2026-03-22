"""Panopticon foundation encoder with segmentation head.

This module provides:

- :class:`PanopticonDataset` - a :class:`~swmaps.models.dataset.SegDataset`
  subclass with Panopticon-specific image loading.
- :class:`PanopticonModel` - a segmentation model combining the Panopticon
  foundation encoder with a lightweight convolutional segmentation head.
  Supports multi-sensor inputs via per-sample channel ID tensors.
"""

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch import nn
from torchgeo.models import Panopticon

from swmaps.core.missions import GLOBAL_BAND_IDS, get_mission_from_id
from swmaps.models.base import BaseSegModel
from swmaps.models.dataset import SegDataset

class PanopticonDataset(SegDataset):
    """Dataset class for Panopticon training and inference.

    Extends :class:`~swmaps.models.dataset.SegDataset` with the same
    image normalisation as :class:`~swmaps.models.farseg.FarSegDataset`.
    NoData pixels are masked with ignore index ``255``.

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
        img_path, mask_path, sat_id = self.data_pairs[idx]

        # ---- Load image ----
        with rasterio.open(img_path) as src:
            img = src.read().transpose(1, 2, 0).astype("float32")
            nodata_mask = np.all(img == 0, axis=-1)

            if img.max() > 1.0:
                img /= 65535.0 if src.dtypes[0] == "uint16" else 255.0

        # ---- Load mask ----
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype("int64")

        mask[nodata_mask] = 255

        # ---- Apply label mapping ----
        if self.lookup is not None:
            mask = np.clip(mask, 0, 255)
            mask = self.lookup[mask]

        # ---- Augmentations ----
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img_tensor = augmented["image"]
            mask_tensor = augmented["mask"].long()
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1))
            mask_tensor = torch.from_numpy(mask).long()

        return img_tensor, mask_tensor, sat_id


class PanopticonModel(BaseSegModel):
    """Panopticon foundation encoder with a convolutional segmentation head.

    The Panopticon encoder processes multi-spectral imagery from multiple
    satellite sensors using per-sample channel ID tensors that encode which
    spectral bands are present. This allows a single model to handle
    Landsat-5, Landsat-7, and Sentinel-2 imagery without separate heads.

    The segmentation head is a two-layer conv block that upsamples encoder
    features back to the input resolution.

    Args:
        num_classes: Number of output segmentation classes. Defaults to ``16``.
        in_channels: Number of input spectral bands. Defaults to ``6``.
        embed_dim: Encoder embedding dimension. Defaults to ``768``.
        attn_dim: Encoder attention dimension. Defaults to ``2304``.
        img_size: Input image size in pixels. Defaults to ``512``.
    """

    def __init__(
        self,
        num_classes=16,
        in_channels=6,
        embed_dim=768,
        attn_dim=2304,
        img_size=512,
    ):
        super().__init__(num_classes)

        self.embed_dim = embed_dim
        self.in_channels = in_channels

        self.encoder = Panopticon(
            embed_dim=embed_dim,
            attn_dim=attn_dim,
            img_size=img_size,
        )

        self.input_proj = nn.Identity()

        self.segmentation_head = self._make_head(
            embed_dim,
            num_classes,
        )

    def _make_head(self, in_channels, num_classes):
        """Build the convolutional segmentation head.

        Args:
            in_channels: Number of input feature channels from the encoder.
            num_classes: Number of output segmentation classes.

        Returns:
            nn.Sequential: Two-layer conv block with BatchNorm and ReLU.
        """
        head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_classes, 1),
        )

        # Proper initialization
        for m in head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        return head

    def forward(self, x, sat_ids):
        """Run a forward pass through the encoder and segmentation head.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.
            sat_ids: Integer satellite ID tensor of shape ``(B,)`` used to
                build per-sample channel ID tensors for the Panopticon encoder.

        Returns:
            torch.Tensor: Logit tensor of shape ``(B, num_classes, H, W)``
            upsampled to the input spatial resolution.
        """
        device = x.device

        chn_ids = self._get_channel_ids(sat_ids, device)

        input_size = x.shape[-2:]

        # Project multispectral → encoder input space
        x = self.input_proj(x)
        x_dict = {
            "imgs": x,
            "chn_ids": chn_ids,
            "sat_ids": torch.tensor(
                sat_ids,
                dtype=torch.long,
                device=device,
            ),
        }

        # Foundation features
        feats = self.encoder.model.forward_features(x_dict)
        feats = feats[:, 1:, :]
        B, N, D = feats.shape
        H = W = int(N**0.5)
        feats = feats.transpose(1, 2).reshape(B, D, H, W)

        # Segmentation logits
        logits = self.segmentation_head(feats)

        # Upsample back to image resolution
        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits

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
        """Configure and launch Panopticon training.

        Dynamically adjusts the segmentation head when *label_map* changes
        the number of output classes, then delegates to
        :meth:`~swmaps.models.base.BaseSegModel.train_core`.

        Args:
            data_pairs: List of ``(image_path, mask_path, sat_id)`` tuples.
            out_dir: Directory for checkpoints and training logs.
            label_map: Optional dict mapping raw mask values to class indices.
                When provided the segmentation head is rebuilt to match the
                number of unique output classes.
            val_pairs: Optional validation pairs.
            epochs: Maximum training epochs. Defaults to ``10``.
            batch_size: Samples per mini-batch. Defaults to ``64``.
            loss_type: One of ``"ce"``, ``"focal"``, ``"dice"``, ``"hybrid"``.
            lr: Initial Adam learning rate. Defaults to ``5e-5``.
            lr_patience: Epochs without improvement before LR reduction.
            stopping_patience: Epochs without improvement before early stop.
            target_size: Square crop size in pixels. Defaults to ``512``.
            **kwargs: Forwarded to
                :meth:`~swmaps.models.base.BaseSegModel.train_core`.
        """
        if label_map:
            unique_superclasses = set(label_map.values())
            new_num_classes = max(unique_superclasses) + 1

            if new_num_classes != self.num_classes:
                print(
                    f"[*] Reconfiguring segmentation head "
                    f"from {self.num_classes} to {new_num_classes} classes."
                )

                self.num_classes = new_num_classes

                self.segmentation_head = self._make_head(
                    self.embed_dim,
                    self.num_classes,
                )

        self.meta = {
            "model_type": "panopticon",
            "model_kwargs": {
                "num_classes": self.num_classes,
                "embed_dim": self.embed_dim,
                "in_channels": self.in_channels,
            },
        }

        dataset_class = PanopticonDataset

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

    def _get_channel_ids(self, sat_ids, device):
        """Build the Panopticon channel ID tensor for a batch.

        Maps each satellite integer ID to its ordered list of global band
        indices using :data:`~swmaps.core.missions.GLOBAL_BAND_IDS`, then
        stacks them into a ``(B, C)`` tensor.

        Args:
            sat_ids: Integer satellite ID tensor of shape ``(B,)``.
            device: Torch device to place the output tensor on.

        Returns:
            torch.Tensor: Long tensor of shape ``(B, C)`` where each row
            contains the global band indices for that sample's sensor.
        """
        chn_rows = []

        for sid in sat_ids:
            mission = get_mission_from_id(int(sid))

            band_names = list(mission.bands().keys())

            chn_rows.append([GLOBAL_BAND_IDS[b] for b in band_names])

        chn_ids = torch.tensor(
            chn_rows,
            dtype=torch.long,
            device=device,
        )

        return chn_ids
