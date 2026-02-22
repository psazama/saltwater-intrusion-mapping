import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch import nn
from torchgeo.models import Panopticon

from swmaps.core.missions import GLOBAL_BAND_IDS, get_mission_from_id
from swmaps.models.base import BaseSegModel
from swmaps.models.dataset import SegDataset

# ============================================================
# Dataset
# ============================================================


class PanopticonDataset(SegDataset):
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


# ============================================================
# Model
# ============================================================


class PanopticonModel(BaseSegModel):
    """
    Panopticon foundation encoder + segmentation head.
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

        # ----------------------------------------------------
        # Panopticon foundation encoder
        # ----------------------------------------------------
        self.encoder = Panopticon(
            embed_dim=embed_dim,
            attn_dim=attn_dim,
            img_size=img_size,
        )

        self.input_proj = nn.Identity()

        # ----------------------------------------------------
        # Segmentation head
        # ----------------------------------------------------
        self.segmentation_head = self._make_head(
            embed_dim,
            num_classes,
        )

    # --------------------------------------------------------
    # Segmentation Head Builder
    # --------------------------------------------------------
    def _make_head(self, in_channels, num_classes):
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

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, x, sat_ids):
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

    # --------------------------------------------------------
    # Training Wrapper
    # --------------------------------------------------------
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

        # ----------------------------------------------------
        # Dynamic class adjustment (HEAD ONLY)
        # ----------------------------------------------------
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

    ######################################
    # Helper Utilities
    ######################################
    def _get_channel_ids(self, sat_ids, device):
        """
        Build Panopticon channel ID tensor.

        Returns:
            chn_ids: (B, C)
        """

        chn_rows = []

        for sid in sat_ids:
            print(sid)
            print(type(sid))
            print(sid.shape)
            mission = get_mission_from_id(int(sid))

            band_names = list(mission.bands().keys())

            chn_rows.append([GLOBAL_BAND_IDS[b] for b in band_names])

        chn_ids = torch.tensor(
            chn_rows,
            dtype=torch.long,
            device=device,
        )

        return chn_ids
