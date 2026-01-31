from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchgeo.models import FarSeg
from tqdm import tqdm

from swmaps.models.base import BaseSegModel


class FarSegDataset(Dataset):
    """Internal helper to load aligned GeoTIFF pairs."""

    def __init__(self, data_pairs):
        self.data_pairs = data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.data_pairs[idx]

        with rasterio.open(img_path) as src:
            # Read all bands, scale to [0, 1] if uint16
            img = src.read().astype("float32")
            if img.max() > 1.0:
                img /= 65535.0 if src.dtypes[0] == "uint16" else 255.0

        with rasterio.open(mask_path) as src:
            # Mask is usually single band
            mask = src.read(1).astype("int64")

        h, w = img.shape[1], img.shape[2]
        new_h = ((h + 31) // 32) * 32
        new_w = ((w + 31) // 32) * 32

        pad_h = new_h - h
        pad_w = new_w - w

        # img shape: [C, H, W] -> pad is (left, right, top, bottom)
        img_tensor = torch.from_numpy(img)
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

        mask_tensor = torch.from_numpy(mask)
        mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

        return img_tensor, mask_tensor


class FarSegModel(BaseSegModel):
    """
    Wrapper around TorchGeo's FarSeg for semantic segmentation.
    https://arxiv.org/pdf/2011.09766
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

    def forward(self, x):
        return self.model(x)

    def _calculate_miou(self, conf_matrix):
        """Calculates Mean IoU from a confusion matrix, ignoring classes with no samples."""
        intersection = np.diag(conf_matrix)
        union = np.sum(conf_matrix, axis=1) + np.sum(conf_matrix, axis=0) - intersection

        # Avoid division by zero
        mask = union > 0
        iou = intersection[mask] / union[mask]
        return np.mean(iou) if len(iou) > 0 else 0.0

    def _save_checkpoint(self, out_dir, filename):
        """Helper to save the model state."""
        out_path = Path(out_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": "farseg",
                "model_kwargs": {
                    "num_classes": self.num_classes,
                    "backbone": self.backbone,
                    "in_channels": self.model.backbone.conv1.in_channels,
                },
                "state_dict": self.state_dict(),
            },
            out_path,
        )
        print(f"Model saved to {out_path}")

    def train_model(
        self,
        data_pairs,
        out_dir,
        epochs=10,
        batch_size=4,
        lr=1e-4,
        val_pairs=None,
        **kwargs,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        num_classes = self.num_classes

        # 1. Prepare Data Loaders
        train_loader = DataLoader(
            FarSegDataset(data_pairs), batch_size=batch_size, shuffle=True
        )
        val_loader = (
            DataLoader(FarSegDataset(val_pairs), batch_size=batch_size)
            if val_pairs
            else None
        )

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=255)

        print(f"Starting FarSeg training on {device} with {num_classes} classes...")
        best_miou = 0.0

        for epoch in range(epochs):
            # --- TRAINING PHASE ---
            self.train()
            train_loss = 0.0
            train_conf_matrix = np.zeros((num_classes, num_classes))

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = self.forward(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Metrics update
                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                targets = masks.detach().cpu().numpy()
                valid = (targets >= 0) & (targets < num_classes)
                # Compute histogram for confusion matrix
                train_conf_matrix += np.bincount(
                    num_classes * targets[valid].astype(int) + preds[valid],
                    minlength=num_classes**2,
                ).reshape(num_classes, num_classes)

            avg_train_loss = train_loss / len(train_loader)
            train_miou = self._calculate_miou(train_conf_matrix)

            # --- VALIDATION PHASE ---
            avg_val_loss = None
            val_miou = None

            if val_loader:
                self.eval()
                val_loss = 0.0
                val_conf_matrix = np.zeros((num_classes, num_classes))

                with torch.no_grad():
                    for v_images, v_masks in tqdm(
                        val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False
                    ):
                        v_images, v_masks = v_images.to(device), v_masks.to(device)
                        v_outputs = self.forward(v_images)
                        val_loss += criterion(v_outputs, v_masks).item()

                        v_preds = torch.argmax(v_outputs, dim=1).cpu().numpy()
                        v_targets = v_masks.cpu().numpy()
                        v_valid = (v_targets >= 0) & (v_targets < num_classes)
                        val_conf_matrix += np.bincount(
                            num_classes * v_targets[v_valid].astype(int)
                            + v_preds[v_valid],
                            minlength=num_classes**2,
                        ).reshape(num_classes, num_classes)

                avg_val_loss = val_loss / len(val_loader)
                val_miou = self._calculate_miou(val_conf_matrix)

            # --- LOGGING ---
            print(f"\n--- Epoch {epoch+1} Summary ---")
            print(f"TRAIN | Loss: {avg_train_loss:.4f} | mIoU: {train_miou:.4f}")

            if val_loader:
                print(f"VAL   | Loss: {avg_val_loss:.4f} | mIoU: {val_miou:.4f}")
                # Save best based on Val mIoU (usually more meaningful than loss)
                if val_miou > best_miou:
                    best_miou = val_miou
                    self._save_checkpoint(out_dir, "farseg_best.pth")
                    print(f"*** New Best mIoU: {val_miou:.4f}! Saved checkpoint. ***")

        self._save_checkpoint(out_dir, "farseg_final.pth")
        return out_dir
