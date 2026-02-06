import json
from pathlib import Path

import albumentations as A
import numpy as np
import rasterio
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchgeo.models import FarSeg
from tqdm import tqdm

from swmaps.models.base import BaseSegModel


class FarSegDataset(Dataset):
    def __init__(self, data_pairs, label_map=None, target_size=512, transform=None):
        self.data_pairs = data_pairs
        self.target_size = target_size
        self.transform = transform

        # Pre-compute lookup table once
        self.lookup = None
        if label_map:
            # Initialize with 255 (ignore index) or 0 depending on your preference
            self.lookup = np.arange(256, dtype="int64")
            for k, v in label_map.items():
                if k < 256:
                    self.lookup[k] = v

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.data_pairs[idx]

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

    def _print_confusion_matrix(self, conf_matrix, epoch, label_names=None):
        """Prints a formatted confusion matrix to the console."""
        print(f"\n--- Confusion Matrix (Epoch {epoch}) ---")
        # If no labels provided, use indices
        if label_names is None:
            label_names = [f"Cls {i}" for i in range(self.num_classes)]

        # Header
        header = "Target \\ Pred".ljust(15) + "".join(
            [str(label).rjust(10) for label in label_names]
        )
        print(header)
        print("-" * len(header))

        for i, row in enumerate(conf_matrix):
            row_str = str(label_names[i]).ljust(15)
            for cell in row:
                row_str += str(int(cell)).rjust(10)
            print(row_str)
        print("-" * len(header) + "\n")

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

    def _save_logs(self, out_dir, history):
        """Saves the training history to a JSON or text file."""
        log_path = Path(out_dir) / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(history, f, indent=4)
        print(f"Metrics log saved to {log_path}")

    def _compute_class_weights(self, data_pairs, label_map, num_classes):
        """Scans the dataset to calculate inverse frequency weights."""
        print("Calculating class weights (this may take a moment)...")
        counts = np.zeros(num_classes)

        # We only scan a subset (e.g., 100 pairs) if the dataset is massive
        subset = data_pairs[: min(len(data_pairs), 200)]

        for _, mask_path in subset:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                if label_map:
                    # Apply mapping logic to match training
                    mapped_mask = np.zeros_like(mask)
                    for k, v in label_map.items():
                        mapped_mask[mask == k] = v
                    mask = mapped_mask

                # Count pixels for valid classes (ignore index 255)
                for c in range(num_classes):
                    counts[c] += np.sum(mask == c)

        # Inverse frequency: weights = total_pixels / (num_classes * class_pixels)
        total_pixels = np.sum(counts)
        weights = total_pixels / (num_classes * counts + 1e-6)

        # Normalize weights so the average weight is 1.0
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32)

    def train_model(
        self,
        data_pairs,
        out_dir,
        label_map=None,
        val_pairs=None,
        epochs=10,
        batch_size=64,
        lr=5e-5,
        patience=7,
        target_size=512,  # Added as a parameter with a sensible default
        **kwargs,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Move to device AFTER potential reconfiguration
        self.to(device)
        num_classes = self.num_classes

        # for m in self.modules():
        #    if isinstance(m, nn.BatchNorm2d):
        #        m.momentum = 0.01

        # 2. CALCULATE WEIGHTS & CRITERION
        # We define this once here to include class weights and ignore_index
        weights = self._compute_class_weights(data_pairs, label_map, self.num_classes)
        weights = weights.to(device)
        print(f"Computed Weights: {weights.cpu().numpy()}")

        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=255)

        # 3. DEFINE TRANSFORMS
        train_transform = A.Compose(
            [
                # 1. Geometric
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.7
                ),
                # 2. Spectral/Atmospheric
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2
                        ),
                        A.RandomGamma(
                            gamma_limit=(80, 120)
                        ),  # Simulates atmospheric haze/clarity
                        A.MultiplicativeNoise(
                            multiplier=(0.9, 1.1), per_channel=True
                        ),  # Spectral variation
                    ],
                    p=0.4,
                ),
                # 3. Noise/Quality
                A.OneOf(
                    [
                        A.GaussNoise(
                            var_limit=(0.001, 0.01)
                        ),  # Adjusted for 0.0-1.0 range
                        A.Blur(blur_limit=3),
                    ],
                    p=0.3,
                ),
                # 4. Structural
                A.CoarseDropout(
                    max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3
                ),
                # 5. Final Sizing
                # Set value=0 for image padding and mask_value=255 for the ignore index
                A.PadIfNeeded(
                    min_height=target_size,
                    min_width=target_size,
                    border_mode=0,
                    value=0,
                    mask_value=255,
                ),
                A.RandomCrop(target_size, target_size),
                ToTensorV2(),
            ]
        )

        val_transform = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=target_size,
                    min_width=target_size,
                    value=0,
                    mask_value=255,
                ),
                A.CenterCrop(target_size, target_size),
                ToTensorV2(),
            ]
        )

        # 4. INITIALIZE DATALOADERS
        train_loader = DataLoader(
            FarSegDataset(
                data_pairs,
                label_map=label_map,
                target_size=target_size,
                transform=train_transform,
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        val_loader = None
        if val_pairs:
            val_loader = DataLoader(
                FarSegDataset(
                    val_pairs,
                    label_map=label_map,
                    target_size=target_size,
                    transform=val_transform,
                ),
                batch_size=batch_size,
                num_workers=8,
                pin_memory=True,
            )

        # 5. OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)

        warmup_epochs = 2
        total_warmup_steps = len(train_loader) * warmup_epochs
        warmup_sched = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: min(1.0, step / total_warmup_steps)
        )
        plateau_sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=patience, factor=0.1
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, plateau_sched],
            milestones=[total_warmup_steps],
        )

        # 6. TRAINING LOOP
        best_val_loss = float("inf")
        best_miou = 0.0
        early_stop_counter = 0
        history = {
            "train_loss": [],
            "train_miou": [],
            "val_loss": [],
            "val_miou": [],
            "lr": [],
        }

        print(f"Starting FarSeg training on {device} with {num_classes} classes...")

        for epoch in range(epochs):
            # --- TRAINING PHASE ---
            self.train()
            train_loss, train_conf_matrix = 0.0, np.zeros((num_classes, num_classes))

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = self.forward(images)
                loss = criterion(
                    outputs, masks.long()
                )  # Ensure masks are long for CELoss
                loss.backward()
                optimizer.step()

                current_step = epoch * len(train_loader) + pbar.n
                if current_step < total_warmup_steps:
                    scheduler.step()

                train_loss += loss.item()

                # Metric calculation
                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                targets = masks.detach().cpu().numpy()

                valid = (targets >= 0) & (targets < num_classes)

                train_conf_matrix += np.bincount(
                    num_classes * targets[valid].astype(int) + preds[valid],
                    minlength=num_classes**2,
                ).reshape(num_classes, num_classes)

            avg_train_loss = train_loss / len(train_loader)
            train_miou = self._calculate_miou(train_conf_matrix)
            current_lr = optimizer.param_groups[0]["lr"]

            # --- VALIDATION PHASE ---
            avg_val_loss, val_miou = 0.0, 0.0
            if val_loader:
                self.eval()
                val_loss, val_conf_matrix = 0.0, np.zeros((num_classes, num_classes))
                with torch.no_grad():
                    for v_images, v_masks in val_loader:
                        v_images, v_masks = v_images.to(device), v_masks.to(device)
                        v_outputs = self.forward(v_images)
                        val_loss += criterion(v_outputs, v_masks.long()).item()

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

                if (epoch + 1) * len(train_loader) >= total_warmup_steps:
                    scheduler.step(avg_val_loss)

                # Checkpoints & Early Stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss, early_stop_counter = avg_val_loss, 0
                else:
                    early_stop_counter += 1

                if val_miou > best_miou:
                    best_miou = val_miou
                    self._save_checkpoint(out_dir, "best_model.pth")

                history["val_loss"].append(avg_val_loss)
                history["val_miou"].append(val_miou)

            history["train_loss"].append(avg_train_loss)
            history["train_miou"].append(train_miou)
            history["lr"].append(current_lr)

            print(
                f"Epoch {epoch+1:02d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f}, mIoU: {train_miou:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}, mIoU: {val_miou:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            if (epoch + 1) % 5 == 0 and val_loader:
                self._print_confusion_matrix(val_conf_matrix, epoch + 1)

            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        self._save_checkpoint(out_dir, "farseg_final.pth")
        self._save_logs(out_dir, history)
        return out_dir
