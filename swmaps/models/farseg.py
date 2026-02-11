import albumentations as A
import hypertune
import numpy as np
import rasterio
import torch
from albumentations.pytorch import ToTensorV2
from torch import optim
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

        # Load Satellite ID
        path_str = str(img_path).lower()
        if "landsat5" in path_str or "lt05" in path_str:
            sat_id = 0  # L5
        elif "landsat7" in path_str or "le07" in path_str:
            sat_id = 1  # L7
        else:
            sat_id = 2  # S2 (default)

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

    def forward(self, x, sat_ids=None):
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

        # Move to device AFTER potential reconfiguration
        self.to(device)
        num_classes = self.num_classes

        # 2. CALCULATE WEIGHTS & CRITERION
        weights = self._compute_class_weights(data_pairs, label_map, self.num_classes)
        weights = weights.to(device)
        print(f"Computed Weights: {weights.cpu().numpy()}")

        criterion = self._get_loss_criterion(loss_type, weights=weights)

        # 3. DEFINE TRANSFORMS
        # TODO: data augmentation transformations probably generalize to other architectures and should be in base.py
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
            num_workers=4,
            pin_memory=False,
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
                num_workers=4,
                pin_memory=False,
            )

        # 5. OPTIMIZER & SCHEDULER
        # TODO: make weight_decay a parameter
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)

        # TODO: make warmup_epochs a parameter
        warmup_epochs = 2
        total_warmup_steps = len(train_loader) * warmup_epochs
        warmup_sched = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: min(1.0, step / total_warmup_steps)
        )
        plateau_sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=lr_patience, factor=0.1
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
        # TODO: get this dictionary programmatically from Missions
        sat_names = {0: "L5", 1: "L7", 2: "S2"}
        for name in sat_names.values():
            history[f"train_miou_{name.lower()}"] = []
            history[f"val_miou_{name.lower()}"] = []

        print(f"Starting FarSeg training on {device} with {num_classes} classes...")

        for epoch in range(epochs):
            # --- TRAINING PHASE ---
            self.train()
            train_loss, train_conf_matrix = 0.0, np.zeros((num_classes, num_classes))
            train_sat_matrices = {
                sid: np.zeros((num_classes, num_classes)) for sid in sat_names.keys()
            }

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for images, masks, sat_ids in pbar:
                images, masks, sat_ids = (
                    images.to(device),
                    masks.to(device),
                    sat_ids.to(device),
                )

                optimizer.zero_grad()
                outputs = self.forward(images, sat_ids)
                loss = criterion(
                    outputs, masks.long()
                )  # Ensure masks are long for CELoss
                loss.backward()
                optimizer.step()

                current_step = epoch * len(train_loader) + pbar.n
                if current_step < total_warmup_steps:
                    warmup_sched.step()

                train_loss += loss.item()

                # Metric calculation
                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                targets = masks.detach().cpu().numpy()
                sids = sat_ids.detach().cpu().numpy()

                self._update_conf_matrices(
                    preds, targets, sids, train_conf_matrix, train_sat_matrices
                )

            avg_train_loss = train_loss / len(train_loader)
            train_miou = self._calculate_miou(train_conf_matrix)
            current_lr = optimizer.param_groups[0]["lr"]

            # --- VALIDATION PHASE ---
            avg_val_loss, val_miou = 0.0, 0.0
            if val_loader:
                self.eval()

                val_loss, val_conf_matrix = 0.0, np.zeros((num_classes, num_classes))
                val_sat_matrices = {
                    sid: np.zeros((num_classes, num_classes))
                    for sid in sat_names.keys()
                }

                with torch.no_grad():
                    for v_images, v_masks, sat_ids in val_loader:
                        v_images, v_masks, sat_ids = (
                            v_images.to(device),
                            v_masks.to(device),
                            sat_ids.to(device),
                        )
                        v_outputs = self.forward(v_images, sat_ids)
                        val_loss += criterion(v_outputs, v_masks.long()).item()

                        v_preds = torch.argmax(v_outputs, dim=1).cpu().numpy()
                        v_targets = v_masks.cpu().numpy()
                        v_sids = sat_ids.cpu().numpy()
                        self._update_conf_matrices(
                            v_preds,
                            v_targets,
                            v_sids,
                            val_conf_matrix,
                            val_sat_matrices,
                        )

                avg_val_loss = val_loss / len(val_loader)
                val_miou = self._calculate_miou(val_conf_matrix)

                if (epoch + 1) * len(train_loader) >= total_warmup_steps:
                    plateau_sched.step(avg_val_loss)

                # Checkpoints & Early Stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss, early_stop_counter = avg_val_loss, 0
                else:
                    early_stop_counter += 1

                if val_miou > best_miou:
                    best_miou = val_miou
                    self._save_checkpoint(
                        out_dir, "best_model.pth", extra_meta=self.meta
                    )

                hpt = hypertune.HyperTune()
                hpt.report_hyperparameter_tuning_metric(
                    hyperparameter_metric_tag="val_miou",
                    metric_value=val_miou,
                    global_step=epoch,
                )
                for sid, name in sat_names.items():
                    s_miou = self._calculate_miou(val_sat_matrices[sid])
                    history[f"val_miou_{name.lower()}"].append(s_miou)
                history["val_loss"].append(avg_val_loss)
                history["val_miou"].append(val_miou)

            for sid, name in sat_names.items():
                s_miou = self._calculate_miou(train_sat_matrices[sid])
                history[f"train_miou_{name.lower()}"].append(s_miou)
            history["train_loss"].append(avg_train_loss)
            history["train_miou"].append(train_miou)
            history["lr"].append(current_lr)

            if val_loader:
                # Build a dynamic string per satellite: "L5: 0.4521 | L7: 0.3982 | S2: 0.6120"
                sat_summary = " | ".join(
                    [
                        f"{name}: {history[f'val_miou_{name.lower()}'][-1]:.4f}"
                        for name in sat_names.values()
                    ]
                )
                print(f"   [Satellite Val mIoU] {sat_summary}")

            print(
                f"Epoch {epoch+1:02d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f}, mIoU: {train_miou:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}, mIoU: {val_miou:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            if (epoch + 1) % 5 == 0 and val_loader:
                print(" Global Confusion Matrix")
                self._print_confusion_matrix(val_conf_matrix, epoch + 1)
                for sid, name in sat_names.items():
                    print(f" SENSOR: {name} Confusion Matrix")
                    self._print_confusion_matrix(
                        val_sat_matrices[sid], f"{epoch+1} ({name})"
                    )

            if early_stop_counter >= stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        self._save_checkpoint(out_dir, "farseg_final.pth", extra_meta=self.meta)
        self._save_logs(out_dir, history)
        return out_dir
