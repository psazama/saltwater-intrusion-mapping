import json
from pathlib import Path
from typing import List, Tuple, Union

import albumentations as A
import hypertune
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from swmaps.models.dataset import data_transforms


class JointLoss(nn.Module):
    def __init__(self, first_loss, second_loss, first_weight=0.5, second_weight=0.5):
        super().__init__()
        self.first_loss = first_loss
        self.second_loss = second_loss
        self.first_weight = first_weight
        self.second_weight = second_weight

    def forward(self, logits, targets):
        loss1 = self.first_loss(logits, targets)
        loss2 = self.second_loss(logits, targets)
        return (self.first_weight * loss1) + (self.second_weight * loss2)


class BaseSegModel(nn.Module):

    ###########################################
    ## Segmentation Model Utilities
    ###########################################

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def _get_loss_criterion(self, loss_type, weights=None):
        if loss_type == "focal":
            print("Using Focal Loss")
            return smp.losses.FocalLoss(mode="multiclass", ignore_index=255)
        elif loss_type == "dice":
            print("Using Dice Loss")
            return smp.losses.DiceLoss(
                mode="multiclass", ignore_index=255, from_logits=True
            )
        elif loss_type == "hybrid":
            print("Using Hybrid Dice-Focal Loss")
            return JointLoss(
                first_loss=smp.losses.DiceLoss(mode="multiclass", ignore_index=255),
                second_loss=smp.losses.FocalLoss(mode="multiclass", ignore_index=255),
                first_weight=0.5,
                second_weight=0.5,
            )
        elif loss_type == "ce":
            print("Using Cross Entropy Loss")
            return nn.CrossEntropyLoss(weight=weights, ignore_index=255)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def _compute_class_weights(self, data_pairs, label_map, num_classes):
        """Scans the dataset to calculate inverse frequency weights."""
        print("Calculating class weights (this may take a moment)...")
        counts = np.zeros(num_classes)

        # We only scan a subset (e.g., 100 pairs) if the dataset is massive
        subset = data_pairs[: min(len(data_pairs), 200)]

        for _, mask_path, _ in subset:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                if label_map:
                    # Apply mapping logic to match training
                    mapped_mask = np.full_like(mask, fill_value=255)
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

    def _calculate_miou(self, conf_matrix):
        """Calculates Mean IoU from a confusion matrix, ignoring classes with no samples."""
        intersection = np.diag(conf_matrix)
        union = np.sum(conf_matrix, axis=1) + np.sum(conf_matrix, axis=0) - intersection

        # Avoid division by zero
        mask = union > 0
        iou = intersection[mask] / union[mask]
        return np.mean(iou) if len(iou) > 0 else 0.0

    def _update_conf_matrices(
        self, preds, targets, sat_ids, global_matrix, sat_matrices
    ):
        """
        Updates global and per-satellite confusion matrices.
        Args:
            preds: numpy array of predictions
            targets: numpy array of ground truth
            sat_ids: numpy array of satellite identifiers
            global_matrix: the main confusion matrix to update
            sat_matrices: dict mapping {id: matrix}
        """
        valid = (targets != 255) & (targets >= 0) & (targets < self.num_classes)
        global_matrix += np.bincount(
            self.num_classes * targets[valid].astype(int) + preds[valid],
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)

        for sid in sat_matrices.keys():
            s_mask = sat_ids == sid
            if not np.any(s_mask):
                continue

            ts_targets = targets[s_mask]
            ts_preds = preds[s_mask]
            ts_valid = (
                (ts_targets != 255)
                & (ts_targets >= 0)
                & (ts_targets < self.num_classes)
            )

            if np.any(ts_valid):
                sat_matrices[sid] += np.bincount(
                    self.num_classes * ts_targets[ts_valid].astype(int)
                    + ts_preds[ts_valid],
                    minlength=self.num_classes**2,
                ).reshape(self.num_classes, self.num_classes)

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

    def _save_checkpoint(self, out_dir, filename, extra_meta=None):
        out_path = Path(out_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Filter out criterion weights from the state_dict
        state_dict = {
            k: v for k, v in self.state_dict().items() if not k.startswith("criterion")
        }

        checkpoint = {
            "num_classes": self.num_classes,
            "state_dict": state_dict,
        }
        if extra_meta:
            checkpoint.update(extra_meta)
        torch.save(checkpoint, out_path)

    def _save_logs(self, out_dir, history):
        """Saves the training history to a JSON or text file."""
        log_path = Path(out_dir) / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(history, f, indent=4)
        print(f"Metrics log saved to {log_path}")

    ###########################################
    ## Segmentation Model Core
    ###########################################

    def train_model(
        self,
        data_pairs: List[Tuple[Union[str, Path], Union[str, Path]]],
        out_dir: Union[str, Path],
        label_map: dict = None,  # Added this
        val_pairs: List[Tuple[Union[str, Path], Union[str, Path]]] = None,  # Added this
        **kwargs,
    ):
        """
        Abstract training method.
        Specific models (like FarSeg) must override this.
        """
        raise NotImplementedError(
            "This model does not implement a custom training loop."
        )

    def train_core(
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
        dataset_class=None,
        training_set=None,
        val_set=None,
        **kwargs,
    ):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        num_classes = self.num_classes

        # 2. CALCULATE WEIGHTS & CRITERION
        weights = self._compute_class_weights(data_pairs, label_map, self.num_classes)
        weights = weights.to(device)
        print(f"Computed Weights: {weights.cpu().numpy()}")

        criterion = self._get_loss_criterion(loss_type, weights=weights)

        # 3. DEFINE TRANSFORMS
        train_transform = data_transforms(target_size)

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
            dataset_class(
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
                dataset_class(
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

        print(f"Starting Model training on {device} with {num_classes} classes...")

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

        self._save_checkpoint(out_dir, "final.pth", extra_meta=self.meta)
        self._save_logs(out_dir, history)
        return out_dir

    def predict(self, x: torch.Tensor, sat_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Standardized prediction interface.
        Returns class indices (the argmax).
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x, sat_ids)
            return torch.argmax(out, dim=1)

    def freeze_backbone(self):
        for name, p in self.named_parameters():
            if "backbone" in name:
                p.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device


class BaseSalinityModel(nn.Module):
    """
    Base class for salinity prediction models.
    """

    def __init__(self, output_dim: int = 1):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x):
        raise NotImplementedError

    def train_model(self, data_pairs, out_dir, **kwargs):
        """Abstract training method for salinity models."""
        raise NotImplementedError(
            "This model does not implement a custom training loop."
        )

    def freeze_backbone(self):
        for name, p in self.named_parameters():
            if "backbone" in name:
                p.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_regression(self):
        return self.output_dim == 1

    @property
    def is_classification(self):
        return self.output_dim > 1
