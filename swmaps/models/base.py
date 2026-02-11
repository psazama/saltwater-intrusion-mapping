import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from torch import nn


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

        for _, mask_path in subset:
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
