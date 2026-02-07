import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import rasterio
import torch
from torch import nn


class BaseSegModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

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
