from pathlib import Path
from typing import List, Tuple, Union

import torch
from torch import nn


class BaseSegModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        raise NotImplementedError

    def train_model(
        self,
        data_pairs: List[Tuple[Union[str, Path], Union[str, Path]]],
        out_dir: Union[str, Path],
        **kwargs
    ):
        """
        Abstract training method.
        Specific models (like FarSeg) must override this to handle their
        own specific loss functions, dataloaders, and hyperparameters.

        Args:
            data_pairs: List of (image_path, mask_path) tuples.
            out_dir: Directory to save weights and logs.
            **kwargs: Flexible arguments for epochs, lr, batch_size, etc.
        """
        raise NotImplementedError(
            "This model does not implement a custom training loop."
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standardized prediction interface.
        Returns class indices (the argmax).
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
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
