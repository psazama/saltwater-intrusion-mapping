"""Torch-free base class for salinity prediction models.

Kept separate from :mod:`swmaps.models.base` (which imports torch,
albumentations, and segmentation-models-pytorch) so that the salinity
pipeline - and therefore the FastAPI service - can be imported without the
deep-learning stack installed.
"""

from __future__ import annotations


class BaseSalinityModel:
    """Base class for salinity prediction models.

    Args:
        output_dim: Number of output dimensions. ``1`` denotes regression;
            values ``> 1`` denote classification.
    """

    def __init__(self, output_dim: int = 1):
        self.output_dim = output_dim

    def forward(self, x):
        """Abstract forward pass."""
        raise NotImplementedError

    def train_model(self, data_pairs, out_dir, **kwargs):
        """Abstract training method for salinity models."""
        raise NotImplementedError(
            "This model does not implement a custom training loop."
        )

    @property
    def is_regression(self) -> bool:
        return self.output_dim == 1

    @property
    def is_classification(self) -> bool:
        return self.output_dim > 1
