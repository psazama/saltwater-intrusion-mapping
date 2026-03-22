"""Dataset utilities for segmentation model training and inference.

This module provides:

- Helper functions for resolving mission slugs and satellite integer IDs from
  file paths (:func:`mission_from_path`, :func:`satellite_id_from_mission`).
- A configurable augmentation pipeline (:func:`data_transforms`) used during
  training to improve generalisation across sensors and geographic regions.
- :class:`SegDataset` - a :class:`torch.utils.data.Dataset` base class that
  loads paired image/mask GeoTIFFs, applies optional label remapping, and
  handles both training and inference modes.
"""

from pathlib import Path

import albumentations as A
import numpy as np
import rasterio
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from swmaps.core.missions import get_mission


def mission_from_path(path: str) -> str:
    """Infer the mission slug from an image file path.

    Checks the file stem and parent directory name (case-insensitive) against
    known mission keywords. Falls back to the first ``_``-delimited token of
    the stem when no keyword matches.

    Args:
        path: Path to an imagery file whose name or parent directory encodes
            the satellite mission.

    Returns:
        str: Mission slug e.g. ``"sentinel-2"``, ``"landsat-5"``,
        ``"landsat-7"``. Returns the raw stem prefix when no known mission
        is detected.
    """

    s2 = ["sentinel-2", "sentinel2"]
    l5 = ["landsat-5", "landsat5", "lt05"]
    l7 = ["landsat-7", "landsat7", "le07"]
    l8 = ["landsat-8", "landsat8"]

    p = Path(path)
    # Check parents or stem for mission keywords
    parts = [p.parent.name.lower(), p.stem.lower()]
    for part in parts:
        if any(item in part for item in s2):
            return "sentinel-2"
        if any(item in part for item in l5):
            return "landsat-5"
        if any(item in part for item in l7):
            return "landsat-7"
        if any(item in part for item in l8):
            return "landsat-8"

    # Fallback to splitting stem
    return p.stem.split("_")[0]


def satellite_id_from_mission(mission: str) -> int:
    """Convert a mission slug to its integer satellite identifier.

    Args:
        mission: One of ``"landsat-5"``, ``"landsat-7"``, ``"sentinel-2"``.

    Returns:
        int: ``0`` = Landsat-5, ``1`` = Landsat-7, ``2`` = Sentinel-2.

    Raises:
        KeyError: If *mission* is not in the known mapping.
    """

    missions = {
        "landsat-5": 0,
        "landsat-7": 1,
        "sentinel-2": 2,
    }

    return missions[mission]


def data_transforms(target_size: int) -> A.Compose:
    """Build the training augmentation pipeline.

    Applies a multi-stage strategy to simulate variability across sensors,
    seasons, and regions:

    1. Geometric - rotations, flips, shift-scale-rotate.
    2. Spectral/atmospheric - brightness, gamma, multiplicative noise.
    3. Noise/quality - Gaussian noise or blur.
    4. Structural - coarse dropout for occlusion robustness.
    5. Final sizing - pad to *target_size* then randomly crop.

    The mask ignore index (``255``) is preserved across all spatial transforms.

    Args:
        target_size: Spatial size in pixels of the output square crop.

    Returns:
        A.Compose: Configured Albumentations augmentation pipeline.
    """

    transforms = A.Compose(
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
                    A.GaussNoise(var_limit=(0.001, 0.01)),  # Adjusted for 0.0-1.0 range
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
    return transforms


class SegDataset(Dataset):
    """PyTorch Dataset for paired imagery/mask GeoTIFF segmentation data.

    Each sample is a tuple ``(image_tensor, mask_tensor, sat_id)`` where:

    - ``image_tensor`` - float32 tensor of shape ``(C, H, W)`` in ``[0, 1]``.
    - ``mask_tensor`` - int64 tensor of shape ``(H, W)``; ``255`` is the
      ignore index used by all loss functions.
    - ``sat_id`` - integer satellite identifier from *data_pairs*.

    Args:
        data_pairs: List of ``(image_path, mask_path, sat_id)`` tuples.
        label_map: Optional dict mapping raw mask integer values to training
            class indices.
        target_size: Spatial size used by :meth:`_crop_center`.
        transform: Optional Albumentations transform applied to image and mask.
        inference_only: When ``True``, returns a dummy zero mask instead of
            reading a mask file. Use for inference pipelines.
    """

    def __init__(
        self,
        data_pairs,
        label_map=None,
        target_size=512,
        transform=None,
        inference_only=False,
    ):
        self.data_pairs = data_pairs
        self.samples = data_pairs  # TODO clean up and rermove
        self.target_size = target_size
        self.crop_size = target_size  # TODO clean up and remove
        self.transform = transform
        self.inference_only = inference_only

        # Pre-compute lookup table once
        self.lookup = None
        if label_map:
            # Initialize with 255 (ignore index) or 0 depending on your preference
            self.lookup = np.arange(256, dtype="int64")
            for k, v in label_map.items():
                if k < 256:
                    self.lookup[k] = v

        # Auto-infer channel count from the first sample
        if len(self.samples) > 0:
            with rasterio.open(self.samples[0][0]) as src:
                self.in_channels = src.count
        else:
            self.in_channels = 3

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""

    def _read_image(self, image_path):
        """Read a GeoTIFF and return a float32 array scaled to ``[0, 1]``.

        In 3-channel mode uses mission-specific RGB band indices. In
        multi-channel mode reads all :attr:`in_channels` bands in order.

        Args:
            image_path: Path to a multiband GeoTIFF.

        Returns:
            np.ndarray: Float32 array of shape ``(C, H, W)`` in ``[0, 1]``.
        """
        with rasterio.open(image_path) as src:
            # If it's a 3-band model, we ensure RGB order via mission logic
            if self.in_channels == 3 and src.count >= 3:
                mission_name = mission_from_path(image_path)
                try:
                    band_index = get_mission(mission_name).band_indices()
                    indices = [
                        band_index["red"],
                        band_index["green"],
                        band_index["blue"],
                    ]
                except (KeyError, ValueError):
                    # Fallback to first 3 bands if mission lookup fails
                    indices = [1, 2, 3]
            else:
                # For FarSeg/Multiband, read all inferred channels
                indices = list(range(1, self.in_channels + 1))

            # Safety check: don't request more bands than the file actually has
            indices = [i for i in indices if i <= src.count]

            arr = src.read(indices).astype("float32")

            # Scale to [0, 1]
            if arr.max() > 1.0:
                arr /= 65535.0 if src.dtypes[0] == "uint16" else 255.0

        return arr

    def _crop_center(self, arr):
        """Centre-crop or pad an array to ``(self.crop_size, self.crop_size)``.

        Handles both ``(C, H, W)`` image arrays and ``(H, W)`` mask arrays.
        Padding uses reflect mode to avoid hard edges.

        Args:
            arr: Input numpy array with shape ``(C, H, W)`` or ``(H, W)``.

        Returns:
            np.ndarray: Cropped/padded array of the target crop size.
        """
        is_2d = arr.ndim == 2
        if is_2d:
            arr = arr[np.newaxis, ...]

        c, h, w = arr.shape
        ch = self.crop_size

        # Padding if the image is smaller than the crop_size
        if h < ch or w < ch:
            pad_h = max(0, ch - h)
            pad_w = max(0, ch - w)
            arr = np.pad(
                arr,
                (
                    (0, 0),
                    (pad_h // 2, pad_h - pad_h // 2),
                    (pad_w // 2, pad_w - pad_w // 2),
                ),
                mode="reflect",
            )
            _, h, w = arr.shape

        # Calculate crop corners
        top = (h - ch) // 2
        left = (w - ch) // 2
        cropped = arr[:, top : top + ch, left : left + ch]

        return cropped[0] if is_2d else cropped

    def __getitem__(self, idx):
        s = self.samples[idx]
        image_path = s[0]
        sat_id = s[2]

        # Read and process image
        image_arr = self._read_image(image_path)
        image_arr = self._crop_center(image_arr)
        image = torch.from_numpy(image_arr)

        if self.inference_only:
            # FarSeg expects a mask in the training loop, so we return a dummy
            # during inference to keep the loader consistent
            dummy_mask = torch.zeros(
                (self.crop_size, self.crop_size), dtype=torch.int64
            )
            return image, dummy_mask, sat_id

        # Read and process mask
        with rasterio.open(s[1]) as src:
            mask_arr = src.read(1).astype("int64")

        mask_arr = self._crop_center(mask_arr)

        return image, torch.from_numpy(mask_arr), sat_id
