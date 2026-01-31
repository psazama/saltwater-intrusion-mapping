from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from swmaps.core.missions import get_mission


def mission_from_path(path: str) -> str:
    p = Path(path)
    # Check parents or stem for mission keywords
    parts = [p.parent.name.lower(), p.stem.lower()]
    for part in parts:
        if "sentinel-2" in part:
            return "sentinel-2"
        if "landsat-5" in part:
            return "landsat-5"
        if "landsat-7" in part:
            return "landsat-7"
        if "landsat-8" in part:
            return "landsat-8"

    # Fallback to splitting stem
    return p.stem.split("_")[0]


class SegDataset(Dataset):
    def __init__(self, samples, crop_size=512, inference_only=False):
        """
        samples: list of dicts with "image_path" and possibly "mask_path"
        crop_size: desired H and W for model input (must be multiple of 32 for FarSeg)
        """
        self.samples = samples
        self.crop_size = crop_size
        self.inference_only = inference_only

        # Auto-infer channel count from the first sample
        if len(samples) > 0:
            with rasterio.open(samples[0]["image_path"]) as src:
                self.in_channels = src.count
        else:
            self.in_channels = 3

    def __len__(self):
        return len(self.samples)

    def _read_image(self, image_path):
        """
        Reads the number of bands inferred during initialization.
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
        """
        Center-crops or pads a NumPy array to self.crop_size.
        Handles both 3D (C, H, W) and 2D (H, W) arrays.
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
        image_path = s["image_path"]

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
            return image, dummy_mask

        # Read and process mask
        with rasterio.open(s["mask_path"]) as src:
            mask_arr = src.read(1).astype("int64")

        mask_arr = self._crop_center(mask_arr)

        return image, torch.from_numpy(mask_arr)
