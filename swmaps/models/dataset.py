from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from swmaps.core.missions import get_mission


def mission_from_path(path: str) -> str:
    p = Path(path)
    parent = p.parent.name
    if parent in ("sentinel-2", "landsat-5", "landsat-7"):
        return parent
    return p.stem.split("_")[0]


class SaltwaterSegDataset(Dataset):
    def __init__(self, samples, crop_size=512, inference_only=False):
        """
        samples: list of dicts with "image_path" and possibly "mask_path"
        crop_size: desired H and W for model input
        """
        self.samples = samples
        self.crop_size = crop_size
        self.inference_only = inference_only

    def __len__(self):
        return len(self.samples)

    def _read_rgb(self, image_path):
        mission = mission_from_path(image_path)
        band_index = get_mission(mission)["band_index"]

        rgb_bands = [
            band_index["red"],
            band_index["green"],
            band_index["blue"],
        ]

        with rasterio.open(image_path) as src:
            arr = src.read(rgb_bands).astype("float32")

        return arr

    def _crop_center(self, arr):
        """
        Center‑crop a NumPy array to self.crop_size.
        """
        _, h, w = arr.shape
        ch = self.crop_size

        if h < ch or w < ch:
            # pad if smaller than requested
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
            h, w = arr.shape[1:]

        # calculate crop corners
        top = (h - ch) // 2
        left = (w - ch) // 2
        return arr[:, top : top + ch, left : left + ch]

    def __getitem__(self, idx):
        s = self.samples[idx]
        image_path = s["image_path"]

        # read the RGB and crop
        image_arr = self._read_rgb(image_path)
        image_arr = self._crop_center(image_arr)

        image = torch.from_numpy(image_arr)

        if self.inference_only:
            dummy_mask = torch.zeros(1, dtype=torch.int64)
            return image, dummy_mask

        with rasterio.open(s["mask_path"]) as src:
            mask_arr = src.read(1).astype("int64")
        mask_arr = self._crop_center(mask_arr[np.newaxis, ...])[0]

        return image, torch.from_numpy(mask_arr)
