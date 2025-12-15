"""Pipeline helpers for deriving NDWI water masks from mosaics."""

import logging
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

from swmaps.core.indices import compute_ndwi
from swmaps.core.water_trend import check_image_for_nans, check_image_for_valid_signal


def generate_masks(center_size=None, input_dir=None):
    """Generate NDWI water masks for all mosaics in the given directory.

    Args:
        center_size (int | None): Optional pixel size of a centred window.
        input_dir (str | Path | None): Directory to search for mosaics.

    Returns:
        None: Masks are written next to their source mosaics.
    """
    if input_dir is None:
        logging.warning("[WARNING] No mask path provided")
        return
    else:
        search_dir = Path(input_dir)

    for tif in tqdm(sorted(search_dir.rglob("*_multiband.tif"))):
        if tif.name.endswith(("_mask.tif", "_features.tif")):
            continue
        if check_image_for_nans(str(tif)) or not check_image_for_valid_signal(str(tif)):
            continue

        if "sentinel" in tif.name:
            mission = "sentinel-2"
        elif "landsat-5" in tif.name:
            mission = "landsat-5"
        elif "landsat-7" in tif.name:
            mission = "landsat-7"
        else:
            continue

        out_mask = tif.with_name(f"{tif.stem}_mask.tif")
        compute_ndwi(
            str(tif), mission, str(out_mask), display=False, center_size=center_size
        )

        # --- write PNG visualization ---
        png_path = out_mask.with_suffix(".png")

        with rasterio.open(out_mask) as src:
            mask = src.read(1)

        # Convert to binary uint8 image for PNG
        mask_png = np.where(mask > 0, 255, 0).astype(np.uint8)

        Image.fromarray(mask_png, mode="L").save(png_path)
    print("Water masks generated")
