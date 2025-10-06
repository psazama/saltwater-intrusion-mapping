"""Spectral index utilities used to derive water masks from imagery."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

from .missions import get_mission


def compute_ndwi(
    path: str | Path,
    mission: str,
    out_path: str | Path | None = None,
    display: bool = False,
    threshold: float = 0.2,
    center_size: int | None = None,
) -> np.ndarray:
    """Compute the Normalized Difference Water Index (NDWI) mask.

    Args:
        path (str | Path): Path to the source GeoTIFF file containing
            multispectral imagery.
        mission (str): Mission identifier, e.g. ``"landsat-5"``,
            ``"landsat-7"``, or ``"sentinel-2"``.
        out_path (str | Path | None): Optional path where the NDWI raster
            should be written.
        display (bool): If ``True``, render the NDWI mask using Matplotlib.
        threshold (float): Threshold applied to the NDWI ratio to classify
            water pixels.
        center_size (int | None): If provided, limit the computation to a
            centered square window with the specified edge length (pixels).

    Returns:
        np.ndarray: Binary NDWI mask where water pixels equal ``1`` and
        other pixels equal ``0``.
    """
    mission_info = get_mission(mission)
    band_index = mission_info["band_index"]
    green_band = band_index["green"]
    nir_band = band_index["nir08"]

    scale_reflectance = "landsat" in mission

    with rasterio.open(path) as src:
        if center_size:
            # Get image size in pixels
            img_width = src.width
            img_height = src.height

            # Compute top-left corner of the centered window
            col_off = (img_width - center_size) // 2
            row_off = (img_height - center_size) // 2

            window = Window(
                col_off=col_off, row_off=row_off, width=center_size, height=center_size
            )

            green = src.read(green_band, window=window).astype(np.float32)
            nir = src.read(nir_band, window=window).astype(np.float32)
        else:
            green = src.read(green_band).astype(np.float32)
            nir = src.read(nir_band).astype(np.float32)

        if scale_reflectance:
            green = green * 0.0000275 - 0.2
            nir = nir * 0.0000275 - 0.2

        ndwi = (green - nir) / (green + nir + 1e-10)
        ndwi_mask = (ndwi > threshold).astype(float)

        profile = src.profile.copy()
        profile.update(
            {
                "count": 1,
                "dtype": "float32",
                "nodata": np.nan,
            }
        )
        if center_size:
            transform = src.window_transform(window)
            profile.update(
                {
                    "height": center_size,
                    "width": center_size,
                    "transform": transform,
                }
            )

        if out_path:
            with rasterio.open(out_path, "w", **profile, BIGTIFF="YES") as dst:
                dst.write(ndwi_mask, 1)

    if display:
        plt.imshow(ndwi_mask, cmap="gray")
        plt.title(f"NDWI Mask ({mission})")
        plt.axis("off")
        plt.show()

    return ndwi_mask
