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
    threshold: float = 0.01,
    center_size: int | None = None,
) -> np.ndarray:
    """Compute the Normalized Difference Water Index (NDWI) mask."""
    mission_info = get_mission(mission)
    band_index = mission_info["band_index"]
    green_band = band_index["green"]
    nir_band = band_index["nir08"]

    scale_reflectance = "landsat" in mission

    with rasterio.open(path) as src:
        if center_size:
            img_width = src.width
            img_height = src.height

            col_off = (img_width - center_size) // 2
            row_off = (img_height - center_size) // 2

            window = Window(
                col_off=col_off,
                row_off=row_off,
                width=center_size,
                height=center_size,
            )

            green_raw = src.read(green_band, window=window)
            nir_raw = src.read(nir_band, window=window)
        else:
            green_raw = src.read(green_band)
            nir_raw = src.read(nir_band)

        # Identify valid pixels BEFORE scaling
        valid = (green_raw != 0) & (nir_raw != 0)

        green = green_raw.astype(np.float32)
        nir = nir_raw.astype(np.float32)

        # Mask NoData pixels explicitly
        green[~valid] = np.nan
        nir[~valid] = np.nan

        if scale_reflectance:
            green = green * 0.0000275 - 0.2
            nir = nir * 0.0000275 - 0.2

        ndwi = (green - nir) / (green + nir)

        # Apply threshold only on valid pixels
        ndwi_mask = ((ndwi > threshold) & valid).astype(np.float32)

        profile = src.profile.copy()
        profile.update(
            {
                "count": 1,
                "dtype": "float32",
                "nodata": 0.0,
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
