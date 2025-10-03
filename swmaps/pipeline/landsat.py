"""
Pipeline utilities for Landsat processing and salinity estimation.
"""

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioError

from swmaps.core.salinity_tools import estimate_salinity_level

# Map salinity classes to integer codes for compact raster storage
SALINITY_CLASS_CODES = {"land": 0, "fresh": 1, "brackish": 2, "saline": 3}


def landsat_reflectance_stack(src: rasterio.io.DatasetReader) -> list[np.ndarray]:
    """
    Convert Landsat Collection 2 Level-2 DN values to reflectance arrays.
    """
    scale = 0.0000275
    offset = -0.2
    return [
        (src.read(i).astype(np.float32) * scale + offset)
        for i in range(1, src.count + 1)
    ]


def write_single_band(
    path: Path,
    profile: dict,
    array: np.ndarray,
    dtype: str,
    nodata: float | int | None = np.nan,
) -> None:
    """
    Save a single-band raster to disk.
    """
    profile = profile.copy()
    profile.update({"count": 1, "dtype": dtype})

    if nodata is None:
        profile.pop("nodata", None)
    else:
        profile["nodata"] = nodata

    path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(path, "w", **profile, BIGTIFF="YES") as dst:
        dst.write(array.astype(dtype, copy=False), 1)


def estimate_salinity_from_mosaic(
    mosaic_path: Path, water_threshold: float = 0.2
) -> dict[str, Path] | None:
    """
    Estimate salinity products from a Landsat mosaic.

    Parameters
    ----------
    mosaic_path : Path
        Path to Landsat mosaic GeoTIFF.
    water_threshold : float, default=0.2
        Threshold for NDWI water detection.

    Returns
    -------
    dict[str, Path] or None
        Paths to salinity rasters: {"score", "class", "water_mask"}, or None on failure.
    """
    try:
        with rasterio.open(mosaic_path) as src:
            profile = src.profile
            reflectance_bands = landsat_reflectance_stack(src)
    except RasterioError as exc:
        logging.warning("Unable to open mosaic %s: %s", mosaic_path, exc)
        return None

    salinity = estimate_salinity_level(
        *reflectance_bands,
        reflectance_scale=None,
        water_threshold=water_threshold,
    )

    if salinity is None:
        logging.warning("Salinity estimation skipped for %s", mosaic_path)
        return None

    base = mosaic_path.with_suffix("")
    score_path = base.with_name(f"{base.stem}_salinity_score.tif")
    class_path = base.with_name(f"{base.stem}_salinity_class.tif")
    water_path = base.with_name(f"{base.stem}_salinity_water_mask.tif")

    logging.info("Writing salinity products for %s", mosaic_path.name)

    # Write score raster
    write_single_band(score_path, profile, salinity["score"], dtype="float32")

    # Convert string labels → codes
    class_codes = np.vectorize(
        lambda v: SALINITY_CLASS_CODES.get(v, 0), otypes=[np.uint8]
    )(salinity["class_map"])
    write_single_band(class_path, profile, class_codes, dtype="uint8", nodata=255)

    # Write combined water mask
    write_single_band(
        water_path,
        profile,
        salinity["water_mask"].astype(np.float32),
        dtype="float32",
        nodata=np.nan,
    )

    return {"score": score_path, "class": class_path, "water_mask": water_path}
