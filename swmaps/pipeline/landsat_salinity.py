"""Pipeline utilities for Landsat processing and salinity estimation."""

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioError

from swmaps.core.salinity.heuristic import estimate_salinity_level

# Map salinity classes to integer codes for compact raster storage
SALINITY_CLASS_CODES = {"land": 0, "fresh": 1, "brackish": 2, "saline": 3}


def landsat_reflectance_stack(
    src: rasterio.io.DatasetReader,
) -> dict[str, np.ndarray]:
    """Convert Landsat Collection 2 Level-2 DN values to surface reflectance bands.

    Parameters
    ----------
    src : rasterio.io.DatasetReader
        Open raster dataset from which to read Landsat Collection 2 Level-2
        spectral bands.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dictionary mapping spectral band names to float32 reflectance arrays.
        The returned keys are:

        - ``"blue"``
        - ``"green"``
        - ``"red"``
        - ``"nir"``
        - ``"swir1"``
        - ``"swir2"``

        Reflectance values are scaled using the documented Collection 2
        scale factor (``0.0000275``) and offset (``-0.2``).
    """
    scale = 0.0000275
    offset = -0.2

    # Landsat Collection 2 Level-2 band ordering:
    # 1 = coastal/aerosol
    # 2 = blue
    # 3 = green
    # 4 = red
    # 5 = nir
    # 6 = swir1
    # 7 = swir2

    return {
        "blue": (src.read(1).astype(np.float32) * scale + offset),
        "green": (src.read(2).astype(np.float32) * scale + offset),
        "red": (src.read(3).astype(np.float32) * scale + offset),
        "nir": (src.read(4).astype(np.float32) * scale + offset),
        "swir1": (src.read(5).astype(np.float32) * scale + offset),
        "swir2": (src.read(6).astype(np.float32) * scale + offset),
    }


def write_single_band(
    path: Path,
    profile: dict,
    array: np.ndarray,
    dtype: str,
    nodata: float | int | None = np.nan,
) -> None:
    """Save a single-band raster to disk using a template profile.

    Parameters
    ----------
    path : pathlib.Path
        Destination file path for the raster.
    profile : dict
        Raster profile to copy/update when writing the output.
    array : numpy.ndarray
        Data array to persist.
    dtype : str
        Target dtype for the raster on disk.
    nodata : float | int | None, optional
        Value to mark as nodata; ``None`` removes nodata from the profile.

    Returns
    -------
    None
        Raster is written as a side effect.
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
            if src.count < 6:
                logging.warning(
                    "Expected ≥6 bands for salinity estimation, found %d in %s",
                    src.count,
                    mosaic_path.name,
                )
                return None
            profile = src.profile
            bands = landsat_reflectance_stack(src)
    except RasterioError as exc:
        logging.warning("Unable to open mosaic %s: %s", mosaic_path, exc)
        return None

    # Defensive check in case a mosaic is missing expected bands
    required = {"blue", "green", "red", "nir", "swir1", "swir2"}
    missing = required - bands.keys()
    if missing:
        logging.warning(
            "Missing required bands %s in %s; skipping salinity estimation",
            missing,
            mosaic_path.name,
        )
        return None

    salinity = estimate_salinity_level(
        blue=bands["blue"],
        green=bands["green"],
        red=bands["red"],
        nir=bands["nir"],
        swir1=bands["swir1"],
        swir2=bands["swir2"],
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

    # Convert string labels → integer codes
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
