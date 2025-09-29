"""Example Landsat processing workflow for the Somerset study region.

This script demonstrates how to:

1. Load a study area from ``config/somerset.geojson``.
2. Query historical Landsat-5 and Landsat-7 imagery for several date ranges.
3. Create mosaics and NDWI-based water masks for each scene.
4. Estimate water salinity levels from the multispectral stacks.
5. Retrieve NLCD and USDA NASS Cropland Data Layer overlays for the region.

The workflow stores all derived rasters underneath ``swmaps/data`` via
:func:`swmaps.config.data_path` so that repeated runs reuse existing downloads.

The script focuses on clarity rather than raw throughput – patch-wise downloads
are executed sequentially and we only request a single STAC item per date range.
Adjust the mission configuration or date ranges as needed for your analyses.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.errors import RasterioError

from swmaps.config import data_path
from swmaps.core.download_tools import (
    compute_ndwi,
    create_mosaic_placeholder,
    download_nass_cdl,
    download_nlcd,
    get_mission,
    patchwise_query_download_mosaic,
    reproject_bbox,
)
from swmaps.core.salinity_tools import estimate_salinity_level

# ---------------------------------------------------------------------------
# Study-area configuration
# ---------------------------------------------------------------------------
GEOJSON_PATH = Path("config/somerset.geojson")
OUTPUT_ROOT = data_path("examples", "somerset_landsat")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Limit each mission to a handful of representative historical ranges.
# The ranges fall inside each mission's operational period to avoid queries
# without data. Adjust or expand to suit the temporal coverage you need.
MISSION_DATE_RANGES: dict[str, list[str]] = {
    "landsat-5": [
        "1990-07-01/1990-07-31",
        "2005-07-01/2005-07-31",
    ],
    "landsat-7": [
        "2000-07-01/2000-07-31",
        "2010-07-01/2010-07-31",
    ],
}

# Amount of temporal padding (in days) applied to each configured date range
# before querying satellite imagery. This helps avoid empty queries by widening
# the search window around the target month.
DATE_RANGE_PADDING_DAYS = 15

# Maximum number of STAC items to request per patch. Allowing multiple
# candidates gives the downloader flexibility to fall back to alternative
# acquisitions when the first item is unavailable.
MAX_ITEMS_PER_PATCH = 3

# Mapping from the salinity class labels returned by ``estimate_salinity_level``
# to compact integer codes for easier raster storage/visualisation.
SALINITY_CLASS_CODES = {"land": 0, "fresh": 1, "brackish": 2, "saline": 3}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _load_region_bounds() -> tuple[gpd.GeoSeries, list[float]]:
    """Load the Somerset AOI and return its geometry plus WGS84 bounds."""

    gdf = gpd.read_file(GEOJSON_PATH).to_crs("EPSG:4326")
    geometry = gdf.geometry.unary_union
    bounds = list(geometry.bounds)
    return geometry, bounds


T = TypeVar("T")


def _safe_execute(
    description: str,
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T | None:
    """Execute ``func`` while logging non-fatal errors and returning ``None`` on failure."""

    try:
        return func(*args, **kwargs)
    except (FileNotFoundError, ValueError) as exc:
        logging.warning("%s skipped: %s", description, exc)
    except Exception:
        logging.exception("%s failed unexpectedly", description)
    return None


def _expand_date_range(date_range: str, buffer_days: int) -> str:
    """Return a widened ISO8601 date interval string.

    Parameters
    ----------
    date_range:
        The original "YYYY-MM-DD/YYYY-MM-DD" interval string.
    buffer_days:
        Number of days to expand before the start date and after the end date.
    """

    start_str, end_str = date_range.split("/")
    start = datetime.strptime(start_str, "%Y-%m-%d") - timedelta(days=buffer_days)
    end = datetime.strptime(end_str, "%Y-%m-%d") + timedelta(days=buffer_days)
    return f"{start.date()}/{end.date()}"


def _landsat_reflectance_stack(src: rasterio.io.DatasetReader) -> list[np.ndarray]:
    """Convert raw Landsat digital numbers into reflectance arrays."""

    # Landsat Collection 2 Level-2 SR scale/offset values from USGS docs.
    scale = 0.0000275
    offset = -0.2

    bands = [src.read(i).astype(np.float32) for i in range(1, src.count + 1)]
    return [(band * scale + offset).astype(np.float32) for band in bands]


def _write_single_band(
    path: Path,
    profile: dict,
    array: np.ndarray,
    *,
    dtype: str,
    nodata: float | int | None = np.nan,
) -> None:
    """Persist a single-band raster array using the provided profile template."""

    profile = profile.copy()
    profile.update({"count": 1, "dtype": dtype})

    if nodata is None:
        profile.pop("nodata", None)
    else:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile, BIGTIFF="YES") as dst:
        dst.write(array.astype(dtype, copy=False), 1)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def process_landsat_history() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    geometry, bounds = _load_region_bounds()
    projected_bounds = reproject_bbox(geometry)

    logging.info("Loaded Somerset AOI with bounds %s", bounds)

    for mission, date_ranges in MISSION_DATE_RANGES.items():
        mission_cfg = get_mission(mission)
        mission_tag = mission.replace("-", "")

        logging.info("Processing %s (%d date ranges)", mission, len(date_ranges))

        for date_range in date_ranges:
            tag = date_range.replace("/", "_")
            mosaic_path = OUTPUT_ROOT / f"{mission_tag}_somerset_{tag}.tif"
            water_mask_path = mosaic_path.with_name(f"{mosaic_path.stem}_ndwi_mask.tif")
            salinity_score_path = mosaic_path.with_name(
                f"{mosaic_path.stem}_salinity_score.tif"
            )
            salinity_class_path = mosaic_path.with_name(
                f"{mosaic_path.stem}_salinity_class.tif"
            )
            salinity_water_mask_path = mosaic_path.with_name(
                f"{mosaic_path.stem}_salinity_water_mask.tif"
            )
            nlcd_path = OUTPUT_ROOT / f"nlcd_{mission_tag}_{tag}.tif"
            cdl_path = OUTPUT_ROOT / f"cdl_{mission_tag}_{tag}.tif"

            # ------------------------------------------------------------------
            # Step 1: Build/refresh the Landsat mosaic for this date range
            # ------------------------------------------------------------------
            expanded_date_range = _expand_date_range(
                date_range, DATE_RANGE_PADDING_DAYS
            )

            if not mosaic_path.exists():
                logging.info("Creating mosaic placeholder at %s", mosaic_path)
                mosaic_path.parent.mkdir(parents=True, exist_ok=True)
                create_mosaic_placeholder(
                    mosaic_path=mosaic_path,
                    bbox=projected_bounds,
                    mission=mission,
                    resolution=mission_cfg["resolution"],
                    crs="EPSG:32618",
                    dtype="float32",
                )

                logging.info(
                    "Downloading %s imagery for %s (expanded to %s)",
                    mission,
                    date_range,
                    expanded_date_range,
                )
                _safe_execute(
                    f"Downloading imagery for {mission} {date_range}",
                    patchwise_query_download_mosaic,
                    mosaic_path=mosaic_path,
                    bbox=bounds,
                    mission=mission,
                    resolution=mission_cfg["resolution"],
                    bands=mission_cfg["bands"],
                    date_range=expanded_date_range,
                    base_output_path=mosaic_path.parent,
                    to_disk=False,
                    multithreaded=False,
                    max_items=MAX_ITEMS_PER_PATCH,
                )
            else:
                logging.info("Reusing existing mosaic at %s", mosaic_path)

            if not mosaic_path.exists():
                logging.warning(
                    "No mosaic produced for %s on %s; skipping.", mission, date_range
                )
                continue

            # ------------------------------------------------------------------
            # Step 2: Generate an NDWI water mask
            # ------------------------------------------------------------------
            if not water_mask_path.exists():
                logging.info("Computing NDWI water mask -> %s", water_mask_path)
                _safe_execute(
                    f"Computing NDWI water mask for {mission} {tag}",
                    compute_ndwi,
                    mosaic_path,
                    mission,
                    out_path=water_mask_path,
                    threshold=0.2,
                )
            else:
                logging.info("Water mask already available at %s", water_mask_path)

            # ------------------------------------------------------------------
            # Step 3: Estimate salinity levels using the mosaic bands
            # ------------------------------------------------------------------
            if (
                not salinity_score_path.exists()
                or not salinity_class_path.exists()
                or not salinity_water_mask_path.exists()
            ):
                try:
                    with rasterio.open(mosaic_path) as src:
                        profile = src.profile
                        reflectance_bands = _landsat_reflectance_stack(src)
                except RasterioError as exc:
                    logging.warning(
                        "Unable to open mosaic %s for salinity estimation: %s",
                        mosaic_path,
                        exc,
                    )
                    continue

                salinity = _safe_execute(
                    f"Estimating salinity levels for {mission} {tag}",
                    estimate_salinity_level,
                    *reflectance_bands,
                    reflectance_scale=None,
                    water_threshold=0.2,
                )

                if salinity is None:
                    logging.warning("Salinity estimation skipped for %s", tag)
                    continue

                score = salinity["score"]
                class_map = salinity["class_map"]
                water_mask = salinity["water_mask"].astype(np.float32)

                logging.info("Writing salinity products for %s", tag)
                _write_single_band(salinity_score_path, profile, score, dtype="float32")

                class_codes = np.vectorize(
                    lambda value: SALINITY_CLASS_CODES.get(value, 0),
                    otypes=[np.uint8],
                )(class_map)
                _write_single_band(
                    salinity_class_path,
                    profile,
                    class_codes,
                    dtype="uint8",
                    nodata=255,
                )

                # Replace/augment the NDWI mask with the combined water mask signal.
                _write_single_band(
                    salinity_water_mask_path,
                    profile,
                    water_mask,
                    dtype="float32",
                    nodata=np.nan,
                )
            else:
                logging.info("Salinity rasters already exist for %s", tag)

            # ------------------------------------------------------------------
            # Step 4: Fetch NLCD & CDL overlays aligned to the date range year
            # ------------------------------------------------------------------
            year = datetime.fromisoformat(date_range.split("/")[0]).year

            if not nlcd_path.exists():
                logging.info("Requesting NLCD overlay for %s", year)
                _safe_execute(
                    f"Downloading NLCD overlay for {year}",
                    download_nlcd,
                    region=geometry,
                    year=year,
                    output_path=nlcd_path,
                )
            else:
                logging.info("NLCD overlay already cached at %s", nlcd_path)

            if not cdl_path.exists():
                logging.info("Requesting CDL overlay for %s", year)
                _safe_execute(
                    f"Downloading CDL overlay for {year}",
                    download_nass_cdl,
                    region=geometry,
                    year=year,
                    output_path=cdl_path,
                )
            else:
                logging.info("CDL overlay already cached at %s", cdl_path)

    logging.info("Processing complete. Outputs saved under %s", OUTPUT_ROOT)


if __name__ == "__main__":
    process_landsat_history()
