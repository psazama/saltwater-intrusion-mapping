############################################################
# swmaps/pipelines/assets.py
#
# • Partitions come from config/date_range.json (StaticPartitions)
# • Uses data_path helper → no hard‑coded "data/…" strings
# • Supports inline‑mask behaviour in process_date
############################################################

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from dagster import (
    StaticPartitionsDefinition,
    asset,
)

from swmaps.config import data_path
from swmaps.core.download_tools import get_mission, process_date

# ---------------------------------------------------------------------------
# 1. Partitions list  (exact same ranges as pipeline_runner)
# ---------------------------------------------------------------------------
_CFG_DIR = Path(__file__).resolve().parents[2] / "config"
with open(_CFG_DIR / "date_range.json", "r") as fh:
    _DATE_RANGES: list[str] = json.load(fh)["date_ranges"]

partitions_def = StaticPartitionsDefinition(_DATE_RANGES)

# ---------------------------------------------------------------------------
# 2. Study‑area bounding box (WGS84)
# ---------------------------------------------------------------------------
_GEOJSON = _CFG_DIR / "easternshore.geojson"
_gdf = gpd.read_file(_GEOJSON)
if _gdf.crs != "EPSG:4326":
    _gdf = _gdf.to_crs("EPSG:4326")
_BBOX = _gdf.total_bounds.tolist()

# ---------------------------------------------------------------------------
# 3. Mission configs + base mosaic filenames (no date suffix)
# ---------------------------------------------------------------------------
_SENTINEL = get_mission("sentinel-2")
_LS5 = get_mission("landsat-5")
_LS7 = get_mission("landsat-7")

_SENTINEL_TIF = data_path("sentinel_eastern_shore.tif")
_LS5_TIF = data_path("landsat5_eastern_shore.tif")
_LS7_TIF = data_path("landsat7_eastern_shore.tif")


# ---------------------------------------------------------------------------
# 4. Asset: one partition == one date‑range string from JSON
# ---------------------------------------------------------------------------
@asset(
    name="masks_by_range",
    partitions_def=partitions_def,
    io_manager_key="local_files",  # swap to gcs_io_manager in prod
)
def masks_by_range(context) -> list[str]:
    """Materialise NDWI water masks for a single date range (e.g. 1984‑03‑01/1984‑03‑31)."""

    date_range: str = context.partition_key  # already in the correct "start/end" form

    result = process_date(
        date=date_range,
        bbox=_BBOX,
        sentinel_mission=_SENTINEL,
        landsat5_mission=_LS5,
        landsat7_mission=_LS7,
        sentinel2_mosaic_path=_SENTINEL_TIF,
        landsat5_mosaic_path=_LS5_TIF,
        landsat7_mosaic_path=_LS7_TIF,
        inline_mask=True,
    )

    tag = date_range.replace("/", "_")
    mask_paths: list[str] = []
    for mission_prefix in ("sentinel", "landsat5", "landsat7"):
        candidate = data_path(f"{mission_prefix}_eastern_shore_{tag}_mask.tif")
        if candidate.exists():
            mask_paths.append(str(candidate))

    context.log.info(f"{len(mask_paths)} mask file(s) written for {date_range}")
    if result["errors"]:
        context.log.warning(f"process_date() raised errors: {result['errors']}")

    return mask_paths
