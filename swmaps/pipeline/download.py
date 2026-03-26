"""Download helpers for acquiring imagery used by the processing pipeline.

Public functions
----------------
:func:`run_download` - primary entry point accepting a typed
:class:`~swmaps.schema.DownloadConfig` and returning a
:class:`~swmaps.schema.PipelineResult`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

from tqdm import tqdm

from swmaps.config import data_path
from swmaps.core.mosaic import process_date
from swmaps.schema import DownloadConfig, PipelineResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _daterange(start: datetime, end: datetime, step_days: int = 1):
    """Yield successive dates from *start* to *end* in *step_days* increments.

    Args:
        start: First date to yield.
        end: Last date to yield (inclusive).
        step_days: Number of days between consecutive yields.

    Yields:
        datetime: Successive dates.
    """
    current = start
    while current <= end:
        yield current
        current = current + timedelta(days=step_days)


def _extract_coords_from_geojson(geojson_path: str | Path) -> tuple[float, float]:
    """Extract the centroid latitude and longitude from a GeoJSON file.

    Args:
        geojson_path: Path to a GeoJSON file.

    Returns:
        tuple[float, float]: ``(latitude, longitude)`` of the union centroid.
    """
    import geopandas as gpd

    gdf = gpd.read_file(geojson_path)
    centroid = gdf.union_all().centroid
    return centroid.y, centroid.x


# ---------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------


def run_download(cfg: DownloadConfig) -> PipelineResult:
    """Download imagery for all missions and dates specified in *cfg*.

    Iterates over every (mission, date) combination defined by the config,
    calls :func:`~swmaps.core.mosaic.process_date` for each, and collects
    output mosaic paths.

    Args:
        cfg: Typed download configuration.

    Returns:
        PipelineResult: ``status="ok"`` with all downloaded mosaic paths,
        ``status="skipped"`` when ``skip_download`` is set, or
        ``status="error"`` on a configuration problem.
    """
    if cfg.skip_download:
        return PipelineResult.skipped("skip_download is True in config")

    if cfg.geometry:
        geojson_file = Path(cfg.geometry)
        if not geojson_file.exists():
            return PipelineResult.failure(f"GeoJSON file not found: {cfg.geometry}")
        lat, lon = _extract_coords_from_geojson(cfg.geometry)
        lats = [lat]
        lons = [lon]
    else:
        lats = cfg.latitude if isinstance(cfg.latitude, list) else [cfg.latitude]
        lons = cfg.longitude if isinstance(cfg.longitude, list) else [cfg.longitude]

    start_date = datetime.fromisoformat(cfg.start_date)
    end_date = datetime.fromisoformat(cfg.end_date)
    out_dir = Path(cfg.out_dir) if cfg.out_dir else data_path("downloads")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[GEE] Downloading %s - %s | missions: %s | out: %s",
        cfg.start_date,
        cfg.end_date,
        cfg.mission,
        out_dir,
    )

    results: list[Path] = []

    for mission in cfg.mission:
        mission_out_dir = out_dir / mission
        mission_out_dir.mkdir(parents=True, exist_ok=True)

        # Sentinel-2 has ~5 day revisit - wider windows add cloud-cover risk
        days_before = 7 if "sentinel" in mission else cfg.days_before
        days_after = 7 if "sentinel" in mission else cfg.days_after

        for lat, lon in zip(lats, lons):
            for date in tqdm(
                _daterange(start_date, end_date, cfg.date_step),
                desc=f"[GEE] {mission} | lat={lat:.2f}, lon={lon:.2f}",
            ):
                output_path = process_date(
                    lat=lat,
                    lon=lon,
                    date=date,
                    buffer_km=cfg.buffer_km,
                    mission=mission,
                    out_dir=mission_out_dir,
                    days_before=days_before,
                    days_after=days_after,
                    cloud_filter=cfg.cloud_filter,
                    samples=cfg.samples_per_date,
                    save_png=cfg.save_png,
                )
                if isinstance(output_path, list):
                    results.extend([Path(p) for p in output_path if p])
                elif output_path:
                    results.append(Path(output_path))

    logger.info(
        "[GEE] Done. %d mosaics across %d mission(s).",
        len(results),
        len(cfg.mission),
    )
    return PipelineResult.ok(
        results,
        mosaic_count=len(results),
        mission_count=len(cfg.mission),
    )
