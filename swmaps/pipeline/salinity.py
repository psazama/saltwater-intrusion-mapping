"""Pipeline functions for salinity ground-truth processing and per-mosaic classification.

This module is the only layer that touches the filesystem on behalf of the
salinity model. It owns:

- :func:`run_salinity_pipeline` - downloads CODC truth data, matches in-situ
  casts to satellite imagery, and produces ``ground_truth_matches.csv``.
- :func:`run_salinity_classification` - iterates over mosaic GeoTIFFs, calls
  :meth:`~swmaps.models.salinity_heuristic.SalinityHeuristicModel.predict` on
  each, and writes score / class / water-mask rasters.

Both functions return a :class:`~swmaps.schema.PipelineResult`.

Model contract
--------------
The salinity model is pure - it only touches in-memory arrays::

    model = SalinityHeuristicModel()
    result = model.predict(bands)   # bands dict in, result dict out - no I/O

This module is responsible for all filesystem access: opening mosaics,
calling ``mission.read_bands(src)`` to get the band dict, then writing
``result["score"]``, ``result["class_codes"]``, and ``result["water_mask"]``
to raster files via the private :func:`_write_band` helper.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioError

from swmaps.config import data_path
from swmaps.core.missions import get_mission_from_path
from swmaps.core.satellite_query import (
    download_matching_gee_images as _download_matching,
)
from swmaps.core.satellite_query import find_gee_coverage as _find_coverage
from swmaps.core.trend import check_image_for_nans, check_image_for_valid_signal
from swmaps.datasets.salinity import (
    build_salinity_truth,
    download_salinity_datasets,
    load_salinity_truth,
)
from swmaps.models.salinity_heuristic import (
    SalinityHeuristicModel,
)
from swmaps.schema import PipelineResult, SalinityConfig

logger = logging.getLogger(__name__)


def _write_band(
    path: Path,
    profile: dict,
    array: np.ndarray,
    dtype: str,
    nodata=np.nan,
) -> None:
    """Write a single-band array to a GeoTIFF using a template profile.

    Args:
        path: Output file path.
        profile: Rasterio profile dict (copied internally before modification).
        array: 2-D array to write.
        dtype: Rasterio dtype string, e.g. ``"float32"`` or ``"uint8"``.
        nodata: Nodata value written to the profile.
    """
    profile = profile.copy()
    profile.update(count=1, dtype=dtype)
    if nodata is None:
        profile.pop("nodata", None)
    else:
        profile["nodata"] = nodata
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile, BIGTIFF="YES") as dst:
        dst.write(array.astype(dtype, copy=False), 1)


def run_salinity_classification(
    cfg: SalinityConfig,
    mosaics: Path | list[Path],
) -> PipelineResult:
    """Run per-mosaic heuristic salinity classification and write rasters.

    For each multiband GeoTIFF this function:

    1. Opens the file and calls ``mission.read_bands(src)`` to get a band
       dict in ``[0, 1]`` reflectance units.
    2. Calls :meth:`~swmaps.models.salinity_heuristic.SalinityHeuristicModel.predict`
       on the band dict - no I/O inside the model.
    3. Writes three output rasters next to each source mosaic:
       ``<stem>_salinity_score.tif``, ``<stem>_salinity_class.tif``,
       ``<stem>_salinity_water_mask.tif``.

    Args:
        cfg: Salinity configuration.
        mosaics: Either a directory path to search recursively for
            ``*_multiband.tif`` files, or a single mosaic path, or a list
            of mosaic paths.

    Returns:
        PipelineResult: ``status="ok"`` with all written file paths, or
        ``status="error"`` if no mosaics were found. Per-file failures are
        logged and counted in ``result.meta["skipped"]``.
    """
    if not cfg.run_salinity_classification:
        return PipelineResult.skipped("run_salinity_classification is False in config")

    # Resolve mosaics argument into a flat list of paths
    if isinstance(mosaics, list):
        mosaic_list = mosaics
    elif mosaics.is_dir():
        mosaic_list = [
            p
            for p in sorted(mosaics.rglob("*_multiband.tif"))
            if not any(
                x in p.name for x in ["_score", "_class", "_mask", "aligned_cdl"]
            )
        ]
    else:
        mosaic_list = [mosaics]

    if not mosaic_list:
        return PipelineResult.failure(
            f"No multiband mosaics found in {mosaics}",
        )

    model = SalinityHeuristicModel()
    output_paths: list[Path] = []
    skipped = 0

    for mosaic_path in mosaic_list:
        if check_image_for_nans(str(mosaic_path)) or not check_image_for_valid_signal(
            str(mosaic_path)
        ):
            logger.debug("Skipping invalid mosaic: %s", mosaic_path.name)
            skipped += 1
            continue

        try:
            mission = get_mission_from_path(mosaic_path)
        except ValueError:
            logger.debug("Cannot infer mission for %s - skipping", mosaic_path.name)
            skipped += 1
            continue

        try:
            with rasterio.open(mosaic_path) as src:
                if src.count < 6:
                    logger.warning(
                        "Expected ≥6 bands, found %d in %s - skipping.",
                        src.count,
                        mosaic_path.name,
                    )
                    skipped += 1
                    continue
                bands = mission.read_bands(src)
                profile = src.profile
        except RasterioError as exc:
            logger.warning("Cannot open %s: %s", mosaic_path.name, exc)
            skipped += 1
            continue

        try:
            result = model.predict(bands, water_threshold=cfg.water_threshold)
        except Exception as exc:
            logger.warning("predict() failed for %s: %s", mosaic_path.name, exc)
            skipped += 1
            continue

        base = mosaic_path.with_suffix("")
        score_path = base.with_name(f"{base.stem}_salinity_score.tif")
        class_path = base.with_name(f"{base.stem}_salinity_class.tif")
        water_path = base.with_name(f"{base.stem}_salinity_water_mask.tif")

        _write_band(score_path, profile, result["score"], dtype="float32")
        _write_band(
            class_path, profile, result["class_codes"], dtype="uint8", nodata=255
        )
        _write_band(
            water_path,
            profile,
            result["water_mask"].astype(np.float32),
            dtype="float32",
        )

        output_paths.extend([score_path, class_path, water_path])
        logger.info("Salinity products written for %s", mosaic_path.name)

    if not output_paths and skipped == len(mosaics):
        return PipelineResult.failure(
            "All mosaics were skipped (band count, quality, or mission errors).",
        )

    return PipelineResult.ok(
        output_paths,
        total=len(mosaic_list),
        processed=len(mosaic_list) - skipped,
        skipped=skipped,
    )


def run_salinity_pipeline(cfg: SalinityConfig) -> PipelineResult:
    """Download CODC truth data, match to satellite imagery, and export a CSV.

    Steps:

    1. Download CODC NetCDF files (skipped when
       :attr:`~SalinityConfig.truth_download_list` is ``None``).
    2. Build the flat salinity truth CSV from downloaded ``.nc`` files
       (skipped when :attr:`~SalinityConfig.truth_dir` is ``None``).
    3. Find satellite coverage for each in-situ cast location using GEE.
    4. Download co-located multiband GeoTIFFs.
    5. Write ``ground_truth_matches.csv`` to the data root.

    Args:
        cfg: Salinity configuration.

    Returns:
        PipelineResult: ``status="ok"`` with the path to the matches CSV, or
        ``status="error"`` if no imagery was matched.
    """
    if not cfg.run_salinity_pipeline:
        return PipelineResult.skipped("run_salinity_pipeline is False in config")

    if cfg.truth_download_list:
        dest = cfg.truth_dir or str(data_path("salinity_labels", "codc"))
        download_salinity_datasets(cfg.truth_download_list, dest)

    truth_file = cfg.truth_file or str(
        data_path("salinity_labels", "codc_salinity_profiles.csv")
    )

    if cfg.truth_dir:
        codc_files = sorted(Path(cfg.truth_dir).glob("*.nc"))
        build_salinity_truth(codc_files, truth_file)

    y = load_salinity_truth(truth_file)
    overlapped = _find_coverage(y)
    overlapped = overlapped[overlapped["covered_by"].apply(len) > 0]

    matches = _download_matching(overlapped)

    csv_path = data_path("ground_truth_matches.csv")
    matches.to_csv(csv_path, index=False)
    logger.info("Ground truth matches written to %s", csv_path)

    if matches.empty:
        return PipelineResult.failure(
            "No matched imagery found. "
            "Try increasing buffer_km or days_before/days_after.",
        )

    return PipelineResult.ok(
        [Path(csv_path)],
        match_count=len(matches),
        truth_file=truth_file,
    )
