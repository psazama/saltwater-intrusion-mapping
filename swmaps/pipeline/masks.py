"""Pipeline helpers for deriving NDWI water masks from mosaics.

Public functions
----------------
:func:`run_water_masks` - batch mask generation accepting a directory,
a single mosaic path, or an explicit list of paths.

:func:`generate_water_mask` - single-mosaic helper used by the processing
trigger (:mod:`swmaps.pipeline.run_pipeline`).

Both return :class:`~swmaps.schema.PipelineResult`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

from swmaps.core.indices import compute_ndwi
from swmaps.core.missions import get_mission_from_path
from swmaps.core.trend import check_image_for_nans, check_image_for_valid_signal
from swmaps.schema import PipelineResult

logger = logging.getLogger(__name__)


def generate_water_mask(mosaic: str | Path) -> PipelineResult:
    """Generate an NDWI water mask for a single mosaic GeoTIFF.

    Validates the mosaic for NaNs and signal before computing the mask.
    Writes a single-band float32 GeoTIFF next to the source file.

    Args:
        mosaic: Path to a multiband GeoTIFF produced by the download pipeline.

    Returns:
        PipelineResult: ``status="ok"`` with the mask path, or
        ``status="error"`` on validation failure or unrecognised mission.
    """
    mosaic = Path(mosaic)

    try:
        mission = get_mission_from_path(mosaic).slug
    except ValueError as exc:
        return PipelineResult.failure(
            f"Cannot infer mission from path '{mosaic.name}': {exc}"
        )

    if check_image_for_nans(str(mosaic)):
        return PipelineResult.failure(f"Mosaic {mosaic.name} contains NaNs - skipping.")
    if not check_image_for_valid_signal(str(mosaic)):
        return PipelineResult.failure(
            f"Mosaic {mosaic.name} has no valid signal - skipping."
        )

    out_mask = mosaic.with_name(f"{mosaic.stem}_mask.tif")
    compute_ndwi(str(mosaic), mission, str(out_mask), display=False)

    return PipelineResult.ok([out_mask], mission=mission)


def run_water_masks(
    mosaics: Path | list[Path],
    center_size: int | None = None,
    write_png: bool = True,
) -> PipelineResult:
    """Generate NDWI water masks for one or more mosaics.

    Accepts a directory path (searched recursively for ``*_multiband.tif``),
    a single mosaic path, or an explicit list of paths - matching the same
    interface as :func:`~swmaps.pipeline.salinity.run_salinity_classification`.

    Args:
        mosaics: Directory to search, single mosaic path, or list of paths.
        center_size: When set, only the central square of this many pixels
            is used for NDWI computation.
        write_png: If ``True``, write a grayscale PNG preview alongside
            each mask GeoTIFF.

    Returns:
        PipelineResult: ``status="ok"`` with all written paths, or
        ``status="error"`` when no mosaics are found. Per-file failures
        are logged and counted in ``result.meta["skipped"]``.
    """
    # Resolve argument into a flat list
    if isinstance(mosaics, list):
        mosaic_list = mosaics
    elif mosaics.is_dir():
        mosaic_list = [
            p
            for p in sorted(mosaics.rglob("*_multiband.tif"))
            if not p.name.endswith(("_mask.tif", "_features.tif"))
        ]
    else:
        mosaic_list = [mosaics]

    if not mosaic_list:
        return PipelineResult.failure(
            f"No multiband mosaics found in {mosaics}",
        )

    output_paths: list[Path] = []
    skipped = 0

    for tif in tqdm(mosaic_list, desc="Generating water masks"):
        if check_image_for_nans(str(tif)) or not check_image_for_valid_signal(str(tif)):
            skipped += 1
            continue

        try:
            mission = get_mission_from_path(tif).slug
        except ValueError:
            logger.debug("Cannot infer mission for %s - skipping", tif.name)
            skipped += 1
            continue

        out_mask = tif.with_name(f"{tif.stem}_mask.tif")
        compute_ndwi(
            str(tif),
            mission,
            str(out_mask),
            display=False,
            center_size=center_size,
        )
        output_paths.append(out_mask)

        if write_png:
            png_path = out_mask.with_suffix(".png")
            with rasterio.open(out_mask) as src:
                mask_arr = src.read(1)
            mask_png = np.where(mask_arr > 0, 255, 0).astype(np.uint8)
            Image.fromarray(mask_png, mode="L").save(png_path)
            output_paths.append(png_path)

    logger.info(
        "Water masks: %d generated, %d skipped.",
        len(mosaic_list) - skipped,
        skipped,
    )

    return PipelineResult.ok(
        output_paths,
        total=len(mosaic_list),
        processed=len(mosaic_list) - skipped,
        skipped=skipped,
    )
