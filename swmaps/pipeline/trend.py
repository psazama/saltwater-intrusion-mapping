"""Pipeline entry point for generating class-trend heatmap products.

Public function: :func:`run_trend_heatmap`.

Returns a :class:`~swmaps.schema.PipelineResult` consistent with all other
pipeline functions in this package.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import rasterio

from swmaps.config import data_path
from swmaps.core.trend import (
    load_class_year,
    pixel_trend,
    plot_trend_heatmap,
    save_trend_results,
)
from swmaps.schema import PipelineResult, TrendConfig

logger = logging.getLogger(__name__)


def discover_masks(
    root: Path,
    *,
    name_contains: Union[str, Iterable[str]] = "_mask",
) -> list[Path]:
    """Discover mask GeoTIFFs in a directory tree.

    Args:
        root: Root directory to search recursively.
        name_contains: Substring or iterable of substrings to match against
            filenames (case-insensitive). Defaults to ``"_mask"``.

    Returns:
        list[Path]: Sorted list of matching ``.tif`` files.
    """
    keys = (
        [name_contains.lower()]
        if isinstance(name_contains, str)
        else [k.lower() for k in name_contains]
    )
    return sorted(
        p for p in root.rglob("*.tif") if any(k in p.name.lower() for k in keys)
    )


def run_trend_heatmap(
    cfg: TrendConfig,
    output_dir: Path | None = None,
) -> PipelineResult:
    """Compute and save a per-pixel trend heatmap from class masks.

    Loads all mask files found in *output_dir*, converts them to binary
    arrays for :attr:`~TrendConfig.trend_class_value`, runs Theil-Sen
    slope and Mann-Kendall significance per pixel, saves the heatmap PNG
    and trend GeoTIFFs, and returns a :class:`~swmaps.schema.PipelineResult`.

    Args:
        cfg: Trend configuration object.
        output_dir: Directory to search for masks and write outputs. When
            ``None``, falls back to :attr:`~TrendConfig.trend_output_dir`
            in *cfg*, then to the project data root.

    Returns:
        PipelineResult: ``status="ok"`` with all written file paths,
        ``status="skipped"`` when the config flag is off, or
        ``status="error"`` if no masks are found.
    """
    if not cfg.run_water_trend:
        return PipelineResult.skipped("run_water_trend is False in config")

    search_dir = Path(output_dir or cfg.trend_output_dir or data_path())
    search_dir.mkdir(parents=True, exist_ok=True)

    mask_files = discover_masks(search_dir, name_contains=cfg.trend_class_name)
    if not mask_files:
        return PipelineResult.failure(
            f"No masks containing '{cfg.trend_class_name}' found "
            f"under {search_dir}.",
        )

    # Convert masks to binary for the target class
    binary_paths: list[str] = []
    for f in mask_files:
        with rasterio.open(f) as src:
            data = src.read(1)
            bin_data = (data == cfg.trend_class_value).astype(np.uint8)
            tmp_path = f.parent / f"{f.stem}_binary.tif"
            with rasterio.open(
                tmp_path,
                "w",
                driver="GTiff",
                height=bin_data.shape[0],
                width=bin_data.shape[1],
                count=1,
                dtype=bin_data.dtype,
                crs=src.crs,
                transform=src.transform,
            ) as dst:
                dst.write(bin_data, 1)
        binary_paths.append(str(tmp_path))

    class_year = load_class_year(binary_paths)
    slope, pval = pixel_trend(class_year)
    signif = pval < 0.05

    # Plot heatmap
    ax = plot_trend_heatmap(
        slope,
        signif,
        title=f"Trend in {cfg.trend_class_name} "
        f"(value={cfg.trend_class_value}) per year",
    )

    heatmap_path = search_dir / f"{cfg.trend_class_name}_trend_heatmap.png"
    ax.figure.savefig(heatmap_path, bbox_inches="tight")
    logger.info("Trend heatmap saved to %s", heatmap_path)

    stem = search_dir / f"{cfg.trend_class_name}_trend"
    slope_tif, pval_tif = save_trend_results(slope, pval, stem)

    return PipelineResult.ok(
        [heatmap_path, slope_tif, pval_tif],
        mask_count=len(mask_files),
    )
