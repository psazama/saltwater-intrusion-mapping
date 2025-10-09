"""Pipeline entry point for generating water-trend heatmap products."""

from pathlib import Path

from swmaps.config import data_path
from swmaps.core.water_trend import (
    load_wet_year,
    pixel_trend,
    plot_trend_heatmap,
    save_trend_results,
)


def trend_heatmap(output_dir: str | Path = None) -> None:
    """Generate and save a trend heatmap from existing water masks.

    Args:
        output_dir (str | Path | None): Directory to search for masks and
            save outputs. Defaults to swmaps/data when not provided.

    Returns:
        None: Heatmap and trend rasters are written to the chosen directory.
    """
    if output_dir is None:
        search_dir = data_path()
    else:
        search_dir = Path(output_dir)
        Path(search_dir).mkdir(parents=True, exist_ok=True)

    mask_files = [str(p) for p in search_dir.glob("*_mask.tif")]
    if not mask_files:
        print(f"No mask files found in {search_dir}")
        return

    wet_year = load_wet_year(mask_files[::20], chunks={"x": 512, "y": 512})
    slope, pval = pixel_trend(wet_year)
    signif = pval < 0.05
    ax = plot_trend_heatmap(slope, signif, title="Trend in % wet months per year")

    heatmap_file = search_dir / "water_trend_heatmap.png"
    ax.figure.savefig(heatmap_file, bbox_inches="tight")
    print(f"Saved heatmap to {heatmap_file}")

    save_trend_results(slope, pval, search_dir / "water_trend")
