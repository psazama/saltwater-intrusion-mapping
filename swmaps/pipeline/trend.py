"""Pipeline entry point for generating water-trend heatmap products."""

from swmaps.config import data_path
from swmaps.core.water_trend import (
    load_wet_year,
    pixel_trend,
    plot_trend_heatmap,
    save_trend_results,
)


def trend_heatmap() -> None:
    """Generate and save a trend heatmap from existing water masks.

    Args:
        None

    Returns:
        None: Heatmap and trend rasters are written to the data directory.
    """
    mask_files = [str(p) for p in data_path().glob("*_mask.tif")]
    if not mask_files:
        print("No mask files found")
        return

    wet_year = load_wet_year(mask_files[::20], chunks={"x": 512, "y": 512})
    slope, pval = pixel_trend(wet_year)
    signif = pval < 0.05
    ax = plot_trend_heatmap(slope, signif, title="Trend in % wet months per year")
    heatmap_file = data_path("water_trend_heatmap.png")
    ax.figure.savefig(heatmap_file, bbox_inches="tight")
    print(f"Saved heatmap to {heatmap_file}")
    save_trend_results(slope, pval, data_path("water_trend"))
