"""Pipeline entry point for generating class-trend heatmap products."""

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


def discover_masks(
    root: Path,
    *,
    name_contains: Union[str, Iterable[str]] = "_mask",
) -> list[Path]:
    """Discover masks in a directory.

    Parameters
    ----------
    root : Path
        Root directory to search.
    name_contains : str or iterable of str, optional
        Substring(s) to match in filenames. Defaults to "_mask".

    Returns
    -------
    list[Path]
        Sorted list of matching files.
    """
    if isinstance(name_contains, str):
        keys = [name_contains.lower()]
    else:
        keys = [k.lower() for k in name_contains]

    matches: list[Path] = []
    for p in root.rglob("*.tif"):
        fname = p.name.lower()
        if any(k in fname for k in keys):
            matches.append(p)

    return sorted(matches)


def trend_heatmap(
    output_dir: Union[str, Path] = None,
    class_value: int | float = 1,
    class_name: str = "mask",
) -> None:
    """Generate and save a trend heatmap from class masks.

    Args:
        output_dir (str | Path | None): Directory to search for masks and
            save outputs. Defaults to swmaps/data when not provided.
        class_value (int | float): Pixel value to track in the masks.
        class_name (str): Substring to filter masks (e.g., "mask", "segmentation").

    Returns:
        None
    """
    # Prepare search directory
    if output_dir is None:
        search_dir = data_path()
    else:
        search_dir = Path(output_dir)
        search_dir.mkdir(parents=True, exist_ok=True)

    # Discover masks
    mask_files = discover_masks(search_dir, name_contains=class_name)
    if not mask_files:
        print(f"No masks found in {search_dir} containing '{class_name}'")
        return

    # Convert masks to binary for the target class
    binary_mask_files = []
    for f in mask_files:
        with rasterio.open(f) as src:
            data = src.read(1)
            bin_data = (data == class_value).astype(np.uint8)

            temp_path = Path(f.parent) / f"{f.stem}_binary.tif"
            with rasterio.open(
                temp_path,
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

            binary_mask_files.append(str(temp_path))

    # Load yearly fraction for the class
    class_year = load_class_year(binary_mask_files)

    # Compute trends
    slope, pval = pixel_trend(class_year)
    signif = pval < 0.05

    # Plot heatmap
    ax = plot_trend_heatmap(
        slope,
        signif,
        title=f"Trend in {class_name} (value={class_value}) per year",
    )

    # Save outputs
    heatmap_file = search_dir / f"{class_name}_trend_heatmap.png"
    ax.figure.savefig(heatmap_file, bbox_inches="tight")
    print(f"Saved heatmap to {heatmap_file}")

    save_trend_results(slope, pval, search_dir / f"{class_name}_trend")
    print(f"Trend rasters saved to {search_dir / f'{class_name}_trend'}")
