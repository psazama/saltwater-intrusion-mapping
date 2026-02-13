"""
GEE-based raster mosaicking utilities.

This replaces the old STAC-based per-band downloading and stacking
logic with Earth Engine imagery retrieval and export.

Outputs are still local multiband GeoTIFFs so the rest of the pipeline
( salinity extraction, NDWI, model features ) continues to work.
"""

from datetime import datetime, timedelta
from pathlib import Path

from tqdm import tqdm

from swmaps.config import data_path
from swmaps.core.missions import get_mission
from swmaps.core.satellite_query import (
    download_gee_multiband,
    get_best_image,
    query_gee_images,
    wait_for_ee_task,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def compute_bbox(lat, lon, buffer_km=1.0):
    """Return a bounding box around (lat, lon) with buffer_km distance."""
    deg = buffer_km / 111.0
    return [lon - deg, lat - deg, lon + deg, lat + deg]


def _write_rgb_png_from_tif(
    tif_path: Path,
    png_path: Path,
    rgb_bands: tuple[int, int, int],
    stretch: bool = True,
):
    """
    Save an RGB PNG from a multiband GeoTIFF.

    Args:
        tif_path: path to multiband GeoTIFF
        png_path: output PNG path
        rgb_bands: 1-based band indices, e.g. (4, 3, 2) for Sentinel-2
        stretch: apply min-max stretch for visualization
    """
    import numpy as np
    import rasterio
    from PIL import Image

    with rasterio.open(tif_path) as src:
        r = src.read(rgb_bands[0]).astype("float32")
        g = src.read(rgb_bands[1]).astype("float32")
        b = src.read(rgb_bands[2]).astype("float32")

    def _norm(x):
        if not stretch:
            return x
        lo, hi = np.percentile(x, (2, 98))
        if hi <= lo:
            return np.zeros_like(x)
        x = (x - lo) / (hi - lo)
        return np.clip(x, 0, 1)

    r = _norm(r)
    g = _norm(g)
    b = _norm(b)

    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb * 255).astype("uint8")

    Image.fromarray(rgb, mode="RGB").save(png_path)


# ---------------------------------------------------------------------
# Main GEE Mosaic Build
# ---------------------------------------------------------------------


def process_date(
    lat: float,
    lon: float,
    date: datetime,
    buffer_km: float = 1.0,
    mission: str = "sentinel-2",
    out_dir: str | Path | None = None,
    days_before: int = 7,
    days_after: int = 7,
    cloud_filter: float = 30,
    samples: int = 1,
    save_png: bool = False,
):
    """
    Build a local multiband GeoTIFF for the given location & date.

    This is the GEE-native replacement for the old STAC-based mosaic
    that used `_stack_bands()`.

    Args:
        lat, lon: center coordinates
        date: datetime object for target date
        buffer_km: bounding-box size
        mission: sentinel-2, landsat-5, landsat-7
        out_dir: where to write output raster
        days_before/days_after: temporal window
        cloud_filter: max cloud percentage allowed
        samples: number of samples to return

    Returns:
        path to the downloaded multiband TIFF
    """
    if out_dir is None:
        out_dir = data_path("mosaics")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build date window
    dt = date
    date_range = (
        f"{(dt - timedelta(days=days_before)).date()}/"
        f"{(dt + timedelta(days=days_after)).date()}"
    )

    bbox = compute_bbox(lat, lon, buffer_km)
    mission_info = get_mission(mission)

    # Default RGB band choices per mission, 1-based indices
    if mission == "sentinel-2":
        rgb_bands = (4, 3, 2)  # B4 red, B3 green, B2 blue
    elif mission in ("landsat-5", "landsat-7"):
        rgb_bands = (3, 2, 1)  # red, green, blue
    else:
        rgb_bands = None

    # Query ImageCollection
    col, band_map = query_gee_images(
        mission=mission,
        bbox=bbox,
        date_range=date_range,
        cloud_filter=cloud_filter,
    )

    size = col.size().getInfo()
    if size == 0:
        return None

    # Select best image
    imgs = get_best_image(col, mission, samples)
    if imgs is None:
        print(
            f"No usable image found after filtering for {mission} around {date_range}"
        )
        return None

    output_paths = []
    async_tasks = []  # Track async tasks that need to complete
    print(f"Begin processing mosaic count: {len(imgs)}")
    for img in imgs:
        # Export clipped multiband raster
        output_path, task = download_gee_multiband(
            image=img,
            mission=mission,
            bands=band_map,
            bbox=bbox,
            out_dir=out_dir,
            scale=mission_info.gee_scale,
        )

        output_paths.append(output_path)

        if task is not None:
            # Store task for later waiting
            async_tasks.append((output_path, task))
            print(f"[GEE] Started async export for: {output_path}")
        else:
            print(f"[GEE] Wrote mosaic to: {output_path}")

    # Wait for all async tasks to complete before proceeding
    if async_tasks:
        print(f"[GEE] Waiting for {len(async_tasks)} async task(s) to complete...")
        for output_path, task in async_tasks:
            try:
                wait_for_ee_task(task, timeout=3600, poll_interval=15)
                print(f"[GEE] Async export completed: {output_path}")
            except (TimeoutError, RuntimeError) as e:
                print(f"[GEE] Error waiting for task: {e}")
                # Remove failed path from output_paths
                if output_path in output_paths:
                    output_paths.remove(output_path)

    # Generate PNGs after all files are ready
    if save_png and rgb_bands is not None:
        for output_path in output_paths:
            png_path = Path(output_path).with_suffix(".png")
            try:
                _write_rgb_png_from_tif(
                    tif_path=Path(output_path),
                    png_path=png_path,
                    rgb_bands=rgb_bands,
                )
                print(f"[GEE] Wrote RGB preview to: {png_path}")
            except Exception as e:
                print(f"[GEE] Failed to create PNG for {output_path}: {e}")

    return output_paths


# ---------------------------------------------------------------------
# Batch processing for many dates/locations (optional API)
# ---------------------------------------------------------------------


def process_multiple(
    df,
    lat_col="latitude",
    lon_col="longitude",
    date_col="date",
    mission="sentinel-2",
    buffer_km=1.0,
    out_dir=None,
    days_before=7,
    days_after=7,
    cloud_filter=30,
    samples=1,
    save_png=False,
):
    """
    Apply GEE mosaic building to every row of a DataFrame.

    Returns a list of file paths.
    """
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        lat = row[lat_col]
        lon = row[lon_col]
        date = row[date_col]
        path = process_date(
            lat=lat,
            lon=lon,
            date=date,
            mission=mission,
            buffer_km=buffer_km,
            out_dir=out_dir,
            days_before=days_before,
            days_after=days_after,
            cloud_filter=cloud_filter,
            samples=samples,
            save_png=save_png,
        )
        results.append(path)
    return results
