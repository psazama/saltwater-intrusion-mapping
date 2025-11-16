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
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _compute_bbox(lat, lon, buffer_km=1.0):
    """Return a bounding box around (lat, lon) with buffer_km distance."""
    deg = buffer_km / 111.0
    return [lon - deg, lat - deg, lon + deg, lat + deg]


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

    bbox = _compute_bbox(lat, lon, buffer_km)
    mission_info = get_mission(mission)

    print(f"[GEE] Querying {mission} for {lat:.4f}, {lon:.4f} @ {date_range}")

    # Query ImageCollection
    col, band_map = query_gee_images(
        mission=mission,
        bbox=bbox,
        date_range=date_range,
        cloud_filter=cloud_filter,
    )

    size = col.size().getInfo()
    if size == 0:
        raise RuntimeError(f"No GEE images found for {mission} around {date_range}")

    # Select best image
    img = get_best_image(col)
    if img is None:
        raise RuntimeError("No usable image found after filtering.")

    # Export clipped multiband raster
    output_path = download_gee_multiband(
        image=img,
        mission=mission,
        bands=band_map,
        bbox=bbox,
        out_dir=out_dir,
        scale=mission_info["gee_scale"],
    )

    print(f"[GEE] Wrote mosaic to: {output_path}")
    return output_path


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
        )
        results.append(path)
    return results
