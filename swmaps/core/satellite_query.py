"""
Earth Engine based querying and downloading of satellite imagery.

This module replaces all STAC, AWS, rasterio-session, and per-band
downloading logic with clean GEE-native helpers.

Downstream code simply receives local multiband GeoTIFFs, same as before.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.merge import merge
from tqdm import tqdm

from swmaps.config import data_path

from .missions import get_mission

# ---------------------------------------------------------
#  EARTH ENGINE INITIALIZATION
# ---------------------------------------------------------


def initialize_ee():
    """Safely initialize Earth Engine, falling back to project from env."""
    try:
        ee.Initialize()
    except ee.EEException:
        project = os.environ.get("EARTHENGINE_PROJECT")
        if not project:
            raise ValueError(
                "Earth Engine initialization failed and EARTHENGINE_PROJECT is not set."
            )
        ee.Initialize(project=project)


# ---------------------------------------------------------
#  GEE QUERY HELPERS
# ---------------------------------------------------------


def query_gee_images(
    mission: str,
    bbox: list[float],
    date_range: str,
    cloud_filter: float | None = None,
):
    """
    Query a mission’s ImageCollection in GEE.

    Args:
        mission (str): Mission slug (sentinel-2, landsat-5, ...)
        bbox (list): [minx, miny, maxx, maxy]
        date_range (str): "YYYY-MM-DD/YYYY-MM-DD"
        cloud_filter (float): Optional cloud threshold

    Returns:
        (ee.ImageCollection, list[str])
    """
    initialize_ee()

    mission_info = get_mission(mission)
    collection_id = mission_info.gee_collection
    bands = mission_info.bands()

    start, end = date_range.split("/")
    region = ee.Geometry.BBox(*bbox)

    col = ee.ImageCollection(collection_id).filterBounds(region).filterDate(start, end)

    if cloud_filter:
        if mission.startswith("sentinel"):
            col = col.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_filter))
        elif mission.startswith("landsat"):
            col = col.filter(ee.Filter.lte("CLOUD_COVER", cloud_filter))

    return col, bands


def get_best_image(collection: ee.ImageCollection, mission: str, samples: int):
    """Pick the 'best' image: lowest cloud cover, most recent.

    Args:
        collection (ee.ImageCollection): The collection to search.
        mission (str): Mission slug (e.g. "sentinel-2", "landsat-5").
        samples (int): Number of samples to return

    Returns:
        ee.Image | None: The selected image or None if no images exist.
    """

    # If the collection is empty, return None
    size = collection.size().getInfo()
    if size == 0:
        return None

    # TODO: move to missions subclasses
    # Pick the correct cloud‐cover property based on the mission
    if mission.startswith("sentinel"):
        cloud_property = "CLOUDY_PIXEL_PERCENTAGE"
    else:
        cloud_property = "CLOUD_COVER"

    collection = collection.map(
        lambda img: img.set(
            "sort_key",
            ee.Number(img.get(cloud_property))
            .multiply(1e15)
            .subtract(ee.Number(img.get("system:time_start"))),
        )
    )

    # Clamp samples to collection size
    n = ee.Number(samples).min(size)

    image_list = collection.sort("sort_key").toList(samples)

    n_client = n.getInfo()
    images = [ee.Image(image_list.get(i)) for i in range(n_client)]
    return images


# ---------------------------------------------------------
#  TILE AND MERGE HELPERS FOR LARGE IMAGES
# ---------------------------------------------------------


def tile_and_download_gee_image(
    image: ee.Image,
    bbox: list[float],
    scale: int,
    crs: str,
    out_dir: Path,
    prefix: str,
    n_tiles: int = 2,
) -> list[Path]:
    """
    Split a large image into tiles and download each synchronously.

    Args:
        image: The clipped GEE image to download
        bbox: [minx, miny, maxx, maxy]
        scale: Resolution in meters
        crs: Target CRS
        out_dir: Directory to save tile files
        prefix: Filename prefix for tiles
        n_tiles: Number of tiles per dimension (default 2x2 grid)

    Returns:
        List of paths to downloaded tile GeoTIFFs
    """
    minx, miny, maxx, maxy = bbox

    # Calculate tile boundaries
    width = (maxx - minx) / n_tiles
    height = (maxy - miny) / n_tiles

    tile_paths = []

    print(f"[GEE] Splitting into {n_tiles}x{n_tiles} grid ({n_tiles*n_tiles} tiles)")

    for row in range(n_tiles):
        for col in range(n_tiles):
            tile_minx = minx + col * width
            tile_miny = miny + row * height
            tile_maxx = tile_minx + width
            tile_maxy = tile_miny + height

            tile_bbox = [tile_minx, tile_miny, tile_maxx, tile_maxy]
            tile_region = ee.Geometry.BBox(*tile_bbox)

            tile_name = f"{prefix}_tile_r{row}_c{col}.tif"
            tile_path = out_dir / tile_name

            print(f"[GEE] Downloading tile ({row},{col})...")

            try:
                url = image.getDownloadURL(
                    {
                        "scale": scale,
                        "crs": crs,
                        "region": tile_region,
                        "format": "GEOTIFF",
                        "filePerBand": False,
                    }
                )

                r = requests.get(url, stream=True)
                r.raise_for_status()

                with open(tile_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

                tile_paths.append(tile_path)
                print(f"[GEE] ✓ Tile ({row},{col}) saved to {tile_path.name}")

            except Exception as e:
                print(f"[GEE] ✗ Failed to download tile ({row},{col}): {e}")
                raise

    return tile_paths


def merge_geotiff_tiles(tile_paths: list[Path], output_path: Path) -> Path:
    """
    Merge a list of GeoTIFF tiles into a single multiband GeoTIFF.

    Args:
        tile_paths: List of paths to tile GeoTIFFs
        output_path: Path for the merged output file

    Returns:
        Path to the merged GeoTIFF
    """
    print(f"[GEE] Merging {len(tile_paths)} tiles into {output_path.name}")

    # Open all tiles
    src_files = []
    for tile_path in tile_paths:
        src = rasterio.open(tile_path)
        src_files.append(src)

    # Merge tiles into a single array
    mosaic, transform = merge(src_files)

    # Get metadata from the first tile
    meta = src_files[0].meta.copy()
    meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
        }
    )

    # Write merged output
    with rasterio.open(output_path, "w", **meta) as dest:
        dest.write(mosaic)

    # Close all source files
    for src in src_files:
        src.close()

    # Clean up tile files
    for tile_path in tile_paths:
        try:
            tile_path.unlink()
            print(f"[GEE] Cleaned up {tile_path.name}")
        except Exception as e:
            print(f"[GEE] Warning: Could not delete {tile_path.name}: {e}")

    print(f"[GEE] ✓ Merged output saved to {output_path.name}")
    return output_path


# ---------------------------------------------------------
#  DOWNLOAD MULTIBAND IMAGE FROM GEE
# ---------------------------------------------------------


def download_gee_multiband(
    image: ee.Image,
    mission: str,
    bands: dict[str, str],
    bbox: list[float],
    out_dir: Path,
    scale: int = 10,
):
    """
    Export a clipped, multiband GeoTIFF for the given GEE image.
    
    For large Sentinel-2 images (>30 MB), automatically splits the request
    into smaller tiles, downloads each synchronously, and merges them into
    a single GeoTIFF. This avoids async Drive exports and ensures files
    are immediately available on the local filesystem.
    
    Args:
        image: GEE Image to download
        mission: Mission slug (e.g., "sentinel-2")
        bands: Dictionary mapping band names
        bbox: [minx, miny, maxx, maxy]
        out_dir: Directory to save output
        scale: Resolution in meters (default 10)
        
    Returns:
        str: Path to the downloaded (or merged) GeoTIFF
    """
    initialize_ee()
    out_dir.mkdir(parents=True, exist_ok=True)

    band_list = list(bands.values())
    region = ee.Geometry.BBox(*bbox)

    clipped = image.select(band_list)

    mission_info = get_mission(mission)
    ANALYSIS_CRS = mission_info.gee_crs

    clipped = clipped.reproject(
        crs=ANALYSIS_CRS,
        scale=scale,
    )
    clipped = clipped.clip(region)

    # -------------------------------------------------
    # Size estimation
    # -------------------------------------------------
    minx, miny, maxx, maxy = bbox

    # Rough meters per degree
    meters_per_deg = 111_000

    width_m = (maxx - minx) * meters_per_deg
    height_m = (maxy - miny) * meters_per_deg

    width_px = int(width_m / scale)
    height_px = int(height_m / scale)

    n_pixels = width_px * height_px
    n_bands = len(band_list)

    bytes_per_pixel = 2  # UInt16
    est_bytes = n_pixels * n_bands * bytes_per_pixel
    est_mb = est_bytes / (1024 * 1024)

    print(
        f"[GEE] Estimated download size for {mission}: "
        f"{width_px} x {height_px} px, "
        f"{n_bands} bands, ~{est_mb:.1f} MB"
    )

    image_id = image.get("system:index").getInfo()
    out_path = out_dir / f"{mission}_{image_id}_multiband.tif"

    # -------------------------------------------------
    # Large image handling with tiling
    # -------------------------------------------------
    if mission.startswith("sentinel") and est_mb > 30:
        print(
            f"[GEE] Size exceeds 30 MB for Sentinel-2, "
            f"splitting into tiles for synchronous download"
        )

        # Determine grid size based on estimated size
        # For ~30 MB threshold, 2x2 grid reduces each tile to ~1/4 size
        n_tiles = 2
        if est_mb > 120:  # Very large images need more tiles
            n_tiles = 3
        if est_mb > 270:  # Extremely large images
            n_tiles = 4

        prefix = f"{mission}_{image_id}"
        tile_paths = tile_and_download_gee_image(
            clipped, bbox, scale, ANALYSIS_CRS, out_dir, prefix, n_tiles
        )

        # Merge tiles into final output
        merge_geotiff_tiles(tile_paths, out_path)

        return str(out_path)

    # -------------------------------------------------
    # Synchronous download path (for small images)
    # -------------------------------------------------
    url = clipped.getDownloadURL(
        {
            "scale": scale,
            "crs": ANALYSIS_CRS,
            "region": region,
            "format": "GEOTIFF",
            "filePerBand": False,
        }
    )

    r = requests.get(url, stream=True)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    return str(out_path)


# ---------------------------------------------------------
#  COVERAGE CHECKING (FOR CLUSTERS)
# ---------------------------------------------------------


def find_gee_coverage(
    df: pd.DataFrame,
    missions: list[str] = ["sentinel-2", "landsat-5", "landsat-7"],
    buffer_km: float = 10,
    days_before: int = 7,
    days_after: int = 7,
    temporal_granularity: str = "M",
):
    """
    Mirror of the old STAC-based `find_satellite_coverage`, but using GEE.

    Returns:
        DataFrame with `covered_by` list per cluster.
    """
    initialize_ee()

    # ---- Same clustering logic as before ----

    buffer_deg = buffer_km / 111.0

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    gdf["date"] = pd.to_datetime(gdf["date"])

    gdf_proj = gdf.to_crs("EPSG:3857")
    gdf_proj["buffer"] = gdf_proj.geometry.buffer(buffer_km * 1000)
    gdf["buffer"] = gdf_proj.to_crs("EPSG:4326")["buffer"]

    dissolved = (
        gdf.set_geometry("buffer")
        .dissolve()
        .explode(index_parts=True)
        .reset_index(drop=True)
    )

    # Assign spatial clusters
    cluster_ids = []
    for pt in tqdm(gdf.geometry, desc="Assigning clusters"):
        cid = next(i for i, poly in enumerate(dissolved.geometry) if pt.within(poly))
        cluster_ids.append(cid)
    gdf["spatial_cluster"] = cluster_ids

    gdf["temporal_cluster"] = gdf["date"].dt.to_period(temporal_granularity)
    gdf["cluster_id"] = (
        gdf["spatial_cluster"].astype(str) + "_" + gdf["temporal_cluster"].astype(str)
    )

    # ---- Coverage check ----

    cluster_results = {}
    for cid, cluster_df in tqdm(
        gdf.groupby("cluster_id"),
        total=gdf["cluster_id"].nunique(),
        desc="Querying GEE",
    ):
        row = cluster_df.iloc[0]
        lat, lon, date = row["latitude"], row["longitude"], row["date"]

        dt = pd.to_datetime(date)
        date_range = (
            f"{(dt - timedelta(days=days_before)).date()}/"
            f"{(dt + timedelta(days=days_after)).date()}"
        )
        bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]

        covered = []
        for mission in missions:
            mission_info = get_mission(mission)
            start, end = mission_info["valid_date_range"]

            if start and dt < datetime.strptime(start, "%Y-%m-%d"):
                continue
            if end and dt > datetime.strptime(end, "%Y-%m-%d"):
                continue

            col, _ = query_gee_images(mission, bbox, date_range)
            if col.size().getInfo() > 0:
                covered.append(mission)

        cluster_results[cid] = covered

    gdf["covered_by"] = gdf["cluster_id"].map(cluster_results)
    return pd.DataFrame(gdf.drop(columns=["geometry", "buffer"]))


# ---------------------------------------------------------
#  DOWNLOAD MATCHING IMAGES
# ---------------------------------------------------------


def download_matching_gee_images(
    df: pd.DataFrame,
    missions: list[str] = ["sentinel-2", "landsat-5", "landsat-7"],
    buffer_km: float = 0.1,
    output_dir: str | Path | None = None,
    days_before: int = 7,
    days_after: int = 7,
    samples: int = 1,
):
    """
    Replacement for the old STAC-based downloader.

    Produces MULTIBAND GeoTIFFs (one per cluster/mission).
    """
    initialize_ee()
    output_dir = Path(output_dir) if output_dir else data_path("gee_downloads")
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_paths = []
    seen = {}  # (mission, cluster_id) -> file path

    buffer_deg = buffer_km / 111.0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        lat, lon, date = row["latitude"], row["longitude"], row["date"]
        cid = row["cluster_id"]
        dt = pd.to_datetime(date)

        date_range = (
            f"{(dt - timedelta(days=days_before)).date()}/"
            f"{(dt + timedelta(days=days_after)).date()}"
        )

        bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]

        row_files = []

        for mission in row.get("covered_by", []):
            key = (mission, cid)

            # Dedup
            if key in seen:
                row_files.append(seen[key])
                continue

            col, bands = query_gee_images(mission, bbox, date_range, cloud_filter=20)
            best = get_best_image(col, mission, samples)

            if best is None:
                continue

            mission_dir = output_dir / mission / cid
            path = download_gee_multiband(
                best,
                mission,
                bands,
                bbox,
                mission_dir,
                scale=get_mission(mission).gee_scale,
            )

            seen[key] = path
            row_files.append(path)

        downloaded_paths.append(row_files)

    df = df.copy()
    df["downloaded_files"] = downloaded_paths
    return df
