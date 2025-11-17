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
import pandas as pd
import requests
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
    collection_id = mission_info["gee_collection"]
    bands = mission_info["bands"]

    start, end = date_range.split("/")
    region = ee.Geometry.BBox(*bbox)

    col = ee.ImageCollection(collection_id).filterBounds(region).filterDate(start, end)

    if cloud_filter:
        if mission.startswith("sentinel"):
            col = col.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_filter))
        elif mission.startswith("landsat"):
            col = col.filter(ee.Filter.lte("CLOUD_COVER", cloud_filter))

    return col, bands


# swmaps/core/satellite_query.py


def get_best_image(collection: ee.ImageCollection, mission: str):
    """Pick the 'best' image: lowest cloud cover, most recent.

    Args:
        collection (ee.ImageCollection): The collection to search.
        mission (str): Mission slug (e.g. "sentinel-2", "landsat-5").

    Returns:
        ee.Image | None: The selected image or None if no images exist.
    """
    # If the collection is empty, return None
    size = collection.size().getInfo()
    if size == 0:
        return None

    # Pick the correct cloud‐cover property based on the mission
    if mission.startswith("sentinel"):
        cloud_property = "CLOUDY_PIXEL_PERCENTAGE"
    else:
        cloud_property = "CLOUD_COVER"

    # Sort ascending by cloud cover, then descending by acquisition time
    image = (
        collection.sort(cloud_property)  # ascending: least cloudy first
        .sort("system:time_start", False)  # descending: newest first
        .first()
    )
    return image


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

    Args:
        image (ee.Image)
        mission (str)
        bands (dict):
        bbox (list): extent
        out_dir (Path): download location
        scale (int): pixel resolution

    Returns:
        str: path to saved GeoTIFF
    """
    initialize_ee()

    out_dir.mkdir(parents=True, exist_ok=True)

    band_list = list(bands.values())
    region = ee.Geometry.BBox(*bbox)

    clipped = image.select(band_list).clip(region)

    url = clipped.getDownloadURL(
        {
            "scale": scale,
            "region": region,
            "format": "GEOTIFF",
            "filePerBand": False,
        }
    )

    out_path = out_dir / f"{image.get('system:index').getInfo()}_multiband.tif"

    # Download binary stream
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
            best = get_best_image(col, mission)

            if best is None:
                continue

            mission_dir = output_dir / mission / cid
            path = download_gee_multiband(
                best,
                mission,
                bands,
                bbox,
                mission_dir,
                scale=get_mission(mission)["gee_scale"],
            )

            seen[key] = path
            row_files.append(path)

        downloaded_paths.append(row_files)

    df = df.copy()
    df["downloaded_files"] = downloaded_paths
    return df
