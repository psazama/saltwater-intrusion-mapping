"""
Earth Engine based querying and downloading of satellite imagery.

This module replaces all STAC, AWS, rasterio-session, and per-band
downloading logic with clean GEE-native helpers.

Downstream code simply receives local multiband GeoTIFFs, same as before.
"""

import os
import time
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
#  TASK TRACKING HELPERS
# ---------------------------------------------------------


def wait_for_ee_task(task: ee.batch.Task, timeout: int = 3600, poll_interval: int = 10):
    """
    Wait for an Earth Engine task to complete.

    Args:
        task (ee.batch.Task): The task to wait for
        timeout (int): Maximum time to wait in seconds (default: 3600 = 1 hour)
        poll_interval (int): Time between status checks in seconds (default: 10)

    Returns:
        bool: True if task completed successfully

    Raises:
        TimeoutError: If task doesn't complete within timeout period
        RuntimeError: If task fails during execution
    """
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Task {task.id} exceeded timeout of {timeout} seconds")

        status = task.status()
        state = status.get("state")

        if state == "COMPLETED":
            print(f"[GEE] Task {task.id} completed successfully")
            return True
        elif state == "FAILED":
            error_msg = status.get("error_message", "Unknown error")
            raise RuntimeError(f"Task {task.id} failed: {error_msg}")
        elif state == "CANCELLED":
            raise RuntimeError(f"Task {task.id} was cancelled")
        elif state in ("READY", "RUNNING"):
            print(
                f"[GEE] Task {task.id} is {state}, waiting... ({elapsed:.0f}s elapsed)"
            )
            time.sleep(poll_interval)
        else:
            # UNSUBMITTED or unknown state
            print(f"[GEE] Task {task.id} in state: {state}")
            time.sleep(poll_interval)


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
    Automatically switches Sentinel-2 to async export if request is large.

    Returns:
        tuple: (str, ee.batch.Task | None) where:
            - str is the output file path
            - ee.batch.Task is the task object for async exports, or None for sync exports
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
    # Sentinel auto-export rule
    # -------------------------------------------------
    if mission.startswith("sentinel") and est_mb > 30:
        print(
            f"[GEE] Size exceeds 30 MB for Sentinel-2, "
            f"exporting asynchronously: {out_path.name}"
        )

        task = ee.batch.Export.image.toDrive(
            image=clipped,
            description=f"{mission}_{image_id}",
            folder=str(out_dir.name),
            fileNamePrefix=out_path.stem,
            region=region,
            scale=scale,
            crs=ANALYSIS_CRS,
            maxPixels=1e13,
        )
        task.start()

        return str(out_path), task

    # -------------------------------------------------
    # Synchronous download path
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

    return str(out_path), None


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
    async_tasks = []  # Track async tasks

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
            path, task = download_gee_multiband(
                best,
                mission,
                bands,
                bbox,
                mission_dir,
                scale=get_mission(mission).gee_scale,
            )

            if task is not None:
                async_tasks.append((path, task))

            seen[key] = path
            row_files.append(path)

        downloaded_paths.append(row_files)

    # Wait for all async tasks to complete
    if async_tasks:
        print(f"[GEE] Waiting for {len(async_tasks)} async task(s) to complete...")
        for path, task in tqdm(async_tasks, desc="Waiting for async tasks"):
            try:
                wait_for_ee_task(task, timeout=3600, poll_interval=15)
            except (TimeoutError, RuntimeError) as e:
                print(f"[GEE] Error waiting for task at {path}: {e}")

    df = df.copy()
    df["downloaded_files"] = downloaded_paths
    return df
