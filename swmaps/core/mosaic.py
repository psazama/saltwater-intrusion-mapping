import logging
import math
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject
from shapely.geometry import MultiPolygon, Polygon
from tqdm import tqdm

from swmaps.core.aoi import iter_square_patches

from .indices import compute_ndwi
from .missions import get_mission
from .raster_utils import init_logger, reproject_bbox
from .satellite_query import _stack_bands, query_satellite_items


def create_mosaic_placeholder(
    mosaic_path: str | Path,
    bbox: tuple[float, float, float, float],
    mission: str,
    resolution: float,
    crs: str = "EPSG:32618",
    dtype: str = "float32",
):
    """
    Create an empty mosaic GeoTIFF file to be filled later.

    Parameters:
        mosaic_path (str): Path where the placeholder file will be saved.
        bbox (tuple): (minx, miny, maxx, maxy) in target CRS.
        mission (str): The satellite mission to use ("sentinel-2" or "landsat-5").
        resolution (float): Target pixel resolution in meters.
        crs (str): Coordinate reference system.
        dtype (str): Data type of the mosaic file.
    """
    mission_specs = get_mission(mission)
    bands = len(mission_specs["bands"])

    minx, miny, maxx, maxy = bbox

    width = math.ceil((maxx - minx) / resolution)
    height = math.ceil((maxy - miny) / resolution)

    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "LZW",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "nodata": np.nan,
    }

    with rasterio.open(
        mosaic_path,
        "w",
        **profile,
        sparse_ok=True,  # <- only real tiles hit disk
        BIGTIFF="YES",
    ) as dst:
        dst.write(np.full((bands, height, width), np.nan, dtype=dtype))

    return transform, width, height, crs, bands


def add_image_to_mosaic(
    band_index: int,
    image_data: np.ndarray,
    src_transform: Affine,
    src_crs: str | CRS,
    mosaic_path: str | Path,
) -> None:
    """
    Reproject and insert a satellite image array into the mosaic placeholder.

    Parameters:
        image_data (np.ndarray): The image array to reproject and insert.
        src_transform (Affine): Affine transform of the source image.
        src_crs (str or CRS): CRS of the source image.
        mosaic_path (str): Path to the mosaic file to update.
    """
    with rasterio.open(mosaic_path, "r+") as mosaic:
        if band_index > mosaic.count:
            raise ValueError(
                f"Band index {band_index} exceeds available bands in mosaic."
            )

        existing_array = mosaic.read(band_index)
        temp_array = np.full(existing_array.shape, np.nan, dtype=image_data.dtype)

        reproject(
            source=image_data,
            destination=temp_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=mosaic.transform,
            dst_crs=mosaic.crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )

        print(
            f"[DEBUG] Band {band_index} reprojected to temp array: {np.isnan(temp_array).sum() / temp_array.size:.2%} NaNs"
        )

        # Combine arrays with averaging
        valid_existing = ~np.isnan(existing_array)
        valid_temp = ~np.isnan(temp_array)

        combined_array = existing_array.copy()
        combined_array[valid_temp & ~valid_existing] = temp_array[
            valid_temp & ~valid_existing
        ]
        combined_array[valid_temp & valid_existing] = (
            existing_array[valid_temp & valid_existing]
            + temp_array[valid_temp & valid_existing]
        ) / 2.0

        mosaic.write(combined_array, band_index)
        print(
            f"[DEBUG] Band {band_index} final mosaic NaN ratio: {np.isnan(combined_array).sum() / combined_array.size:.2%}"
        )


def compress_mosaic(mosaic_path: str | Path) -> None:
    """
    Rewrites the mosaic file with compression.
    """
    with rasterio.open(mosaic_path) as src:
        profile = src.profile.copy()
        profile.update(
            {
                "dtype": "float32",
                "nodata": np.nan,
            }
        )
        data = src.read()

    profile.update(
        {
            "compress": "lzw",  # or "deflate", "zstd", etc.
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }
    )

    # Use a temporary file for atomic write
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        temp_path = tmp.name

    with rasterio.open(
        temp_path,
        "w",
        **profile,
        predictor=2,  # horizontal differencing
        sparse_ok=True,  # <- only real tiles hit disk
        BIGTIFF="YES",
    ) as dst:
        dst.write(data)

    shutil.move(temp_path, mosaic_path)
    print(f"[INFO] Compressed mosaic saved: {mosaic_path}")


def _download_patch(
    idx_patch: tuple[int, gpd.GeoSeries],
    mission: str,
    date_range: str,
    bands: dict[str, str],
    max_items: int,
) -> tuple[int, np.ndarray, Affine, CRS] | None:
    """
    Runs in its own process.
    Returns (patch_index, stack, transform, crs)  OR  None if no imagery.
    """
    i, patch = idx_patch
    sub_bbox = patch.geometry.bounds

    items, _ = query_satellite_items(
        mission=mission, bbox=sub_bbox, date_range=date_range, max_items=max_items
    )
    if not items:
        return None

    stack, tfm, crs = _stack_bands(items[0], bands)  # (bands, h, w)
    return i, stack, tfm, crs


def patchwise_query_download_mosaic(
    mosaic_path: str | Path,
    bbox: Sequence[float] | Polygon | MultiPolygon | gpd.GeoDataFrame | gpd.GeoSeries,
    mission: str,
    resolution: float,
    bands: dict[str, str],
    date_range: str,
    base_output_path: str | Path,
    to_disk: bool = False,
    patch_size_meters: float | None = None,
    multithreaded: bool = False,
    max_items: int = 1,
) -> dict:
    """
    Breaks region into patches and processes each separately,
    then compresses the resulting mosaic.
    """
    # If caller passes ``None`` we default to 20× native pixel size
    if patch_size_meters is None:
        patch_size_meters = max(15_000, resolution * 600)

    patches = list(iter_square_patches(bbox, patch_size_meters))
    gdf_patches = gpd.GeoDataFrame(geometry=patches, crs="EPSG:32618")
    gdf_patches = gdf_patches.to_crs("EPSG:4326")  # back to WGS84 for querying
    total_patches = len(gdf_patches)
    logging.warning(f"[INFO] Total Patch Count: {total_patches}")

    if multithreaded:
        with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as pool:
            patch_runner = partial(
                _download_patch,
                mission=mission,
                date_range=date_range,
                bands=bands,
                max_items=max_items,
            )
            futures = {
                pool.submit(patch_runner, p): p[0] for p in gdf_patches.iterrows()
            }

            for fut in as_completed(futures):
                out = fut.result()
                if out is None:  # no imagery found for this patch
                    continue
                i, stack, tfm, crs = out
                for b in range(stack.shape[0]):  # write band-by-band
                    add_image_to_mosaic(b + 1, stack[b], tfm, crs, mosaic_path)

                size_gb = os.path.getsize(mosaic_path) / (1024**3)
                print(f"[PATCH {i+1}/{total_patches}] mosaic ≈ {size_gb:,.2f} GB")
    else:
        patch_runner = partial(
            _download_patch,
            mission=mission,
            date_range=date_range,
            bands=bands,
            max_items=max_items,
        )

        for i, patch in tqdm(
            gdf_patches.iterrows(), total=len(gdf_patches), desc="Patches"
        ):
            out = patch_runner((i, patch))
            if out is None:  # no imagery found for this patch
                continue
            i, stack, tfm, crs = out
            for b in range(stack.shape[0]):  # write band-by-band
                add_image_to_mosaic(b + 1, stack[b], tfm, crs, mosaic_path)

            size_gb = os.path.getsize(mosaic_path) / (1024**3)
            print(f"[PATCH {i+1}/{total_patches}] mosaic ≈ {size_gb:,.2f} GB")
        print("finished downloading mosaic")

    # Final step: compress the mosaic after all patches are added
    # compress_mosaic(mosaic_path) # commenting out to test speed change.


def should_skip_mosaic(
    path: str | Path,
    mission_config: dict,
    date_str: str,
    threshold: float = 0.8,
) -> bool:
    """
    Determines if mosaic processing should be skipped based on:
    1. Date being outside the valid mission date range
    2. Existing file with high NaN ratio

    Parameters:
        path (str): Path to the mosaic file
        mission_config (dict): Mission configuration from get_mission()
        date_str (str): Date string in format "YYYY-MM-DD/YYYY-MM-DD" or "YYYY-MM-DD"
        threshold (float): NaN ratio threshold above which to skip

    Returns:
        bool: True if processing should be skipped, False otherwise
    """
    # Extract start date from date string (handles both single dates and ranges)
    if "/" in date_str:
        date_str = date_str.split("/")[0]

    try:
        query_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        logging.warning(f"[WARNING] Invalid date format: {date_str}")
        return True  # Skip if date format is invalid

    # Check if date is within valid mission range
    dates = mission_config["valid_date_range"]
    mission_start = datetime.strptime(dates[0], "%Y-%m-%d")
    if dates[1]:
        mission_end = datetime.strptime(dates[1], "%Y-%m-%d")
    else:
        mission_end = datetime.now()

    if query_date < mission_start or query_date > mission_end:
        print(query_date, mission_start, mission_end)
        logging.info(
            f"[SKIP] Date {date_str} is outside mission date range {dates[0]}, {dates[1]}"
        )
        return True

    # Check if file exists and has high NaN ratio (original logic)
    if not os.path.exists(path):
        return False

    try:
        with rasterio.open(path) as src:
            data = src.read()
            nan_ratio = np.isnan(data).sum() / data.size
            if nan_ratio > threshold:
                logging.info(f"[SKIP] {path} already exists with {nan_ratio:.2%} NaNs.")
                return True
    except Exception as e:
        logging.warning(f"[WARNING] Could not read {path}: {e}")

    return False


def process_date(
    date: str,
    bbox: Sequence[float] | Polygon | MultiPolygon | gpd.GeoDataFrame | gpd.GeoSeries,
    sentinel_mission: dict,
    landsat5_mission: dict,
    landsat7_mission: dict,
    sentinel2_mosaic_path: str | Path,
    landsat5_mosaic_path: str | Path,
    landsat7_mosaic_path: str | Path,
    inline_mask: bool = False,
    multithreaded: bool = False,
    max_items: int = 1,
):
    """
    Processes satellite data for a single date by creating and populating mosaics for Landsat-5, Landsat-7, and Sentinel-2.

    For each mission:
    - Constructs a dated filename for the output mosaic.
    - Reprojects the bounding box to the target CRS (EPSG:32618).
    - Creates an empty mosaic placeholder GeoTIFF.
    - Downloads and inserts patchwise image data into the mosaic for the given date.

    Errors encountered during processing are caught and stored in the returned result dictionary.

    Parameters:
        date (str): The date range to process (e.g., "2020-06-01/2020-06-15").
        bbox (list): Bounding box in WGS84 [min_lon, min_lat, max_lon, max_lat].
        sentinel_mission (dict): Sentinel-2 mission specs from get_mission().
        landsat5_mission (dict): Landsat-5 mission specs from get_mission().
        landsat7_mission (dict): Landsat-7 mission specs from get_mission().
        sentinel2_mosaic_path (str): Base path to output Sentinel-2 mosaics.
        landsat5_mosaic_path (str): Base path to output Landsat-5 mosaics.
        landsat7_mosaic_path (str): Base path to output Landsat-7 mosaics.
        inline_mask (bool): If True, write an NDWI water mask next to every finished mosaic and then delete the mosaic to save disk space.


    Returns:
        dict: A result dictionary with the date and any errors encountered.
    """
    init_logger("download_worker_log.txt")  # optional: make this configurable

    logging.info(f"Started processing for date: {date}")
    missions = ["sentinel-2", "landsat-5", "landsat-7"]
    mission_paths = [sentinel2_mosaic_path, landsat5_mosaic_path, landsat7_mosaic_path]

    result = {"date": date, "errors": []}
    for mission_number, mission_name in enumerate(missions):
        logging.info(f"[MOSAIC] Starting download for {mission_name}")
        try:
            mission_config = get_mission(mission_name)
            base, _ = os.path.splitext(mission_paths[mission_number])
            mname = f"{base}_{date.replace('/', '_')}.tif"
            if should_skip_mosaic(mname, mission_config, date):
                continue
            data_type = "float32"
            create_mosaic_placeholder(
                mosaic_path=mname,
                bbox=reproject_bbox(bbox),
                resolution=mission_config["resolution"],
                mission=mission_name,
                crs="EPSG:32618",
                dtype=data_type,
            )
            patchwise_query_download_mosaic(
                mname,
                bbox,
                mission_name,
                mission_config["resolution"],
                mission_config["bands"],
                date,
                None,
                to_disk=False,
                multithreaded=multithreaded,
                max_items=max_items,
            )
            logging.info(f"[MOSAIC] Saved {mname}")

            # ---- optional mask-and-cleanup ----
            if inline_mask:
                mask_path = f"{os.path.splitext(mname)[0]}_mask.tif"
                compute_ndwi(
                    mname,
                    mission_name,
                    out_path=mask_path,
                    display=False,
                    threshold=0.2,
                )
                os.remove(mname)
                logging.info(f"[MASK] Saved {mask_path} and removed {mname}")

        except Exception as e:
            result["errors"].append(f"{mission_name} error: {e}")

    return result
