"""Utilities for building multi-band mosaics from STAC imagery downloads."""

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
    """Create an empty mosaic GeoTIFF file to be filled later.

    Args:
        mosaic_path (str | Path): Path where the placeholder file will be
            saved.
        bbox (tuple[float, float, float, float]): Bounding box expressed as
            ``(minx, miny, maxx, maxy)`` in the target CRS.
        mission (str): Satellite mission slug used to determine band count.
        resolution (float): Target pixel resolution in metres.
        crs (str): Coordinate reference system for the mosaic.
        dtype (str): Data type to allocate for the raster bands.

    Returns:
        tuple[Affine, int, int, str | CRS, int]: The raster transform, width,
        height, CRS, and number of bands for the mosaic.
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
    """Reproject and insert a satellite image array into the mosaic placeholder.

    Args:
        band_index (int): One-based band index within the mosaic.
        image_data (np.ndarray): Image array to reproject and merge.
        src_transform (Affine): Affine transform of the source image.
        src_crs (str | CRS): Coordinate reference system of the source image.
        mosaic_path (str | Path): Path to the mosaic file being updated.

    Returns:
        None: The mosaic file is updated in place.

    Raises:
        ValueError: If ``band_index`` exceeds the number of bands in the
            mosaic file.
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
    """Rewrite the mosaic file with compression to reduce disk usage.

    Args:
        mosaic_path (str | Path): Path to the mosaic GeoTIFF to compress.

    Returns:
        None: The file on disk is replaced with a compressed copy.
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
    """Download imagery for a single AOI patch.

    Args:
        idx_patch (tuple[int, gpd.GeoSeries]): Tuple containing the patch
            index and its geometry.
        mission (str): Mission slug passed through to the query helper.
        date_range (str): Date or date range filter in ISO format.
        bands (dict[str, str]): Mapping of band labels to STAC asset names.
        max_items (int): Maximum number of STAC items to request.

    Returns:
        tuple[int, np.ndarray, Affine, CRS] | None: Tuple containing the
        patch index, stacked band array, affine transform, and CRS, or
        ``None`` if no imagery satisfied the query.
    """
    i, patch = idx_patch
    sub_bbox = patch.geometry.bounds

    items, _ = query_satellite_items(
        mission=mission, bbox=sub_bbox, date_range=date_range, max_items=max_items
    )
    if not items:
        return None

    all_stacks = []
    for item in items:
        arr, tfm, crs = _stack_bands(item, bands)  # (bands, h, w)
        all_stacks.append(arr)

    # Shape: (num_items, bands, h, w)
    data = np.stack(all_stacks, axis=0)

    # Mask zeros as NaN so they don’t count
    data_masked = np.where(data == 0, np.nan, data)

    # Median along the item dimension (axis=0), ignoring NaNs
    median_stack = np.nanmedian(data_masked, axis=0)  # shape: (bands, h, w)

    return i, median_stack, tfm, crs


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
    """Download, mosaic, and optionally persist imagery patch by patch.

    Args:
        mosaic_path (str | Path): Path to the mosaic GeoTIFF that will be
            created or updated.
        bbox (Sequence[float] | Polygon | MultiPolygon | gpd.GeoDataFrame |
            gpd.GeoSeries): AOI definition used to generate patches.
        mission (str): Mission slug controlling band configuration and query
            filters.
        resolution (float): Target output pixel resolution in metres.
        bands (dict[str, str]): Mapping of band aliases to STAC asset keys.
        date_range (str): Date range string to filter imagery.
        base_output_path (str | Path): Directory where optional exports are
            saved when ``to_disk`` is ``True``.
        to_disk (bool): If ``True``, persist each patch stack alongside the
            mosaic.
        patch_size_meters (float | None): Optional override for the patch
            edge length in metres. Defaults to a multiple of ``resolution``.
        multithreaded (bool): If ``True``, process patches in parallel using
            a process pool.
        max_items (int): Maximum number of STAC items requested per patch.

    Returns:
        dict: Summary metadata describing the download session.
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
    """Determine whether mosaic processing should be skipped for a date.

    Args:
        path (str | Path): Path to the mosaic file that would be created.
        mission_config (dict): Mission configuration returned by
            :func:`get_mission`.
        date_str (str): Date string in ``YYYY-MM-DD`` or date-range form.
        threshold (float): Maximum acceptable NaN ratio for existing mosaics.

    Returns:
        bool: ``True`` when processing should be skipped, ``False`` otherwise.
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
    output_dir: str = None,
):
    """Process imagery for a single date across multiple missions.

    Args:
        date (str): Date or date range (``YYYY-MM-DD`` or
            ``YYYY-MM-DD/YYYY-MM-DD``).
        bbox (Sequence[float] | Polygon | MultiPolygon | gpd.GeoDataFrame |
            gpd.GeoSeries): AOI definition in WGS84 coordinates.
        sentinel_mission (dict): Sentinel-2 mission metadata from
            :func:`get_mission`.
        landsat5_mission (dict): Landsat-5 mission metadata from
            :func:`get_mission`.
        landsat7_mission (dict): Landsat-7 mission metadata from
            :func:`get_mission`.
        sentinel2_mosaic_path (str | Path): Base directory for Sentinel-2
            mosaics.
        landsat5_mosaic_path (str | Path): Base directory for Landsat-5
            mosaics.
        landsat7_mosaic_path (str | Path): Base directory for Landsat-7
            mosaics.
        inline_mask (bool): If ``True``, write NDWI masks next to each
            completed mosaic and optionally delete the mosaic afterwards.
        multithreaded (bool): If ``True``, process patches in parallel.
        max_items (int): Maximum STAC items requested per patch.

    Returns:
        dict: Dictionary containing the processed date and any captured
        errors.
    """
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        init_logger(str(Path(output_dir) / "download_worker_log.txt"))
    else:
        init_logger("download_worker_log.txt")

    logging.info(f"Started processing for date: {date}")
    missions = ["sentinel-2", "landsat-5", "landsat-7"]
    mission_paths = [sentinel2_mosaic_path, landsat5_mosaic_path, landsat7_mosaic_path]

    result = {"date": date, "errors": []}
    for mission_number, mission_name in enumerate(missions):
        logging.info(f"[MOSAIC] Starting download for {mission_name}")
        try:
            mission_config = get_mission(mission_name)
            base, _ = os.path.splitext(mission_paths[mission_number])

            filename = f"{Path(base).name}_{date.replace('/', '_')}.tif"

            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                mname = str(Path(output_dir) / filename)
            else:
                mname = str(Path(base).parent / filename)

            if should_skip_mosaic(mname, mission_config, date):
                logging.info(f"[MOSAIC] Skipping download for {mname}")
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
