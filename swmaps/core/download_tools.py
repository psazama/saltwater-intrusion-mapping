import logging
import math
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Optional

import pystac

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.session import AWSSession
from rasterio.transform import Affine, from_bounds
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window
from rasterio.crs import CRS
from shapely.geometry import MultiPolygon, Polygon, box
from tqdm import tqdm

from swmaps.config import data_path
from swmaps.core.aoi import iter_square_patches


def get_mission(mission: str) -> dict[str, object]:
    if mission == "sentinel-2":
        collection = "sentinel-2-l2a"
        query_filter = {"eo:cloud_cover": {"lt": 10}}
        bands = {
            "blue": "blue",
            "green": "green",
            "red": "red",
            "nir08": "nir",
            "swir16": "swir1",
            "swir22": "swir2",
        }
        band_index = {
            "blue": 1,
            "green": 2,
            "red": 3,
            "nir08": 4,
            "swir16": 5,
            "swir22": 6,
        }
        resolution = 10
        valid_date_range = ["2015-06-23", None]
    elif mission == "landsat-5":
        collection = "landsat-c2-l2"
        query_filter = {"eo:cloud_cover": {"lt": 10}, "platform": {"eq": "landsat-5"}}
        bands = {
            "blue": "blue",
            "green": "green",
            "red": "red",
            "nir08": "nir",
            "swir16": "swir1",
            "swir22": "swir2",
        }
        band_index = {
            "blue": 1,
            "green": 2,
            "red": 3,
            "nir08": 4,
            "swir16": 5,
            "swir22": 6,
        }
        resolution = 30
        valid_date_range = ["1984-03-01", "2013-01-01"]
    elif mission == "landsat-7":
        collection = "landsat-c2-l2"
        query_filter = {"eo:cloud_cover": {"lt": 10}, "platform": {"eq": "landsat-7"}}
        bands = {
            "blue": "blue",
            "green": "green",
            "red": "red",
            "nir08": "nir",
            "swir16": "swir1",
            "swir22": "swir2",
        }
        band_index = {
            "blue": 1,
            "green": 2,
            "red": 3,
            "nir08": 4,
            "swir16": 5,
            "swir22": 6,
        }
        resolution = 30
        valid_date_range = ["1999-04-15", "2022-03-31"]
    else:
        raise ValueError("Unsupported mission")
    return {
        "bands": bands,
        "band_index": band_index,
        "collection": collection,
        "query_filter": query_filter,
        "resolution": resolution,
        "valid_date_range": valid_date_range,
    }


def create_coastal_poly(
    bounding_box_file: str | Path,
    out_file: str | Path | None = None,
    buf_km: float = 2,
    offshore_km: float = 1,
) -> gpd.GeoDataFrame:
    """
    Build a coastal band polygon (buffered coastline, clipped to bbox) and save it once.

    Parameters
    ----------
    bounding_box_file : str | Path
        Vector file containing your big bbox (GeoJSON, Shapefile, etc.).
    out_file : str | Path | None
        Where to save the band (defaults to  config/coastal_band.gpkg).
    buf_km, offshore_km : float
        Width of inland / offshore buffers (kilometres).

    Returns
    -------
    GeoDataFrame with a single Polygon feature (lat/long, EPSG:4326).
    """

    bbox_gdf = gpd.read_file(bounding_box_file).to_crs("EPSG:4326")
    bbox = bbox_gdf.total_bounds  # [minx, miny, maxx, maxy]

    # Expand by a generous margin (≈ 10 km) so we don’t miss any coast
    margin_deg = 10 / 111.0  # ~10 km in degrees
    qbox = box(
        bbox[0] - margin_deg,
        bbox[1] - margin_deg,
        bbox[2] + margin_deg,
        bbox[3] + margin_deg,
    )

    coast_file = data_path("coastline/ne_10m_coastline.shp")
    coast = gpd.read_file(
        coast_file,
        bbox=qbox,
    ).to_crs("EPSG:4326")

    if coast.empty:
        raise ValueError("No coastline segments intersect the provided bbox.")

    #  Project to metric CRS
    utm_zone = int((bbox[0] + 180) // 6) + 1
    utm_crs = f"EPSG:{32600 + utm_zone}"
    coast_m = coast.to_crs(utm_crs)

    inland = coast_m.buffer(buf_km * 1_000, cap_style=2)
    offshore = coast_m.buffer(-offshore_km * 1_000, cap_style=2)
    band_m = inland.union(offshore).unary_union

    # back to WGS-84
    band = gpd.GeoSeries([band_m], crs=utm_crs).to_crs("EPSG:4326")

    # final clip to exact bbox
    band_clip = gpd.clip(band, bbox_gdf)
    out_gdf = gpd.GeoDataFrame(geometry=band_clip, crs="EPSG:4326")

    out_file = Path(out_file or "config/coastal_band.gpkg")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_file(out_file, driver="GPKG", layer="coastal_band")

    return out_gdf


def warp_to_wgs84(
    src_path: str,
    dst_path: Optional[str] = None,
    resampling: Resampling = Resampling.nearest,
    dst_nodata: float = np.nan,
) -> str:
    """Reproject a raster to WGS‑84 (EPSG:4326) for web‑map visualisation.

    Parameters
    ----------
    src_path : str
        Path to the input GeoTIFF (any projected or geographic CRS).
    dst_path : str, optional
        Path for the re‑projected output. If *None* (default) the function
        appends ``_wgs84`` to *src_path* before the file extension.
    resampling : rasterio.enums.Resampling, optional
        Resampling algorithm (default: ``Resampling.nearest``).
    dst_nodata : float, optional
        Nodata value written to the output (default: ``numpy.nan``).

    Returns
    -------
    str
        Path to the generated WGS‑84 GeoTIFF.
    """

    if dst_path is None:
        base, ext = os.path.splitext(src_path)
        dst_path = f"{base}_wgs84{ext}"

    with rasterio.open(src_path) as src:
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        profile = src.profile.copy()
        profile.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "nodata": dst_nodata,
            }
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    dst_nodata=dst_nodata,
                )

    return dst_path


def downsample_to_landsat(
    data: np.ndarray,
    transform: Affine,
    crs,
    target_resolution: float = 30,
    resampling: Resampling = Resampling.average,
) -> tuple:
    """Resample Sentinel-2 data to Landsat 30 m resolution."""

    src_res = max(abs(transform.a), abs(transform.e))
    scale = target_resolution / src_res
    dst_height = max(1, int(np.ceil(data.shape[0] / scale)))
    dst_width = max(1, int(np.ceil(data.shape[1] / scale)))
    dst_transform = transform * Affine.scale(scale)

    dst = np.empty((dst_height, dst_width), dtype=data.dtype)

    reproject(
        source=data,
        destination=dst,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=crs,
        resampling=resampling,
        dst_nodata=np.nan,
    )

    return dst, dst_transform


def init_logger(log_path: str = "process_log.txt") -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(process)d] %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
    )


def _stack_bands(
    item: pystac.Item,
    bands: dict[str, str],
) -> tuple[np.ndarray, Affine, rasterio.crs.CRS]:
    session = AWSSession(requester_pays=True)
    arrays = []
    transforms = None
    crs = None
    for key in bands.keys():
        href = item.assets[key].href
        with rasterio.Env(session), rasterio.open(href) as src:
            arr = src.read(1).astype(np.float32)
            if transforms is None:
                transforms, crs = src.transform, src.crs
            arrays.append(arr)
    return np.stack(arrays), transforms, crs


def find_non_nan_window(
    tif_path: str,
    bands: list[int] | None = None,
    window_size: int = 512,
    stride: int = 256,
    threshold_ratio: float = 0.5,
) -> tuple[np.ndarray | list[np.ndarray], dict, Window] | None:
    """
    Finds a window in a raster where the data is mostly valid (not NaN or near-zero).

    Parameters:
        tif_path (str): Path to the raster file.
        bands (list or None): List of 1-based band indices to read and validate. If None, reads band 1.
        window_size (int): Size of the square window.
        stride (int): Step size for moving the window.
        threshold_ratio (float): Minimum proportion of valid data required.

    Returns:
        Tuple of:
            - data (np.ndarray or list of np.ndarrays): Array(s) for the selected window.
            - profile (dict): Raster profile updated for the window.
            - window (Window): The selected rasterio window.
    """
    with rasterio.open(tif_path) as src:
        scale_reflectance = False
        if "landsat" in tif_path:
            scale_reflectance = True

        width, height = src.width, src.height
        band_list = bands if bands is not None else [1]

        for y in range(0, height - window_size + 1, stride):
            for x in range(0, width - window_size + 1, stride):
                window = Window(x, y, window_size, window_size)

                # Read and validate bands
                data_list = []
                valid = True

                for b in band_list:
                    data = src.read(b, window=window)
                    if scale_reflectance:
                        # Landsat Collection 2 Level-2 surface reflectance (SR) products store reflectance as integers and apply the scale factor = 0.0000275, offset = -0.2 as per USGS Landsat documentation
                        data = data.astype(np.float32) * 0.0000275 - 0.2
                    data_list.append(data)

                    valid_mask = ~np.isnan(data)
                    if (
                        np.sum(valid_mask) < threshold_ratio * data.size
                        or np.sum(np.isclose(data, 0))
                        > (1 - threshold_ratio) * data.size
                    ):
                        valid = False
                        break  # this window is not good

                if valid:
                    print(f"Found valid window at x={x}, y={y}")
                    transform = src.window_transform(window)
                    profile = src.profile.copy()
                    profile.update(
                        {
                            "height": window.height,
                            "width": window.width,
                            "transform": transform,
                        }
                    )
                    profile.update(
                        {
                            "dtype": "float32",
                            "nodata": np.nan,
                        }
                    )

                    # Return single array if only one band, else list
                    return (
                        (data_list[0] if len(data_list) == 1 else data_list),
                        profile,
                        window,
                    )

    print("No valid window found.")
    return None, None, None


def find_satellite_coverage(
    df: pd.DataFrame,
    missions: list[str] = ["sentinel-2", "landsat-5", "landsat-7"],
    buffer_km: float = 5,
) -> pd.DataFrame:
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        lat, lon, date = row["latitude"], row["longitude"], row["date"]
        try:
            # Buffer in degrees (approx)
            buffer_deg = buffer_km / 111.0
            bbox = [
                lon - buffer_deg,
                lat - buffer_deg,
                lon + buffer_deg,
                lat + buffer_deg,
            ]

            # ±1 day range
            dt = datetime.strptime(date, "%Y-%m-%d")
            date_range = (
                f"{(dt - timedelta(days=1)).date()}/{(dt + timedelta(days=1)).date()}"
            )

            covered_by = []
            for mission in missions:
                try:
                    mission_info = get_mission(mission)
                    start_date, end_date = mission_info["valid_date_range"]
                    if start_date and dt < datetime.strptime(start_date, "%Y-%m-%d"):
                        continue
                    if end_date and dt > datetime.strptime(end_date, "%Y-%m-%d"):
                        continue

                    items, _ = query_satellite_items(
                        mission=mission, bbox=bbox, date_range=date_range, max_items=1
                    )
                    if items:
                        covered_by.append(mission)
                except Exception:
                    pass

            results.append(covered_by)
        except Exception as e:
            print(f"Failed on row {row}: {e}")
            results.append([])

    df["covered_by"] = results
    return df


def download_matching_images(
    df: pd.DataFrame,
    missions: list[str] = ["sentinel-2", "landsat-5", "landsat-7"],
    buffer_km: float = 0.1,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir) if output_dir else data_path("matched_downloads")
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        lat, lon, date = row["latitude"], row["longitude"], row["date"]
        dt = datetime.strptime(date, "%Y-%m-%d")
        date_range = (
            f"{(dt - timedelta(days=1)).date()}/{(dt + timedelta(days=1)).date()}"
        )
        buffer_deg = buffer_km / 111.0
        bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]

        row_downloads = []

        for mission in row.get("covered_by", []):
            try:
                items, bands = query_satellite_items(
                    mission=mission, bbox=bbox, date_range=date_range, max_items=1
                )
                if not items:
                    continue

                item = items[0]
                date_tag = dt.strftime("%Y%m%d")
                download_dir = os.path.join(
                    output_dir, f"{mission}_{date_tag}_{lat:.4f}_{lon:.4f}"
                )
                os.makedirs(download_dir, exist_ok=True)

                band_files = []
                for band_key in bands.values():
                    path = os.path.join(download_dir, f"{item.id}_{band_key}.tif")
                    band_files.append(path)

                multi_band_path = os.path.join(download_dir, f"{item.id}_multiband.tif")

                if os.path.exists(multi_band_path):
                    print(f"[SKIP] Multiband file already exists: {multi_band_path}")
                    row_downloads.append(multi_band_path)
                    continue

                # Download each band
                download_satellite_bands_from_item(
                    item,
                    bands,
                    to_disk=True,
                    data_dir=download_dir,
                    mission=mission,
                    downsample_to_landsat_res=(mission == "sentinel-2"),
                )

                # Read all bands and stack
                band_arrays = []
                first_profile = None
                for path in band_files:
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"Expected band missing: {path}")
                    with rasterio.open(path) as src:
                        data = src.read(1)
                        band_arrays.append(data)
                        if first_profile is None:
                            first_profile = src.profile.copy()

                # Update profile for multiband
                first_profile.update(count=len(band_arrays))
                with rasterio.open(multi_band_path, "w", **first_profile) as dst:
                    for i, band_array in enumerate(band_arrays, start=1):
                        dst.write(band_array, i)

                row_downloads.append(multi_band_path)

            except Exception as e:
                print(
                    f"[ERROR] Could not download or combine {mission} for {lat},{lon} on {date}: {e}"
                )

        downloaded_paths.append(row_downloads)

    df = df.copy()
    df["downloaded_files"] = downloaded_paths
    return df


def reproject_bbox(
    bbox: (
        list[float]
        | tuple[float, float, float, float]
        | gpd.GeoDataFrame
        | gpd.GeoSeries
        | Polygon
        | MultiPolygon
    ),
    src_crs: str = "EPSG:4326",
    dst_crs: str = "EPSG:32618",
) -> list[float]:
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        geom = None
        minx, miny = bbox[0], bbox[1]
        maxx, maxy = bbox[2], bbox[3]
    else:
        # shapely geometry OR Geo(Data)Frame
        if hasattr(bbox, "geometry"):  # GeoDataFrame / GeoSeries
            geom = bbox.unary_union
        else:  # Polygon / MultiPolygon
            geom = bbox
        minx, miny, maxx, maxy = geom.bounds

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    minx_proj, miny_proj = transformer.transform(minx, miny)
    maxx_proj, maxy_proj = transformer.transform(maxx, maxy)

    return [
        min(minx_proj, maxx_proj),
        min(miny_proj, maxy_proj),
        max(minx_proj, maxx_proj),
        max(miny_proj, maxy_proj),
    ]


def query_satellite_items(
    mission: str = "sentinel-2",
    bbox: list[float] | None = None,
    date_range: str | None = None,
    max_items: int | None = None,
    debug: bool = False,
) -> tuple[list[pystac.Item], dict[str, str]]:
    """
    Queries available satellite imagery items using the AWS Earth Search STAC API.

    Parameters:
        mission (str): The satellite mission to use ("sentinel-2" or "landsat-5" or "landsat-7").
        bbox (list): Bounding box [min_lon, min_lat, max_lon, max_lat].
        date_range (str): ISO8601 date range string.
        debug (bool): Whether to print debug output.

    Returns:
        tuple: (list of STAC items, band mapping dictionary)
    """

    mission_specs = get_mission(mission)
    bands = mission_specs["bands"]
    collection = mission_specs["collection"]
    query_filter = mission_specs["query_filter"]

    catalog = Client.open("https://earth-search.aws.element84.com/v1")
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=date_range,
        query=query_filter,
        max_items=max_items,
    )

    items = list(search.items())
    if not items:
        raise ValueError("No items found for your search.")

    if debug:
        for item in items:
            print(f"Found item: {item.id}")

    return items, bands


def download_satellite_bands_from_item(
    item: pystac.Item,
    bands: dict[str, str],
    to_disk: bool = True,
    data_dir: str | Path | None = None,
    debug: bool = False,
    mission: str | None = None,
    downsample_to_landsat_res: bool = False,
    target_resolution: float = 30,
) -> list[tuple[int, np.ndarray, Affine, CRS]]:
    """
    Downloads selected bands from a single STAC item.
    Optionally downsamples Sentinel-2 imagery to Landsat
    resolution before returning the array.

    Parameters:
        item (dict): A STAC item.
        bands (dict): Mapping of asset keys to band names.
        data_dir (str): Local directory for output.
        debug (bool): Whether to print debug output.
        mission (str, optional): Mission name (e.g. "sentinel-2").
        downsample_to_landsat_res (bool): If True and mission is
            "sentinel-2", resample the band to ``target_resolution``.
        target_resolution (float): Target pixel size in meters
            when downsampling (default 30).

    Returns:
        str: Path to the last band written for this item.
    """

    session = AWSSession(requester_pays=True)
    data_dir = Path(data_dir) if data_dir else data_path()
    if to_disk:
        data_dir.mkdir(parents=True, exist_ok=True)
    out_path = None

    def _fetch_one(k_n):
        key, name = k_n
        href = item.assets[key].href
        with rasterio.Env(session):
            with rasterio.open(href) as src:
                arr = src.read(1).astype(np.float32)
                nod = src.nodata or 0
                arr[arr == nod] = np.nan
                return key, name, arr, src.transform, src.crs

    band_order = list(bands.keys())
    band_data: list[tuple[int, np.ndarray, Affine, str]] = []

    with ThreadPoolExecutor(max_workers=len(bands)) as pool:
        for key, name, data, trf, crs in pool.map(_fetch_one, bands.items()):
            if downsample_to_landsat_res and mission == "sentinel-2":
                data, trf = downsample_to_landsat(data, trf, crs, target_resolution)

            band_index = band_order.index(key) + 1
            band_data.append((band_index, data, trf, crs))

            if to_disk:
                out_path = data_dir / f"{item.id}_{name}.tif"
                profile = {
                    "driver": "GTiff",
                    "dtype": "float32",
                    "nodata": np.nan,
                    "crs": crs,
                    "transform": trf,
                    "height": data.shape[0],
                    "width": data.shape[1],
                }
                with rasterio.open(out_path, "w", **profile, BIGTIFF="YES") as dst:
                    dst.write(data, 1)

            if debug:
                nan_pct = np.isnan(data).sum() / data.size
                logging.debug(f"{item.id} – {name}: {data.shape}  NaNs={nan_pct:.2%}")

    return band_data


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
    bbox: (
        Sequence[float]
        | Polygon
        | MultiPolygon
        | gpd.GeoDataFrame
        | gpd.GeoSeries
    ),
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
    bbox: (
        Sequence[float]
        | Polygon
        | MultiPolygon
        | gpd.GeoDataFrame
        | gpd.GeoSeries
    ),
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


def compute_ndwi(
    path: str | Path,
    mission: str,
    out_path: str | Path | None = None,
    display: bool = False,
    threshold: float = 0.2,
) -> np.ndarray:
    """
    Computes the NDWI mask from a GeoTIFF based on mission-specific green and NIR bands.

    Parameters:
        path (str): Path to GeoTIFF file.
        mission (str): One of ["landsat-5", "landsat-7", "sentinel-2"].
        out_path (str, optional): Where to save NDWI GeoTIFF.
        display (bool): Whether to show the NDWI mask.
        threshold (float): NDWI threshold to define water (default 0.2).

    Returns:
        np.ndarray: Binary NDWI mask (1 = water, 0 = non-water).
    """
    mission_info = get_mission(mission)
    band_index = mission_info["band_index"]
    green_band = band_index["green"]
    nir_band = band_index["nir08"]

    scale_reflectance = "landsat" in mission

    with rasterio.open(path) as src:
        green = src.read(green_band).astype(np.float32)
        nir = src.read(nir_band).astype(np.float32)

        if scale_reflectance:
            green = green * 0.0000275 - 0.2
            nir = nir * 0.0000275 - 0.2

        ndwi = (green - nir) / (green + nir + 1e-10)
        ndwi_mask = (ndwi > threshold).astype(float)

        profile = src.profile.copy()
        profile.update({"count": 1, "dtype": "float32", "nodata": np.nan})

        if out_path:
            with rasterio.open(out_path, "w", **profile, BIGTIFF="YES") as dst:
                dst.write(ndwi_mask, 1)

    if display:
        plt.imshow(ndwi_mask, cmap="gray")
        plt.title(f"NDWI Mask ({mission})")
        plt.axis("off")
        plt.show()

    return ndwi_mask
