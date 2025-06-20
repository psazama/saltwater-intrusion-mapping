import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.session import AWSSession
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from shapely.geometry import box
from tqdm import tqdm


def get_mission(mission):
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


def init_logger(log_path="process_log.txt"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(process)d] %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
    )


def find_non_nan_window(
    tif_path, bands=None, window_size=512, stride=256, threshold_ratio=0.5
):
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
                    profile.update({"BIGTIFF": "IF_SAFER"})

                    # Return single array if only one band, else list
                    return (
                        (data_list[0] if len(data_list) == 1 else data_list),
                        profile,
                        window,
                    )

    print("No valid window found.")
    return None, None, None


def find_satellite_coverage(
    df, missions=["sentinel-2", "landsat-5", "landsat-7"], buffer_km=5
):
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
    df,
    missions=["sentinel-2", "landsat-5", "landsat-7"],
    buffer_km=2,
    output_dir="data/matched_downloads",
):
    os.makedirs(output_dir, exist_ok=True)
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
                if items:
                    item = items[0]
                    date_tag = dt.strftime("%Y%m%d")
                    download_dir = os.path.join(
                        output_dir, f"{mission}_{date_tag}_{lat:.4f}_{lon:.4f}"
                    )
                    os.makedirs(download_dir, exist_ok=True)

                    download_satellite_bands_from_item(
                        item, bands, to_disk=True, data_dir=download_dir
                    )

                    # Record each band’s path
                    for band_key in bands.values():
                        band_file = os.path.join(
                            download_dir, f"{item.id}_{band_key}.tif"
                        )
                        if os.path.exists(band_file):
                            row_downloads.append(band_file)
            except Exception as e:
                print(
                    f"[ERROR] Could not download {mission} for {lat},{lon} on {date}: {e}"
                )

        downloaded_paths.append(row_downloads)

    df = df.copy()
    df["downloaded_files"] = downloaded_paths
    return df


def reproject_bbox(bbox, src_crs="EPSG:4326", dst_crs="EPSG:32618"):
    minx, miny = bbox[0], bbox[1]
    maxx, maxy = bbox[2], bbox[3]

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
    mission="sentinel-2", bbox=None, date_range=None, max_items=None, debug=False
):
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
    item, bands, to_disk=True, data_dir="data", debug=False
):
    """
    Downloads selected bands from a single STAC item.

    Parameters:
        item (dict): A STAC item.
        bands (dict): Mapping of asset keys to band names.
        data_dir (str): Local directory for output.
        debug (bool): Whether to print debug output.

    Returns:
        str: Path to the last band written for this item.
    """

    session = AWSSession(requester_pays=True)
    if to_disk:
        os.makedirs(data_dir, exist_ok=True)
    out_path = None

    band_data = []
    band_order = list(bands.keys())
    for key, name in bands.items():
        if key in item.assets:
            href = item.assets[key].href
            if to_disk:
                out_path = os.path.join(data_dir, f"{item.id}_{name}.tif")

                if os.path.exists(out_path):
                    if debug:
                        print(f"[SKIP] {out_path} already exists.")
                    continue

            if debug:
                print(f"Accessing {name} from {href} ...")

            with rasterio.Env(session):
                with rasterio.open(href) as src:
                    profile = src.profile
                    profile.update({"BIGTIFF": "IF_SAFER"})
                    if src.count == 1:
                        data = src.read(1)
                    else:
                        raise ValueError(
                            f"Expected single-band image, but got {src.count} bands in {href}"
                        )

                    band_index = band_order.index(key) + 1
                    if band_index > len(bands):
                        raise ValueError(
                            f"Invalid band index {band_index} for item {item.id}"
                        )
                    band_data.append((band_index, data, src.transform, src.crs))

                if to_disk:
                    with rasterio.open(
                        out_path, "w", **profile, BIGTIFF="IF_SAFER"
                    ) as dst:
                        dst.write(data, 1)
        else:
            print(
                f"[WARNING] Band {key} not available in item {item.id}. Available bands: {list(item.assets.keys())}"
            )

    return band_data


def create_mosaic_placeholder(
    mosaic_path, bbox, mission, resolution, crs="EPSG:32618", dtype="float32"
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
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "nodata": np.nan,
    }
    profile.update({"BIGTIFF": "IF_SAFER"})

    profile.update(
        {
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }
    )

    with rasterio.open(mosaic_path, "w", **profile) as dst:
        nan_data = np.full((bands, height, width), np.nan, dtype=dtype)
        dst.write(nan_data)

    return transform, width, height, crs, bands


def add_image_to_mosaic(band_index, image_data, src_transform, src_crs, mosaic_path):
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
        mosaic_array = mosaic.read(band_index)

        reproject(
            source=image_data,
            destination=mosaic_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=mosaic.transform,
            dst_crs=mosaic.crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )

        mosaic.write(mosaic_array, band_index)


def divide_bbox_into_patches(bbox, patch_size_meters, crs="EPSG:32618"):
    """
    Divide a bounding box into smaller square patches in the target CRS.

    Parameters:
        bbox (list): [min_lon, min_lat, max_lon, max_lat] in WGS84.
        patch_size_meters (float): Patch size in meters.
        crs (str): Target projected CRS.

    Returns:
        list of shapely.geometry.Polygon: Patch polygons in the target CRS.
    """
    # Create GeoDataFrame in WGS84
    gdf = gpd.GeoDataFrame(geometry=[box(*bbox)], crs="EPSG:4326")
    gdf = gdf.to_crs(crs)
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy

    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx, patch_size_meters)
    ys = np.arange(miny, maxy, patch_size_meters)

    patches = []
    for x in xs:
        for y in ys:
            patch = box(x, y, x + patch_size_meters, y + patch_size_meters)
            patches.append(patch)

    return patches


def compress_mosaic(mosaic_path):
    """
    Rewrites the mosaic file with compression.
    """
    with rasterio.open(mosaic_path) as src:
        profile = src.profile.copy()
        profile.update({"BIGTIFF": "IF_SAFER"})
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

    with rasterio.open(temp_path, "w", **profile) as dst:
        dst.write(data)

    shutil.move(temp_path, mosaic_path)
    print(f"[INFO] Compressed mosaic saved: {mosaic_path}")


def patchwise_query_download_mosaic(
    mosaic_path,
    bbox,
    mission,
    patch_size_meters,
    resolution,
    bands,
    date_range,
    base_output_path,
    to_disk=False,
):
    """
    Breaks region into patches and processes each separately,
    then compresses the resulting mosaic.
    """
    patches = divide_bbox_into_patches(bbox, patch_size_meters)
    gdf_patches = gpd.GeoDataFrame(geometry=patches, crs="EPSG:32618")
    gdf_patches = gdf_patches.to_crs("EPSG:4326")  # back to WGS84 for querying

    for i, patch in gdf_patches.iterrows():
        sub_bbox = patch.geometry.bounds
        try:
            items, _ = query_satellite_items(
                mission=mission, bbox=sub_bbox, date_range=date_range, max_items=3
            )
            if to_disk:
                patch_output_path = os.path.join(base_output_path, f"patch_{i}")
                os.makedirs(patch_output_path, exist_ok=True)

            for item in items:
                if to_disk:
                    band_data = download_satellite_bands_from_item(
                        item, bands, to_disk=to_disk, data_dir=patch_output_path
                    )
                else:
                    band_data = download_satellite_bands_from_item(
                        item, bands, to_disk=to_disk, data_dir=None
                    )
                for band_name, band_data, src_transform, src_crs in band_data:
                    add_image_to_mosaic(
                        band_name, band_data, src_transform, src_crs, mosaic_path
                    )

            if os.path.exists(mosaic_path):
                size_gb = os.path.getsize(mosaic_path) / (1024**3)
                print(f"[PATCH {i}] Mosaic size so far: {size_gb:.2f} GB")

        except Exception as e:
            print(f"[ERROR] Patch {i} failed: {e}")

    # Final step: compress the mosaic after all patches are added
    compress_mosaic(mosaic_path)


def should_skip_mosaic(path, mission_config, date_str, threshold=0.8):
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
    date,
    bbox,
    sentinel_mission,
    landsat5_mission,
    landsat7_mission,
    sentinel2_mosaic_path,
    landsat5_mosaic_path,
    landsat7_mosaic_path,
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

    Returns:
        dict: A result dictionary with the date and any errors encountered.
    """
    init_logger("download_worker_log.txt")  # optional: make this configurable

    logging.info(f"Started processing for date: {date}")
    missions = ["sentinel-2", "landsat-5", "landsat-7"]
    mission_paths = [sentinel2_mosaic_path, landsat5_mosaic_path, landsat7_mosaic_path]

    result = {"date": date, "errors": []}
    for mission_number, mission_name in enumerate(missions):
        try:
            mission_config = get_mission(mission_name)
            base, _ = os.path.splitext(mission_paths[mission_number])
            mname = f"{base}_{date.replace('/', '_')}.tif"
            if should_skip_mosaic(mname, mission_config, date):
                continue
            create_mosaic_placeholder(
                mosaic_path=mname,
                bbox=reproject_bbox(bbox),
                resolution=mission_config["resolution"],
                mission=mission_name,
                crs="EPSG:32618",
                dtype="float32",
            )
            patchwise_query_download_mosaic(
                mname,
                bbox,
                mission_name,
                mission_config["resolution"] * 1000,
                mission_config["resolution"],
                mission_config["bands"],
                date,
                None,
                to_disk=False,
            )
        except Exception as e:
            result["errors"].append(f"{mission_name} error: {e}")

    return result


def compute_ndwi(green, nir, profile, out_path=None, display=False, threshold=0.2):
    """
    Computes the Normalized Difference Water Index (NDWI) from Green and NIR bands.

    NDWI highlights surface water by leveraging the reflectance difference between the green and near-infrared bands:
        NDWI = (Green - NIR) / (Green + NIR)

    A threshold (e.g., NDWI > 0.2) can be used to create a binary water mask for Sentinel-2.

    Parameters:
        green_image_data (np.ndarray): The (Green) GeoTIFF image array.
        nir_image_data (np.ndarray): The (NIR) GeoTIFF image array.
        out_path (str, optional): Path to save the output NDWI binary mask (GeoTIFF). If None, the result is not saved.
        display (bool, optional): If True, displays the binary mask using matplotlib.

    Returns:
        numpy.ndarray: Binary NDWI water mask array (1 = water, 0 = non-water).
    """
    ndwi = (green - nir) / (green + nir + 1e-10)
    ndwi_mask = (ndwi > threshold).astype(float)
    profile.update({"BIGTIFF": "IF_SAFER"})

    if out_path:
        with rasterio.open(out_path, "w", **profile, BIGTIFF="YES") as dst:
            dst.write(ndwi_mask, 1)

    if display:
        plt.imshow(ndwi_mask, cmap="gray")
        plt.title("NDWI Water Mask")
        plt.axis("off")
        plt.show()

    return ndwi_mask
