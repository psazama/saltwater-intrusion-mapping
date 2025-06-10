import logging
import os

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
    else:
        raise ValueError("Unsupported mission")
    return {
        "bands": bands,
        "band_index": band_index,
        "collection": collection,
        "query_filter": query_filter,
        "resolution": resolution,
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

                    # Return single array if only one band, else list
                    return (
                        (data_list[0] if len(data_list) == 1 else data_list),
                        profile,
                        window,
                    )

    print("No valid window found.")
    return None, None, None


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
        mission (str): The satellite mission to use ("sentinel-2" or "landsat-5").
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
                    if src.count == 1:
                        data = src.read(1)
                    else:
                        raise ValueError(
                            f"Expected single-band image, but got {src.count} bands in {href}"
                        )

                    band_index = band_order.index(key) + 1
                    band_data.append((band_index, data, src.transform, src.crs))

                if to_disk:
                    with rasterio.open(out_path, "w", **profile, BIGTIFF="YES") as dst:
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

    with rasterio.open(mosaic_path, "w", **profile, BIGTIFF="YES") as dst:
        dst.write(np.full((height, width), np.nan, dtype=dtype), 1)

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
    with rasterio.open(mosaic_path, "r+", BIGTIFF="YES") as mosaic:
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
    Breaks region into patches and processes each separately.

    Parameters:
        bbox (list): [min_lon, min_lat, max_lon, max_lat]
        mission (str): Satellite mission
        patch_size_meters (int): Size of square patch
        resolution (int): Resolution in meters per pixel
        bands (dict): STAC asset key to friendly name
        date_range (str): Date range for querying imagery
        base_output_path (str): Where to store patches and mosaics
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

        except Exception as e:
            print(f"[ERROR] Patch {i} failed: {e}")


def process_date(
    date,
    bbox,
    sentinel_mission,
    landsat5_mission,
    sentinel2_mosaic_path,
    landsat5_mosaic_path,
):
    """
    Processes satellite data for a single date by creating and populating mosaics for both Landsat-5 and Sentinel-2.

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
        sentinel2_mosaic_path (str): Base path to output Sentinel-2 mosaics.
        landsat5_mosaic_path (str): Base path to output Landsat-5 mosaics.

    Returns:
        dict: A result dictionary with the date and any errors encountered.
    """
    init_logger("download_worker_log.txt")  # optional: make this configurable

    logging.info(f"Started processing for date: {date}")

    result = {"date": date, "errors": []}
    try:
        mname = landsat5_mosaic_path[:-4] + "_" + date.replace("/", "_") + ".tif"
        create_mosaic_placeholder(
            mosaic_path=mname,
            bbox=reproject_bbox(bbox),
            resolution=landsat5_mission["resolution"],
            mission="landsat-5",
            crs="EPSG:32618",
            dtype="float32",
        )
        patchwise_query_download_mosaic(
            mname,
            bbox,
            "landsat-5",
            landsat5_mission["resolution"] * 1000,
            landsat5_mission["resolution"],
            landsat5_mission["bands"],
            date,
            None,
            to_disk=False,
        )
    except Exception as e:
        result["errors"].append(f"Landsat5 error: {e}")

    try:
        mname = sentinel2_mosaic_path[:-4] + "_" + date.replace("/", "_") + ".tif"
        create_mosaic_placeholder(
            mosaic_path=mname,
            bbox=reproject_bbox(bbox),
            resolution=sentinel_mission["resolution"],
            mission="sentinel-2",
            crs="EPSG:32618",
            dtype="float32",
        )
        patchwise_query_download_mosaic(
            mname,
            bbox,
            "sentinel-2",
            sentinel_mission["resolution"] * 1000,
            sentinel_mission["resolution"],
            sentinel_mission["bands"],
            date,
            None,
            to_disk=False,
        )
    except Exception as e:
        result["errors"].append(f"Sentinel2 error: {e}")

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

    if out_path:
        with rasterio.open(out_path, "w", **profile, BIGTIFF="YES") as dst:
            dst.write(ndwi_mask, 1)

    if display:
        plt.imshow(ndwi_mask, cmap="gray")
        plt.title("NDWI Water Mask")
        plt.axis("off")
        plt.show()

    return ndwi_mask
