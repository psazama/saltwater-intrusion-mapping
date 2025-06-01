import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.session import AWSSession
import matplotlib.pyplot as plt
from pystac_client import Client
from tqdm import tqdm
import numpy as np
import boto3
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pyproj import Transformer

def reproject_bbox(bbox, src_crs="EPSG:4326", dst_crs="EPSG:32618"):
    minx, miny = bbox[0], bbox[1]
    maxx, maxy = bbox[2], bbox[3]

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    minx_proj, miny_proj = transformer.transform(minx, miny)
    maxx_proj, maxy_proj = transformer.transform(maxx, maxy)

    return [min(minx_proj, maxx_proj), min(miny_proj, maxy_proj),
            max(minx_proj, maxx_proj), max(miny_proj, maxy_proj)]

def query_satellite_items(mission="sentinel-2", bbox=None, date_range=None, debug=False):
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
    if mission == "sentinel-2":
        collection = "sentinel-2-l2a"
        query_filter = {"eo:cloud_cover": {"lt": 10}}
        bands = {
            "green": "green",
            "red": "red",
            "nir08": "nir",
            "swir16": "swir1",
            "swir22": "swir2"
        }
    elif mission == "landsat-5":
        collection = "landsat-c2-l2"
        query_filter = {
            "eo:cloud_cover": {"lt": 10},
            "platform": {"eq": "landsat-5"}
        }
        bands = {
            "blue": "blue",
            "green": "green",
            "red": "red",
            "nir08": "nir",
            "swir16": "swir1",
            "swir22": "swir2"
        }
    else:
        raise ValueError("Unsupported mission")

    catalog = Client.open("https://earth-search.aws.element84.com/v1")
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=date_range,
        query=query_filter,
        max_items=None
    )

    items = list(search.get_items())
    if not items:
        raise ValueError("No items found for your search.")

    if debug:
        for item in items:
            print(f"Found item: {item.id}")

    return items, bands

def download_satellite_bands_from_item(item, bands, to_disk=True, data_dir="data", debug=False):
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
    #from pystac_client import AWSSession

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
                    data = src.read(1)
                    band_index = band_order.index(key) + 1
                    band_data.append((band_index, data, src.transform, src.crs))

                if to_disk:
                    with rasterio.open(out_path, 'w', **profile) as dst:
                        dst.write(data, 1)
        else:
            print(f"[WARNING] Band {key} not available in item {item.id}. Available bands: {list(item.assets.keys())}")

    return band_data

def create_mosaic_placeholder(mosaic_path, bbox, resolution, crs="EPSG:32618", dtype="float32"):
    """
    Create an empty mosaic GeoTIFF file to be filled later.

    Parameters:
        mosaic_path (str): Path where the placeholder file will be saved.
        bbox (tuple): (minx, miny, maxx, maxy) in target CRS.
        resolution (float): Target pixel resolution in meters.
        crs (str): Coordinate reference system.
        dtype (str): Data type of the mosaic file.
    """
    minx, miny, maxx, maxy = bbox
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': dtype,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw',
        'nodata': np.nan
    }

    with rasterio.open(mosaic_path, "w", **profile) as dst:
        dst.write(np.full((height, width), np.nan, dtype=dtype), 1)

    return transform, width, height, crs

def add_image_to_mosaic(band_index, image_data, src_transform, src_crs, mosaic_path):
    """
    Reproject and insert a satellite image array into the mosaic placeholder.

    Parameters:
        image_data (np.ndarray): The image array to reproject and insert.
        src_transform (Affine): Affine transform of the source image.
        src_crs (str or CRS): CRS of the source image.
        mosaic_path (str): Path to the mosaic file to update.
    """
    with rasterio.open(mosaic_path, 'r+') as mosaic:
        mosaic_array = mosaic.read(band_index)

        reproject(
            source=image_data,
            destination=mosaic_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=mosaic.transform,
            dst_crs=mosaic.crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest
        )

        mosaic.write(mosaic_array, band_index)

def extract_and_save_patch(args):
    """
    Extracts a patch from a GeoTIFF and saves it to a new file.

    Parameters:
        args (tuple): (tiff_path, x, y, patch_size, output_path)

    Returns:
        dict: Dictionary with origin coordinates and file path of saved patch.
    """
    tiff_path, x, y, patch_size, output_path = args

    with rasterio.open(tiff_path) as src:
        window = rasterio.windows.Window(x, y, patch_size, patch_size)
        patch = src.read(1, window=window)

        # Recalculate the transform for this window
        transform = src.window_transform(window)

        # Update the profile with the new transform and dimensions
        updated_profile = src.profile.copy()
        updated_profile.update({
            "height": patch.shape[0],
            "width": patch.shape[1],
            "transform": transform,
            "count": 1,
            "dtype": patch.dtype,
        })

        if output_path is not None:
            patch_filename = os.path.join(output_path, f"{x}_{y}.tif")
            with rasterio.open(patch_filename, "w", **updated_profile) as dst:
                dst.write(patch, 1)

        return {
            "origin": (x, y),
            "filepath": patch_filename
        }

def parallel_extract_patches(tiff_path, patch_size=224, overlap=0.5, output_path=None, max_workers=4):
    """
    Extracts overlapping patches from a GeoTIFF and saves them in parallel.

    Parameters:
        tiff_path (str): Path to the input GeoTIFF.
        patch_size (int): Width and height of each patch.
        overlap (float): Fractional overlap between patches.
        output_path (str, optional): Directory to save patches.
        max_workers (int): Number of parallel workers.

    Returns:
        tuple: (List of patch metadata, image width, height, and original profile)
    """
    # Create output directory
    if output_path is not None:
        print(output_path)
        os.makedirs(output_path, exist_ok=True)
    
    stride = int(patch_size * (1 - overlap))
    patches = []
    tasks = []

    with rasterio.open(tiff_path) as src:
        width, height = src.width, src.height
        orig_profile = src.profile.copy()

        xs = list(range(0, width - patch_size + 1, stride))
        ys = list(range(0, height - patch_size + 1, stride))

        # Ensure right and bottom edges are included
        if xs[-1] + patch_size < width:
            xs.append(width - patch_size)
        if ys[-1] + patch_size < height:
            ys.append(height - patch_size)

        for y in ys:
            for x in xs:
                tasks.append((tiff_path, x, y, patch_size, output_path))

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_and_save_patch, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    return results, width, height, orig_profile

def compute_ndwi(green_path, nir_path, out_path=None, display=False):
    """
    Computes the Normalized Difference Water Index (NDWI) from Sentinel-2 Green (B03) and NIR (B08) bands.

    NDWI highlights surface water by leveraging the reflectance difference between the green and near-infrared bands:
        NDWI = (Green - NIR) / (Green + NIR)

    A threshold (e.g., NDWI > 0.2) can be used to create a binary water mask.

    Parameters:
        green_path (str): Path to the Sentinel-2 Band 3 (Green) GeoTIFF.
        nir_path (str): Path to the Sentinel-2 Band 8 (NIR) GeoTIFF.
        out_path (str, optional): Path to save the output NDWI binary mask (GeoTIFF). If None, the result is not saved.
        display (bool, optional): If True, displays the binary mask using matplotlib.

    Returns:
        numpy.ndarray: Binary NDWI water mask array (1 = water, 0 = non-water).
    """
    with rasterio.open(green_path) as gsrc, rasterio.open(nir_path) as nsrc:
        green = gsrc.read(1).astype(float)
        nir = nsrc.read(1).astype(float)
        profile = gsrc.profile.copy()

    ndwi = (green - nir) / (green + nir + 1e-10)
    ndwi_mask = (ndwi > 0.2).astype(float)

    if out_path:
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(ndwi_mask, 1)

    if display:
        plt.imshow(ndwi_mask, cmap='gray')
        plt.title("NDWI Water Mask")
        plt.axis('off')
        plt.show()

    return ndwi_mask


def stitch_patches_to_geotiff(output_path, patches_paths, orig_width, orig_height, orig_profile):
    """
    Reconstructs a full GeoTIFF from patch files, averaging overlaps.

    Parameters:
        output_path (str): Path to write the stitched GeoTIFF.
        patches_paths (list): List of dictionaries with patch file paths and origins.
        orig_width (int): Width of the original image.
        orig_height (int): Height of the original image.
        orig_profile (dict): Rasterio profile to apply to the output.
    """
    # Open destination file for writing
    with rasterio.open(output_path, "w", **orig_profile) as dst:
        
        # Create a separate weight array to average overlapping regions
        weight_array = np.zeros((orig_height, orig_width), dtype=np.float32)
        data_array = np.zeros((orig_height, orig_width), dtype=np.float32)

        for patch_dict in tqdm(patches_paths):
            patch_path = patch_dict["filepath"]
            x, y = patch_dict["origin"]

            with rasterio.open(patch_path) as src:
                patch = src.read(1)

            if patch.ndim == 3:
                patch = patch.squeeze()  # For shape (1, H, W)

            h, w = patch.shape
            data_array[y:y+h, x:x+w] += patch
            weight_array[y:y+h, x:x+w] += 1

        # Avoid division by zero
        weight_array[weight_array == 0] = 1
        averaged_result = data_array / weight_array

        # Write the final stitched data in one go or tile it as well
        dst.write(averaged_result, 1)
