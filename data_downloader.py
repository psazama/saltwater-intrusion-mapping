import rasterio
from pystac_client import Client
from rasterio.session import AWSSession
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import boto3
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def download_satellite_bands(mission="sentinel-2", bbox=None, date_range=None, data_dir="data"):
    """
    Downloads selected bands for either Sentinel-2 or Landsat-5 imagery using the AWS Earth Search STAC API.

    This function dynamically selects the appropriate STAC collection and band identifiers based on the chosen
    satellite mission. Currently supports:
        - Sentinel-2 Level-2A (e.g., B03 = green, B08 = NIR)
        - Landsat-5 Level-2 (e.g., green, nir08)

    Parameters:
        mission (str): The satellite mission to use ("sentinel-2" or "landsat-5").
        bbox (list): Bounding box [min_lon, min_lat, max_lon, max_lat] defining the area of interest.
        date_range (str): ISO8601 date range string (e.g., '2010-01-01/2010-12-31').
        data_dir (str): Local directory where downloaded TIFF files will be stored.

    Returns:
        dict: Mapping of band short names to file paths (e.g., {"green": "/path/to/green.tif", ...}).
    """
    if mission == "sentinel-2":
        collection = "sentinel-2-l2a"
        bands = {
            "green": "green",
            "red": "red",
            "nir08": "nir",
            "swir16": "swir1",
            "swir22": "swir2"
        }
    elif mission == "landsat-5":
        collection = "landsat-5-l1"  # Or "landsat-5-c2-l1" if using Collection 2
        bands = {
            "B1": "blue",
            "B2": "green",
            "B3": "red",
            "B4": "nir",
            "B5": "swir1",
            "B7": "swir2"
        }
    else:
        raise ValueError("Unsupported mission")

    catalog = Client.open("https://earth-search.aws.element84.com/v1")
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 10}},
        max_items=1
    )
    
    items = list(search.get_items())
    if not items:
        raise ValueError("No items found for your search.")
    
    item = items[0]
    print(f"Found item: {item.id}")
    
    # Create output directory
    output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Enable unsigned S3 access
    session = boto3.Session()
    aws_session = AWSSession(session=session, requester_pays=True)
    
    last_out_path = None    
    for key, name in bands.items():
        if key in item.assets:
            href = item.assets[key].href
            out_path = os.path.join(output_dir, f"{item.id}_{name}.tif")
        
            print(f"Downloading {name} from {href} ...")
            with rasterio.Env(aws_session):
                with rasterio.open(href) as src:
                    profile = src.profile
                    data = src.read(1)
        
                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(data, 1)
            last_out_path = out_path
        else:
            print(f"[WARNING] Band {key} not available in item {item.id}. Available bands: {list(item.assets.keys())}")
    
    return last_out_path

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
