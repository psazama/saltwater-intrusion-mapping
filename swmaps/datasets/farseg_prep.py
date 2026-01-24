from pathlib import Path

import rasterio
from rasterio.windows import Window

from swmaps.datasets.cdl import download_cdl_and_imagery


def prepare_farseg_dataset(region, year, mission, output_dir, tile_size=512):
    """
    Orchestrates downloading, aligning, and tiling for FarSeg training.
    """
    data = download_cdl_and_imagery(mission, region, year)

    for img_path, mask_path in zip(data["imagery"], data["cdl_aligned"]):
        create_farseg_tiles(img_path, mask_path, output_dir, tile_size)


def create_farseg_tiles(image_path, mask_path, output_dir, tile_size):
    """
    Slices large GeoTIFFs into the patches expected by FarSeg.
    """
    img_out = Path(output_dir) / "train" / "images"
    mask_out = Path(output_dir) / "train" / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    base_name = Path(image_path).stem

    with rasterio.open(image_path) as src_img, rasterio.open(mask_path) as src_mask:
        # Get dimensions of the large image
        h, w = src_img.height, src_img.width

        # Iterate through the image in steps of tile_size
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Define the window
                # We use min() to ensure we don't go out of bounds
                window = Window(x, y, min(tile_size, w - x), min(tile_size, h - y))

                # Check if the tile is full size (FarSeg prefers consistent shapes)
                if window.width != tile_size or window.height != tile_size:
                    continue

                # Read the data from the windows
                img_data = src_img.read(window=window)
                mask_data = src_mask.read(1, window=window)

                # Skip tiles that are purely NoData (all zeros or all 255)
                if img_data.max() == 0 or mask_data.max() == 0:
                    continue

                # Create unique filenames for each tile
                tile_id = f"{base_name}_y{y}_x{x}"
                img_tile_path = img_out / f"{tile_id}.tif"
                mask_tile_path = mask_out / f"{tile_id}.tif"

                # Define profiles for the new small GeoTIFFs
                img_profile = src_img.profile.copy()
                img_profile.update(
                    {
                        "height": tile_size,
                        "width": tile_size,
                        "transform": src_img.window_transform(window),
                    }
                )

                mask_profile = src_mask.profile.copy()
                mask_profile.update(
                    {
                        "height": tile_size,
                        "width": tile_size,
                        "transform": src_mask.window_transform(window),
                        "count": 1,
                    }
                )

                # Write the tiles
                with rasterio.open(img_tile_path, "w", **img_profile) as dst_img:
                    dst_img.write(img_data)

                with rasterio.open(mask_tile_path, "w", **mask_profile) as dst_mask:
                    dst_mask.write(mask_data, 1)
