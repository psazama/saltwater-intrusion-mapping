"""Pipeline helpers for deriving NDWI water masks from mosaics."""

from tqdm import tqdm

from swmaps.config import data_path
from swmaps.core.indices import compute_ndwi
from swmaps.core.water_trend import check_image_for_nans, check_image_for_valid_signal


def generate_masks(center_size=None):
    """Generate NDWI water masks for all mosaics in the data directory.

    Args:
        center_size (int | None): Optional pixel size of a centred window to
            analyse when computing NDWI.

    Returns:
        None: Masks are written next to their source mosaics.
    """
    for tif in tqdm(sorted(data_path().glob("*.tif"))):
        if tif.name.endswith(("_mask.tif", "_features.tif")):
            continue
        if check_image_for_nans(str(tif)) or not check_image_for_valid_signal(str(tif)):
            continue

        if "sentinel" in tif.name:
            mission = "sentinel-2"
        elif "landsat5" in tif.name:
            mission = "landsat-5"
        elif "landsat7" in tif.name:
            mission = "landsat-7"
        else:
            continue

        out_mask = tif.with_name(f"{tif.stem}_mask.tif")
        compute_ndwi(
            str(tif), mission, str(out_mask), display=False, center_size=center_size
        )
    print("Water masks generated")
