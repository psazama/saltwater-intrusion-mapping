"""
Example Landsat workflow for the Somerset study region.

Steps:
1. Load AOI from config/somerset.geojson
2. Build Landsat mosaics for configured date ranges
3. Compute NDWI masks
4. Estimate salinity rasters
5. Fetch NLCD and CDL overlays
"""

import logging
from datetime import datetime
from pathlib import Path

import geopandas as gpd

from swmaps.config import data_path
from swmaps.core.indices import compute_ndwi
from swmaps.core.missions import get_mission
from swmaps.core.mosaic import (
    create_mosaic_placeholder,
    patchwise_query_download_mosaic,
)
from swmaps.core.raster_utils import reproject_bbox
from swmaps.pipeline.landsat import estimate_salinity_from_mosaic
from swmaps.pipeline.overlays import fetch_cdl_overlay, fetch_nlcd_overlay

GEOJSON_PATH = Path("config/somerset.geojson")
OUTPUT_ROOT = data_path("examples", "somerset_landsat")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MISSION_DATE_RANGES = {
    "landsat-5": ["1990-07-01/1990-07-31", "2005-07-01/2005-07-31"],
    "landsat-7": ["2000-07-01/2000-07-31", "2010-07-01/2010-07-31"],
}


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    gdf = gpd.read_file(GEOJSON_PATH).to_crs("EPSG:4326")
    geometry = gdf.unary_union
    bounds = list(geometry.bounds)
    projected_bounds = reproject_bbox(geometry)

    for mission, date_ranges in MISSION_DATE_RANGES.items():
        mission_cfg = get_mission(mission)
        mission_tag = mission.replace("-", "")

        for date_range in date_ranges:
            tag = date_range.replace("/", "_")
            mosaic_path = OUTPUT_ROOT / f"{mission_tag}_somerset_{tag}.tif"

            if not mosaic_path.exists():
                logging.info("Creating mosaic for %s %s", mission, date_range)
                create_mosaic_placeholder(
                    mosaic_path=mosaic_path,
                    bbox=projected_bounds,
                    mission=mission,
                    resolution=mission_cfg["resolution"],
                    crs="EPSG:32618",
                    dtype="float32",
                )
                patchwise_query_download_mosaic(
                    mosaic_path=mosaic_path,
                    bbox=bounds,
                    mission=mission,
                    resolution=mission_cfg["resolution"],
                    bands=mission_cfg["bands"],
                    date_range=date_range,
                    base_output_path=mosaic_path.parent,
                    to_disk=True,
                    multithreaded=False,
                    max_items=3,
                )

            # Step 2: NDWI mask
            ndwi_path = mosaic_path.with_name(f"{mosaic_path.stem}_ndwi_mask.tif")
            if not ndwi_path.exists():
                logging.info("Computing NDWI for %s", mosaic_path.name)
                compute_ndwi(
                    str(mosaic_path), mission, out_path=str(ndwi_path), threshold=0.2
                )

            # Step 3: Salinity estimation
            estimate_salinity_from_mosaic(mosaic_path)

            # Step 4: NLCD/CDL overlays
            year = datetime.fromisoformat(date_range.split("/")[0]).year
            fetch_nlcd_overlay(geometry, year)
            fetch_cdl_overlay(geometry, year)


if __name__ == "__main__":
    main()
