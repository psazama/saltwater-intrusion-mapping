import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
from tqdm import tqdm

from download_tools import get_mission, process_date
from salinity_tools import (
    extract_salinity_features_from_mosaic,
    salinity_truth,
)


def main():
    parser = argparse.ArgumentParser(
        description="Saltwater Intrustion Detection Runner"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        required=False,
        choices=[0, 1, 2],
        help="Processing step to begin on (0 = data download, 1 = water mask creation, 2 = water map time comparison)",
    )
    args = parser.parse_args()

    # Load the GeoJSON
    gdf = gpd.read_file("easternshore.geojson")

    # Ensure it's in WGS84 (EPSG:4326) for STAC API compatibility
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Get bounding box: [min_lon, min_lat, max_lon, max_lat]
    bbox = gdf.total_bounds.tolist()
    sentinel_mission = get_mission("sentinel-2")
    landsat5_mission = get_mission("landsat-5")
    landsat7_mission = get_mission("landsat-7")
    landsat5_mosaic_path = "data/landsat5_eastern_shore.tif"
    landsat7_mosaic_path = "data/landsat7_eastern_shore.tif"
    sentinel2_mosaic_path = "data/sentinel_eastern_shore.tif"

    if args.step <= 0:
        with open("date_range.json", "r") as f:
            dates = json.load(f)

        dates_to_run = dates["date_ranges"][::3]
        results = []

        max_workers = os.cpu_count() // 2
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_date,
                    date,
                    bbox,
                    sentinel_mission,
                    landsat5_mission,
                    landsat7_mission,
                    sentinel2_mosaic_path,
                    landsat5_mosaic_path,
                    landsat7_mosaic_path,
                ): date
                for date in dates_to_run
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results.append(result)
                if result["errors"]:
                    print(f"[{result['date']}] Errors: {result['errors']}")

    if args.step <= 1:
        test_mosaic = "data/landsat5_eastern_shore_2006-09-01_2006-09-30.tif"
        test_mission = landsat5_mission

        y = salinity_truth()
        print(y.head())

        base, _ = os.path.splitext(test_mosaic)
        output_feature_path = f"{base}_features.tif"
        output_mask_path = f"{base}_mask.tif"
        X = extract_salinity_features_from_mosaic(
            test_mosaic,
            test_mission["band_index"],
            output_feature_path,
            output_mask_path,
        )
        print(X)

        # model, metrics = train_salinity_deng(X, y)


if __name__ == "__main__":
    main()
