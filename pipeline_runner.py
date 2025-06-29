import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
from tqdm import tqdm

from swmaps.config import data_path
from swmaps.core.download_tools import (
    download_matching_images,
    find_satellite_coverage,
    get_mission,
    process_date,
)
from swmaps.core.salinity_tools import (
    build_salinity_truth,
    extract_salinity_features_from_mosaic,
    load_salinity_truth,
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
    ),
    parser.add_argument(
        "--salinity_truth_directory",
        type=str,
        default=None,
        required=False,
        help="Build groundtruth salinity dataframe with .nc files in this directory",
    )
    parser.add_argument(
        "--salinity_truth_file",
        type=str,
        default=str(data_path("salinity_labels", "codc_salinity_profiles.csv")),
        required=False,
        help="Load salinity ground truth from file",
    )
    parser.add_argument(
        "--inline_mask",
        action="store_true",
        help="If set, create a water-mask after each mosaic is finished and delete the mosaic.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=max(os.cpu_count() // 2, 1),
        help="Parallel workers for imagery processing",
    )
    args = parser.parse_args()

    # Load the GeoJSON
    geojson = Path(__file__).resolve().parent / "config" / "easternshore.geojson"
    gdf = gpd.read_file(geojson)

    # Ensure it's in WGS84 (EPSG:4326) for STAC API compatibility
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Get bounding box: [min_lon, min_lat, max_lon, max_lat]
    bbox = gdf.total_bounds.tolist()
    missions = {
        "sentinel-2": get_mission("sentinel-2"),
        "landsat-5": get_mission("landsat-5"),
        "landsat-7": get_mission("landsat-7"),
    }

    # base mosaic filenames (no date suffix)
    sentinel2_mosaic = data_path("sentinel_eastern_shore.tif")
    landsat5_mosaic = data_path("landsat5_eastern_shore.tif")
    landsat7_mosaic = data_path("landsat7_eastern_shore.tif")

    ##### Data Downloading ######
    if args.step == 0:
        with open(
            Path(__file__).resolve().parent / "config" / "date_range.json", "r"
        ) as fh:
            dates = json.load(fh)["date_ranges"]

        dates_to_run = dates[:36:120]
        results = []

        max_workers = os.cpu_count() // 2
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_date,
                    date,
                    bbox,
                    missions["sentinel-2"],
                    missions["landsat-5"],
                    missions["landsat-7"],
                    sentinel2_mosaic,
                    landsat5_mosaic,
                    landsat7_mosaic,
                    args.inline_mask,
                ): date
                for date in dates_to_run
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results.append(result)
                if result["errors"]:
                    print(f"[{result['date']}] Errors: {result['errors']}")
        return

    ### Get Water Masks ###
    if args.step == 1:
        return

    ### Find Overlaps of Salinity Measures ###
    if args.step == 2:
        if args.salinity_truth_directory:
            codc_files = sorted(Path(args.salinity_truth_directory).glob("*.nc"))
            build_salinity_truth(
                dataset_files=codc_files, output_csv=args.salinity_truth_file
            )
        y = load_salinity_truth(args.salinity_truth_file)

        overlapped_measurements = find_satellite_coverage(y)
        overlapped_measurements_filter = overlapped_measurements[
            overlapped_measurements["covered_by"].apply(lambda x: len(x) > 0)
        ]

        datapath_df = download_matching_images(overlapped_measurements_filter)
        datapath_df = data_path("ground_truth_matches.csv")
        datapath_df.to_csv(datapath_df, index=False)
        print(f"Downloaded matches listed in {datapath_df}")

        test_data = datapath_df.iloc[0][0]
        test_mission = get_mission(test_data["covered_by"][0])
        test_mosaic = test_data["downloaded_files"][0]

        base = Path(test_mosaic).with_suffix("")
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

        return


if __name__ == "__main__":
    main()
