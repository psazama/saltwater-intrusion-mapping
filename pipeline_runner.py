"""Command-line entry point for executing the saltwater intrusion pipeline."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
from tqdm import tqdm

from swmaps.config import data_path
from swmaps.core.download_tools import (
    compute_ndwi,
    create_coastal_poly,
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
from swmaps.core.water_trend import (
    check_image_for_nans,
    check_image_for_valid_signal,
    load_wet_year,
    pixel_trend,
    plot_trend_heatmap,
    save_trend_results,
)
def main() -> None:
    """Parse CLI arguments and orchestrate the pipeline execution steps.

    Args:
        None

    Returns:
        None: The CLI performs work as a side effect—downloading imagery,
        generating mosaics, and computing salinity/trend outputs based on the
        requested step.
    """
    parser = argparse.ArgumentParser(
        description="Saltwater Intrustion Detection Runner"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        required=False,
        choices=[0, 1, 2, 3, 4],
        help=(
            "Processing step to begin on ("
            "0 = coastal polygon, 1 = data download, 2 = water mask creation, "
            "3 = water trend heatmap, 4 = salinity pipeline)"
        ),
    )
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
    parser.add_argument(
        "--bbox",
        action="store_true",
        help="Use full bounding box instead of coastal band.",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=1,
        help="Number of items to get for each patch region during imagery download",
    )
    parser.add_argument(
        "--multithreaded",
        action="store_true",
        help="Use the multithreaded version of the download functions",
    )
    parser.add_argument(
        "--center_size",
        type=int,
        default=None,
        help="Number of items to get for each patch region during imagery download",
    )
    args = parser.parse_args()

    if args.bbox:
        # Load the GeoJSON
        geojson = Path(__file__).resolve().parent / "config" / "somerset.geojson"
        gdf = gpd.read_file(geojson)

        # Ensure it's in WGS84 (EPSG:4326) for STAC API compatibility
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        # Get bounding box: [min_lon, min_lat, max_lon, max_lat]
        bbox = gdf.total_bounds.tolist()
    else:
        band_path = Path(__file__).resolve().parent / "config" / "coastal_band.gpkg"
        bbox = gpd.read_file(band_path, layer="coastal_band").geometry.iloc[0]
        geojson = band_path

    missions = {
        "sentinel-2": get_mission("sentinel-2"),
        "landsat-5": get_mission("landsat-5"),
        "landsat-7": get_mission("landsat-7"),
    }

    # base mosaic filenames (no date suffix)
    sentinel2_mosaic = data_path("sentinel_eastern_shore.tif")
    landsat5_mosaic = data_path("landsat5_eastern_shore.tif")
    landsat7_mosaic = data_path("landsat7_eastern_shore.tif")

    ##### Generate Bounding Box #####
    if args.step == 0:
        print("Creating Coastal Polygon")
        create_coastal_poly(geojson)
        return

    ##### Data Downloading ######
    if args.step == 1:
        with open(
            Path(__file__).resolve().parent / "config" / "date_range.json", "r"
        ) as fh:
            dates = json.load(fh)["date_ranges"]

        # dates_to_run = dates[7::12]
        dates_to_run = dates[6::12] + dates[7::12] + dates[8::12]
        print(dates_to_run)
        results = []

        if args.multithreaded:
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
                        max_items=args.max_items,
                    ): date
                    for date in dates_to_run
                }

                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    results.append(result)
                    if result["errors"]:
                        print(f"[{result['date']}] Errors: {result['errors']}")
        else:
            for date in tqdm(dates_to_run):
                print(date)
                result = process_date(
                    date,
                    bbox,
                    missions["sentinel-2"],
                    missions["landsat-5"],
                    missions["landsat-7"],
                    sentinel2_mosaic,
                    landsat5_mosaic,
                    landsat7_mosaic,
                    args.inline_mask,
                    max_items=args.max_items,
                )
                print(f"finished processing: {date}")
                results.append(result)
                if result["errors"]:
                    print(f"[{result['date']}] Errors: {result['errors']}")

        return

    ### Generate Water Masks ###
    if args.step == 2:
        data_dir = data_path()
        tifs = sorted(data_dir.glob("*.tif"))

        for tif in tqdm(tifs):
            if tif.name.endswith("_mask.tif") or tif.name.endswith("_features.tif"):
                continue
            if check_image_for_nans(str(tif)) or not check_image_for_valid_signal(
                str(tif)
            ):
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
            try:
                compute_ndwi(
                    str(tif),
                    mission,
                    out_path=str(out_mask),
                    display=False,
                    center_size=args.center_size,
                )
            except Exception as e:
                # skip any invalid tifs
                print(f"[ERROR] skipping invalid tiff: {tif}, {e}")
                continue

        print("Water masks generated")
        return

    ### Water Trend Heatmap ###
    if args.step == 3:
        data_dir = data_path()
        mask_files = sorted(data_dir.glob("*_mask.tif"))

        mask_glob = [str(p) for p in mask_files if p.exists()]
        if mask_glob:
            wet_year = load_wet_year(mask_glob[::20], chunks={"x": 512, "y": 512})
            slope, pval = pixel_trend(wet_year)
            signif = pval < 0.05
            ax = plot_trend_heatmap(
                slope, signif, title="Trend in % wet months per year"
            )
            heatmap_file = data_path("water_trend_heatmap.png")
            ax.figure.savefig(heatmap_file, bbox_inches="tight")
            print(f"Saved trend heatmap to {heatmap_file}")

            save_trend_results(slope, pval, data_path("water_trend"))
        else:
            print("No mask files found for water trend analysis")
        return

    ### Find Overlaps of Salinity Measures ###
    if args.step == 4:
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
