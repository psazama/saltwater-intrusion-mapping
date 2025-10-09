import argparse
import logging
import tomllib
from pathlib import Path

import geopandas as gpd

from swmaps.config import data_path, set_data_root
from swmaps.pipeline.coastal import create_coastal
from swmaps.pipeline.download import download_data
from swmaps.pipeline.landsat import estimate_salinity_from_mosaic
from swmaps.pipeline.masks import generate_masks
from swmaps.pipeline.overlays import fetch_cdl_overlay, fetch_nlcd_overlay
from swmaps.pipeline.salinity import salinity_pipeline
from swmaps.pipeline.trend import trend_heatmap


def main():
    parser = argparse.ArgumentParser(description="General saltwater intrusion workflow")
    parser.add_argument("--config", required=True, help="Path to config file (TOML)")
    parser.add_argument(
        "--profile",
        default="demo",
        help="Profile name under [steps.<profile>] in the config file (default: demo)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (e.g. -v for DEBUG, -vv for very detailed)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    profile = args.profile
    if profile not in cfg.get("steps", {}):
        raise ValueError(f"Profile {profile!r} not found in config file")

    # Determine output root and ensure downstream helpers write there
    output_root = Path(cfg.get("output_root", data_path())).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    set_data_root(output_root)

    # Set logging level based on verbosity
    if args.verbose >= 1:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")

    steps = cfg["steps"][profile]
    params = cfg.get("parameters", {})

    # Coastal AOI
    if steps.get("coastal"):
        logging.info("Creating coastal AOI polygon")
        create_coastal(use_bbox=True, output_root=output_root)

    # Download imagery
    if steps.get("download"):
        logging.info("Downloading imagery")
        download_data(
            dates=[
                f"{d}"
                for mission, dranges in cfg.get("missions", {}).items()
                for d in dranges
            ],
            inline_mask=params.get("inline_mask", False),
            max_items=params.get("max_items", 1),
            multithreaded=params.get("multithreaded", False),
        )

    # NDWI masks
    if steps.get("ndwi"):
        logging.info("Generating NDWI water masks")
        generate_masks(center_size=params.get("center_size"))

    # Trend
    if steps.get("trend"):
        logging.info("Computing trend heatmap")
        trend_heatmap()

    # Overlay
    if steps.get("overlays"):
        logging.info("Fetching NLCD/CDL overlays")
        region = cfg["region"]
        gdf = gpd.read_file(region).to_crs("EPSG:4326")
        geometry = gdf.unary_union
        for mission, dranges in cfg.get("missions", {}).items():
            for date_range in dranges:
                year = int(date_range.split("-")[0])
                fetch_nlcd_overlay(geometry, year)
                fetch_cdl_overlay(geometry, year)

    # Salinity Overlay
    if steps.get("salinity-overlay"):
        for mission, dranges in cfg.get("missions", {}).items():
            for date_range in dranges:
                tag = date_range.replace("/", "_")
                mosaic_file = output_root / f"{mission.replace('-', '')}_salinity_{tag}.tif"
                if mosaic_file.exists():
                    estimate_salinity_from_mosaic(mosaic_file)

    # Get Salinity Labels
    if steps.get("salinity-labels"):
        logging.info("Estimating salinity")
        salinity_pipeline(
            truth_dir=params.get("truth_dir"),
            truth_file=params.get("truth_file"),
        )

    # Create Salinity Groundtruth
    if steps.get("salinity-truth"):
        logging.info("Building/loading salinity truth data")
        build_salinity_truth(dataset_files=params.get("truth_nc_files"))
        df = load_salinity_truth(params.get("truth_file"))
        logging.info("Loaded %d truth samples", len(df))

    # Extract Salinity Features
    if steps.get("salinty-features"):
        logging.info("Extracting salinity features from mosaic")
        extract_salinity_features_from_mosaic(
            mosaic_path=params["mosaic_file"],
            mission_band_index=params["band_index"],
            output_feature_path=params["feature_tif"],
            output_mask_path=params["mask_tif"],
        )

    # Train Deng et al. XGBoost
    if steps.get("xgboost-train"):
        logging.info("Training Deng et al. XGBoost model")
        X = ...  # load feature arrays
        y = ...  # load labels
        train_salinity_deng(X, y, save_model_path=params.get("model_out"))

    logging.info("✅ Workflow finished successfully")


if __name__ == "__main__":
    main()
