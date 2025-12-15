#!/usr/bin/env python3

"""
Workflow runner for the saltwater intrusion mapping pipeline (GEE version).

This script:
  1. Loads a TOML config
  2. Generates coastal AOI shapefile (if enabled)
  3. Downloads imagery using the GEE-based pipeline
  4. Executes salinity truth-matching + feature extraction pipeline
"""

import argparse
import logging
import tomllib  # Python 3.11+ TOML parser
from pathlib import Path

from swmaps.config import data_path
from swmaps.pipeline.download import download_data
from swmaps.pipeline.landsat_salinity import estimate_salinity_from_mosaic
from swmaps.pipeline.masks import generate_masks
from swmaps.pipeline.salinity import (
    salinity_pipeline,
)
from swmaps.pipeline.trend import trend_heatmap

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def ensure_output_dirs(cfg: dict):
    """Create directories listed in the config."""
    out = cfg.get("out_dir")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="GEE Saltwater Intrusion Pipeline Runner"
    )
    parser.add_argument(
        "--config", required=True, help="Path to TOML configuration file"
    )
    args = parser.parse_args()

    # -----------------------------------------------------------
    # Load configuration
    # -----------------------------------------------------------
    cfg = load_config(args.config)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # -----------------------------------------------------------
    # Prepare output directories
    # -----------------------------------------------------------
    ensure_output_dirs(cfg)

    # -----------------------------------------------------------
    # Step 1 — Create AOI (if configured)
    # -----------------------------------------------------------
    if cfg.get("make_coastal_aoi", False):
        logging.info("Creating coastal AOI polygon")
        print("Creating Coastal Polygon")

        from swmaps.pipeline.coastal import build_coastal_polygon

        out_dir = cfg.get("aoi_out", data_path("aoi"))

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        build_coastal_polygon(
            region=cfg["region"],
            output_path=Path(out_dir) / "coastal_aoi.geojson",
            buffer_km=cfg.get("coastal_buffer_km", 10),
        )

        logging.info("AOI creation complete")

    # -----------------------------------------------------------
    # Step 2 — Download imagery using the GEE pipeline
    # -----------------------------------------------------------
    logging.info("Downloading imagery")
    print("Downloading imagery")

    try:
        download_results = download_data(cfg)
    except Exception as e:
        logging.error(f"Download step failed: {e}")
        raise

    logging.info(f"Download step complete: {len(download_results)} files")

    # -----------------------------------------------------------
    # Step 3 — Optional salinity truth processing
    # -----------------------------------------------------------

    if cfg.get("run_salinity_pipeline", False):
        logging.info("Running salinity pipeline")
        try:
            salinity_pipeline(
                truth_download_list=cfg.get("truth_download_list"),
                truth_dir=cfg.get("truth_dir"),
                truth_file=cfg.get("truth_file"),
            )
        except Exception as e:
            logging.error(f"Salinity pipeline failed: {e}")
            raise

        logging.info("Salinity pipeline complete")

    logging.info("Workflow complete!")

    # -----------------------------------------------------------
    # Step 4 — Salinity Classification
    # -----------------------------------------------------------

    download_dir = Path(cfg.get("out_dir", data_path("mosaics")))

    if cfg.get("run_salinity_classification", False):
        logging.info("Running salinity classification")

        for mosaic_path in sorted(download_dir.rglob("*.tif")):
            # Skip derived products
            if mosaic_path.name.endswith(
                (
                    "_salinity_score.tif",
                    "_salinity_class.tif",
                    "_salinity_water_mask.tif",
                )
            ):
                continue

            name = mosaic_path.name.lower()

            if "landsat" in name:
                estimate_salinity_from_mosaic(
                    mosaic_path=mosaic_path,
                    water_threshold=cfg.get("water_threshold", 0.2),
                )
            elif "sentinel" in name:
                logging.info(
                    "Sentinel mosaic detected for %s; salinity not yet implemented",
                    mosaic_path.name,
                )
            else:
                logging.warning(
                    "Unable to infer sensor for %s; skipping salinity classification",
                    mosaic_path.name,
                )

    # -----------------------------------------------------------
    # Step 5 — Water Trend Analysis
    # -----------------------------------------------------------

    if cfg.get("run_water_trend", False):
        logging.info("Generating water trend products")
        generate_masks(input_dir=download_dir)
        trend_heatmap(output_dir=cfg.get("trend_output_dir", download_dir))

    # ---------------------------------------------------------------------


if __name__ == "__main__":
    main()
