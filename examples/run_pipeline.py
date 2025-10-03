import argparse
from pathlib import Path

from swmaps.pipeline.coastal import create_coastal
from swmaps.pipeline.download import download_data
from swmaps.pipeline.masks import generate_masks
from swmaps.pipeline.salinity import salinity_pipeline
from swmaps.pipeline.trend import trend_heatmap


def main():
    parser = argparse.ArgumentParser(
        description="Run full saltwater intrusion pipeline"
    )
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        help="Region identifier (expects matching config/ file, e.g. 'somerset')",
    )
    parser.add_argument(
        "--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, required=True, help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--truth-dir",
        type=str,
        default=None,
        help="Optional salinity ground truth .nc directory",
    )
    parser.add_argument(
        "--truth-file",
        type=str,
        default=None,
        help="Optional path to salinity truth CSV",
    )
    parser.add_argument(
        "--inline-mask",
        action="store_true",
        help="If set, generate water masks immediately after mosaics",
    )
    parser.add_argument(
        "--multithreaded", action="store_true", help="Enable parallel downloads"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=1,
        help="Number of items per patch for downloads",
    )
    parser.add_argument(
        "--center-size", type=int, default=None, help="Center crop size for NDWI"
    )
    args = parser.parse_args()

    # -----------------------------
    # Step 0: Create/load region AOI
    # -----------------------------
    config_dir = Path(__file__).resolve().parents[1] / "config"
    region_file = config_dir / f"{args.region}.geojson"
    if not region_file.exists():
        raise FileNotFoundError(f"Region file not found: {region_file}")
    print(f"Using region file: {region_file}")

    create_coastal(use_bbox=True)  # here we just trigger coastal poly

    # -----------------------------
    # Step 1: Download imagery
    # -----------------------------
    print(f"Downloading imagery from {args.start_date} to {args.end_date}...")
    # Here we build the date list directly instead of using pre-saved JSON
    dates = [
        f"{args.start_date} to {args.end_date}"
    ]  # Replace with actual date splitting if needed
    download_data(
        dates=dates,
        inline_mask=args.inline_mask,
        max_items=args.max_items,
        multithreaded=args.multithreaded,
    )

    # -----------------------------
    # Step 2: Generate water masks
    # -----------------------------
    print("Generating NDWI water masks...")
    generate_masks(center_size=args.center_size)

    # -----------------------------
    # Step 3: Compute trend heatmap
    # -----------------------------
    print("Computing water trend heatmap...")
    trend_heatmap()

    # -----------------------------
    # Step 4: Salinity pipeline
    # -----------------------------
    if args.truth_dir or args.truth_file:
        print("Running salinity pipeline...")
        salinity_pipeline(truth_dir=args.truth_dir, truth_file=args.truth_file)

    print("✅ Pipeline finished successfully.")


if __name__ == "__main__":
    main()
