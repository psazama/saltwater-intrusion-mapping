import argparse
from pathlib import Path

from swmaps.pipeline.landsat import estimate_salinity_from_mosaic


def main():
    parser = argparse.ArgumentParser(
        description="Estimate salinity from Landsat mosaic"
    )
    parser.add_argument(
        "--mosaic", type=str, required=True, help="Path to Landsat mosaic GeoTIFF"
    )
    parser.add_argument(
        "--water-threshold",
        type=float,
        default=0.2,
        help="NDWI water threshold (default: 0.2)",
    )
    args = parser.parse_args()

    mosaic_path = Path(args.mosaic)

    print(f"Estimating salinity products for {mosaic_path}...")
    outputs = estimate_salinity_from_mosaic(
        mosaic_path, water_threshold=args.water_threshold
    )

    if outputs:
        for k, v in outputs.items():
            print(f"{k}: {v}")
    else:
        print("Salinity estimation failed.")


if __name__ == "__main__":
    main()
