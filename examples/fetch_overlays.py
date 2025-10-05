"""CLI for downloading NLCD and CDL overlays for a provided region geometry."""

import argparse
from pathlib import Path

import geopandas as gpd

from swmaps.pipeline.overlays import fetch_cdl_overlay, fetch_nlcd_overlay
def main() -> None:
    """Download NLCD and CDL rasters for the requested region and year.

    Args:
        None

    Returns:
        None: File paths for downloaded rasters are printed to stdout.
    """
    parser = argparse.ArgumentParser(description="Fetch NLCD/CDL overlays")
    parser.add_argument(
        "--region", type=str, required=True, help="Path to region GeoJSON or GPKG file"
    )
    parser.add_argument(
        "--year", type=int, required=True, help="Year for overlay (e.g. 2016)"
    )
    args = parser.parse_args()

    # Load AOI geometry
    gdf = gpd.read_file(Path(args.region)).to_crs("EPSG:4326")
    geom = gdf.unary_union

    print(f"Fetching NLCD for {args.year}...")
    nlcd_path = fetch_nlcd_overlay(geom, args.year)
    print(f"Saved NLCD to {nlcd_path}")

    print(f"Fetching CDL for {args.year}...")
    cdl_path = fetch_cdl_overlay(geom, args.year)
    print(f"Saved CDL to {cdl_path}")


if __name__ == "__main__":
    main()
