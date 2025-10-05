"""CLI wrapper for generating NDWI water masks from downloaded mosaics."""

import argparse

from swmaps.pipeline.masks import generate_masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--center-size", type=int, default=None)
    args = parser.parse_args()
    generate_masks(center_size=args.center_size)
