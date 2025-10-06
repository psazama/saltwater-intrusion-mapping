"""CLI helper for building the study region's coastal polygon."""

import argparse

from swmaps.pipeline.coastal import create_coastal

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-bbox", action="store_true")
    args = parser.parse_args()
    create_coastal(args.use_bbox)
