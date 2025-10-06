"""Minimal CLI wrapper around the salinity pipeline entry point."""

import argparse

from swmaps.pipeline.salinity import salinity_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-dir", type=str)
    parser.add_argument("--truth-file", type=str)
    args = parser.parse_args()
    salinity_pipeline(truth_dir=args.truth_dir, truth_file=args.truth_file)
