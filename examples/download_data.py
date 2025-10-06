"""CLI helper for downloading imagery inputs for the pipeline."""

import argparse

from swmaps.pipeline.download import download_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inline-mask", action="store_true")
    parser.add_argument("--max-items", type=int, default=1)
    parser.add_argument("--multithreaded", action="store_true")
    args = parser.parse_args()
    download_data(
        inline_mask=args.inline_mask,
        max_items=args.max_items,
        multithreaded=args.multithreaded,
    )
