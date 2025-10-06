"""Download helpers for acquiring imagery used by the processing pipeline."""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from swmaps.config import data_path
from swmaps.core.missions import get_mission
from swmaps.core.mosaic import process_date


def download_data(dates=None, inline_mask=False, max_items=1, multithreaded=False):
    """Download imagery for one or more date ranges.

    Args:
        dates (Iterable[str] | None): Dates or date ranges to download. When
            ``None``, load the default schedule from ``config/date_range.json``.
        inline_mask (bool): If ``True``, generate NDWI masks immediately after
            building each mosaic.
        max_items (int): Maximum STAC items per patch request.
        multithreaded (bool): If ``True``, process dates in parallel using a
            process pool.

    Returns:
        list[dict]: Result records produced by :func:`process_date` for each
        processed date.
    """
    if dates is None:
        with open(
            Path(__file__).resolve().parents[2] / "config" / "date_range.json"
        ) as fh:
            dates = json.load(fh)["date_ranges"]
        dates = dates[6::12] + dates[7::12] + dates[8::12]

    missions = {
        "sentinel-2": get_mission("sentinel-2"),
        "landsat-5": get_mission("landsat-5"),
        "landsat-7": get_mission("landsat-7"),
    }

    results = []
    if multithreaded:
        with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
            futures = {
                executor.submit(
                    process_date,
                    date,
                    None,
                    missions["sentinel-2"],
                    missions["landsat-5"],
                    missions["landsat-7"],
                    data_path("sentinel_eastern_shore.tif"),
                    data_path("landsat5_eastern_shore.tif"),
                    data_path("landsat7_eastern_shore.tif"),
                    inline_mask,
                    max_items=max_items,
                ): date
                for date in dates
            }
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results.append(result)
    else:
        for date in tqdm(dates):
            result = process_date(
                date,
                None,
                missions["sentinel-2"],
                missions["landsat-5"],
                missions["landsat-7"],
                data_path("sentinel_eastern_shore.tif"),
                data_path("landsat5_eastern_shore.tif"),
                data_path("landsat7_eastern_shore.tif"),
                inline_mask,
                max_items=max_items,
            )
            results.append(result)
    return results
