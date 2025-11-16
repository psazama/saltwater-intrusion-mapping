"""Download helpers for acquiring imagery used by the processing pipeline."""

from datetime import datetime, timedelta
from pathlib import Path

from tqdm import tqdm

from swmaps.config import data_path
from swmaps.core.mosaic import process_date

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _daterange(start: datetime, end: datetime):
    """Yield a date for every day in [start, end]."""
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------


def download_data(cfg: dict):
    """
    Main entry point used by `examples/workflow_runner.py`.

    Args:
        cfg (dict): Loaded TOML config
            Must include keys:
                - start_date
                - end_date
                - latitude
                - longitude
                - mission
                - out_dir (optional)
                - buffer_km (optional)
                - cloud_filter (optional)
    """
    # -------------------------------------------------------
    # Extract config values
    # -------------------------------------------------------
    start_date = datetime.fromisoformat(cfg["start_date"])
    end_date = datetime.fromisoformat(cfg["end_date"])
    lat = cfg["latitude"]
    lon = cfg["longitude"]
    mission = cfg.get("mission", "sentinel-2")

    out_dir = cfg.get("out_dir")
    if out_dir is None:
        out_dir = data_path("downloads")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buffer_km = cfg.get("buffer_km", 1.0)
    cloud_filter = cfg.get("cloud_filter", 30)
    days_before = cfg.get("days_before", 7)
    days_after = cfg.get("days_after", 7)

    print("------------------------------------------------")
    print(f"[GEE] Downloading imagery for mission: {mission}")
    print(f"[GEE] AOI: lat={lat}, lon={lon}")
    print(f"[GEE] Date range: {start_date.date()} to {end_date.date()}")
    print(f"[GEE] Output directory: {out_dir}")
    print("------------------------------------------------")

    # -------------------------------------------------------
    # Iterate over dates and build mosaics
    # -------------------------------------------------------

    results = []

    for date in tqdm(_daterange(start_date, end_date), desc="[GEE] Processing dates"):
        try:
            output_path = process_date(
                lat=lat,
                lon=lon,
                date=date,
                buffer_km=buffer_km,
                mission=mission,
                out_dir=out_dir,
                days_before=days_before,
                days_after=days_after,
                cloud_filter=cloud_filter,
            )
            results.append(output_path)

        except Exception as e:
            print(f"[WARN] Failed to process {date.date()}: {e}")

    print("------------------------------------------------")
    print(f"[GEE] Completed. Built {len(results)} mosaics.")
    print("------------------------------------------------")

    return results
