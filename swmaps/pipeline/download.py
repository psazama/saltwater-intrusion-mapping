"""Download helpers for acquiring imagery used by the processing pipeline."""

from datetime import datetime, timedelta
from pathlib import Path

from tqdm import tqdm

from swmaps.config import data_path
from swmaps.core.mosaic import process_date

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _daterange(start: datetime, end: datetime, step_days: int = 1):
    """Yield dates from start to end using an integer day step."""
    current = start
    while current <= end:
        yield current
        current = current + timedelta(days=step_days)


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
                - date_step (optional)
                - out_dir (optional)
                - buffer_km (optional)
                - cloud_filter (optional)
    """
    # -------------------------------------------------------
    # Extract config values
    # -------------------------------------------------------
    start_date = datetime.fromisoformat(cfg["start_date"])
    end_date = datetime.fromisoformat(cfg["end_date"])
    date_step = cfg.get("date_step", 1)
    samples_per_date = cfg.get("samples_per_date", 1)
    lat = cfg["latitude"]
    lon = cfg["longitude"]
    missions_cfg = cfg.get("mission", "sentinel-2")
    if isinstance(missions_cfg, (list, tuple)):
        mission_list = list(missions_cfg)
    else:
        # If a comma-separated string is provided, split into individual missions.
        if isinstance(missions_cfg, str) and "," in missions_cfg:
            mission_list = [m.strip() for m in missions_cfg.split(",") if m.strip()]
        else:
            mission_list = [missions_cfg]

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
    print(f"[GEE] Downloading imagery for missions: {', '.join(mission_list)}")
    print(f"[GEE] AOI: lat={lat}, lon={lon}")
    print(f"[GEE] Date range: {start_date.date()} to {end_date.date()}")
    print(f"[GEE] Base output directory: {out_dir}")
    print("------------------------------------------------")

    # -------------------------------------------------------
    # Iterate over dates and build mosaics
    # -------------------------------------------------------

    results = []

    # Iterate over each mission and process full date range.
    for mission in mission_list:
        # Create mission-specific directory
        mission_out_dir = out_dir / mission
        mission_out_dir.mkdir(parents=True, exist_ok=True)

        for date in tqdm(
            _daterange(start_date, end_date, date_step),
            desc=f"[GEE] Processing dates for {mission}",
        ):
            # try:
            output_path = process_date(
                lat=lat,
                lon=lon,
                date=date,
                buffer_km=buffer_km,
                mission=mission,
                out_dir=mission_out_dir,
                days_before=days_before,
                days_after=days_after,
                cloud_filter=cloud_filter,
                samples=samples_per_date,
            )
            if isinstance(output_path, list):
                results.extend(output_path)
            else:
                results.append(output_path)
            # except Exception as e:
            #     print(f"[WARN] {mission}: Failed to process {date.date()}: {e}")

    print("------------------------------------------------")
    print(
        f"[GEE] Completed. Built {len(results)} mosaics across {len(mission_list)} mission(s)."
    )
    print("------------------------------------------------")

    return results
