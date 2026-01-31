"""Download helpers for acquiring imagery used by the processing pipeline."""

import logging
from datetime import datetime, timedelta
from pathlib import Path

from tqdm import tqdm

from swmaps.config import data_path
from swmaps.core.mosaic import compute_bbox, process_date

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


def download_data(cfg: dict, val=False):
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
    if not val:
        start_date = datetime.fromisoformat(cfg["start_date"])
        end_date = datetime.fromisoformat(cfg["end_date"])
        date_step = cfg.get("date_step", 1)
        lat = cfg["latitude"]
        lon = cfg["longitude"]
        out_dir = cfg.get("out_dir")
    else:
        start_date = datetime.fromisoformat(cfg["val_start_date"])
        end_date = datetime.fromisoformat(cfg["val_end_date"])
        date_step = cfg.get("val_date_step", 1)
        lat = cfg["val_latitude"]
        lon = cfg["val_longitude"]
        out_dir = cfg.get("val_dir")

    samples_per_date = cfg.get("samples_per_date", 1)
    missions_cfg = cfg.get("mission", "sentinel-2")
    save_png = bool(cfg.get("save_png", False))

    if isinstance(missions_cfg, (list, tuple)):
        mission_list = list(missions_cfg)
    else:
        # If a comma-separated string is provided, split into individual missions.
        if isinstance(missions_cfg, str) and "," in missions_cfg:
            mission_list = [m.strip() for m in missions_cfg.split(",") if m.strip()]
        else:
            mission_list = [missions_cfg]

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

        if "sentinel" in mission:
            curr_days_before = 7
            curr_days_after = 7
        else:
            curr_days_before = days_before
            curr_days_after = days_after

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
                days_before=curr_days_before,
                days_after=curr_days_after,
                cloud_filter=cloud_filter,
                samples=samples_per_date,
                save_png=save_png,
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


# ---------------------------------------------------------------------
# CDL helper: add CDL download support via config
# ---------------------------------------------------------------------
def download_cdl(cfg: dict, val=False):
    """
    If config contains download_cdl = true, download the USDA NASS CDL for
    the requested region/year and save to cdl_out (or cwd).
    """
    if not cfg.get("download_cdl", False):
        return None

    if not val:
        cdl_out = cfg.get("cdl_out")
        if cdl_out is None:
            cdl_out = data_path("downloads")
    else:
        cdl_out = cfg.get("val_cdl_out")

    cdl_out = Path(cdl_out)
    cdl_out.parent.mkdir(parents=True, exist_ok=True)

    # Lazy import to avoid adding a hard dependency before it's needed
    import geopandas as gpd

    from swmaps.datasets.cdl import download_nass_cdl

    cdl_year = int(cfg.get("cdl_year", datetime.utcnow().year))
    # Region can be:
    #  - path to GeoJSON (string)
    #  - bounds sequence: [xmin, ymin, xmax, ymax]
    #  - else fallback to center latitude/longitude + buffer (km)
    if not val:
        region_cfg = cfg.get("cdl_region", cfg.get("region", None))
    else:
        region_cfg = cfg.get("val_cdl_region", cfg.get("val_region", None))

    region = None
    if isinstance(region_cfg, str) and Path(region_cfg).exists():
        gdf = gpd.read_file(region_cfg)
        region = gdf.unary_union  # shapely geometry
    elif isinstance(region_cfg, (list, tuple)) and len(region_cfg) == 4:
        region = region_cfg  # pass bounds through
    else:
        if not val:
            lat = cfg.get("latitude")
            lon = cfg.get("longitude")
        else:
            lat = cfg.get("val_latitude")
            lon = cfg.get("val_longitude")
        if lat is None or lon is None:
            raise ValueError(
                "download_cdl requested but no cdl_region/path and no latitude/longitude in config"
            )
        buffer_km = float(cfg.get("cdl_buffer_km", cfg.get("buffer_km", 1.0)))
        region = tuple(compute_bbox(lat, lon, buffer_km))

    print(f"[CDL] Downloading CDL for year {cdl_year}; saving to {cdl_out or 'cwd'}")
    try:
        cdl_path = download_nass_cdl(region=region, year=cdl_year, output_path=cdl_out)
        print(f"[CDL] Saved: {cdl_path}")
        return cdl_path
    except Exception:
        logging.exception("CDL download failed")
        return None
