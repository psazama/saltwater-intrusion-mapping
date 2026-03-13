from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

from swmaps.core.missions import get_mission_from_path
from swmaps.infra.db import get_connection, register_scene


def _parse_date_from_filename(stem: str) -> str:
    parts = stem.split("_")
    if parts[0] == "sentinel-2":
        date_str = parts[1][:8]
    else:
        date_str = parts[3]
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"


def backfill_directory(data_dir: str):
    data_path = Path(data_dir)
    tifs = [
        p for p in data_path.rglob("*_multiband.tif") if "aligned_cdl" not in str(p)
    ]

    print(f"Found {len(tifs)} scenes to backfill")

    with get_connection() as conn:
        for tif in tifs:
            try:
                with rasterio.open(tif) as src:
                    crs = src.crs.to_string()
                    # Reproject bounds to EPSG:4326 for storage and validation
                    bounds = transform_bounds(
                        src.crs,
                        "EPSG:4326",
                        src.bounds.left,
                        src.bounds.bottom,
                        src.bounds.right,
                        src.bounds.top,
                    )
                    bbox = list(bounds)  # [minx, miny, maxx, maxy] in degrees

                mission = get_mission_from_path(tif).slug
                # Use filenmae stem as scene_id since we don't have GEE metadata
                scene_id = tif.stem.replace("_multiband", "")
                # Strip mission prefix if present e.g. "landsat-7_" or "sentinel-2_"
                if "_" in scene_id:
                    parts = scene_id.split("_", 1)
                    if parts[0] in {
                        "landsat-5",
                        "landsat-7",
                        "landsat-8",
                        "sentinel-2",
                    }:
                        scene_id = parts[1]

                register_scene(
                    conn=conn,
                    image_id=scene_id,
                    mission=mission,
                    bbox=bbox,
                    out_path=str(tif),
                    crs=crs,
                    acquisition_date=_parse_date_from_filename(tif.stem),
                )
                print(f"Registered: {scene_id}")
            except Exception as e:
                print(f"Skipped {tif.name}: {e}")


if __name__ == "__main__":
    backfill_directory("data/")
