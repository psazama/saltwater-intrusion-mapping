"""Mission configuration using subclass methods for satellite-specific logic."""

from .satellites.base import Mission
from .satellites.landsat5 import Landsat5
from .satellites.landsat7 import Landsat7
from .satellites.sentinel2 import Sentinel2

# --------------------------
# Factory function
# --------------------------
GLOBAL_BAND_IDS = {
    "blue": 0,
    "green": 1,
    "red": 2,
    "nir08": 3,
    "swir16": 4,
    "swir22": 5,
}


def get_mission(mission: str) -> Mission:
    mission_map = {
        "sentinel-2": Sentinel2,
        "landsat-5": Landsat5,
        "landsat-7": Landsat7,
    }

    if mission not in mission_map:
        raise ValueError(f"Unsupported mission: {mission}")

    return mission_map[mission]()


def get_mission_from_id(mission: int) -> Mission:
    missions = {
        0: Sentinel2(),
        1: Landsat5(),
        2: Landsat7(),
    }

    return missions[mission]
