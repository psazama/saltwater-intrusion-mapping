"""Mission configuration using subclass methods for satellite-specific logic."""

from .satellites.base import Mission
from .satellites.landsat5 import Landsat5
from .satellites.landsat7 import Landsat7
from .satellites.sentinel2 import Sentinel2


# --------------------------
# Factory function
# --------------------------
def get_mission(mission: str) -> Mission:
    mission_map = {
        "sentinel-2": Sentinel2,
        "landsat-5": Landsat5,
        "landsat-7": Landsat7,
    }

    if mission not in mission_map:
        raise ValueError(f"Unsupported mission: {mission}")

    return mission_map[mission]()
