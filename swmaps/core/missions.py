"""Mission configuration and factory utilities for satellite-specific logic.

This module provides a registry of supported satellite missions and a unified
factory interface for resolving :class:`~swmaps.core.satellites.base.Mission`
instances from a slug string, an integer ID, or a file path.  All three lookup
styles delegate to a single internal registry so the supported-mission list
only needs to be maintained in one place.
"""

from pathlib import Path

from .satellites.base import Mission
from .satellites.landsat5 import Landsat5
from .satellites.landsat7 import Landsat7
from .satellites.sentinel2 import Sentinel2

# ---------------------------------------------------------------------------
# Global band ordering
# ---------------------------------------------------------------------------

GLOBAL_BAND_IDS: dict[str, int] = {
    "blue": 0,
    "green": 1,
    "red": 2,
    "nir08": 3,
    "swir16": 4,
    "swir22": 5,
}

# ---------------------------------------------------------------------------
# Internal registry
# ---------------------------------------------------------------------------

_MISSION_REGISTRY: dict[str, type[Mission]] = {
    "sentinel-2": Sentinel2,
    "landsat-5": Landsat5,
    "landsat-7": Landsat7,
}

_MISSION_BY_ID: dict[int, type[Mission]] = {
    0: Sentinel2,
    1: Landsat5,
    2: Landsat7,
}

# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def get_mission(mission: str) -> Mission:
    """Return a Mission for the given slug.

    Args:
        mission: e.g. ``"sentinel-2"``, ``"landsat-5"``, ``"landsat-7"``.

    Returns:
        Mission: A fresh instance of the corresponding Mission subclass.

    Raises:
        ValueError: If *mission* is not in the registry.
    """
    if mission not in _MISSION_REGISTRY:
        raise ValueError(
            f"Unsupported mission: '{mission}'. "
            f"Valid options are: {sorted(_MISSION_REGISTRY)}"
        )
    return _MISSION_REGISTRY[mission]()


def get_mission_from_id(mission_id: int) -> Mission:
    """Return a Mission for the given integer ID.

    Args:
        mission_id: ``0`` → Sentinel-2, ``1`` → Landsat-5, ``2`` → Landsat-7.

    Returns:
        Mission: A fresh instance of the corresponding Mission subclass.

    Raises:
        KeyError: If *mission_id* has no registered mapping.
    """
    if mission_id not in _MISSION_BY_ID:
        raise KeyError(
            f"Unknown mission ID: {mission_id}. "
            f"Valid IDs are: {sorted(_MISSION_BY_ID)}"
        )
    return _MISSION_BY_ID[mission_id]()


def get_mission_from_path(path) -> Mission:
    """Infer a Mission from a file path.

    Searches the filename and parent directory name for a known mission slug.

    Args:
        path: Path to a GeoTIFF whose name or parent directory encodes the mission.

    Returns:
        Mission: A fresh instance of the matched Mission subclass.

    Raises:
        ValueError: If no registered mission slug is found in the path.
    """
    p = Path(path)
    search_str = (p.name + "/" + p.parent.name).lower()

    for slug, cls in _MISSION_REGISTRY.items():
        if slug in search_str:
            return cls()

    raise ValueError(
        f"Cannot derive mission from path: '{path}'. "
        f"Expected one of {sorted(_MISSION_REGISTRY)} to appear in the "
        f"filename or parent directory name."
    )
