"""Mission configuration using subclass methods for satellite-specific logic."""

from __future__ import annotations

from typing import Dict, Optional, Tuple


class Mission:
    """Base class for satellite missions."""

    name: str

    def __init__(self, name: str):
        self.name = name

    # Standardized reflectance bands from source data
    def reflectance_stack(self, src) -> Dict[str, object]:
        raise NotImplementedError("Subclasses must implement reflectance_stack")

    # Mapping from common band names to band names
    def bands(self) -> Dict[str, int]:
        raise NotImplementedError("Subclasses must implement bands")

    # Mapping from common band names to band numbers
    def band_indices(self) -> Dict[str, int]:
        raise NotImplementedError("Subclasses must implement band_indices")

    # Optional mission-specific preprocessing hook
    def preprocess(self, src) -> Dict[str, object]:
        return self.reflectance_stack(src)

    # GEE-specific attributes
    @property
    def gee_collection(self) -> str:
        raise NotImplementedError

    @property
    def gee_scale(self) -> int:
        raise NotImplementedError

    # Legacy STAC fields
    @property
    def collection(self) -> str:
        raise NotImplementedError

    @property
    def query_filter(self) -> Dict:
        return {"eo:cloud_cover": {"lt": 10}}

    @property
    def resolution(self) -> int:
        raise NotImplementedError

    @property
    def valid_date_range(self) -> Tuple[Optional[str], Optional[str]]:
        raise NotImplementedError
