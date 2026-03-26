"""Abstract base class for satellite mission configurations.

All concrete missions (Landsat-5, Landsat-7, Sentinel-2) inherit from
:class:`Mission` and override the abstract properties and methods defined here.
This guarantees a uniform interface across the rest of the pipeline regardless
of which sensor produced the imagery.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


class Mission:
    """Abstract base class representing a satellite mission's configuration.

    A Mission encapsulates everything the pipeline needs to know about a
    specific sensor: which GEE ImageCollection to query, how to map logical
    band names (``"green"``, ``"nir08"``) to the sensor's native band names,
    the native resolution, the valid acquisition date range, and the CRS to
    reproject to before download.

    Subclasses must implement :meth:`bands`, :meth:`band_indices`, and
    :meth:`reflectance_stack`, plus the abstract properties
    :attr:`gee_collection`, :attr:`gee_scale`, :attr:`collection`,
    :attr:`resolution`, and :attr:`valid_date_range`.

    Args:
        name: Human-readable mission slug, e.g. ``"sentinel-2"``.
    """

    name: str

    def __init__(self, name: str):
        self.name = name

    @property
    def slug(self) -> str:
        """Human-readable mission slug identical to :attr:`name`.

        Returns:
            str: Mission slug, e.g. ``"landsat-5"``.
        """
        return self.name

    def reflectance_stack(self, src) -> Dict[str, object]:
        """Extract surface-reflectance arrays from an open rasterio dataset.

        Args:
            src: An open rasterio DatasetReader for a multiband GeoTIFF.

        Returns:
            Dict[str, np.ndarray]: Logical band name to float32 array.

        Raises:
            NotImplementedError: Always - must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement reflectance_stack")

    def bands(self) -> Dict[str, int]:
        """Return a mapping from logical band names to native GEE band names.

        Returns:
            Dict[str, str]: e.g. ``{"blue": "B2", "green": "B3", ...}``.

        Raises:
            NotImplementedError: Always - must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement bands")

    def band_indices(self) -> Dict[str, int]:
        """Return a mapping from logical band names to 1-based rasterio band indices.

        Returns:
            Dict[str, int]: e.g. ``{"blue": 1, "green": 2, "red": 3, ...}``.

        Raises:
            NotImplementedError: Always - must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement band_indices")

    def preprocess(self, src) -> Dict[str, object]:
        """Apply mission-specific preprocessing and return reflectance arrays.

        Defaults to calling :meth:`reflectance_stack`. Subclasses may override
        to inject additional steps before arrays are consumed downstream.

        Args:
            src: An open rasterio DatasetReader.

        Returns:
            Dict[str, np.ndarray]: Same structure as :meth:`reflectance_stack`.
        """
        return self.reflectance_stack(src)

    def read_bands(self, src) -> Dict[str, np.ndarray]:
        """Read and scale band arrays from an open rasterio dataset.

        This is the standard way for the pipeline layer to get a band dict
        ready for :meth:`~swmaps.models.salinity_heuristic.SalinityHeuristicModel.predict`.
        Each subclass implements the correct band ordering and scale factor
        for its sensor.

        Args:
            src: An open :class:`rasterio.io.DatasetReader` for a multiband
                GeoTIFF produced by this pipeline.

        Returns:
            Dict[str, np.ndarray]: Band dict with keys ``"blue"``, ``"green"``,
            ``"red"``, ``"nir"``, ``"swir1"``, ``"swir2"`` as float32 arrays
            in the ``[0, 1]`` reflectance range.

        Raises:
            NotImplementedError: Always - must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement read_bands")

    @property
    def gee_crs(self) -> str:
        """Default CRS used when reprojecting imagery in GEE before export.

        Defaults to ``"EPSG:32618"`` (UTM Zone 18N - US East Coast).

        Returns:
            str: EPSG CRS string.
        """
        return "EPSG:32618"

    @property
    def gee_collection(self) -> str:
        """GEE ImageCollection asset ID for this mission.

        Raises:
            NotImplementedError: Always - must be overridden by subclasses.
        """
        raise NotImplementedError

    @property
    def gee_scale(self) -> int:
        """Native pixel resolution in metres used when exporting from GEE.

        Raises:
            NotImplementedError: Always - must be overridden by subclasses.
        """
        raise NotImplementedError

    @property
    def collection(self) -> str:
        """STAC collection identifier (retained for legacy compatibility).

        Raises:
            NotImplementedError: Always - must be overridden by subclasses.
        """
        raise NotImplementedError

    @property
    def resolution(self) -> int:
        """Native sensor resolution in metres.

        Raises:
            NotImplementedError: Always - must be overridden by subclasses.
        """
        raise NotImplementedError

    @property
    def valid_date_range(self) -> Tuple[Optional[str], Optional[str]]:
        """Operational date range as ``(start_date, end_date)`` ISO-8601 strings.

        ``None`` indicates an open bound (mission still active or no cutoff).

        Raises:
            NotImplementedError: Always - must be overridden by subclasses.
        """
        raise NotImplementedError

    @property
    def query_filter(self) -> Dict:
        """Default STAC query filter for this collection.

        Returns:
            Dict: Defaults to ``{"eo:cloud_cover": {"lt": 10}}``.
        """
        return {"eo:cloud_cover": {"lt": 10}}
