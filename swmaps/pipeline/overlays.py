"""
Pipeline utilities for overlay datasets (NLCD and CDL).
"""

from pathlib import Path
from typing import Union

from shapely.geometry import MultiPolygon, Polygon

from swmaps.config import data_path
from swmaps.core.nlcd_cdl import download_nass_cdl, download_nlcd


def fetch_nlcd_overlay(
    region: Union[Polygon, MultiPolygon, list[float]],
    year: int,
    allow_closest: bool = True,
) -> Path | None:
    """
    Fetch NLCD land cover overlay for a region and year.

    Parameters
    ----------
    region : Polygon | MultiPolygon | list[float]
        AOI geometry or bounding box [xmin, ymin, xmax, ymax] in WGS84.
    year : int
        Requested NLCD year.
    allow_closest : bool, default=True
        If True, downloads nearest available NLCD product if exact year is missing.

    Returns
    -------
    Path or None
        Path to the downloaded NLCD GeoTIFF, or None if unavailable.
    """
    output_path = data_path(f"nlcd_{year}.tif")
    return download_nlcd(
        region, year, output_path=output_path, allow_closest=allow_closest
    )


def fetch_cdl_overlay(
    region: Union[Polygon, MultiPolygon, list[float]], year: int
) -> Path | None:
    """
    Fetch USDA Cropland Data Layer (CDL) overlay.

    Parameters
    ----------
    region : Polygon | MultiPolygon | list[float]
        AOI geometry or bounding box [xmin, ymin, xmax, ymax] in WGS84.
    year : int
        CDL product year.

    Returns
    -------
    Path or None
        Path to the downloaded CDL GeoTIFF, or None if unavailable.
    """
    output_path = data_path(f"cdl_{year}.tif")
    return download_nass_cdl(region, year, output_path=output_path)
