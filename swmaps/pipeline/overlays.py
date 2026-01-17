"""
Pipeline utilities for overlay datasets (NLCD and CDL).
"""

from pathlib import Path
from typing import Union

from shapely.geometry import MultiPolygon, Polygon

from swmaps.config import data_path
from swmaps.datasets.cdl import download_nass_cdl
from swmaps.datasets.nlcd import download_nlcd


def fetch_nlcd_overlay(
    region: Union[Polygon, MultiPolygon, list[float]],
    year: int,
    allow_closest: bool = True,
    output_dir: str | Path = None,
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
    output_dir : str | Path | None, default=None
        Directory where the overlay will be saved. Defaults to swmaps/data.

    Returns
    -------
    Path or None
        Path to the downloaded NLCD GeoTIFF, or None if unavailable.
    """
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / f"nlcd_{year}.tif"
    else:
        output_path = data_path(f"nlcd_{year}.tif")

    return download_nlcd(
        region, year, output_path=output_path, allow_closest=allow_closest
    )


def fetch_cdl_overlay(
    region: Union[Polygon, MultiPolygon, list[float]],
    year: int,
    output_dir: str | Path = None,
) -> Path | None:
    """
    Fetch USDA Cropland Data Layer (CDL) overlay.

    Parameters
    ----------
    region : Polygon | MultiPolygon | list[float]
        AOI geometry or bounding box [xmin, ymin, xmax, ymax] in WGS84.
    year : int
        CDL product year.
    output_dir : str | Path | None, default=None
        Directory where the overlay will be saved. Defaults to swmaps/data.

    Returns
    -------
    Path or None
        Path to the downloaded CDL GeoTIFF, or None if unavailable.
    """
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / f"cdl_{year}.tif"
    else:
        output_path = data_path(f"cdl_{year}.tif")

    return download_nass_cdl(region, year, output_path=output_path)
