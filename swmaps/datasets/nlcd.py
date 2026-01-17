"""Utilities for downloading NLCD land-cover products."""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence, Union

import requests
from shapely.geometry import MultiPolygon, Polygon

from swmaps.config import data_path


def validate_nlcd_year(
    given_year: int,
):
    """Check if the requested NLCD year is available.

    Args:
        given_year (int): Desired NLCD land-cover year.

    Returns:
        tuple[int | None, str | None]: The closest available year and the
        identifier of the coverage offering, or ``(None, None)`` when no
        coverage is found.
    """
    url = "https://www.mrlc.gov/geoserver/mrlc_download/wcs"

    params = {
        "service": "WCS",
        "request": "GetCapabilities",
        "version": "1.0.0",
    }

    # Fetch the capabilities document
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()

    # Parse the XML
    root = ET.fromstring(resp.content)

    # Extract all <name> tags under CoverageOfferingBrief
    names = [elem.text for elem in root.findall(".//{http://www.opengis.net/wcs}name")]

    # Filter just NLCD land cover coverages
    nlcd_coverages = [n for n in names if "NLCD_" in n and "Land_Cover" in n]

    closest_year = None
    min_diff = float("inf")
    curr_cov = None
    # TODO: update to allow for non-continental US
    for cov in nlcd_coverages:
        if (
            "NLCD" not in cov
            or "Land_Cover_Change" in cov
            or "L48" not in cov
            or "ANLCD" in cov
        ):
            continue
        try:
            cov_year = int(cov.split("_")[2])  # e.g. "NLCD_2019_Land_Cover_L48"
        except (IndexError, ValueError):
            continue
        diff = abs(given_year - cov_year)
        if diff < min_diff:
            min_diff = diff
            closest_year = cov_year
            curr_cov = cov

    return closest_year, curr_cov


def download_nlcd(
    region: Union[Sequence[float], Polygon, MultiPolygon],
    year: int,
    output_path: Union[str, Path] | None = None,
    overwrite: bool = False,
    allow_closest: bool = True,
) -> Path | None:
    """Download the annual NLCD land-cover product for the requested year.

    Args:
        region (Sequence[float] | Polygon | MultiPolygon): AOI specified as a
            bounding box in WGS84 or a Shapely geometry.
        year (int): Target NLCD year.
        output_path (Union[str, Path] | None): Optional destination path for
            the GeoTIFF.
        overwrite (bool): Currently unused placeholder for API compatibility.
        allow_closest (bool): If ``True``, download the nearest available
            year when the exact one is missing.

    Returns:
        Path | None: Path to the downloaded raster or ``None`` if the product
        is unavailable and ``allow_closest`` is ``False``.
    """

    if output_path is None:
        output_path = data_path(f"nlcd/NLCD_{year}_Land_Cover_L48.tif")

    closest_year, product = validate_nlcd_year(year)
    if not allow_closest and closest_year != year:
        logging.warning("Exact NLCD product not available for %s", year)
        return None

    if isinstance(region, (Polygon, MultiPolygon)):
        bounds = region.bounds
    elif isinstance(region, (list, tuple)) and len(region) == 4:
        bounds = tuple(region)
    else:
        raise ValueError(
            "Region must be a bounding box sequence or a Polygon/MultiPolygon"
        )
    bounds = ",".join(map(str, bounds))

    # NLCD WCS (Geoserver GetCoverage)
    wcs_url = "https://www.mrlc.gov/geoserver/mrlc_download/wcs"
    params = {
        "service": "WCS",
        "request": "GetCoverage",
        "coverage": product,
        "crs": "EPSG:4326",
        "bbox": bounds,
        "format": "image/tiff",
        "width": 512,
        "height": 512,
        "version": "1.0.0",
    }

    resp = requests.get(wcs_url, params=params, stream=True)
    resp.raise_for_status()

    if closest_year != year:
        p = Path(output_path)
        output_path = p.with_name(p.stem + "_approx" + p.suffix)

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved {output_path}")
