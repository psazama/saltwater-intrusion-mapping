"""Utilities for downloading NLCD and CDL land-cover products."""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence, Union

import pyproj
import requests
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import transform

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


def download_nass_cdl(
    region: Sequence[float] | Polygon | MultiPolygon,
    year: int,
    output_path: Union[str, Path] | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Download the USDA NASS Cropland Data Layer for the provided region.

    Args:
        region (Sequence[float] | Polygon | MultiPolygon): AOI in WGS84
            coordinates.
        year (int): Target CDL year.
        output_path (Union[str, Path] | None): Optional destination path.
        overwrite (bool): Placeholder argument for compatibility.

    Returns:
        Path | None: Path to the downloaded CDL raster, or ``None`` if the
        download fails.
    """
    cdl_url = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"

    proj_wgs84 = pyproj.CRS("EPSG:4326")
    proj_albers = pyproj.CRS("EPSG:5070")
    transformer = pyproj.Transformer.from_crs(
        proj_wgs84, proj_albers, always_xy=True
    ).transform

    # bounds should be xmin, ymin, xmax, ymax
    if isinstance(region, (Polygon, MultiPolygon)):
        region = transform(transformer, region)
        bounds = region.bounds
    else:
        region = box(*region)
        region = transform(transformer, region)
        bounds = region
    bbox_str = ",".join(map(str, bounds))

    params = {"year": str(year), "bbox": bbox_str, "epsg": "4326"}

    if output_path is None:
        output_path = Path(f"cdl_{year}.tif")
    else:
        output_path = Path(output_path)

    resp = requests.get(cdl_url, params=params, stream=True)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    download_url = root.find(".//returnURL").text
    print("Download URL:", download_url)

    tif_resp = requests.get(download_url, stream=True)
    tif_resp.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in tif_resp.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Saved {output_path}")
    return output_path
