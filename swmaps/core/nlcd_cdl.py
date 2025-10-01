import logging
import math
from pathlib import Path
from typing import Sequence, Union

import requests
from bs4 import BeautifulSoup
from shapely.geometry import MultiPolygon, Polygon

from swmaps.config import data_path
from swmaps.core.aoi import to_polygon

from .download_helpers import (
    _region_tag,
    _request_raster_with_format_fallback,
    _save_response_to_raster,
)

LEGACY_YEARS = {2001, 2006, 2011, 2016, 2019, 2021}
ANNUAL_RANGE = range(1985, 2025)

ANNUAL_SCIENCEBASE_ITEM = (
    "https://www.sciencebase.gov/catalog/item/63475ce5d34ed907bf70c6d5?format=json"
)
MRLC_ANNUAL_URL = "https://www.mrlc.gov/data?f%5B0%5D=category%3Aannual-nlcd"
MRLC_BASE = "https://s3-us-west-2.amazonaws.com/mrlc/nlcd/annual/"


def _detect_region(bounds: tuple[float, float, float, float]) -> str:
    """Infer NLCD region code from bounding box (EPSG:4326)."""
    minx, miny, maxx, maxy = bounds
    if -170 <= minx <= -154 and 18 <= miny <= 23:
        return "HI"
    elif -180 <= minx <= -129 and 50 <= miny <= 72:
        return "AK"
    elif -67 <= minx <= -65 and 17 <= miny <= 19:
        return "PR"
    else:
        return "L48"  # default CONUS


def _download_file(url: str, destination: Path) -> Path:
    logging.info("Downloading from %s", url)
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(destination, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return destination


def _get_annual_nlcd_url(year: int) -> str:
    """Try ScienceBase first, then fall back to scraping MRLC's annual NLCD page."""

    # --- Primary: ScienceBase ---
    try:
        r = requests.get(ANNUAL_SCIENCEBASE_ITEM, timeout=30)
        r.raise_for_status()
        data = r.json()
        for f in data.get("files", []):
            if f["name"].startswith(f"{year}_NLCD_Land_Cover_L48"):
                return f["url"]
    except Exception as e:
        logging.warning(f"[NLCD] ScienceBase lookup failed for {year}: {e}")

    # --- Fallback: MRLC webpage scrape ---
    try:
        r = requests.get(MRLC_ANNUAL_URL, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Look for hrefs that look like NLCD annual files
        links = [
            a["href"]
            for a in soup.find_all("a", href=True)
            if f"{year}_NLCD_Land_Cover_L48" in a["href"]
        ]
        if links:
            return links[0]

        # If no link found, construct the expected S3 path and test it
        candidate_url = f"{MRLC_BASE}{year}_NLCD_Land_Cover_L48.tif"
        test = requests.head(candidate_url)
        if test.status_code == 200:
            return candidate_url

    except Exception as e:
        logging.warning(f"[NLCD] MRLC scrape fallback failed for {year}: {e}")

    raise ValueError(f"Could not resolve Annual NLCD URL for {year}")


def download_nlcd(
    region: Union[Sequence[float], Polygon, MultiPolygon],
    year: int = 2021,
    product: str = "land_cover",
    resolution_m: float | None = 30.0,
    output_path: Union[str, Path] | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Download NLCD for the given region and year.

    Sources:
      - Epoch NLCD (2001–2021): MRLC WCS
      - Annual NLCD (1985–2024): USGS ScienceBase
    Returns:
      Path to the downloaded file, or None if not available.
    """

    polygon = to_polygon(region)
    bounds = polygon.bounds
    region_code = _detect_region(bounds)

    # Epoch products → WCS
    if year in LEGACY_YEARS:
        coverage = f"NLCD_{year}_Land_Cover_{region_code}"
        base_url = "https://www.mrlc.gov/geoserver/mrlc_download/wcs"

        if output_path is None:
            tag = _region_tag(bounds)
            output_path = data_path(f"nlcd/{coverage}_{tag}.tif")

        destination = Path(output_path)
        if destination.exists() and not overwrite:
            logging.info(
                "NLCD file already exists at %s; skipping download.", destination
            )
            return destination

        params: dict[str, object] = {
            "service": "WCS",
            "version": "1.0.0",
            "request": "GetCoverage",
            "coverage": coverage,
            "crs": "EPSG:4326",
            "bbox": ",".join(map(str, bounds)),
        }

        if resolution_m is not None:
            METERS_PER_DEGREE_AT_EQUATOR = 111_320.0
            center_lat = (bounds[1] + bounds[3]) / 2.0
            res_y = resolution_m / METERS_PER_DEGREE_AT_EQUATOR
            res_x = resolution_m / (
                METERS_PER_DEGREE_AT_EQUATOR * math.cos(math.radians(center_lat))
            )
            params["resx"] = f"{res_x:.8f}"
            params["resy"] = f"{res_y:.8f}"

        logging.info("Requesting NLCD coverage '%s' for bounds %s", coverage, bounds)
        response, _ = _request_raster_with_format_fallback(
            base_url,
            params,
            (
                "COG",
                "image/tiff;subtype=cloud-optimized",
                "image/tiff; application=geotiff; profile=cloud-optimized",
                "GeoTIFF",
            ),
        )
        return _save_response_to_raster(response.content, destination)

    # Annual products → ScienceBase
    elif year in ANNUAL_RANGE:
        if region_code != "L48":
            logging.warning(
                "Annual NLCD only available for CONUS (L48). Skipping year %s.", year
            )
            return None
        if product != "land_cover":
            logging.warning(
                "Annual NLCD downloader currently supports only 'land_cover'. Skipping year %s.",
                year,
            )
            return None

        try:
            url = _get_annual_nlcd_url(year)
        except Exception as e:
            logging.error("Could not resolve Annual NLCD URL for %s: %s", year, e)
            return None

        if output_path is None:
            output_path = data_path(f"nlcd/NLCD_{year}_Land_Cover_L48.tif")

        destination = Path(output_path)
        if destination.exists() and not overwrite:
            logging.info(
                "NLCD file already exists at %s; skipping download.", destination
            )
            return destination

        return _download_file(url, destination)

    else:
        logging.warning("No NLCD available for year %s. Skipping.", year)
        return None


def download_nass_cdl(
    region: Sequence[float] | Polygon | MultiPolygon,
    year: int,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Download the USDA NASS Cropland Data Layer for the provided region.

    Available only for 2008 onward. Returns None if not available.
    """

    if year < 2008:
        logging.warning(
            "Skipping CDL: CDL is only available from 2008 onward (requested %s).", year
        )
        return None

    base_url = "https://nassgeodata.gmu.edu/axis/services/CDLService"

    polygon = to_polygon(region)
    bounds = polygon.bounds

    if output_path is None:
        tag = _region_tag(bounds)
        output_path = data_path(f"cdl/CDL_{year}_{tag}.tif")

    destination = Path(output_path)
    if destination.exists() and not overwrite:
        logging.info("CDL file already exists at %s; skipping download.", destination)
        return destination

    params = {
        "year": str(year),
        "bbox": ",".join(map(str, bounds)),
        "epsg": "4326",
    }

    logging.info("Requesting CDL for year %s and bounds %s", year, bounds)
    response, _ = _request_raster_with_format_fallback(
        base_url,
        params,
        (
            "cog",
            "COG",
            "image/tiff;subtype=cloud-optimized",
            "geotiff",
        ),
    )

    return _save_response_to_raster(response.content, destination)
