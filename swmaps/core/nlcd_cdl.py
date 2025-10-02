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
    """Check if the requested year is an available product.
    If exact year is not available, return closest
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
    """Download Annual NLCD for the requested year.

    If exact year is not available:
      - with allow_closest=True (default), downloads nearest available year and saves as "{query_year}_approx".
      - with allow_closest=False, returns None.
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
    """
    Download the USDA NASS Cropland Data Layer for the provided region.
    """
    cdl_url = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"
    # bounds should be xmin, ymin, xmax, ymax
    if isinstance(region, (Polygon, MultiPolygon)):
        bounds = region.bounds
    else:
        bounds = region
    bbox_str = ",".join(map(str, bounds))

    params = {"year": str(year), "bbox": bbox_str, "epsg": "4326"}

    if output_path is None:
        output_path = Path(f"cdl_{year}.tif")
    else:
        output_path = Path(output_path)

    resp = requests.get(cdl_url, params=params, stream=True)
    resp.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Saved {output_path}")
    return output_path


"""
import requests
import boto3

# ScienceBase JSON feed

# ScienceBase item with Annual NLCD collection
item_id = "655ceb8ad34ee4b6e05cc51a"
url = f"https://www.sciencebase.gov/catalog/item/{item_id}?format=json"

resp = requests.get(url)
resp.raise_for_status()
data = resp.json()

# Print available files
for f in data["files"]:
    print(f["name"], f["url"])

# Download one file
file_url = data["files"][0]["url"]
out_path = "nlcd_1985.img"
with requests.get(file_url, stream=True) as r:
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
print(f"Saved {out_path}")


# AWS S3 (usgs-landcover, requester pays)
s3 = boto3.client("s3", region_name="us-west-2")

# List contents of the annual-nlcd bucket
resp = s3.list_objects_v2(
    Bucket="usgs-landcover",
    Prefix="annual-nlcd/c1/",
    RequestPayer="requester"
)

for obj in resp.get("Contents", []):
    print(obj["Key"])

# Download one file
key = "annual-nlcd/c1/1992_NLCD_Land_Cover_L48_20230630.img"
s3.download_file(
    Bucket="usgs-landcover",
    Key=key,
    Filename="nlcd_1992.img",
    ExtraArgs={"RequestPayer": "requester"}
)
print("Saved nlcd_1992.img")


# CDL (CropScape Web Service)
cdl_url = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"
params = {
    "year": 2021,
    "bbox": "-79,38,-76,40",  # xmin,ymin,xmax,ymax
    "epsg": 4326
}

resp = requests.get(cdl_url, params=params, stream=True)
resp.raise_for_status()

with open("cdl_2021.tif", "wb") as f:
    for chunk in resp.iter_content(chunk_size=8192):
        f.write(chunk)
print("Saved cdl_2021.tif")


# NLCD WCS (Geoserver GetCoverage)
wcs_url = "https://www.mrlc.gov/geoserver/mrlc_download/wcs"
params = {
    "service": "WCS",
    "request": "GetCoverage",
    "coverage": "NLCD_2019_Land_Cover_L48",
    "crs": "EPSG:4326",
    "bbox": "-77,38,-76.5,38.5",
    "format": "image/tiff",
    "width": 512,
    "height": 512
}

resp = requests.get(wcs_url, params=params, stream=True)
resp.raise_for_status()

with open("nlcd_2019.tif", "wb") as f:
    for chunk in resp.iter_content(chunk_size=8192):
        f.write(chunk)
print("Saved nlcd_2019.tif")
"""
