"""Utilities for downloading CDL land-cover products."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence, Union

import pyproj
import requests
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import transform


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
