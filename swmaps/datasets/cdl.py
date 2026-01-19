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


def download_cdl_and_imagery(
    mission: str,
    region: Sequence[float] | Polygon | MultiPolygon,
    year: int,
    cdl_output_path: Union[str, Path] | None = None,
    imagery_output_dir: Union[str, Path] | None = None,
    samples: int = 1,
    cloud_filter: float | None = 20,
):
    """Download the CDL for `region`/`year` then fetch matching satellite imagery.

    Workflow:
    - Calls download_nass_cdl(...) to fetch the CDL raster.
    - Builds a WGS84 bbox from the provided `region`.
    - Queries Earth Engine for images for `mission` over the calendar year.
    - Downloads selected multiband GeoTIFF(s) into imagery_output_dir/mission/<year>/.

    Args:
        mission (str): mission slug accepted by swmaps (e.g. "sentinel-2").
        region (Sequence[float] | Polygon | MultiPolygon): AOI in WGS84 coords.
        year (int): Target year for CDL and imagery (imagery searched across the year).
        cdl_output_path: Optional path for the CDL GeoTIFF.
        imagery_output_dir: Optional base output dir for imagery.
        samples: Number of images to request (passed to get_best_image).
        cloud_filter: Optional cloud threshold passed to query_gee_images.

    Returns:
        dict: {
            "cdl": Path to downloaded CDL (str),
            "imagery": list of downloaded imagery paths (list[str])
        }
        If imagery download fails or none found, "imagery" will be an empty list.
    """
    from swmaps.config import data_path
    from swmaps.core.missions import get_mission
    from swmaps.core.satellite_query import (
        download_gee_multiband,
        get_best_image,
        query_gee_images,
    )

    # Keep a copy of the original WGS84 region to compute bbox
    if isinstance(region, (Polygon, MultiPolygon)):
        wgs_region = region
    else:
        # assume region is sequence [minx, miny, maxx, maxy]
        wgs_region = box(*region)

    # 1) Download the CDL raster
    cdl_path = download_nass_cdl(region, year, output_path=cdl_output_path)
    if cdl_path is None:
        print("[CDL] CDL download failed; aborting imagery fetch.")
        return {"cdl": None, "imagery": []}

    # 2) Build bbox in WGS84 (satellite helpers expect [minx, miny, maxx, maxy])
    bbox = list(wgs_region.bounds)  # xmin, ymin, xmax, ymax

    # 3) Query GEE for images over the calendar year
    date_range = f"{year}-01-01/{year}-12-31"

    col, bands = query_gee_images(
        mission=mission, bbox=bbox, date_range=date_range, cloud_filter=cloud_filter
    )
    best = get_best_image(col, mission, samples)

    if best is None:
        print(
            f"[GEE] No imagery found for mission={mission} bbox={bbox} date_range={date_range}"
        )
        return {"cdl": str(cdl_path), "imagery": []}

    # Prepare imagery output directory
    if imagery_output_dir:
        base_out = Path(imagery_output_dir)
    else:
        base_out = Path(data_path("cdl_imagery"))

    mission_dir = base_out / mission / str(year)
    mission_dir.mkdir(parents=True, exist_ok=True)

    # Determine gee scale (fallback to 10)
    mission_info = get_mission(mission)
    gee_scale = getattr(mission_info, "gee_scale", None)
    if callable(gee_scale):
        try:
            gee_scale = gee_scale()
        except TypeError:
            # some mission classes expose gee_scale as an attribute
            pass
    if gee_scale is None:
        gee_scale = 10

    # Download selected images
    imagery_paths = []
    # best may be a single ee.Image or a list/iterable of images; handle both
    try:
        iterable = list(best)
    except TypeError:
        iterable = [best]

    for img in iterable:
        out_path = download_gee_multiband(
            image=img,
            mission=mission,
            bands=bands,
            bbox=bbox,
            out_dir=mission_dir,
            scale=gee_scale,
        )
        imagery_paths.append(out_path)
        print(f"[CDL->GEE] Wrote imagery to: {out_path}")

    return {"cdl": str(cdl_path), "imagery": imagery_paths}
