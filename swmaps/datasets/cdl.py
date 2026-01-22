"""Utilities for downloading CDL land-cover products."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence, Union

import pyproj
import requests
import rioxarray
from PIL import Image
from rasterio.enums import Resampling
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import transform


def align_cdl_to_imagery(cdl_path, imagery_path, output_path):
    imagery = rioxarray.open_rasterio(imagery_path)
    cdl = rioxarray.open_rasterio(cdl_path)

    # Reproject and match the CDL to the imagery exactly
    # Use 'nearest' resampling for CDL because it's categorical data (classes)
    cdl_aligned = cdl.rio.reproject_match(imagery, resampling=Resampling.nearest)
    cdl_aligned.rio.to_raster(output_path)
    return output_path


def download_nass_cdl(
    region: Sequence[float] | Polygon | MultiPolygon,
    year: int,
    output_path: Union[str, Path] | None = None,
    overwrite: bool = False,
    save_png: bool = True,
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
        bounds = region.bounds
    bbox_str = ",".join(map(str, bounds))

    params = {"year": str(year), "bbox": bbox_str, "epsg": "5070"}

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

    if save_png:
        png_output_path = output_path.with_suffix(".png")

        with Image.open(output_path) as img:
            img.save(png_output_path, format="PNG")

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
    import rasterio
    from rasterio.warp import transform_bounds

    from swmaps.core.missions import get_mission
    from swmaps.core.satellite_query import (
        download_gee_multiband,
        get_best_image,
        query_gee_images,
    )

    # 1) Setup original WGS84 region for the initial GEE query
    if isinstance(region, (Polygon, MultiPolygon)):
        wgs_region = region
    else:
        wgs_region = box(*region)
    bbox = list(wgs_region.bounds)

    # 2) Query GEE for images
    date_range = f"{year}-01-01/{year}-12-31"
    col, bands = query_gee_images(
        mission=mission, bbox=bbox, date_range=date_range, cloud_filter=cloud_filter
    )
    best = get_best_image(col, mission, samples)

    if best is None:
        return {"cdl": None, "imagery": []}

    # Prepare output dirs
    base_out = (
        Path(imagery_output_dir)
        if imagery_output_dir
        else Path("cdl_imagery")  # TODO: Path(data_path("cdl_imagery"))
    )
    mission_dir = base_out / mission / str(year)
    mission_dir.mkdir(parents=True, exist_ok=True)

    mission_info = get_mission(mission)
    gee_scale = getattr(mission_info, "gee_scale", 10)

    imagery_paths = []
    aligned_cdl_paths = []

    iterable = list(best) if isinstance(best, (list, tuple)) else [best]

    for img in iterable:
        # 3) DOWNLOAD IMAGERY FIRST
        out_path = download_gee_multiband(
            image=img,
            mission=mission,
            bands=bands,
            bbox=bbox,
            out_dir=mission_dir,
            scale=gee_scale,
        )
        imagery_paths.append(out_path)

        # 4) EXTRACT EXACT BOUNDS FROM THE SAVED TIF
        # This ensures we request the CDL for the exact area Earth Engine delivered
        with rasterio.open(out_path) as src:
            # Transform imagery bounds back to WGS84 for the CDL API
            exact_wgs_bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

        # 5) DOWNLOAD CDL FOR THE EXACT BOUNDS
        temp_cdl_path = mission_dir / f"temp_cdl_{Path(out_path).name}"
        cdl_path = download_nass_cdl(
            region=exact_wgs_bounds, year=year, output_path=temp_cdl_path
        )

        # 6) ALIGN
        aligned_cdl_path = Path(out_path).with_name(
            f"aligned_cdl_{Path(out_path).name}"
        )
        try:
            align_cdl_to_imagery(cdl_path, out_path, aligned_cdl_path)
            aligned_cdl_paths.append(str(aligned_cdl_path))
            # Clean up temp file
            if cdl_path.exists():
                cdl_path.unlink()
        except Exception as e:
            print(f"[WARN] Alignment failed: {e}")

    return {"cdl_aligned": aligned_cdl_paths, "imagery": imagery_paths}
