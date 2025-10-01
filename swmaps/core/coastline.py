import os
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import box

from swmaps.config import data_path


def download_coastal_poly() -> None:
    file_types = ["cpg", "dbf", "prj", "shp", "shx"]

    for file_type in file_types:
        # URL and destination path
        url = f"https://raw.githubusercontent.com/nvkelso/natural-earth-vector/refs/heads/master/10m_physical/ne_10m_coastline.{file_type}"
        extract_dir = data_path("coastline/")
        extract_path = data_path(f"coastline/ne_10m_coastline.{file_type}")

        # Download the file
        os.makedirs(extract_dir, exist_ok=True)
        response = requests.get(url)
        with open(extract_path, "wb") as f:
            f.write(response.content)


def create_coastal_poly(
    bounding_box_file: str | Path,
    out_file: str | Path | None = None,
    buf_km: float = 2,
    offshore_km: float = 1,
) -> gpd.GeoDataFrame:
    """
    Build a coastal band polygon (buffered coastline, clipped to bbox) and save it once.

    Parameters
    ----------
    bounding_box_file : str | Path
        Vector file containing your big bbox (GeoJSON, Shapefile, etc.).
    out_file : str | Path | None
        Where to save the band (defaults to  config/coastal_band.gpkg).
    buf_km, offshore_km : float
        Width of inland / offshore buffers (kilometres).

    Returns
    -------
    GeoDataFrame with a single Polygon feature (lat/long, EPSG:4326).
    """

    bbox_gdf = gpd.read_file(bounding_box_file).to_crs("EPSG:4326")
    bbox = bbox_gdf.total_bounds  # [minx, miny, maxx, maxy]

    # Expand by a generous margin (≈ 10 km) so we don’t miss any coast
    margin_deg = 10 / 111.0  # ~10 km in degrees
    qbox = box(
        bbox[0] - margin_deg,
        bbox[1] - margin_deg,
        bbox[2] + margin_deg,
        bbox[3] + margin_deg,
    )

    coast_file = data_path("coastline/ne_10m_coastline.shp")
    if not os.path.exists(coast_file):
        download_coastal_poly()

    coast = gpd.read_file(
        coast_file,
        bbox=qbox,
    ).to_crs("EPSG:4326")

    if coast.empty:
        raise ValueError("No coastline segments intersect the provided bbox.")

    #  Project to metric CRS
    utm_zone = int((bbox[0] + 180) // 6) + 1
    utm_crs = f"EPSG:{32600 + utm_zone}"
    coast_m = coast.to_crs(utm_crs)

    inland = coast_m.buffer(buf_km * 1_000, cap_style=2)
    offshore = coast_m.buffer(-offshore_km * 1_000, cap_style=2)
    band_m = inland.union(offshore).unary_union

    # back to WGS-84
    band = gpd.GeoSeries([band_m], crs=utm_crs).to_crs("EPSG:4326")

    # final clip to exact bbox
    band_clip = gpd.clip(band, bbox_gdf)
    out_gdf = gpd.GeoDataFrame(geometry=band_clip, crs="EPSG:4326")

    out_file = Path(out_file or "config/coastal_band.gpkg")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_file(out_file, driver="GPKG", layer="coastal_band")

    return out_gdf
