"""Pipeline helpers for preparing coastal region geometry inputs."""

from pathlib import Path

import geopandas as gpd

from swmaps.core.coastline import create_coastal_poly


def create_coastal(use_bbox: bool = False) -> None:
    """Create the study coastal polygon or bounding-box region.

    Args:
        use_bbox (bool): If ``True``, load the configured bounding box instead
            of generating the buffered coastal polygon.

    Returns:
        None: Output files are written under ``config/``.
    """
    if use_bbox:
        geojson = Path(__file__).resolve().parents[2] / "config" / "somerset.geojson"
        gdf = gpd.read_file(geojson)
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        geojson = geojson
    else:
        geojson = Path(__file__).resolve().parents[2] / "config" / "coastal_band.gpkg"

    print("Creating Coastal Polygon")
    create_coastal_poly(geojson)
