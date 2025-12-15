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


def build_coastal_polygon(
    region: str | Path,
    output_path: Path,
    buffer_km: float = 10.0,
    offshore_km: float = 1.0,
) -> None:
    """
    Pipeline wrapper for creating a coastal AOI polygon.

    Args:
        region: Path to region AOI or bounding box geometry.
        output_path: Destination GeoJSON or GPKG for the coastal AOI.
        buffer_km: Inland buffer distance in kilometers.
        offshore_km: Offshore buffer distance in kilometers.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    create_coastal_poly(
        bounding_box_file=region,
        out_file=output_path,
        buffer_km=buffer_km,
        offshore_km=offshore_km,
    )
