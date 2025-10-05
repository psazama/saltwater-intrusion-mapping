"""Area-of-interest helpers for working with bounding boxes and patches."""

from __future__ import annotations

from typing import Iterable, Sequence

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box


def to_polygon(aoi: Sequence[float] | Polygon | MultiPolygon) -> Polygon | MultiPolygon:
    """Normalize an AOI specification into a Shapely geometry.

    Args:
        aoi (Sequence[float] | Polygon | MultiPolygon): Either a bounding
            box as ``(minx, miny, maxx, maxy)`` or an existing Shapely
            polygon/multipolygon expressed in EPSG:4326.

    Returns:
        Polygon | MultiPolygon: A geometry representing the AOI in
        EPSG:4326 coordinates.

    Raises:
        ValueError: If ``aoi`` cannot be interpreted as a bounding box or
            geometry.
    """
    if isinstance(aoi, (Polygon, MultiPolygon)):
        return aoi
    try:
        minx, miny, maxx, maxy = map(float, aoi)
        return box(minx, miny, maxx, maxy)  # shapely polygon
    except Exception as exc:
        raise ValueError(
            "AOI must be bbox [minx,miny,maxx,maxy] "
            "or a shapely Polygon/MultiPolygon"
        ) from exc


def iter_square_patches(
    aoi: Sequence[float] | Polygon | MultiPolygon,
    patch_size_m: float,
    metric_crs: str = "EPSG:32618",
) -> Iterable[Polygon]:
    """Yield square patches that cover the AOI in the requested CRS.

    Args:
        aoi (Sequence[float] | Polygon | MultiPolygon): The AOI definition
            accepted by :func:`to_polygon`.
        patch_size_m (float): The edge length of the patches expressed in
            meters.
        metric_crs (str): Projected CRS used to build the grid of square
            patches.

    Yields:
        Polygon: Square polygons in ``metric_crs`` that intersect the AOI.
    """
    poly = to_polygon(aoi)
    band_proj = gpd.GeoSeries([poly], crs="EPSG:4326").to_crs(metric_crs).iloc[0]
    minx, miny, maxx, maxy = band_proj.bounds

    xs = np.arange(minx, maxx, patch_size_m)
    ys = np.arange(miny, maxy, patch_size_m)

    from shapely.geometry import box

    for x in xs:
        for y in ys:
            patch = box(x, y, x + patch_size_m, y + patch_size_m)
            if patch.intersects(band_proj):
                yield patch
