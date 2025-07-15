from __future__ import annotations

from typing import Iterable, Sequence

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box


def to_polygon(aoi: Sequence[float] | Polygon | MultiPolygon) -> Polygon | MultiPolygon:
    """
    Accepts either a 4-tuple bbox or a shapely polygon (EPSG:4326).
    Returns a shapely Polygon/MultiPolygon (still in EPSG:4326).
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
    """
    Yield **patch polygons in metric_crs** that intersect the AOI.

    The AOI may be bbox or polygon (WGS-84).
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
