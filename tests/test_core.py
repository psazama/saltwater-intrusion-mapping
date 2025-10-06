"""Tests covering basic AOI utilities used throughout the project."""

import pytest

pytest.importorskip("shapely")
pytest.importorskip("geopandas")

from swmaps.core import aoi


def test_to_polygon_bbox() -> None:
    """Ensure the bounding box helper creates a shapely polygon.

    Args:
        None

    Returns:
        None: Assertions raise if behaviour regresses.
    """
    poly = aoi.to_polygon([0, 0, 1, 1])
    from shapely.geometry import Polygon

    assert isinstance(poly, Polygon)
    assert poly.bounds == (0.0, 0.0, 1.0, 1.0)


"""
def test_iter_square_patches_simple() -> None:
    "#""Verify iter_square_patches yields shapely polygons for each patch.

    Args:
        None

    Returns:
        None: Fails via assertion when polygons are not produced.
    "#""
    patches = list(aoi.iter_square_patches([0, 0, 2, 2], 1.0, metric_crs="EPSG:3857"))
    assert len(patches) > 0
    from shapely.geometry import Polygon

    for p in patches:
        assert isinstance(p, Polygon)
"""
