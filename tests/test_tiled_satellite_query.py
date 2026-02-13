"""Tests for tile-based download strategy in satellite_query module."""

from swmaps.core.satellite_query import _get_tile_grid


def test_get_tile_grid_2x2():
    """Test that _get_tile_grid generates correct 2x2 tile grid."""
    bbox = [0.0, 0.0, 10.0, 10.0]
    tiles = _get_tile_grid(bbox, 2)

    assert len(tiles) == 4

    # Check first tile (bottom-left)
    assert tiles[0] == [0.0, 0.0, 5.0, 5.0]

    # Check second tile (bottom-right)
    assert tiles[1] == [5.0, 0.0, 10.0, 5.0]

    # Check third tile (top-left)
    assert tiles[2] == [0.0, 5.0, 5.0, 10.0]

    # Check fourth tile (top-right)
    assert tiles[3] == [5.0, 5.0, 10.0, 10.0]


def test_get_tile_grid_3x3():
    """Test that _get_tile_grid generates correct 3x3 tile grid."""
    bbox = [0.0, 0.0, 9.0, 9.0]
    tiles = _get_tile_grid(bbox, 3)

    assert len(tiles) == 9

    # Check first tile
    assert tiles[0] == [0.0, 0.0, 3.0, 3.0]

    # Check last tile
    assert tiles[8] == [6.0, 6.0, 9.0, 9.0]


def test_get_tile_grid_single():
    """Test that _get_tile_grid works with single tile (1x1)."""
    bbox = [0.0, 0.0, 10.0, 10.0]
    tiles = _get_tile_grid(bbox, 1)

    assert len(tiles) == 1
    assert tiles[0] == bbox


def test_get_tile_grid_negative_coords():
    """Test _get_tile_grid with negative coordinates."""
    bbox = [-10.0, -10.0, 10.0, 10.0]
    tiles = _get_tile_grid(bbox, 2)

    assert len(tiles) == 4
    assert tiles[0] == [-10.0, -10.0, 0.0, 0.0]
    assert tiles[3] == [0.0, 0.0, 10.0, 10.0]
