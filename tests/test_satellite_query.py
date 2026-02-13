"""
Tests for satellite_query module, specifically tile-based download functions.
"""


import pytest

from swmaps.core.satellite_query import merge_geotiff_tiles, tile_bbox


class TestTileBbox:
    """Tests for tile_bbox function."""

    def test_tile_bbox_2x2(self):
        """Test splitting bbox into 2x2 grid."""
        bbox = [0.0, 0.0, 2.0, 2.0]
        tiles = tile_bbox(bbox, grid_size=2)

        assert len(tiles) == 4

        # Check that tiles cover the original bbox
        expected = [
            [0.0, 0.0, 1.0, 1.0],  # bottom-left
            [0.0, 1.0, 1.0, 2.0],  # top-left
            [1.0, 0.0, 2.0, 1.0],  # bottom-right
            [1.0, 1.0, 2.0, 2.0],  # top-right
        ]

        for tile in tiles:
            assert tile in expected

    def test_tile_bbox_3x3(self):
        """Test splitting bbox into 3x3 grid."""
        bbox = [0.0, 0.0, 3.0, 3.0]
        tiles = tile_bbox(bbox, grid_size=3)

        assert len(tiles) == 9

        # Each tile should be 1.0 x 1.0
        for tile in tiles:
            minx, miny, maxx, maxy = tile
            assert abs((maxx - minx) - 1.0) < 1e-10
            assert abs((maxy - miny) - 1.0) < 1e-10

    def test_tile_bbox_single(self):
        """Test single tile (grid_size=1)."""
        bbox = [10.0, 20.0, 30.0, 40.0]
        tiles = tile_bbox(bbox, grid_size=1)

        assert len(tiles) == 1
        assert tiles[0] == bbox

    def test_tile_bbox_asymmetric(self):
        """Test with asymmetric bbox."""
        bbox = [-5.0, 10.0, 5.0, 30.0]
        tiles = tile_bbox(bbox, grid_size=2)

        assert len(tiles) == 4

        # Check width and height of each tile
        for tile in tiles:
            minx, miny, maxx, maxy = tile
            assert abs((maxx - minx) - 5.0) < 1e-10  # width = 10/2 = 5
            assert abs((maxy - miny) - 10.0) < 1e-10  # height = 20/2 = 10


class TestMergeGeotiffTiles:
    """Tests for merge_geotiff_tiles function."""

    def test_merge_requires_actual_files(self):
        """Test that merge function requires real GeoTIFF files."""
        # This is a placeholder test since we can't create actual GeoTIFF files
        # without Earth Engine access. The actual merging will be tested
        # through integration tests.

        # Just verify the function exists and has correct signature
        import inspect

        sig = inspect.signature(merge_geotiff_tiles)
        params = list(sig.parameters.keys())

        assert "tile_paths" in params
        assert "output_path" in params
        assert "bbox" in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
