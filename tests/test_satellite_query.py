"""Tests for satellite query and tiling functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_bounds  # noqa: E402


def create_mock_geotiff(
    path: Path, bounds: list, width: int = 100, height: int = 100, bands: int = 3
):
    """Create a mock GeoTIFF file for testing.

    Args:
        path: Output path for the GeoTIFF
        bounds: [minx, miny, maxx, maxy]
        width: Width in pixels
        height: Height in pixels
        bands: Number of bands
    """
    minx, miny, maxx, maxy = bounds
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Create random data
    data = np.random.randint(0, 10000, (bands, height, width), dtype=np.uint16)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=rasterio.uint16,
        crs="EPSG:32610",
        transform=transform,
    ) as dst:
        dst.write(data)


def test_merge_geotiff_tiles():
    """Test that GeoTIFF tiles can be merged correctly."""
    from swmaps.core.satellite_query import merge_geotiff_tiles

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create 4 mock tiles in a 2x2 grid
        bbox = [-120.0, 36.0, -119.0, 37.0]
        minx, miny, maxx, maxy = bbox

        # Split into 2x2 tiles
        width = (maxx - minx) / 2
        height = (maxy - miny) / 2

        tile_paths = []
        for row in range(2):
            for col in range(2):
                tile_minx = minx + col * width
                tile_miny = miny + row * height
                tile_maxx = tile_minx + width
                tile_maxy = tile_miny + height

                tile_path = tmpdir / f"tile_r{row}_c{col}.tif"
                create_mock_geotiff(
                    tile_path,
                    [tile_minx, tile_miny, tile_maxx, tile_maxy],
                    width=50,
                    height=50,
                    bands=3,
                )
                tile_paths.append(tile_path)

        # Merge tiles
        output_path = tmpdir / "merged.tif"
        result_path = merge_geotiff_tiles(tile_paths, output_path)

        # Verify merged file exists
        assert result_path.exists()

        # Verify merged file has correct properties
        with rasterio.open(result_path) as src:
            assert src.count == 3  # 3 bands
            assert src.width >= 50  # At least as wide as one tile
            assert src.height >= 50  # At least as tall as one tile

        # Verify tile cleanup (tiles should be deleted)
        for tile_path in tile_paths:
            assert not tile_path.exists(), f"Tile {tile_path} was not cleaned up"


def test_tile_and_download_gee_image():
    """Test that large images are split into tiles correctly."""
    from swmaps.core.satellite_query import tile_and_download_gee_image

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Mock the GEE image and download
        mock_image = MagicMock()

        # Create mock response for each tile download
        def mock_get_download_url(params):
            # Return a fake URL
            return "http://fake-gee-url.com/download"

        mock_image.getDownloadURL = mock_get_download_url

        bbox = [-120.0, 36.0, -119.0, 37.0]

        # Mock the requests.get to create actual tile files
        with patch("swmaps.core.satellite_query.requests.get") as mock_requests, patch(
            "swmaps.core.satellite_query.ee.Geometry.BBox"
        ) as mock_bbox:

            # Setup mock response
            mock_response = Mock()
            mock_response.iter_content = lambda chunk_size: [b"fake_geotiff_data"]
            mock_response.raise_for_status = Mock()
            mock_requests.return_value = mock_response

            # Mock ee.Geometry.BBox to avoid EE initialization
            mock_bbox.return_value = MagicMock()

            # Call the function
            tile_paths = tile_and_download_gee_image(
                mock_image,
                bbox,
                scale=10,
                crs="EPSG:32610",
                out_dir=tmpdir,
                prefix="test_image",
                n_tiles=2,
            )

            # Verify we got 4 tiles (2x2 grid)
            assert len(tile_paths) == 4

            # Verify all tiles exist
            for tile_path in tile_paths:
                assert tile_path.exists()

            # Verify tile naming convention
            assert any("r0_c0" in str(p) for p in tile_paths)
            assert any("r0_c1" in str(p) for p in tile_paths)
            assert any("r1_c0" in str(p) for p in tile_paths)
            assert any("r1_c1" in str(p) for p in tile_paths)


def test_download_gee_multiband_small_image():
    """Test that small images use direct download without tiling."""
    from swmaps.core.satellite_query import download_gee_multiband

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Mock GEE components
        mock_image = MagicMock()
        mock_image.get = MagicMock(return_value=MagicMock(getInfo=lambda: "test_id"))
        mock_image.select = MagicMock(return_value=mock_image)
        mock_image.reproject = MagicMock(return_value=mock_image)
        mock_image.clip = MagicMock(return_value=mock_image)
        mock_image.getDownloadURL = MagicMock(return_value="http://fake-url.com")

        # Small bbox that should not trigger tiling
        bbox = [-120.0, 36.0, -120.01, 36.01]  # Very small area

        with patch("swmaps.core.satellite_query.initialize_ee"), patch(
            "swmaps.core.satellite_query.get_mission"
        ) as mock_get_mission, patch(
            "swmaps.core.satellite_query.requests.get"
        ) as mock_requests, patch(
            "swmaps.core.satellite_query.ee.Geometry.BBox"
        ) as mock_bbox:

            # Setup mission mock
            mock_mission = MagicMock()
            mock_mission.gee_crs = "EPSG:32610"
            mock_get_mission.return_value = mock_mission

            # Setup requests mock
            mock_response = Mock()
            mock_response.iter_content = lambda chunk_size: [b"fake_geotiff_data"]
            mock_requests.return_value = mock_response

            # Mock ee.Geometry.BBox
            mock_bbox.return_value = MagicMock()

            # Call function
            result = download_gee_multiband(
                mock_image,
                "sentinel-2",
                {"B2": "B2", "B3": "B3", "B4": "B4"},
                bbox,
                tmpdir,
                scale=10,
            )

            # Verify a file was created
            assert result is not None
            assert "test_id" in result

            # Verify tiling was NOT used (direct download should be called)
            assert mock_image.getDownloadURL.called


def test_download_gee_multiband_large_image_triggers_tiling():
    """Test that large Sentinel-2 images trigger tiling."""
    from swmaps.core.satellite_query import (
        download_gee_multiband,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Mock GEE components
        mock_image = MagicMock()
        mock_image.get = MagicMock(
            return_value=MagicMock(getInfo=lambda: "large_test_id")
        )
        mock_image.select = MagicMock(return_value=mock_image)
        mock_image.reproject = MagicMock(return_value=mock_image)
        mock_image.clip = MagicMock(return_value=mock_image)

        # Large bbox that should trigger tiling (>30 MB)
        # For scale=10m, need ~0.5 degree bbox to exceed 30 MB with 12 bands
        bbox = [-120.0, 36.0, -119.5, 36.5]

        with patch("swmaps.core.satellite_query.initialize_ee"), patch(
            "swmaps.core.satellite_query.get_mission"
        ) as mock_get_mission, patch(
            "swmaps.core.satellite_query.tile_and_download_gee_image"
        ) as mock_tile_download, patch(
            "swmaps.core.satellite_query.merge_geotiff_tiles"
        ) as mock_merge, patch(
            "swmaps.core.satellite_query.ee.Geometry.BBox"
        ) as mock_bbox:

            # Setup mission mock
            mock_mission = MagicMock()
            mock_mission.gee_crs = "EPSG:32610"
            mock_get_mission.return_value = mock_mission

            # Mock ee.Geometry.BBox
            mock_bbox.return_value = MagicMock()

            # Mock tile_and_download to return fake tile paths
            fake_tiles = [tmpdir / f"tile_{i}.tif" for i in range(4)]
            for tile in fake_tiles:
                tile.touch()  # Create empty files
            mock_tile_download.return_value = fake_tiles

            # Mock merge to return output path
            output_path = tmpdir / "sentinel-2_large_test_id_multiband.tif"
            mock_merge.return_value = output_path

            # Call function with many bands to increase size estimate
            _result = download_gee_multiband(
                mock_image,
                "sentinel-2",
                {f"B{i}": f"B{i}" for i in range(1, 13)},  # 12 bands
                bbox,
                tmpdir,
                scale=10,
            )

            # Verify tiling was triggered
            assert mock_tile_download.called
            assert mock_merge.called
