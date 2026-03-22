"""Tests for download.py functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from swmaps.pipeline.download import _extract_coords_from_geojson, run_download
from swmaps.schema import DownloadConfig

# Test GeoJSON content for a simple polygon
TEST_GEOJSON = """{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [-76.0, 38.0],
          [-75.0, 38.0],
          [-75.0, 39.0],
          [-76.0, 39.0],
          [-76.0, 38.0]
        ]]
      }
    }
  ]
}"""


def test_extract_coords_from_geojson():
    """Test extracting coordinates from a GeoJSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as f:
        f.write(TEST_GEOJSON)
        temp_path = f.name

    try:
        lat, lon = _extract_coords_from_geojson(temp_path)

        # The centroid of a square from (-76, 38) to (-75, 39) should be roughly (-75.5, 38.5)
        assert abs(lat - 38.5) < 0.01, f"Expected lat ~38.5, got {lat}"
        assert abs(lon - (-75.5)) < 0.01, f"Expected lon ~-75.5, got {lon}"
    finally:
        Path(temp_path).unlink()


def test_extract_coords_from_real_geojson():
    """Test extracting coordinates from a real GeoJSON file in the repo."""
    # This test uses the actual somerset_train.geojson file
    geojson_path = Path(__file__).parent.parent / "config" / "somerset_train.geojson"

    if not geojson_path.exists():
        pytest.skip("somerset_train.geojson not found")

    lat, lon = _extract_coords_from_geojson(geojson_path)

    # Based on the coordinates in somerset_train.geojson:
    # Bounds: [-76.236458, 37.886605, -75.539633, 38.298944]
    # Expected centroid: approximately (-75.888, 38.093)
    assert 37.8 < lat < 38.4, f"Latitude {lat} outside expected range"
    assert -76.3 < lon < -75.5, f"Longitude {lon} outside expected range"


def test_run_download_with_geojson():
    """Test that run_download() uses GeoJSON when provided."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as f:
        f.write(TEST_GEOJSON)
        temp_path = f.name

    try:
        cfg = DownloadConfig(
            start_date="2020-01-01",
            end_date="2020-01-02",
            mission=["sentinel-2"],
            geometry=temp_path,
        )
        with patch("swmaps.pipeline.download.process_date") as mock_process:
            mock_process.return_value = "dummy_path.tif"
            result = run_download(cfg)
            assert mock_process.called
            call_args = mock_process.call_args
            assert abs(call_args.kwargs["lat"] - 38.5) < 0.01
            assert abs(call_args.kwargs["lon"] - (-75.5)) < 0.01
            assert result.is_ok
    finally:
        Path(temp_path).unlink()


def test_run_download_with_explicit_latlon():
    """Test that run_download() works with explicit lat/lon."""
    cfg = DownloadConfig(
        start_date="2020-01-01",
        end_date="2020-01-02",
        mission=["sentinel-2"],
        latitude=38.0,
        longitude=-75.0,
    )
    with patch("swmaps.pipeline.download.process_date") as mock_process:
        mock_process.return_value = "dummy_path.tif"
        result = run_download(cfg)
        assert mock_process.called
        call_args = mock_process.call_args
        assert call_args.kwargs["lat"] == 38.0
        assert call_args.kwargs["lon"] == -75.0
        assert result.is_ok


def test_run_download_geojson_not_found():
    """Test that download_data() raises FileNotFoundError when GeoJSON doesn't exist."""
    cfg = DownloadConfig(
        start_date="2020-01-01",
        end_date="2020-01-02",
        mission=["sentinel-2"],
        geometry="/nonexistent/path/to/file.geojson",
    )
    result = run_download(cfg)
    assert result.status == "error"
    assert "GeoJSON file not found" in result.error


def test_run_download_skip():
    """Test that run_download() skips when skip_download is True."""
    cfg = DownloadConfig(
        start_date="2020-01-01",
        end_date="2020-01-02",
        mission=["sentinel-2"],
        latitude=38.0,
        longitude=-75.0,
        skip_download=True,
    )
    result = run_download(cfg)
    assert result.status == "skipped"
