"""Tests for download.py functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from swmaps.pipeline.download import _extract_coords_from_geojson, download_data


def test_extract_coords_from_geojson():
    """Test extracting coordinates from a GeoJSON file."""
    # Create a temporary GeoJSON file
    geojson_content = """{
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

    with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as f:
        f.write(geojson_content)
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


def test_download_data_with_geojson():
    """Test that download_data() correctly uses GeoJSON when provided."""
    # Create a temporary GeoJSON file
    geojson_content = """{
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

    with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as f:
        f.write(geojson_content)
        temp_path = f.name

    try:
        cfg = {
            "start_date": "2020-01-01",
            "end_date": "2020-01-02",
            "mission": "sentinel-2",
            "geometry": temp_path,  # Using GeoJSON instead of lat/lon
        }

        # Mock process_date to avoid actual GEE calls
        with patch("swmaps.pipeline.download.process_date") as mock_process:
            mock_process.return_value = "dummy_path.tif"

            download_data(cfg)

            # Verify that process_date was called with coordinates from GeoJSON
            assert mock_process.called
            call_args = mock_process.call_args

            # Check that lat and lon are approximately at the centroid
            lat_arg = call_args.kwargs["lat"]
            lon_arg = call_args.kwargs["lon"]

            assert abs(lat_arg - 38.5) < 0.01, f"Expected lat ~38.5, got {lat_arg}"
            assert abs(lon_arg - (-75.5)) < 0.01, f"Expected lon ~-75.5, got {lon_arg}"
    finally:
        Path(temp_path).unlink()


def test_download_data_with_explicit_latlon():
    """Test that download_data() still works with explicit lat/lon (backward compatibility)."""
    cfg = {
        "start_date": "2020-01-01",
        "end_date": "2020-01-02",
        "mission": "sentinel-2",
        "latitude": 38.0,
        "longitude": -75.0,
    }

    # Mock process_date to avoid actual GEE calls
    with patch("swmaps.pipeline.download.process_date") as mock_process:
        mock_process.return_value = "dummy_path.tif"

        download_data(cfg)

        # Verify that process_date was called with the explicit coordinates
        assert mock_process.called
        call_args = mock_process.call_args

        assert call_args.kwargs["lat"] == 38.0
        assert call_args.kwargs["lon"] == -75.0


def test_download_data_val_with_geojson():
    """Test that download_data() works with GeoJSON for validation data."""
    # Create a temporary GeoJSON file
    geojson_content = """{
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

    with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as f:
        f.write(geojson_content)
        temp_path = f.name

    try:
        cfg = {
            "val_start_date": "2020-01-01",
            "val_end_date": "2020-01-02",
            "mission": "sentinel-2",
            "val_region": temp_path,  # Using GeoJSON for validation
        }

        # Mock process_date to avoid actual GEE calls
        with patch("swmaps.pipeline.download.process_date") as mock_process:
            mock_process.return_value = "dummy_path.tif"

            download_data(cfg, val=True)

            # Verify that process_date was called with coordinates from GeoJSON
            assert mock_process.called
            call_args = mock_process.call_args

            # Check that lat and lon are approximately at the centroid
            lat_arg = call_args.kwargs["lat"]
            lon_arg = call_args.kwargs["lon"]

            assert abs(lat_arg - 38.5) < 0.01, f"Expected lat ~38.5, got {lat_arg}"
            assert abs(lon_arg - (-75.5)) < 0.01, f"Expected lon ~-75.5, got {lon_arg}"
    finally:
        Path(temp_path).unlink()


def test_download_data_geojson_not_found():
    """Test that download_data() raises FileNotFoundError when GeoJSON doesn't exist."""
    cfg = {
        "start_date": "2020-01-01",
        "end_date": "2020-01-02",
        "mission": "sentinel-2",
        "geometry": "/nonexistent/path/to/file.geojson",
    }

    with pytest.raises(FileNotFoundError) as exc_info:
        download_data(cfg)

    assert "GeoJSON file not found" in str(exc_info.value)
    assert "/nonexistent/path/to/file.geojson" in str(exc_info.value)


def test_download_data_val_geojson_not_found():
    """Test that download_data() raises FileNotFoundError when validation GeoJSON doesn't exist."""
    cfg = {
        "val_start_date": "2020-01-01",
        "val_end_date": "2020-01-02",
        "mission": "sentinel-2",
        "val_region": "/nonexistent/path/to/val_file.geojson",
    }

    with pytest.raises(FileNotFoundError) as exc_info:
        download_data(cfg, val=True)

    assert "Validation GeoJSON file not found" in str(exc_info.value)
    assert "/nonexistent/path/to/val_file.geojson" in str(exc_info.value)
