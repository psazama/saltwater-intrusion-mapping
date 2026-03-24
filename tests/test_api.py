"""Tests for the swmaps FastAPI application.

Uses FastAPI's TestClient for endpoint testing and patches db.py functions
so no real database connection is needed.
"""

from datetime import date
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from swmaps.api import app

client = TestClient(app)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def scene_row():
    return {
        "scene_id": "LE07_014033_19990806",
        "sensor": "landsat-7",
        "acquisition_date": date(1999, 8, 6),
        "band_count": 6,
        "crs": "EPSG:32618",
        "status": "active",
        "file_locations": ["data/landsat-7/LE07_014033_19990806_multiband.tif"],
        "ingest_timestamp": None,
        "version_no": 1,
    }


@pytest.fixture
def salinity_row():
    return {
        "cast_id": "WOD_CAS_T_S_2018_2_000042",
        "sample_date": date(2018, 6, 15),
        "surface_salinity": 28.5,
        "max_depth": 1.0,
        "source_file": "WOD_CAS_T_S_2018_2.nc",
        "ingested_at": None,
    }


@pytest.fixture
def depth_row():
    return {
        "cast_id": "WOD_CAS_T_S_2018_2_000042",
        "depth_m": 0.5,
        "salinity": 28.5,
        "temperature": 22.1,
    }


@pytest.fixture
def run_row():
    return {
        "product_id": "abc123",
        "base_scene_id": "LE07_014033_19990806",
        "task": "water_mask",
        "status": "complete",
        "started_at": None,
        "completed_at": None,
        "output_paths": ["data/outputs/LE07_014033_19990806_mask.tif"],
        "error_message": None,
        "parameters": {},
    }


# ------------------------------------------------------------------
# Health / status
# ------------------------------------------------------------------


def test_health():
    """GET /health returns ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_tasks():
    """GET /tasks returns registered task names."""
    response = client.get("/tasks")
    assert response.status_code == 200
    assert "water_mask" in response.json()["tasks"]


# ------------------------------------------------------------------
# Scene endpoints
# ------------------------------------------------------------------


def test_get_scenes_requires_spatial():
    """GET /scenes without spatial params returns 422."""
    response = client.get("/scenes")
    assert response.status_code == 422


def test_get_scenes_with_bbox(scene_row):
    """GET /scenes with bbox returns list of scenes."""
    with patch("swmaps.api.get_connection"), patch(
        "swmaps.api.fetch_scenes", return_value=[scene_row]
    ):
        response = client.get("/scenes?bbox=-76.5,38.0,-75.5,39.0")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["scene_id"] == "LE07_014033_19990806"
    assert data[0]["sensor"] == "landsat-7"


def test_get_scenes_with_latlon(scene_row):
    """GET /scenes with lat/lon/radius_km returns list of scenes."""
    with patch("swmaps.api.get_connection"), patch(
        "swmaps.api.fetch_scenes", return_value=[scene_row]
    ):
        response = client.get("/scenes?lat=38.5&lon=-76.0&radius_km=10")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_get_scenes_invalid_bbox():
    """GET /scenes with malformed bbox returns 422."""
    response = client.get("/scenes?bbox=invalid")
    assert response.status_code == 422


def test_get_scene(scene_row):
    """GET /scenes/{scene_id} returns a single scene."""
    with patch("swmaps.api.get_connection"), patch(
        "swmaps.api.fetch_scene", return_value=scene_row
    ):
        response = client.get("/scenes/LE07_014033_19990806")
    assert response.status_code == 200
    assert response.json()["scene_id"] == "LE07_014033_19990806"


def test_get_scene_not_found():
    """GET /scenes/{scene_id} returns 404 when scene does not exist."""
    with patch("swmaps.api.get_connection"), patch(
        "swmaps.api.fetch_scene", return_value=None
    ):
        response = client.get("/scenes/nonexistent")
    assert response.status_code == 404


def test_get_scene_products(run_row):
    """GET /scenes/{scene_id}/products returns processing runs."""
    with patch("swmaps.api.get_connection"), patch(
        "swmaps.api.fetch_scene_products", return_value=[run_row]
    ):
        response = client.get("/scenes/LE07_014033_19990806/products")
    assert response.status_code == 200
    assert response.json()[0]["task"] == "water_mask"


# ------------------------------------------------------------------
# Salinity endpoints
# ------------------------------------------------------------------


def test_get_salinity_profiles_requires_spatial():
    """GET /salinity/profiles without spatial params returns 422."""
    response = client.get("/salinity/profiles")
    assert response.status_code == 422


def test_get_salinity_profiles(salinity_row):
    """GET /salinity/profiles with bbox returns list of profiles."""
    with patch("swmaps.api.get_connection"), patch(
        "swmaps.api.fetch_salinity_profiles", return_value=[salinity_row]
    ):
        response = client.get("/salinity/profiles?bbox=-76.5,38.0,-75.5,39.0")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["cast_id"] == "WOD_CAS_T_S_2018_2_000042"
    assert data[0]["surface_salinity"] == 28.5


def test_get_salinity_profile(salinity_row):
    """GET /salinity/profiles/{cast_id} returns a single profile."""
    with patch("swmaps.api.get_connection"), patch(
        "swmaps.api.fetch_salinity_profile", return_value=salinity_row
    ):
        response = client.get("/salinity/profiles/WOD_CAS_T_S_2018_2_000042")
    assert response.status_code == 200
    assert response.json()["cast_id"] == "WOD_CAS_T_S_2018_2_000042"


def test_get_salinity_profile_not_found():
    """GET /salinity/profiles/{cast_id} returns 404 when not found."""
    with patch("swmaps.api.get_connection"), patch(
        "swmaps.api.fetch_salinity_profile", return_value=None
    ):
        response = client.get("/salinity/profiles/nonexistent")
    assert response.status_code == 404


def test_get_depth_profile(depth_row):
    """GET /salinity/profiles/{cast_id}/depth returns depth levels."""
    with patch("swmaps.api.get_connection"), patch(
        "swmaps.api.fetch_depth_profile", return_value=[depth_row]
    ):
        response = client.get("/salinity/profiles/WOD_CAS_T_S_2018_2_000042/depth")
    assert response.status_code == 200
    data = response.json()
    assert data[0]["depth_m"] == 0.5
    assert data[0]["salinity"] == 28.5


def test_get_imagery_near_cast(scene_row):
    """GET /salinity/profiles/{cast_id}/imagery returns nearby scenes."""
    with patch("swmaps.api.get_connection"), patch(
        "swmaps.api.fetch_imagery_near_sample", return_value=[scene_row]
    ):
        response = client.get("/salinity/profiles/WOD_CAS_T_S_2018_2_000042/imagery")
    assert response.status_code == 200
    assert response.json()[0]["scene_id"] == "LE07_014033_19990806"


# ------------------------------------------------------------------
# Processing run endpoints
# ------------------------------------------------------------------


def test_get_processing_runs(run_row):
    """GET /runs returns list of processing runs."""
    with patch("swmaps.api.get_connection"), patch(
        "swmaps.api.fetch_processing_runs", return_value=[run_row]
    ):
        response = client.get("/runs")
    assert response.status_code == 200
    assert response.json()[0]["product_id"] == "abc123"


def test_get_processing_runs_filtered(run_row):
    """GET /runs with task and status filters."""
    with patch("swmaps.api.get_connection"), patch(
        "swmaps.api.fetch_processing_runs", return_value=[run_row]
    ):
        response = client.get("/runs?task=water_mask&status=complete")
    assert response.status_code == 200
    assert response.json()[0]["status"] == "complete"


def test_get_processing_run(run_row):
    """GET /runs/{product_id} returns a single run."""
    with patch("swmaps.api.get_connection"), patch(
        "swmaps.api.fetch_processing_run", return_value=run_row
    ):
        response = client.get("/runs/abc123")
    assert response.status_code == 200
    assert response.json()["product_id"] == "abc123"


def test_get_processing_run_not_found():
    """GET /runs/{product_id} returns 404 when not found."""
    with patch("swmaps.api.get_connection"), patch(
        "swmaps.api.fetch_processing_run", return_value=None
    ):
        response = client.get("/runs/nonexistent")
    assert response.status_code == 404


# ------------------------------------------------------------------
# Pipeline endpoints
# ------------------------------------------------------------------


def test_trigger_download():
    """POST /run/download returns a PipelineResult."""
    from swmaps.schema import PipelineResult

    with patch(
        "swmaps.api.run_download",
        return_value=PipelineResult.ok([]),
    ):
        response = client.post(
            "/run/download",
            json={
                "start_date": "2021-01-01",
                "end_date": "2021-12-31",
                "mission": ["sentinel-2"],
                "latitude": 38.5,
                "longitude": -76.0,
            },
        )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_trigger_download_invalid_config():
    """POST /run/download with missing required fields returns 422."""
    response = client.post(
        "/run/download",
        json={"mission": ["sentinel-2"]},
    )
    assert response.status_code == 422


def test_trigger_salinity():
    """POST /run/salinity returns a PipelineResult."""
    from swmaps.schema import PipelineResult

    with patch(
        "swmaps.api.run_salinity_pipeline",
        return_value=PipelineResult.skipped("disabled"),
    ):
        response = client.post(
            "/run/salinity",
            json={"run_salinity_pipeline": False},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "skipped"


def test_trigger_trend():
    """POST /run/trend returns a PipelineResult."""
    from swmaps.schema import PipelineResult

    with patch(
        "swmaps.api.run_trend_heatmap",
        return_value=PipelineResult.skipped("disabled"),
    ):
        response = client.post(
            "/run/trend",
            json={"run_water_trend": False},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
