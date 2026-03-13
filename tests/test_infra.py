from datetime import date
from unittest.mock import MagicMock, patch

import pytest

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_conn():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = {
        "id": 1,
        "scene_id": "LE07_014033_19990806",
        "sensor": "landsat-7",
        "acquisition_date": date(1999, 8, 6),
        "version_no": 1,
        "status": "active",
        "file_hash": "abc123",
    }
    cursor.fetchall.return_value = [cursor.fetchone.return_value]
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor


@pytest.fixture
def valid_wkt():
    return "POLYGON((-76.5 38.5, -76.0 38.5, -76.0 39.0, -76.5 39.0, -76.5 38.5))"


@pytest.fixture
def valid_scene():
    return {
        "scene_id": "LE07_014033_19990806",
        "location": "POLYGON((-76.5 38.5, -76.0 38.5, -76.0 39.0, -76.5 39.0, -76.5 38.5))",
        "band_count": 6,
        "acquisition_date": "1999-08-06",
        "sensor": "landsat-7",
        "file_locations": ["data/test/landsat-7_LE07_014033_19990806_multiband.tif"],
        "crs": "EPSG:32618",
    }


# ------------------------------------------------------------------
# Validation tests
# ------------------------------------------------------------------


class TestImageryValidation:
    def test_valid_record_passes(self, valid_scene):
        from swmaps.infra.validate import ImageryRecord

        record = ImageryRecord(
            scene_id=valid_scene["scene_id"],
            location_wkt=valid_scene["location"],
            band_count=valid_scene["band_count"],
            acquisition_date=valid_scene["acquisition_date"],
            sensor=valid_scene["sensor"],
            file_locations=valid_scene["file_locations"],
            crs=valid_scene["crs"],
        )
        assert record.scene_id == "LE07_014033_19990806"

    def test_invalid_scene_id_fails(self, valid_scene):
        from pydantic import ValidationError

        from swmaps.infra.validate import ImageryRecord

        with pytest.raises(ValidationError) as exc_info:
            ImageryRecord(
                scene_id="not_a_valid_id",
                location_wkt=valid_scene["location"],
                band_count=valid_scene["band_count"],
                acquisition_date=valid_scene["acquisition_date"],
                sensor=valid_scene["sensor"],
                file_locations=valid_scene["file_locations"],
            )
        assert "scene_id" in str(exc_info.value)

    def test_future_date_fails(self, valid_scene):
        from pydantic import ValidationError

        from swmaps.infra.validate import ImageryRecord

        with pytest.raises(ValidationError):
            ImageryRecord(
                scene_id=valid_scene["scene_id"],
                location_wkt=valid_scene["location"],
                band_count=valid_scene["band_count"],
                acquisition_date="2099-01-01",
                sensor=valid_scene["sensor"],
                file_locations=valid_scene["file_locations"],
            )

    def test_empty_file_locations_fails(self, valid_scene):
        from pydantic import ValidationError

        from swmaps.infra.validate import ImageryRecord

        with pytest.raises(ValidationError):
            ImageryRecord(
                scene_id=valid_scene["scene_id"],
                location_wkt=valid_scene["location"],
                band_count=valid_scene["band_count"],
                acquisition_date=valid_scene["acquisition_date"],
                sensor=valid_scene["sensor"],
                file_locations=[],
            )

    def test_coordinates_outside_study_area_fails(self, valid_scene):
        from pydantic import ValidationError

        from swmaps.infra.validate import ImageryRecord

        # Polygon somewhere in the Pacific
        out_of_bounds_wkt = (
            "POLYGON((-150.0 20.0, -149.0 20.0, -149.0 21.0, -150.0 21.0, -150.0 20.0))"
        )
        with pytest.raises(ValidationError):
            ImageryRecord(
                scene_id=valid_scene["scene_id"],
                location_wkt=out_of_bounds_wkt,
                band_count=valid_scene["band_count"],
                acquisition_date=valid_scene["acquisition_date"],
                sensor=valid_scene["sensor"],
                file_locations=valid_scene["file_locations"],
            )

    def test_unknown_sensor_fails(self, valid_scene):
        from pydantic import ValidationError

        from swmaps.infra.validate import ImageryRecord

        with pytest.raises(ValidationError):
            ImageryRecord(
                scene_id=valid_scene["scene_id"],
                location_wkt=valid_scene["location"],
                band_count=valid_scene["band_count"],
                acquisition_date=valid_scene["acquisition_date"],
                sensor="unknown-satellite-99",
                file_locations=valid_scene["file_locations"],
            )

    def test_negative_band_count_fails(self, valid_scene):
        from pydantic import ValidationError

        from swmaps.infra.validate import ImageryRecord

        with pytest.raises(ValidationError):
            ImageryRecord(
                scene_id=valid_scene["scene_id"],
                location_wkt=valid_scene["location"],
                band_count=-1,
                acquisition_date=valid_scene["acquisition_date"],
                sensor=valid_scene["sensor"],
                file_locations=valid_scene["file_locations"],
            )


# ------------------------------------------------------------------
# DB function tests
# ------------------------------------------------------------------


class TestDbFunctions:
    def test_fetch_scenes_intersecting_returns_list(self, mock_conn):
        from swmaps.infra.db import fetch_scenes_intersecting

        conn, cursor = mock_conn
        result = fetch_scenes_intersecting(conn, (-76.6, 38.4, -75.9, 39.1))
        assert isinstance(result, list)
        cursor.execute.assert_called_once()

    def test_insert_record_calls_execute(self, mock_conn, valid_scene):
        from swmaps.infra.db import insert_record

        conn, cursor = mock_conn
        # Patch file hash so we don't need a real file
        with patch("swmaps.infra.db.compute_file_hash", return_value="abc123"):
            result = insert_record(conn=conn, **valid_scene)
        cursor.execute.assert_called_once()
        assert result is not None

    def test_scene_exists_true(self, mock_conn):
        from swmaps.infra.db import scene_exists

        conn, cursor = mock_conn
        cursor.fetchone.return_value = {"1": 1}
        result = scene_exists(conn, "LE07_014033_19990806")
        assert result is True

    def test_scene_exists_false(self, mock_conn):
        from swmaps.infra.db import scene_exists

        conn, cursor = mock_conn
        cursor.fetchone.return_value = None
        result = scene_exists(conn, "nonexistent_scene")
        assert result is False

    def test_insert_record_publishes_on_success(self, mock_conn, valid_scene):
        from swmaps.infra.db import insert_record

        conn, cursor = mock_conn
        with patch("swmaps.infra.db.compute_file_hash", return_value="abc123"):
            with patch("swmaps.infra.db.publish_scene_message") as mock_publish:
                insert_record(conn=conn, publish=True, **valid_scene)
                mock_publish.assert_called_once_with(
                    scene_id=valid_scene["scene_id"],
                    sensor=valid_scene["sensor"],
                    acquisition_date=str(valid_scene["acquisition_date"]),
                )

    def test_insert_record_skips_publish_when_opted_out(self, mock_conn, valid_scene):
        from swmaps.infra.db import insert_record

        conn, cursor = mock_conn
        with patch("swmaps.infra.db.compute_file_hash", return_value="abc123"):
            with patch("swmaps.infra.db.publish_scene_message") as mock_publish:
                insert_record(conn=conn, publish=False, **valid_scene)
                mock_publish.assert_not_called()


# ------------------------------------------------------------------
# Processing run tests
# ------------------------------------------------------------------


class TestProcessingRuns:
    def test_generate_product_id_stable(self):
        """Same inputs always produce the same hash."""
        from swmaps.infra.db import generate_product_id

        id1 = generate_product_id(
            "LE07_014033_19990806",
            "segmentation",
            {"model_type": "farseg", "num_classes": 256},
        )
        id2 = generate_product_id(
            "LE07_014033_19990806",
            "segmentation",
            {"model_type": "farseg", "num_classes": 256},
        )
        assert id1 == id2

    def test_generate_product_id_parameter_order_invariant(self):
        """Parameter order should not affect the hash."""
        from swmaps.infra.db import generate_product_id

        id1 = generate_product_id(
            "LE07_014033_19990806",
            "segmentation",
            {"model_type": "farseg", "num_classes": 256},
        )
        id2 = generate_product_id(
            "LE07_014033_19990806",
            "segmentation",
            {"num_classes": 256, "model_type": "farseg"},
        )
        assert id1 == id2

    def test_generate_product_id_different_params_differ(self):
        """Different parameters should produce different hashes."""
        from swmaps.infra.db import generate_product_id

        id1 = generate_product_id(
            "LE07_014033_19990806", "segmentation", {"model_type": "farseg"}
        )
        id2 = generate_product_id(
            "LE07_014033_19990806", "segmentation", {"model_type": "panopticon"}
        )
        assert id1 != id2

    def test_generate_product_id_different_tasks_differ(self):
        """Different tasks should produce different hashes."""
        from swmaps.infra.db import generate_product_id

        id1 = generate_product_id("LE07_014033_19990806", "segmentation", {})
        id2 = generate_product_id("LE07_014033_19990806", "salinity", {})
        assert id1 != id2

    def test_register_processing_run_calls_execute(self, mock_conn):
        from swmaps.infra.db import register_processing_run

        conn, cursor = mock_conn
        cursor.fetchone.return_value = {
            "id": 1,
            "product_id": "abc123",
            "base_scene_id": "LE07_014033_19990806",
            "task": "segmentation",
            "status": "not_started",
            "parameters": {"model_type": "farseg"},
        }
        result = register_processing_run(
            conn=conn,
            scene_id="LE07_014033_19990806",
            task="segmentation",
            parameters={"model_type": "farseg"},
        )
        cursor.execute.assert_called_once()
        assert result is not None

    def test_update_processing_run_complete(self, mock_conn):
        from swmaps.infra.db import update_processing_run

        conn, cursor = mock_conn
        cursor.fetchone.return_value = {
            "product_id": "abc123",
            "status": "complete",
            "completed_at": "2026-01-01T00:00:00Z",
            "output_paths": ["data/outputs/segmentation/test.tif"],
        }
        result = update_processing_run(
            conn=conn,
            product_id="abc123",
            status="complete",
            output_paths=["data/outputs/segmentation/test.tif"],
        )
        cursor.execute.assert_called_once()
        assert result["status"] == "complete"

    def test_update_processing_run_failed(self, mock_conn):
        from swmaps.infra.db import update_processing_run

        conn, cursor = mock_conn
        cursor.fetchone.return_value = {
            "product_id": "abc123",
            "status": "failed",
            "error_message": "CUDA out of memory",
            "completed_at": "2026-01-01T00:00:00Z",
        }
        result = update_processing_run(
            conn=conn,
            product_id="abc123",
            status="failed",
            error_message="CUDA out of memory",
        )
        assert result["status"] == "failed"
        assert result["error_message"] == "CUDA out of memory"

    def test_fetch_unprocessed_scenes_returns_list(self, mock_conn):
        from swmaps.infra.db import fetch_unprocessed_scenes

        conn, cursor = mock_conn
        result = fetch_unprocessed_scenes(conn, task="segmentation")
        assert isinstance(result, list)
        cursor.execute.assert_called_once()

    def test_fetch_unprocessed_scenes_with_parameters(self, mock_conn):
        from swmaps.infra.db import fetch_unprocessed_scenes

        conn, cursor = mock_conn
        result = fetch_unprocessed_scenes(
            conn, task="segmentation", parameters={"model_type": "farseg"}
        )
        assert isinstance(result, list)
        # Verify parameters were passed to the query
        call_args = cursor.execute.call_args[0][1]
        assert "segmentation" in call_args


# ------------------------------------------------------------------
# Salinity db function tests
# ------------------------------------------------------------------


class TestSalinityFunctions:
    def test_insert_salinity_profile_calls_execute(self, mock_conn):
        from swmaps.infra.db import insert_salinity_profile

        conn, cursor = mock_conn
        cursor.fetchone.return_value = {
            "id": 1,
            "cast_id": "WOD_CAS_T_S_2018_2_000042",
            "sample_date": date(2018, 6, 15),
            "surface_salinity": 28.5,
            "sensor": "WOD",
        }
        result = insert_salinity_profile(
            conn=conn,
            cast_id="WOD_CAS_T_S_2018_2_000042",
            longitude=-76.0,
            latitude=38.5,
            sample_date="2018-06-15",
            surface_salinity=28.5,
            max_depth=1.0,
            source_file="WOD_CAS_T_S_2018_2.nc",
        )
        cursor.execute.assert_called_once()
        assert result is not None

    def test_insert_depth_profile_calls_executemany(self, mock_conn):
        from swmaps.infra.db import insert_depth_profile

        conn, cursor = mock_conn
        insert_depth_profile(
            conn=conn,
            cast_id="WOD_CAS_T_S_2018_2_000042",
            depths=[0.5, 1.0, 2.0, 5.0],
            salinities=[28.5, 28.6, 28.8, 29.1],
            temperatures=[22.1, 21.9, 21.5, 20.8],
        )
        cursor.executemany.assert_called_once()

    def test_insert_depth_profile_none_temperatures(self, mock_conn):
        from swmaps.infra.db import insert_depth_profile

        conn, cursor = mock_conn
        # Should not raise even without temperatures
        insert_depth_profile(
            conn=conn,
            cast_id="WOD_CAS_T_S_2018_2_000042",
            depths=[0.5, 1.0],
            salinities=[28.5, 28.6],
            temperatures=None,
        )
        cursor.executemany.assert_called_once()
        # Verify None temperatures were filled in
        call_args = cursor.executemany.call_args[0][1]
        assert all(row[3] is None for row in call_args)

    def test_fetch_imagery_near_sample_returns_list(self, mock_conn):
        from swmaps.infra.db import fetch_imagery_near_sample

        conn, cursor = mock_conn
        result = fetch_imagery_near_sample(
            conn=conn, cast_id="WOD_CAS_T_S_2018_2_000042", radius_km=50, days_window=30
        )
        assert isinstance(result, list)
        cursor.execute.assert_called_once()


# ------------------------------------------------------------------
# Backfill helpers
# ------------------------------------------------------------------


class TestBackfillHelpers:
    def test_parse_landsat_date(self):
        from swmaps.infra.backfill import _parse_date_from_filename

        result = _parse_date_from_filename("landsat-7_LE07_014033_19990806_multiband")
        assert result == "1999-08-06"

    def test_parse_sentinel_date(self):
        from swmaps.infra.backfill import _parse_date_from_filename

        result = _parse_date_from_filename(
            "sentinel-2_20190730T154819_20190730T155818_T18SVJ_multiband"
        )
        assert result == "2019-07-30"

    def test_parse_mission_landsat7(self):
        from pathlib import Path

        from swmaps.core.missions import get_mission_from_path

        result = get_mission_from_path(
            Path("data/landsat-7_LE07_014033_19990806_multiband.tif")
        ).slug
        assert result == "landsat-7"

    def test_parse_mission_sentinel2(self):
        from pathlib import Path

        from swmaps.core.missions import get_mission_from_path

        result = get_mission_from_path(
            Path("data/sentinel-2_20190730T154819_multiband.tif")
        ).slug
        assert result == "sentinel-2"

    def test_parse_mission_unknown_raises(self):
        from pathlib import Path

        from swmaps.core.missions import get_mission_from_path

        with pytest.raises(ValueError):
            get_mission_from_path(Path("data/unknown_sensor_multiband.tif")).slug
