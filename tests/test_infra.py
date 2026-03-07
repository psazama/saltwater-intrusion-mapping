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

        from swmaps.infra.backfill import parse_mission_from_path

        result = parse_mission_from_path(
            Path("data/landsat-7_LE07_014033_19990806_multiband.tif")
        )
        assert result == "landsat-7"

    def test_parse_mission_sentinel2(self):
        from pathlib import Path

        from swmaps.infra.backfill import parse_mission_from_path

        result = parse_mission_from_path(
            Path("data/sentinel-2_20190730T154819_multiband.tif")
        )
        assert result == "sentinel-2"

    def test_parse_mission_unknown_raises(self):
        from pathlib import Path

        from swmaps.infra.backfill import parse_mission_from_path

        with pytest.raises(ValueError):
            parse_mission_from_path(Path("data/unknown_sensor_multiband.tif"))
