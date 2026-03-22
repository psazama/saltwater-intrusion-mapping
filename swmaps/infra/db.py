"""PostgreSQL database helpers for the swmaps imagery and processing catalog.

This module provides functions for:

- Imagery catalog - registering downloaded scenes, querying by spatial extent,
  and checking for duplicates (:func:`register_scene`, :func:`insert_record`,
  :func:`fetch_scenes_intersecting`, :func:`scene_exists`).
- Salinity profiles - inserting in-situ cast records and depth profiles
  (:func:`insert_salinity_profile`, :func:`insert_depth_profile`).
- Processing runs - tracking pipeline task execution against the
  ``processed_products`` table (:func:`register_processing_run`,
  :func:`update_processing_run`, :func:`fetch_unprocessed_scenes`).
- Pub/Sub - publishing scene-ingestion notifications to downstream consumers
  (:func:`publish_scene_message`).

All functions expect an open psycopg2 connection. Use :func:`get_connection`
to obtain one. Connections use ``RealDictCursor`` so rows are returned as
dicts rather than tuples.
"""

import hashlib
import json
import os
import tomllib
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from google.cloud import pubsub_v1
from psycopg2.extras import RealDictCursor

load_dotenv()


def compute_file_hash(path: str) -> str:
    """Compute the MD5 hash of a file's contents.

    Args:
        path: Path to the file to hash.

    Returns:
        str: Hex-encoded MD5 digest.
    """
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_connection():
    """Return a psycopg2 connection using environment variables.

    Reads ``DB_HOST``, ``DB_PORT``, ``DB_NAME``, ``DB_USER``, and
    ``DB_PASSWORD`` from the environment. Uses ``RealDictCursor`` so
    all rows are returned as dicts.

    Returns:
        psycopg2.connection: Open database connection.

    Raises:
        KeyError: If any required environment variable is missing.
    """
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=os.environ.get("DB_PORT", 5432),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        cursor_factory=RealDictCursor,
    )


def run_migration(sql_file: str) -> None:
    """Read and execute a SQL file against the database.

    Args:
        sql_file: Path to a ``.sql`` file containing one or more statements.

    Returns:
        None
    """
    with open(sql_file, "r") as f:
        sql = f.read()

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
        conn.commit()


def publish_scene_message(
    scene_id: str,
    sensor: str,
    acquisition_date: str,
    topic_id: str = "swmaps-new-scenes",
) -> None:
    """Publish a new scene notification to a Pub/Sub topic.

    Args:
        scene_id: GEE scene identifier.
        sensor: Mission slug, e.g. ``"sentinel-2"``.
        acquisition_date: ISO-8601 date string, e.g. ``"2021-06-01"``.
        topic_id: Pub/Sub topic name. Defaults to ``"swmaps-new-scenes"``.

    Raises:
        ValueError: If ``GOOGLE_CLOUD_PROJECT`` is not set.
    """
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set.")

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)

    message = {
        "scene_id": scene_id,
        "sensor": sensor,
        "acquisition_date": acquisition_date,
    }

    future = publisher.publish(topic_path, json.dumps(message).encode("utf-8"))
    future.result()


def insert_record(
    conn,
    scene_id: str,
    location: str,
    band_count: int,
    acquisition_date: str,
    sensor: str,
    file_locations: list,
    crs: str = None,
    publish: bool = True,
    file_hash: str = None,
) -> dict:
    """Insert or upsert a single imagery record into the catalog.

    On conflict (duplicate ``scene_id``) the row is updated and
    ``version_no`` is incremented only when the file hash changes.

    Args:
        conn: Open psycopg2 connection.
        scene_id: Unique GEE scene identifier.
        location: WKT polygon string e.g. ``"POLYGON((...))"`` in EPSG:4326.
        band_count: Number of spectral bands in the imagery file.
        acquisition_date: ISO-8601 acquisition date string.
        sensor: Mission slug, e.g. ``"sentinel-2"``.
        file_locations: List of file paths or ``gs://`` URIs for this scene.
        crs: Coordinate reference system string, e.g. ``"EPSG:32618"``.
        publish: Whether to publish a Pub/Sub notification on success.
        file_hash: Pre-computed MD5 hash. Computed from ``file_locations[0]``
            when ``None``.

    Returns:
        dict: The inserted or updated row.
    """
    from swmaps.infra.validate import ImageryRecord

    # validate ImageryRecord fields
    ImageryRecord(
        scene_id=scene_id,
        location_wkt=location,
        band_count=band_count,
        acquisition_date=acquisition_date,
        sensor=sensor,
        file_locations=file_locations,
        crs=crs,
    )

    sql = """
        INSERT INTO imagery
            (scene_id, location, band_count, acquisition_date, sensor, file_locations, crs, file_hash)
        VALUES
            (%s, ST_GeomFromText(%s, 4326), %s, %s, %s, %s, %s, %s)
        ON CONFLICT (scene_id) DO UPDATE SET
            version_no = CASE
                WHEN imagery.file_hash != EXCLUDED.file_hash
                THEN imagery.version_no + 1
                ELSE imagery.version_no
            END,
            file_locations = EXCLUDED.file_locations,
            file_hash = EXCLUDED.file_hash,
            ingest_timestamp = NOW(),
            status = 'active'
        RETURNING *;
    """
    with conn.cursor() as cursor:
        file_hash = file_hash or compute_file_hash(file_locations[0])

        cursor.execute(
            sql,
            (
                scene_id,
                location,
                band_count,
                acquisition_date,
                sensor,
                file_locations,
                crs,
                file_hash,
            ),
        )
        conn.commit()
        result = cursor.fetchone()

    if publish and result is not None:
        try:
            publish_scene_message(
                scene_id=scene_id, sensor=sensor, acquisition_date=acquisition_date
            )
        except Exception as e:
            print(
                f"[Pub/Sub] Warning: Failed to publish message " f"for {scene_id}: {e}"
            )

    return result


def fetch_scenes_intersecting(conn, bbox: tuple) -> list:
    """Return all active imagery records intersecting a bounding box.

    Args:
        conn: Open psycopg2 connection.
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.

    Returns:
        list[dict]: Matching imagery rows.
    """
    sql = """
        SELECT * FROM imagery
        WHERE ST_Intersects(
            location,
            ST_MakeEnvelope(%s, %s, %s, %s, 4326)
        )
        AND status = 'active'
    """

    with conn.cursor() as cursor:
        cursor.execute(sql, bbox)
        return cursor.fetchall()


def register_scene(
    conn,
    image_id: str,
    mission: str,
    bbox: list,
    out_path: str,
    crs: str,
    acquisition_date: str,
) -> dict:
    """Register a downloaded GEE scene in the imagery catalog.

    Reads the band count from the GeoTIFF, optionally uploads to GCS,
    converts the bounding box to a WKT polygon, and calls
    :func:`insert_record`.

    Args:
        conn: Open psycopg2 connection.
        image_id: GEE scene identifier used as ``scene_id``.
        mission: Mission slug, e.g. ``"sentinel-2"``.
        bbox: ``[minx, miny, maxx, maxy]`` in EPSG:4326 degrees.
        out_path: Local path to the downloaded multiband GeoTIFF.
        crs: CRS string of the downloaded file, e.g. ``"EPSG:32618"``.
        acquisition_date: ISO-8601 acquisition date string.

    Returns:
        dict: The inserted or updated imagery row.
    """
    import rasterio
    from shapely.geometry import box
    from shapely.wkt import dumps

    from swmaps.infra.storage import raw_blob_path, upload_file

    # Get band count from file
    with rasterio.open(out_path) as src:
        band_count = src.count
    local_file_hash = compute_file_hash(out_path)

    # Upload to GCS if bucket is configured, else fall back to local path
    bucket = os.environ.get("GCS_BUCKET")
    if bucket:
        try:
            file_path = upload_file(
                local_path=out_path,
                blob_path=raw_blob_path(image_id, mission, Path(out_path).name),
            )
            print(f"[GCS] Uploaded to {file_path}")
        except Exception as e:
            print(f"[GCS] Warning: upload failed, falling back to local path: {e}")
            file_path = out_path
    else:
        file_path = out_path

    # Convert bbox list to WKT polygon
    location_wkt = dumps(box(*bbox))

    return insert_record(
        conn=conn,
        scene_id=image_id,
        location=location_wkt,
        band_count=band_count,
        acquisition_date=acquisition_date,
        sensor=mission,
        file_locations=[file_path],
        crs=crs,
        file_hash=local_file_hash,
    )


def scene_exists(conn, scene_id: str) -> bool:
    """Check whether an active scene already exists in the catalog.

    Args:
        conn: Open psycopg2 connection.
        scene_id: GEE scene identifier to check.

    Returns:
        bool: ``True`` if an active record exists, ``False`` otherwise.
    """
    sql = """
        SELECT 1 FROM imagery
        WHERE scene_id = %s
        AND status = 'active'
        LIMIT 1;
    """
    with conn.cursor() as cursor:
        cursor.execute(sql, (scene_id,))
        return cursor.fetchone() is not None


def insert_salinity_profile(
    conn,
    cast_id: str,
    longitude: float,
    latitude: float,
    sample_date: str,
    surface_salinity: float,
    max_depth: float,
    source_file: str,
) -> dict:
    """Insert a single in-situ salinity cast into ``salinity_profiles``.

    Silently skips duplicate ``cast_id`` values via ``ON CONFLICT DO NOTHING``.

    Args:
        conn: Open psycopg2 connection.
        cast_id: Stable unique identifier for this cast, e.g.
            ``"WOD_CAS_T_S_2018_2_000042"``.
        longitude: Cast longitude in decimal degrees (EPSG:4326).
        latitude: Cast latitude in decimal degrees (EPSG:4326).
        sample_date: ISO-8601 date string, e.g. ``"2018-06-15"``.
        surface_salinity: Near-surface salinity observation in PSU.
        max_depth: Maximum sampling depth in metres.
        source_file: Source NetCDF filename, e.g. ``"WOD_CAS_T_S_2018_2.nc"``.

    Returns:
        dict | None: The inserted row, or ``None`` if the cast already existed.
    """
    sql = """
        INSERT INTO salinity_profiles
            (cast_id, location, sample_date, surface_salinity,
            max_depth, source_file)
        VALUES
            (%s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s, %s, %s)
        ON CONFLICT (cast_id) DO NOTHING
        RETURNING *;
    """
    with conn.cursor() as cursor:
        cursor.execute(
            sql,
            (
                cast_id,
                longitude,
                latitude,
                sample_date,
                surface_salinity,
                max_depth,
                source_file,
            ),
        )
        conn.commit()
        return cursor.fetchone()


def insert_depth_profile(
    conn, cast_id: str, depths: list, salinities: list, temperatures: list = None
) -> None:
    """Insert all depth levels for a single cast into ``salinity_depth_profiles``.

    Args:
        conn: Open psycopg2 connection.
        cast_id: Cast identifier matching a row in ``salinity_profiles``.
        depths: Sampling depths in metres, parallel to *salinities*.
        salinities: Salinity observations in PSU, parallel to *depths*.
        temperatures: Optional temperature observations in °C, parallel to
            *depths*. Stored as ``NULL`` when ``None``.

    Returns:
        None
    """
    if temperatures is None:
        temperatures = [None] * len(depths)

    sql = """
        INSERT INTO salinity_depth_profiles
            (cast_id, depth_m, salinity, temperature)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT DO NOTHING;
    """
    with conn.cursor() as cursor:
        cursor.executemany(
            sql,
            [(cast_id, d, s, t) for d, s, t in zip(depths, salinities, temperatures)],
        )
        conn.commit()


def fetch_imagery_near_sample(
    conn, cast_id: str, radius_km: float = 50, days_window: int = 30
) -> list:
    """Return imagery records that spatially and temporally overlap a salinity cast.

    Args:
        conn: Open psycopg2 connection.
        cast_id: Cast identifier to search around.
        radius_km: Search radius in kilometres. Defaults to ``50``.
        days_window: Number of days before and after the sample date to
            include. Defaults to ``30``.

    Returns:
        list[dict]: Matching imagery rows with an additional ``distance_km``
        field indicating proximity to the cast location.
    """
    sql = """
        SELECT
            i.scene_id,
            i.sensor,
            i.acquisition_date,
            ST_Distance(
                i.location::geography,
                s.location::geography
            ) / 1000 as distance_km
        FROM imagery i
        JOIN salinity_profiles s ON ST_DWithin(
            i.location::geography,
            s.location::geography,
            %s
        )
        WHERE s.cast_id = %s
        AND i.acquisition_date BETWEEN
            s.sample_date - (%s * INTERVAL '1 day')
            AND s.sample_date + (%s * INTERVAL '1 day')
        AND i.status = 'active'
        ORDER BY i.acquisition_date;
    """
    with conn.cursor() as cursor:
        cursor.execute(sql, (radius_km * 1000, cast_id, days_window, days_window))
        return cursor.fetchall()


def seed_task_types(conn) -> None:
    """Seed the ``task_types`` table from ``config/processing_tasks.toml``.

    Safe to run multiple times - uses ``ON CONFLICT DO NOTHING``.

    Args:
        conn: Open psycopg2 connection.

    Returns:
        None
    """
    with open("config/processing_tasks.toml", "rb") as f:
        config = tomllib.load(f)

    sql = """
        INSERT INTO task_types (task, description)
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING;
    """
    with conn.cursor() as cursor:
        for task, details in config["tasks"].items():
            cursor.execute(sql, (task, details["description"]))
    conn.commit()


def generate_product_id(scene_id: str, task: str, parameters: dict) -> str:
    """Generate a stable unique product ID from scene, task, and parameters.

    Parameter key order does not affect the output - the dict is serialised
    with ``sort_keys=True`` before hashing.

    Args:
        scene_id: GEE scene identifier.
        task: Pipeline task name, e.g. ``"water_mask"``.
        parameters: Task parameter dict.

    Returns:
        str: Hex-encoded MD5 digest used as ``product_id``.
    """
    content = f"{scene_id}:{task}:{json.dumps(parameters, sort_keys=True)}"
    return hashlib.md5(content.encode()).hexdigest()


def register_processing_run(
    conn, scene_id: str, task: str, parameters: dict = None
) -> dict:
    """Create a new processing run record with status ``not_started``.

    If a record with the same ``product_id`` already exists (i.e. identical
    scene, task, and parameters), the existing row is returned unchanged.

    Args:
        conn: Open psycopg2 connection.
        scene_id: GEE scene identifier.
        task: Pipeline task name, e.g. ``"water_mask"``.
        parameters: Optional task parameter dict stored as JSONB.

    Returns:
        dict: The inserted or existing ``processed_products`` row, including
        the generated ``product_id``.
    """
    parameters = parameters or {}
    product_id = generate_product_id(scene_id, task, parameters)

    sql = """
        INSERT INTO processed_products
            (product_id, base_scene_id, task, parameters)
        VALUES
        (%s, %s, %s, %s)
        ON CONFLICT (product_id) DO NOTHING
        RETURNING *;
    """
    sql_fetch = """
        SELECT * FROM processed_products WHERE product_id = %s
    """
    with conn.cursor() as cursor:
        cursor.execute(sql, (product_id, scene_id, task, json.dumps(parameters)))
        conn.commit()
        result = cursor.fetchone()
        if result is None:
            cursor.execute(sql_fetch, (product_id,))
            result = cursor.fetchone()
        return result


def update_processing_run(
    conn,
    product_id: str,
    status: str,
    output_paths: list = None,
    error_message: str = None,
) -> dict:
    """Update the status, output paths, and error message of a processing run.

    Sets ``completed_at`` automatically when *status* is ``"complete"``
    or ``"failed"``.

    Args:
        conn: Open psycopg2 connection.
        product_id: Product identifier returned by :func:`register_processing_run`.
        status: New status - one of ``"not_started"``, ``"complete"``,
            ``"failed"``.
        output_paths: List of output file paths or ``gs://`` URIs.
        error_message: Human-readable error description when *status* is
            ``"failed"``.

    Returns:
        dict: The updated ``processed_products`` row.
    """
    sql = """
        UPDATE processed_products SET
            status = %s,
            output_paths = %s,
            error_message = %s,
            completed_at = CASE
                WHEN %s IN ('complete', 'failed') THEN NOW()
                ELSE NULL
        END
        WHERE product_id = %s
        RETURNING *;   
    """
    with conn.cursor() as cursor:
        cursor.execute(sql, (status, output_paths, error_message, status, product_id))
        conn.commit()
        return cursor.fetchone()


def fetch_unprocessed_scenes(conn, task: str, parameters: dict = None) -> list:
    """Return active imagery scenes with no completed run for a given task.

    Uses a JSONB containment check (``@>``) so a scene is considered
    processed only when its parameters are a superset of the requested ones.

    Args:
        conn: Open psycopg2 connection.
        task: Pipeline task name to check, e.g. ``"water_mask"``.
        parameters: Optional parameter dict - only scenes with no completed
            run matching these parameters are returned.

    Returns:
        list[dict]: Unprocessed imagery rows.
    """
    parameters = parameters or {}
    sql = """
        SELECT i.* FROM imagery i
        WHERE i.status = 'active'
        AND NOT EXISTS (
            SELECT 1 FROM processed_products p
            WHERE p.base_scene_id = i.scene_id
            AND p.task = %s
            AND p.status = 'complete'
            AND p.parameters @> %s
        );
    """
    with conn.cursor() as cursor:
        cursor.execute(sql, (task, json.dumps(parameters)))
        return cursor.fetchall()
