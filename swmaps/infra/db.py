import hashlib
import json
import os
import tomllib

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()


def compute_file_hash(path: str) -> str:
    """Compute MD5 hash of file contents."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_connection():
    """
    Returns a psycopg2 connection using environment variables.
    Expected vars: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
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
    """
    Reads and executes a SQL file against the database.
    Used to apply schema definitions and migrations.
    """
    with open(sql_file, "r") as f:
        sql = f.read()

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
        conn.commit()


def insert_record(
    conn,
    scene_id: str,
    location: str,
    band_count: int,
    acquisition_date: str,
    sensor: str,
    file_locations: list,
    crs: str = None,
) -> dict:
    """
    Inserts a single imagery record into the database.
    location should be a WKT polygon string e.g. 'POLYGON((...))'
    Returns the inserted row.
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
        file_hash = compute_file_hash(file_locations[0])

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
        return cursor.fetchone()


def fetch_scenes_intersecting(conn, bbox: tuple) -> list:
    """
    Returns all imagery records whose location intersects the
    given bounding box.
    bbox should be a tuple of (min_lon, min_lat, max_lon, max_lat)
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
    """
    Registers a downloaded scene in the imagery catalog.
    Derives band_count from the output_file.
    bbox should be [minx, miny, maxx, maxy]
    """
    import rasterio
    from shapely.geometry import box
    from shapely.wkt import dumps

    # Get band count from file
    with rasterio.open(out_path) as src:
        band_count = src.count

    # Convert bbox list to WKT polygon
    location_wkt = dumps(box(*bbox))

    return insert_record(
        conn=conn,
        scene_id=image_id,
        location=location_wkt,
        band_count=band_count,
        acquisition_date=acquisition_date,
        sensor=mission,
        file_locations=[out_path],
        crs=crs,
    )


def scene_exists(conn, scene_id: str) -> bool:
    """
    Returns True if a scene with the given scene_id already exists
    in the catalog with status 'active'.
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


############################
# Salinity Table Functions #
############################


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
    """
    Inserts a single salinity cast into salinity_profiles
    Returns the inserted row.
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
    """
    Inserts all depth levels for a single cast into salinity_depth_profiles.
    depths, salinities, and temperatures should be parallel lists.
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
    """
    Returns imagery records that spatially and temporally overlap a given
    salinity cast.
    radius_km: search radius around sample location
    days_window: number of days before and after sample date to search
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


############################
# Processing Table Functions #
############################


def seed_task_types(conn) -> None:
    """
    Seeds task_types table from processing_tasks config.
    Safe to run multiple times -- uses ON CONFLICT DO NOTHING.
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
    """
    Generates a stable unique product ID from scene_id, task, and parameters
    sort_keys=True ensures parameter order doesn't affect the hash
    """
    content = f"{scene_id}:{task}:{json.dumps(parameters, sort_keys=True)}"
    return hashlib.md5(content.encode()).hexdigest()


def register_processing_run(
    conn, scene_id: str, task: str, parameters: dict = None
) -> dict:
    """
    Creates a new processing run record with status 'not_started'
    Returns the inserted row including the generated product_id
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
    with conn.cursor() as cursor:
        cursor.execute(sql, (product_id, scene_id, task, json.dumps(parameters)))
        conn.commit()
        return cursor.fetchone()


def update_processing_run(
    conn,
    product_id: str,
    status: str,
    output_paths: list = None,
    error_message: str = None,
) -> dict:
    """
    Updates a processing run's status, output paths, and error message.
    Sets completed_at automatically when status is 'complete' or 'failed'
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
    """
    Returns all active imagery scenes that have no completed process runs
    for a giving task and parameter set.
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
