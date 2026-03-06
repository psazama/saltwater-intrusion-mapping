import hashlib
import os

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
