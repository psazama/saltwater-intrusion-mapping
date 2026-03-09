
CREATE TABLE IF NOT EXISTS imagery (
    id  SERIAL PRIMARY KEY,
    scene_id    TEXT NOT NULL UNIQUE,
    location    GEOMETRY(POLYGON, 4326) NOT NULL,
    band_count  INTEGER,
    acquisition_date DATE NOT NULL,
    ingest_timestamp    TIMESTAMPTZ DEFAULT NOW(),
    version_no  INTEGER NOT NULL DEFAULT 1,
    sensor  TEXT NOT NULL,  -- source satellite/sensor
    file_locations  TEXT[] NOT NULL, -- local file paths
    crs TEXT,
    file_hash TEXT,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'missing', 'archived'))
);

CREATE INDEX IF NOT EXISTS imagery_location_idx ON imagery USING GIST (location);

CREATE TABLE IF NOT EXISTS salinity_profiles (
    id  SERIAL PRIMARY KEY,
    cast_id TEXT NOT NULL UNIQUE,
    location    GEOMETRY(POINT, 4326) NOT NULL,
    sample_date DATE NOT NULL,
    surface_salinity    FLOAT NOT NULL,
    max_depth   FLOAT,
    source_file TEXT NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS salinity_profiles_location_idx
    ON salinity_profiles USING GIST (location);
CREATE INDEX IF NOT EXISTS salinity_profiles_date_idx
    ON salinity_profiles (sample_date);

CREATE TABLE IF NOT EXISTS salinity_depth_profiles (
    id  SERIAL PRIMARY KEY,
    cast_id TEXT NOT NULL REFERENCES salinity_profiles(cast_id),
    depth_m FLOAT NOT NULL,
    salinity FLOAT,
    temperature FLOAT
);
CREATE INDEX IF NOT EXISTS salinity_depth_profiles_cast_idx
    ON salinity_depth_profiles (cast_id);

CREATE TABLE IF NOT EXISTS task_types (
    task TEXT PRIMARY KEY,
    description TEXT
);

CREATE TABLE IF NOT EXISTS processed_products (
    id SERIAL PRIMARY KEY,
    product_id TEXT NOT NULL UNIQUE,
    base_scene_id TEXT NOT NULL REFERENCES imagery(scene_id), -- the scene_id that the run ran on
    task TEXT NOT NULL REFERENCES task_types(task), -- the processing task
    status TEXT NOT NULL DEFAULT 'not_started'
        CHECK (status IN ('not_started', 'running', 'failed', 'complete')),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    output_paths TEXT[],
    error_message TEXT,
    parameters JSONB
);

CREATE INDEX IF NOT EXISTS processed_products_scene_idx
    ON processed_products (base_scene_id);

CREATE INDEX IF NOT EXISTS processed_products_task_status
    ON processed_products (task, status);