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