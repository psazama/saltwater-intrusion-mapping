CREATE TABLE IF NOT EXISTS salinity_profiles (
    id               SERIAL PRIMARY KEY,
    cast_id          TEXT NOT NULL UNIQUE,
    location         GEOMETRY(POINT, 4326) NOT NULL,
    sample_date      DATE NOT NULL,
    surface_salinity FLOAT NOT NULL,
    max_depth        FLOAT,
    source_file      TEXT NOT NULL,
    ingested_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS salinity_depth_profiles (
    id          SERIAL PRIMARY KEY,
    cast_id     TEXT NOT NULL REFERENCES salinity_profiles(cast_id),
    depth_m     FLOAT NOT NULL,
    salinity    FLOAT,
    temperature FLOAT
);