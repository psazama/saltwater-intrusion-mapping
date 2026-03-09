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