"""Cloud Run job entry point for the swmaps processing pipeline.

This module is the main executable for the Docker-based processing pipeline.
It polls the database for unprocessed imagery scenes, runs the configured
task on each, and updates processing-run records with the results.

Task callables must accept a single :class:`~pathlib.Path` and return a
:class:`~swmaps.schema.PipelineResult`. This contract lets pipeline functions
be registered here without modification.

Environment variables
---------------------
``TASK``
    Name of the task to run (default: ``"water_mask"``). Must be a key in
    :data:`task_dict`.
``TASK_PARAMETERS``
    JSON-encoded dict of task parameters (default: ``"{}"``).
``GCS_BUCKET``
    When set, outputs are uploaded to GCS and their ``gs://`` URIs are stored.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from swmaps.infra.db import (
    fetch_unprocessed_scenes,
    get_connection,
    register_processing_run,
    update_processing_run,
)
from swmaps.pipeline.registry import task_dict
from swmaps.schema import PipelineResult

logger = logging.getLogger(__name__)


def _resolve_input(uri: str) -> tuple[Path, bool]:
    """Resolve a URI to a local path, downloading from GCS if necessary.

    Args:
        uri: Either a ``gs://`` URI or a local filesystem path.

    Returns:
        tuple[Path, bool]: ``(local_path, is_tmp)`` where *is_tmp* is
        ``True`` when the caller is responsible for cleaning up the file.
    """
    if uri.startswith("gs://"):
        from swmaps.infra.storage import blob_path_from_uri, download_file

        suffix = Path(uri).suffix
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        local_path = download_file(blob_path_from_uri(uri), tmp.name)
        return local_path, True
    return Path(uri), False


def _upload_output(local_path: Path, scene_id: str, task: str) -> str:
    """Upload a pipeline output to GCS if a bucket is configured.

    Args:
        local_path: Local file to upload.
        scene_id: Scene identifier used to build the blob path.
        task: Task name used to build the blob path.

    Returns:
        str: ``gs://`` URI if uploaded, otherwise the local path as a string.
    """
    if os.environ.get("GCS_BUCKET"):
        from swmaps.infra.storage import processed_blob_path, upload_file

        uri = upload_file(
            local_path=local_path,
            blob_path=processed_blob_path(scene_id, task, local_path.name),
        )
        logger.info("[GCS] Uploaded output to %s", uri)
        return uri
    return str(local_path)


def run_pipeline() -> None:
    """Fetch unprocessed scenes and run the configured task on each.

    Reads ``TASK`` and ``TASK_PARAMETERS`` environment variables, queries
    the database for imagery records with no completed run for that task,
    and processes each in turn. Results are written back to the
    ``processed_products`` table.
    """
    task_id = os.environ.get("TASK", "water_mask")
    param_str = os.environ.get("TASK_PARAMETERS", "{}")
    parameters = json.loads(param_str)

    if task_id not in task_dict:
        raise ValueError(
            f"Unknown task '{task_id}'. Registered tasks: {sorted(task_dict)}"
        )
    task_func = task_dict[task_id]

    with get_connection() as conn:
        unprocessed_scenes = fetch_unprocessed_scenes(conn, task_id, parameters)
        for scene in unprocessed_scenes:
            run_rec = register_processing_run(
                conn,
                scene_id=scene["scene_id"],
                task=task_id,
                parameters=parameters,
            )
            processed_locations: list[str] = []

            for f in scene["file_locations"]:
                local_path, is_tmp = _resolve_input(f)
                try:
                    result: PipelineResult = task_func(local_path)

                    if not result.is_ok:
                        update_processing_run(
                            conn,
                            run_rec["product_id"],
                            status="failed",
                            output_paths=[None],
                            error_message=result.error,
                        )
                        conn.commit()
                        break

                    for out_path in result.output_paths:
                        gcs_uri = _upload_output(
                            out_path, scene["scene_id"], task_id
                        )
                        processed_locations.append(gcs_uri)

                except Exception as exc:
                    update_processing_run(
                        conn,
                        run_rec["product_id"],
                        status="failed",
                        output_paths=[None],
                        error_message=str(exc),
                    )
                    conn.commit()
                    logger.exception(
                        "Task %s failed for scene %s", task_id, scene["scene_id"]
                    )
                    break
                finally:
                    if is_tmp:
                        local_path.unlink(missing_ok=True)
            else:
                update_processing_run(
                    conn,
                    run_rec["product_id"],
                    status="complete",
                    output_paths=processed_locations,
                )
                conn.commit()


if __name__ == "__main__":
    run_pipeline()