import json
import os
import tempfile
from pathlib import Path

from swmaps.infra.db import (
    fetch_unprocessed_scenes,
    get_connection,
    register_processing_run,
    update_processing_run,
)
from swmaps.pipeline.masks import generate_water_mask

task_dict = {"water_mask": generate_water_mask}


def _resolve_input(uri: str) -> tuple[Path, bool]:
    """
    If uri is a gs:// path, download to a temp file and return (local_path, True).
    If uri is a local path, return (Path(uri), False).
    The bool indicates whether the caller owns the temp file and should clean it up.
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
    """
    If GCS_BUCKET is set, upload the output file and return gs:// URI.
    Otherwise return the local path string.
    """
    if os.environ.get("GCS_BUCKET"):
        from swmaps.infra.storage import processed_blob_path, upload_file

        uri = upload_file(
            local_path=local_path,
            blob_path=processed_blob_path(scene_id, task, local_path.name),
        )
        print(f"[GCS] Uploaded output to {uri}")
        return uri
    return str(local_path)


def run_pipeline() -> None:
    task_id = os.environ.get("TASK", "water_mask")
    param_str = os.environ.get("TASK_PARAMETERS", "{}")
    parameters = json.loads(param_str)
    task_func = task_dict[task_id]
    with get_connection() as conn:
        unprocessed_scenes = fetch_unprocessed_scenes(conn, task_id, parameters)
        for scene in unprocessed_scenes:
            run_rec = register_processing_run(
                conn, scene_id=scene["scene_id"], task=task_id, parameters=parameters
            )
            processed_locations = []
            for f in scene["file_locations"]:
                local_path, is_tmp = _resolve_input(f)
                try:
                    output_path = Path(task_func(local_path))
                    gcs_uri = _upload_output(output_path, scene["scene_id"], task_id)
                    processed_locations.append(gcs_uri)

                except Exception as e:
                    update_processing_run(
                        conn,
                        run_rec["product_id"],
                        status="failed",
                        output_paths=[None],
                        error_message=str(e),
                    )
                    conn.commit()
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
