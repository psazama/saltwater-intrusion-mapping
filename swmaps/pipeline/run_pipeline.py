import json
import os

from swmaps.infra.db import (
    fetch_unprocessed_scenes,
    get_connection,
    register_processing_run,
    update_processing_run,
)
from swmaps.pipeline.masks import generate_water_mask

task_dict = {"water_mask": generate_water_mask}


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
                try:
                    processed_locations.append(str(task_func(f)))

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
            update_processing_run(
                conn,
                run_rec["product_id"],
                status="complete",
                output_paths=processed_locations,
            )

            conn.commit()


if __name__ == "__main__":
    run_pipeline()
