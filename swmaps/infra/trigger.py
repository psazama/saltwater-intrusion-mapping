"""Cloud Run HTTP trigger for the swmaps processing pipeline.

This module exposes a Flask endpoint that receives Pub/Sub push notifications
when new imagery scenes are ingested. For each notification it validates the
message, then launches a Cloud Run job for every task configured in
``config/processing_tasks.toml`` under ``pipeline.tasks_on_ingest``.

Environment variables
---------------------
``REGION``
    GCP region where the Cloud Run job is deployed. Defaults to
    ``"us-central1"``.
``PIPELINE_JOB``
    Cloud Run job name. Defaults to ``"swmaps-pipeline"``.
``GOOGLE_CLOUD_PROJECT``
    GCP project ID. Required for job dispatch.

Usage::

    # Run locally for development
    flask --app swmaps.infra.trigger run
"""

import base64
import json
import os

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flask import Flask, request
from google.cloud import run_v2
from pydantic import BaseModel

with open("config/validation.toml", "rb") as f:
    config = tomllib.load(f)

KNOWN_SENSORS = config["validation"]["known_sensors"]

with open("config/processing_tasks.toml", "rb") as f:
    config = tomllib.load(f)


class Message(BaseModel):
    data: str
    messageId: str
    publishTime: str


class Blob(BaseModel):
    message: Message
    subscription: str


required_keys = ["scene_id", "sensor"]


def validate_message(blob: dict) -> tuple | None:
    """Validate an incoming Pub/Sub push message.

    Checks that the message conforms to the expected Pub/Sub envelope
    structure and that the decoded payload contains the required fields
    with known sensor values.

    Args:
        blob: Raw JSON body from the Pub/Sub push request.

    Returns:
        tuple[str, int]: An error response tuple ``(message, status_code)``
        if validation fails, or ``None`` if the message is valid.
    """
    try:
        Blob.model_validate(blob)
    except Exception:
        return "Invalid json format", 400

    blob_data = json.loads(base64.b64decode(blob["message"]["data"]).decode("utf-8"))
    if all(key in blob_data for key in required_keys):
        if blob_data["sensor"] not in KNOWN_SENSORS:
            return "Invalid json format", 400
    else:
        return "Invalid json format", 400


app = Flask(__name__)


@app.route("/trigger", methods=["POST"])
def sub_message() -> tuple:
    """Handle a Pub/Sub push notification and launch pipeline jobs.

    Validates the incoming message, then dispatches a Cloud Run job
    execution for each task listed under ``pipeline.tasks_on_ingest``
    in ``config/processing_tasks.toml``.

    Returns:
        tuple[str, int]: ``("OK", 200)`` on success, or an error message
        and status code on failure.
    """
    blob = request.get_json()
    if blob is None:
        return "Invalid json format", 400

    error = validate_message(blob)
    if error:
        return error

    region = os.environ.get("REGION", "us-central1")
    job_name = os.environ.get("PIPELINE_JOB", "swmaps-pipeline")
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")

    try:
        client = run_v2.JobsClient()

        for task in config["pipeline"]["tasks_on_ingest"]:

            task_request = run_v2.RunJobRequest(
                name=f"projects/{project_id}/locations/{region}/jobs/{job_name}",
                overrides=run_v2.RunJobRequest.Overrides(
                    container_overrides=[
                        run_v2.RunJobRequest.Overrides.ContainerOverride(
                            env=[run_v2.EnvVar(name="TASK", value=task)]
                        )
                    ]
                ),
            )

            client.run_job(request=task_request)

        return "OK", 200
    except Exception as e:
        return f"Failed to trigger job: {e}", 500
