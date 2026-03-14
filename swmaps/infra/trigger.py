import base64
import json
import os
import tomllib

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


def validate_message(blob):
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
def sub_message():
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
