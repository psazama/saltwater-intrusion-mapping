"""GCS storage helpers for swmaps imagery and processed products."""

import os
from pathlib import Path

from google.cloud import storage

BUCKET_NAME = os.environ.get("GCS_BUCKET", "eo-ml-data")


def get_client() -> storage.Client:
    return storage.Client()


def raw_blob_path(scene_id: str, mission: str, filename: str) -> str:
    return f"swmaps/imagery/raw/{scene_id}/{mission}/{filename}"


def processed_blob_path(scene_id: str, task: str, filename: str) -> str:
    return f"swmaps/imagery/processed/{scene_id}/{task}/{filename}"


def upload_file(local_path: str | Path, blob_path: str) -> str:
    """
    Upload a local file to GCS.
    Returns the gs:// URI.
    """
    client = get_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(local_path))
    return f"gs://{BUCKET_NAME}/{blob_path}"


def download_file(blob_path: str, local_path: str | Path) -> Path:
    """
    Download a GCS blob to a local path.
    Returns the local Path.
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client = get_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(str(local_path))
    return local_path


def blob_path_from_uri(uri: str) -> str:
    """Strip gs://bucket-name/ prefix to get the blob path."""
    prefix = f"gs://{BUCKET_NAME}/"
    if not uri.startswith(prefix):
        raise ValueError(f"URI {uri} does not match bucket {BUCKET_NAME}")
    return uri[len(prefix) :]


def blob_exists(blob_path: str) -> bool:
    client = get_client()
    bucket = client.bucket(BUCKET_NAME)
    return bucket.blob(blob_path).exists()
