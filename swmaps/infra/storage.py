"""GCS storage helpers for swmaps imagery and processed products.

All blob paths follow the convention::

    swmaps/imagery/raw/<scene_id>/<mission>/<filename>      # raw downloads
    swmaps/imagery/processed/<scene_id>/<task>/<filename>   # pipeline outputs

The GCS bucket is read from the ``GCS_BUCKET`` environment variable,
defaulting to ``"eo-ml-data"`` for local development.
"""

import os
from pathlib import Path

import rasterio
from google.cloud import storage
from rasterio.enums import Resampling

BUCKET_NAME = os.environ.get("GCS_BUCKET", "eo-ml-data")


def get_client() -> storage.Client:
    """Return an authenticated Google Cloud Storage client.

    Uses Application Default Credentials (ADC). Locally, run
    ``gcloud auth application-default login`` to configure ADC.

    Returns:
        storage.Client: Authenticated GCS client instance.
    """

    return storage.Client()


def raw_blob_path(scene_id: str, mission: str, filename: str) -> str:
    """Build the GCS blob path for a raw imagery file.

    Args:
        scene_id: GEE scene identifier, e.g. ``"S2B_20230601T..."``.
        mission: Mission slug, e.g. ``"sentinel-2"``.
        filename: Local filename, e.g. ``"sentinel-2_..._multiband.tif"``.

    Returns:
        str: Blob path relative to the bucket root.
    """

    return f"swmaps/imagery/raw/{scene_id}/{mission}/{filename}"


def processed_blob_path(scene_id: str, task: str, filename: str) -> str:
    """Build the GCS blob path for a processed pipeline product.

    Args:
        scene_id: GEE scene identifier.
        task: Pipeline task name, e.g. ``"water_mask"`` or ``"salinity"``.
        filename: Output filename.

    Returns:
        str: Blob path relative to the bucket root.
    """

    return f"swmaps/imagery/processed/{scene_id}/{task}/{filename}"


def upload_file(local_path: str | Path, blob_path: str) -> str:
    """Upload a local file to GCS and return its ``gs://`` URI.

    Args:
        local_path: Path to the local file to upload.
        blob_path: Destination blob path within the bucket.

    Returns:
        str: Full ``gs://`` URI of the uploaded object.
    """

    client = get_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(local_path))
    return f"gs://{BUCKET_NAME}/{blob_path}"


def download_file(blob_path: str, local_path: str | Path) -> Path:
    """Download a GCS blob to a local path.

    Parent directories are created automatically if they do not exist.

    Args:
        blob_path: Blob path within the bucket.
        local_path: Destination path on the local filesystem.

    Returns:
        Path: The resolved local path after download.
    """

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client = get_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(str(local_path))
    return local_path


def blob_path_from_uri(uri: str) -> str:
    """Strip the ``gs://<bucket-name>/`` prefix from a GCS URI.

    Args:
        uri: Full GCS URI, e.g. ``"gs://eo-ml-data/swmaps/imagery/..."``.

    Returns:
        str: Blob path without the bucket prefix.

    Raises:
        ValueError: If *uri* does not match the configured bucket.
    """

    prefix = f"gs://{BUCKET_NAME}/"
    if not uri.startswith(prefix):
        raise ValueError(f"URI {uri} does not match bucket {BUCKET_NAME}")
    return uri[len(prefix) :]


def blob_exists(blob_path: str) -> bool:
    """Check whether a blob exists in the configured GCS bucket.

    Args:
        blob_path: Blob path within the bucket.

    Returns:
        bool: ``True`` if the blob exists, ``False`` otherwise.
    """

    client = get_client()
    bucket = client.bucket(BUCKET_NAME)
    return bucket.blob(blob_path).exists()


def add_overviews(path: Path) -> None:
    """Add overviews to a GeoTIFF for efficient tile serving.

    Adds overview levels 2, 4, 8, 16, 32 using average resampling.
    Safe to call on files that already have overviews -- existing ones
    are replaced.

    Args:
        path: Path to an existing GeoTIFF.
    """

    with rasterio.open(path, "r+") as dst:
        dst.build_overviews([2, 4, 8, 16, 32], Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")
