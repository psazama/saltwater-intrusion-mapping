"""Dagster Definitions configured for GKE."""

from dagster import Definitions, ResourceDefinition
from dagster_gcp.gcs import gcs_pickle_io_manager

from .pipelines.assets import download_range, masks_by_range
from .pipelines.resources import data_root_res

dummy_gcs_resource = ResourceDefinition.hardcoded_resource("mock_gcs")

# Same assets as ``swmaps.defs`` but with a GCS-backed IO manager for cloud runs.
defs = Definitions(
    assets=[download_range, masks_by_range],
    resources={
        "data_root": data_root_res,
        "local_files": gcs_pickle_io_manager,
        "gcs": dummy_gcs_resource,
    },
)
