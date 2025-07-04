"""Dagster Definitions configured for GKE."""

from dagster import Definitions
from dagster_gcp.gcs import gcs_pickle_io_manager

from .pipelines.assets import masks_by_range
from .pipelines.resources import data_root_res

defs = Definitions(
    assets=[masks_by_range],
    resources={
        "data_root": data_root_res,
        "local_files": gcs_pickle_io_manager,
    },
)
