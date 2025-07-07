# swmaps/__init__.py

from dagster import Definitions, fs_io_manager

from .pipelines.assets import download_range, masks_by_range
from .pipelines.resources import data_root_res

# Expose all pipeline assets for local execution with a filesystem IO manager.
defs = Definitions(
    assets=[download_range, masks_by_range],
    resources={
        "data_root": data_root_res,
        "local_files": fs_io_manager,
    },
)
