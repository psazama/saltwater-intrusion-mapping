# swmaps/__init__.py

from dagster import Definitions, fs_io_manager

from .pipelines.assets import masks_by_range
from .pipelines.resources import data_root_res

defs = Definitions(
    assets=[masks_by_range],
    resources={
        "data_root": data_root_res,
        "local_files": fs_io_manager,
    },
)
