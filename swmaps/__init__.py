# swmaps/__init__.py

try:
    from dagster import Definitions, fs_io_manager
except Exception:  # pragma: no cover - allow missing optional dependency
    Definitions = None  # type: ignore[assignment]
    fs_io_manager = None  # type: ignore[assignment]

if Definitions is not None:
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
else:  # pragma: no cover - dagster is optional for non-pipeline usage
    defs = None
