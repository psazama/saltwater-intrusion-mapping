"""Registry of available pipeline tasks.

Each entry maps a task name to a callable that accepts a single
:class:`~pathlib.Path` and returns a :class:`~swmaps.schema.PipelineResult`.

To add a new task:

1. Implement the pipeline function in the appropriate module.
2. Import it here and add it to :data:`task_dict`.
3. Add a corresponding entry to ``config/processing_tasks.toml``.
"""

from __future__ import annotations

from swmaps.pipeline.masks import generate_water_mask

#: Registry of available pipeline tasks.
#: Keys are task name strings passed via the ``TASK`` environment variable.
#: Values are callables that accept a :class:`~pathlib.Path` and return a
#: :class:`~swmaps.schema.PipelineResult`.
task_dict: dict[str, callable] = {
    "water_mask": generate_water_mask,
}
