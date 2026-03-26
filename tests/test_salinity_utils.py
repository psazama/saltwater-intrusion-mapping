"""Tests for salinity utility helper safeguards."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pydantic")
pytest.importorskip("torch")

from swmaps.datasets import salinity
from swmaps.models.salinity_heuristic import SalinityHeuristicModel


def test_build_salinity_truth_skips_when_output_exists(tmp_path: Path) -> None:
    """If the truth CSV already exists, the builder should not redo work."""

    output_csv = tmp_path / "existing_truth.csv"
    output_csv.write_text("dummy")

    # Should return early without raising even if no datasets are supplied.
    salinity.build_salinity_truth(dataset_files=[], output_csv=output_csv)

    assert output_csv.read_text() == "dummy"


def test_salinity_predict_requires_valid_bands() -> None:
    """predict() should raise ValueError when band dict is missing keys."""
    model = SalinityHeuristicModel()

    with pytest.raises(ValueError, match="missing required keys"):
        model.predict({"blue": np.zeros((4, 4), dtype=np.float32)})
