"""Tests for salinity utility helper safeguards."""

from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from swmaps.core.salinity import utils


def test_build_salinity_truth_skips_when_output_exists(tmp_path: Path) -> None:
    """If the truth CSV already exists, the builder should not redo work."""

    output_csv = tmp_path / "existing_truth.csv"
    output_csv.write_text("dummy")

    # Should return early without raising even if no datasets are supplied.
    utils.build_salinity_truth(dataset_files=[], output_csv=output_csv)

    assert output_csv.read_text() == "dummy"


def test_extract_salinity_features_skip_when_outputs_exist(tmp_path: Path) -> None:
    """Feature extraction should be skipped when outputs are already on disk."""

    feature_path = tmp_path / "features.tif"
    mask_path = tmp_path / "mask.tif"
    feature_path.write_bytes(b"")
    mask_path.write_bytes(b"")

    # No mosaic is needed because the function should return early.
    with pytest.raises(FileNotFoundError):
        Path("nonexistent").resolve(strict=True)

    utils.extract_salinity_features_from_mosaic(
        mosaic_path="nonexistent-mosaic.tif",
        mission_band_index={},
        output_feature_path=feature_path,
        output_mask_path=mask_path,
    )

    # Files should remain untouched (still empty) after the no-op call.
    assert feature_path.read_bytes() == b""
    assert mask_path.read_bytes() == b""
