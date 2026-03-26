"""Tests for the salinity estimation utilities."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")


def test_estimate_salinity_level_classification() -> None:
    """Check salinity classification using reflectance-range band values."""
    from swmaps.models.salinity_heuristic import SalinityHeuristicModel

    model = SalinityHeuristicModel()
    result = model.predict(
        {
            "blue": np.array([[0.05, 0.05, 0.05, 0.05]], dtype=np.float32),
            "green": np.array([[0.6, 0.4, 0.28, 0.1]], dtype=np.float32),
            "red": np.array([[0.05, 0.3, 0.25, 0.2]], dtype=np.float32),
            "nir": np.array([[0.2, 0.25, 0.26, 0.5]], dtype=np.float32),
            "swir1": np.array([[0.05, 0.3, 0.45, 0.4]], dtype=np.float32),
            "swir2": np.array([[0.05, 0.4, 0.55, 0.4]], dtype=np.float32),
        }
    )

    class_map = result["class_map"]
    score = result["score"]
    water_mask = result["water_mask"]

    assert class_map.shape == (1, 4)
    assert water_mask.dtype == bool
    assert water_mask.tolist() == [[True, True, True, False]]
    assert class_map.tolist() == [["fresh", "brackish", "saline", "land"]]
    assert np.isnan(score[0, 3])
    assert np.all(~np.isnan(score[:, :3]))


def test_estimate_salinity_level_reflectance_inputs() -> None:
    """Ensure reflectance inputs produce correct intermediate indices."""
    from swmaps.models.salinity_heuristic import SalinityHeuristicModel

    model = SalinityHeuristicModel()
    result = model.predict(
        {
            "blue": np.array([[0.05, 0.05]], dtype=np.float32),
            "green": np.array([[0.5, 0.25]], dtype=np.float32),
            "red": np.array([[0.05, 0.2]], dtype=np.float32),
            "nir": np.array([[0.2, 0.3]], dtype=np.float32),
            "swir1": np.array([[0.05, 0.35]], dtype=np.float32),
            "swir2": np.array([[0.05, 0.3]], dtype=np.float32),
        }
    )

    indices = result["indices"]
    assert set(indices) >= {
        "ndwi",
        "mndwi",
        "ndvi",
        "ndti",
        "turbidity_ratio",
        "chlorophyll_ratio",
        "salinity_proxy",
        "salinity_proxy_norm",
    }
    assert indices["ndwi"].shape == (1, 2)
