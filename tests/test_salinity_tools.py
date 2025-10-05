"""Tests for the salinity estimation utilities."""

import pytest

np = pytest.importorskip("numpy")

from swmaps.core.salinity_tools import estimate_salinity_level
def _scaled(values):
    """Scale lists to the integer reflectance space used by Landsat sensors.

    Args:
        values (Iterable[float]): Reflectance values in the 0–1 range.

    Returns:
        numpy.ndarray: Values scaled to the integer reflectance domain.
    """
    return (np.array(values, dtype=np.float32) * 10000.0).reshape(1, -1)


def test_estimate_salinity_level_classification() -> None:
    """Check salinity classification results when using scaled reflectance values.

    Args:
        None

    Returns:
        None: Assertions validate the classification outputs.
    """
    blue = _scaled([0.05, 0.05, 0.05, 0.05])
    green = _scaled([0.6, 0.4, 0.28, 0.1])
    red = _scaled([0.05, 0.3, 0.25, 0.2])
    nir = _scaled([0.2, 0.25, 0.26, 0.5])
    swir1 = _scaled([0.05, 0.3, 0.45, 0.4])
    swir2 = _scaled([0.05, 0.4, 0.55, 0.4])

    result = estimate_salinity_level(blue, green, red, nir, swir1, swir2)

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
    """Ensure raw reflectance inputs bypass scaling and produce expected indices.

    Args:
        None

    Returns:
        None: Assertions confirm intermediate indices are present and shaped correctly.
    """
    blue = np.array([[0.05, 0.05]], dtype=np.float32)
    green = np.array([[0.5, 0.25]], dtype=np.float32)
    red = np.array([[0.05, 0.2]], dtype=np.float32)
    nir = np.array([[0.2, 0.3]], dtype=np.float32)
    swir1 = np.array([[0.05, 0.35]], dtype=np.float32)
    swir2 = np.array([[0.05, 0.3]], dtype=np.float32)

    result = estimate_salinity_level(
        blue, green, red, nir, swir1, swir2, reflectance_scale=None
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
    assert indices["ndwi"].shape == blue.shape
