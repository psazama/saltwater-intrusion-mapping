import pytest

np = pytest.importorskip("numpy")

from swmaps.core.salinity_tools import estimate_salinity_level


def test_salinity_override_requires_water_evidence():
    ndwi = np.array([[0.35, -0.6], [0.05, -0.1]])
    mndwi = np.array([[0.05, -0.4], [-0.05, 0.08]])
    salinity_proxy = np.array([[0.5, 3.5], [3.2, 4.1]])

    mask = estimate_salinity_level(
        ndwi,
        mndwi,
        salinity_proxy,
        ndwi_threshold=0.2,
        mndwi_threshold=0.2,
        salinity_override_threshold=2.5,
    )

    expected = np.array([[True, False], [True, True]])
    np.testing.assert_array_equal(mask, expected)


def test_salinity_override_with_mndwi_only():
    ndwi = np.array([-0.2, -0.05, -0.3])
    mndwi = np.array([0.01, 0.12, -0.4])
    salinity_proxy = np.array([3.0, 3.2, 5.0])

    mask = estimate_salinity_level(
        ndwi,
        mndwi,
        salinity_proxy,
        ndwi_threshold=0.2,
        mndwi_threshold=0.2,
        salinity_override_threshold=2.5,
    )

    expected = np.array([False, True, False])
    np.testing.assert_array_equal(mask, expected)


def test_salinity_override_optional_inputs():
    ndwi = np.array([0.1, 0.25])

    mask = estimate_salinity_level(
        ndwi,
        mndwi=None,
        salinity_proxy=np.array([3.0, 1.0]),
        ndwi_threshold=0.2,
        salinity_override_threshold=2.0,
    )

    expected = np.array([True, True])
    np.testing.assert_array_equal(mask, expected)
