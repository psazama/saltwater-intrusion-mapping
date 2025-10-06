"""Tests for trend analysis utilities that operate on water observations."""

import pytest

from swmaps.core import water_trend

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
def test_theil_sen_slope_linear() -> None:
    """Confirm the Theil–Sen slope detects a perfect linear increase.

    Args:
        None

    Returns:
        None: Assertions validate the computed slope value.
    """
    data = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    slope = water_trend.theil_sen_slope(data)
    assert slope == pytest.approx(1.0)


def test_mk_p_trend() -> None:
    """Ensure the Mann–Kendall p-value falls within the valid probability range.

    Args:
        None

    Returns:
        None: Assertions confirm the p-value is between 0 and 1.
    """
    ts = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    years = np.arange(5, dtype=np.float32)
    p = water_trend.mk_p(ts, years)
    assert 0 <= p <= 1


def test_pixel_trend_small() -> None:
    """Validate pixel-wise trend output shapes on a tiny synthetic dataset.

    Args:
        None

    Returns:
        None: Assertions ensure the slope and p-value arrays match expectations.
    """
    arr = xr.DataArray(
        np.stack(
            [np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)]
        ),
        dims=("time", "y", "x"),
    )
    slope, pval = water_trend.pixel_trend(arr, progress=False)
    assert slope.shape == (2, 2)
    assert pval.shape == (2, 2)
