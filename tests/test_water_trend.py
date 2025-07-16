import pytest

from swmaps.core import water_trend

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")


def test_theil_sen_slope_linear():
    data = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    slope = water_trend.theil_sen_slope(data)
    assert slope == pytest.approx(1.0)


def test_mk_p_trend():
    ts = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    years = np.arange(5, dtype=np.float32)
    p = water_trend.mk_p(ts, years)
    assert 0 <= p <= 1


def test_pixel_trend_small():
    arr = xr.DataArray(
        np.stack(
            [np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)]
        ),
        dims=("time", "y", "x"),
    )
    slope, pval = water_trend.pixel_trend(arr, progress=False)
    assert slope.shape == (2, 2)
    assert pval.shape == (2, 2)
