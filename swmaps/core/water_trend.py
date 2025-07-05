"""Utilities for modelling long‑term water‑cover trends."""

from __future__ import annotations

import numpy as np
import xarray as xr
from numba import njit
from scipy.stats import kendalltau
import matplotlib.pyplot as plt


__all__ = [
    "load_wet_year",
    "theil_sen_slope",
    "mk_p",
    "pixel_trend",
    "plot_trend_heatmap",
]


def load_wet_year(mask_glob: str) -> xr.DataArray:
    """Load monthly water masks and convert to yearly wet fraction.

    Parameters
    ----------
    mask_glob:
        Glob pattern or list of files for monthly binary masks. Files must be
        readable by :func:`xarray.open_mfdataset` with ``engine='rasterio'`` and
        contain a ``time`` dimension.

    Returns
    -------
    xr.DataArray
        DataArray of yearly water fraction with dimensions (time, y, x).
    """
    ds = xr.open_mfdataset(
        mask_glob,
        concat_dim="time",
        combine="nested",
        engine="rasterio",
    )
    var = list(ds.data_vars)[0]
    da = ds[var]
    if "band" in da.dims:
        da = da.squeeze("band", drop=True)

    wet_year = da.resample(time="1Y", label="left").mean(dim="time")
    return wet_year


@njit(parallel=True, fastmath=True)
def theil_sen_slope(ts: np.ndarray) -> np.float32:
    """Return Theil–Sen slope for a 1‑D array."""
    n = ts.shape[0]
    k = 0
    slopes = np.empty(n * (n - 1) // 2, dtype=np.float32)
    for i in range(n - 1):
        for j in range(i + 1, n):
            slopes[k] = (ts[j] - ts[i]) / (j - i)
            k += 1
    return np.median(slopes)


def mk_p(ts: np.ndarray, years: np.ndarray) -> np.float32:
    """Mann‑Kendall p‑value for a time series."""
    tau, p = kendalltau(years, ts)
    return np.float32(1.0 if np.isnan(p) else p)


def pixel_trend(wet_year: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Calculate per‑pixel Theil–Sen slope and MK p‑value."""
    years = np.arange(wet_year.shape[0], dtype=np.float32)

    slope = xr.apply_ufunc(
        theil_sen_slope,
        wet_year,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
    )

    def _mk(ts):
        return mk_p(ts, years)

    pval = xr.apply_ufunc(
        _mk,
        wet_year,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
    )

    return slope, pval


def plot_trend_heatmap(
    slope: xr.DataArray,
    signif: xr.DataArray | np.ndarray,
    vmin: float = -0.05,
    vmax: float = 0.05,
    title: str | None = None,
    ax: plt.Axes | None = None,
):
    """Plot a heatmap of water trend with significance mask."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    slope.plot(ax=ax, cmap="coolwarm", vmin=vmin, vmax=vmax)
    slope.where(~signif).plot(ax=ax, cmap="coolwarm", alpha=0.25)
    if title:
        ax.set_title(title)
    ax.axis("equal")
    ax.axis("off")
    return ax
