"""Utilities for modelling long‑term water‑cover trends."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import xarray as xr
from numba import njit
from scipy.stats import kendalltau

__all__ = [
    "load_wet_year",
    "theil_sen_slope",
    "mk_p",
    "pixel_trend",
    "plot_trend_heatmap",
]


def load_wet_year(paths):
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
    datasets = [rxr.open_rasterio(p, masked=True) for p in paths]

    # Extract start date from filename: sentinel_eastern_shore_YYYY-MM-DD_YYYY-MM-DD_mask.tif
    def extract_start_date(path):
        parts = Path(path).stem.split("_")
        return np.datetime64(parts[-3])  # third-to-last is the start date

    times = [extract_start_date(p) for p in paths]
    ds = xr.concat(datasets, dim="time")
    ds = ds.assign_coords(time=("time", times))
    ds = ds.sortby("time")

    da = ds.squeeze("band", drop=True) if "band" in ds.dims else ds
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


def mk_p(ts, years):
    """Mann‑Kendall p‑value for a time series."""
    tau, p = kendalltau(years, ts)
    return np.float32(1.0 if np.isnan(p) else p)


def pixel_trend(wet_year):
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
    slope,
    signif,
    vmin=-0.05,
    vmax=0.05,
    title=None,
    ax=None,
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
