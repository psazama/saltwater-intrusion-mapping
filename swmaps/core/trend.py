"""Utilities for modeling long-term class-mask trends."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr
from numba import njit
from rasterio.windows import Window
from scipy.stats import kendalltau

__all__ = [
    "load_class_year",
    "theil_sen_slope",
    "mk_p",
    "pixel_trend",
    "plot_trend_heatmap",
    "save_trend_results",
]


def check_center_for_nans(path: str, center_size: int = 100) -> bool:
    """Check if the center of the image contains any NaNs."""
    try:
        with rasterio.open(path) as src:
            img_width, img_height = src.width, src.height
            col_off = (img_width - center_size) // 2
            row_off = (img_height - center_size) // 2
            window = Window(
                col_off=col_off, row_off=row_off, width=center_size, height=center_size
            )
            data = src.read(window=window)
            return np.isnan(data).any()
    except Exception as e:
        logging.warning(f"[WARNING] Could not read {path}: {e}")
        return False


def check_image_for_nans(path: str, nan_threshold: float = 0.1) -> bool:
    """Check if the image contains at least a threshold fraction of NaNs."""
    try:
        with rasterio.open(path) as src:
            data = src.read()
            nan_frac = np.isnan(data).sum() / data.size
            return nan_frac >= nan_threshold
    except Exception as e:
        logging.warning(f"[WARNING] Could not read {path}: {e}")
        return False


def check_image_for_valid_signal(
    path: str, variance_threshold: float = 1e-6, nonzero_threshold: float = 0.01
) -> bool:
    """Check if the image contains meaningful signal rather than being empty or constant."""
    try:
        with rasterio.open(path) as src:
            data = src.read()
            if np.var(data) < variance_threshold:
                return False
            if np.count_nonzero(data) / data.size < nonzero_threshold:
                return False
            return True
    except Exception as e:
        logging.warning(f"[WARNING] Could not read {path}: {e}")
        return False


def load_class_year(
    paths: Sequence[str | Path],
    chunks: dict[str, int] | None = None,
    class_value: int | float = 1,
) -> xr.DataArray:
    """Load mask files and convert to yearly fraction of the target class value.

    Parameters
    ----------
    paths : sequence of str or Path
        Paths to mask files.
    chunks : dict, optional
        Dask chunk sizes for lazy loading.
    class_value : int or float
        Pixel value to consider as "present" for the class.

    Returns
    -------
    xr.DataArray
        Yearly fraction of pixels equal to class_value, dimensions (time, y, x).
    """
    datasets = []
    for p in paths:
        da = rxr.open_rasterio(p, masked=True)
        if chunks:
            da = da.chunk(chunks)
        # Convert to binary for the target class
        da = (da == class_value).astype(np.float32)
        datasets.append(da)

    def extract_start_date(path: str | Path) -> np.datetime64:
        filename = Path(path).stem
        match = re.search(r"(\d{8})", filename)
        if match:
            date_str = match.group(1)
            year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
            if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                return np.datetime64(f"{year:04d}-{month:02d}-{day:02d}")
        logging.warning(f"Invalid or missing date in filename: {filename}")
        return np.datetime64("NaT")

    times = [extract_start_date(p) for p in paths]
    ds = xr.concat(datasets, dim="time")
    ds = ds.assign_coords(time=("time", times))
    ds = ds.sortby("time")
    da = ds.squeeze("band", drop=True) if "band" in ds.dims else ds
    return da.resample(time="1Y", label="left").mean(dim="time")


@njit(parallel=True, fastmath=True)
def theil_sen_slope(ts: np.ndarray) -> np.float32:
    """Return the Theil–Sen slope for a 1-D array of observations."""
    if np.isnan(ts).all() or np.nanstd(ts) == 0:
        return np.nan
    n = ts.shape[0]
    slopes = np.empty(n * (n - 1) // 2, dtype=np.float32)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            slopes[k] = (ts[j] - ts[i]) / (j - i)
            k += 1
    return np.median(slopes)


def mk_p(ts: np.ndarray, years: np.ndarray) -> np.float32:
    """Compute the Mann-Kendall p-value for a time series."""
    tau, p = kendalltau(years, ts)
    return np.float32(1.0 if np.isnan(p) else p)


def pixel_trend(
    wet_year: xr.DataArray, progress: bool = True
) -> tuple[xr.DataArray, xr.DataArray]:
    """Calculate per-pixel Theil-Sen slope and MK p-value."""
    years = np.arange(wet_year.shape[0], dtype=np.float32)
    wet_year = wet_year.chunk({"time": -1})

    slope = xr.apply_ufunc(
        theil_sen_slope,
        wet_year,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
    )

    def _mk(ts: np.ndarray) -> np.float32:
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

    if progress:
        try:
            import dask
            from dask.diagnostics import ProgressBar
        except Exception:
            slope, pval = xr.compute(slope, pval)
        else:
            with ProgressBar():
                slope, pval = dask.compute(slope, pval)

    return slope, pval


def plot_trend_heatmap(
    slope: xr.DataArray,
    signif: xr.DataArray,
    vmin: float = -0.05,
    vmax: float = 0.05,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a heatmap of class trend with significance mask."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    slope.plot(ax=ax, cmap="coolwarm", vmin=vmin, vmax=vmax)
    slope.where(~signif).plot(ax=ax, cmap="coolwarm", alpha=0.25)
    if title:
        ax.set_title(title)
    ax.axis("equal")
    ax.axis("off")
    return ax


def save_trend_results(
    slope: xr.DataArray, pval: xr.DataArray, output_stem: str | Path
) -> tuple[Path, Path]:
    """Save slope and p-value arrays to GeoTIFF and NumPy files."""
    output_stem = Path(output_stem)

    slope_tif = output_stem.with_name(output_stem.name + "_slope.tif")
    pval_tif = output_stem.with_name(output_stem.name + "_pval.tif")
    slope_npy = output_stem.with_name(output_stem.name + "_slope.npy")
    pval_npy = output_stem.with_name(output_stem.name + "_pval.npy")

    slope, pval = slope.load(), pval.load()

    try:
        slope.rio.to_raster(slope_tif)
        pval.rio.to_raster(pval_tif)
    except Exception:
        slope.to_netcdf(slope_tif.with_suffix(".nc"))
        pval.to_netcdf(pval_tif.with_suffix(".nc"))

    np.save(slope_npy, slope.values)
    np.save(pval_npy, pval.values)

    return slope_tif, pval_tif
