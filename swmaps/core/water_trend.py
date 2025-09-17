"""Utilities for modelling long‑term water‑cover trends."""

from __future__ import annotations

import logging
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
    "load_wet_year",
    "theil_sen_slope",
    "mk_p",
    "pixel_trend",
    "plot_trend_heatmap",
    "save_trend_results",
]


def check_center_for_nans(
    path: str,
    center_size: int = 100,
) -> bool:
    """Check if the center of the given image contains any nans

    Parameters
    ----------
    path:
        The path to the image to check, passed as a string
    center_size:
        The size of the center chunk to check for NaNs as an int. Default
        value is set to 100px by 100px square.

    Returns
    -------
        bool
            A boolean value that indicates if there are any NaNs in the center
            square of the image.
    """
    try:
        with rasterio.open(path) as src:
            # Get image size in pixels
            img_width = src.width
            img_height = src.height

            # Compute top-left corner of the centered window
            col_off = (img_width - center_size) // 2
            row_off = (img_height - center_size) // 2

            window = Window(
                col_off=col_off, row_off=row_off, width=center_size, height=center_size
            )

            data = src.read(window=window)

            nan_total = np.isnan(data).sum()
            if nan_total > 0:
                return True
            else:
                return False
    except Exception as e:
        logging.warning(f"[WARNING] Could not read {path}: {e}")


def check_image_for_nans(
    path: str,
    nan_threshold: float = 0.1,
) -> bool:
    """Check if the entire image contains at least a threshold
    fraction of NaNs.

    Parameters
    ----------
    path : str
        Path to the image to check.
    nan_threshold : float, optional
        Fraction of NaNs (0–1) allowed before returning True.
        Default is 0.1 (10%).

    Returns
    -------
    bool
        True if the fraction of NaNs in the image >= nan_threshold,
        False otherwise.
    """
    try:
        with rasterio.open(path) as src:
            data = src.read()  # reads all bands

            nan_frac = np.isnan(data).sum() / data.size

            return nan_frac >= nan_threshold

    except Exception as e:
        logging.warning(f"[WARNING] Could not read {path}: {e}")
        return False


def check_image_for_valid_signal(
    path: str,
    variance_threshold: float = 1e-6,
    nonzero_threshold: float = 0.01,
) -> bool:
    """Check if the entire image contains meaningful signal
    rather than being empty or constant.

    Parameters
    ----------
    path : str
        Path to the image to check.
    variance_threshold : float, optional
        Minimum variance required to consider the image valid.
        Default is 1e-6.
    nonzero_threshold : float, optional
        Minimum fraction of nonzero pixels required.
        Default is 0.01 (1%).

    Returns
    -------
    bool
        True if the image has valid signal, False if it appears empty or constant.
    """
    try:
        with rasterio.open(path) as src:
            data = src.read()  # all bands

            # Compute variance across all pixels
            if np.var(data) < variance_threshold:
                return False

            # Compute fraction of nonzero pixels
            nonzero_frac = np.count_nonzero(data) / data.size
            if nonzero_frac < nonzero_threshold:
                return False

            return True

    except Exception as e:
        logging.warning(f"[WARNING] Could not read {path}: {e}")
        return False


def load_wet_year(
    paths: Sequence[str | Path],
    chunks: dict[str, int] | None = None,
) -> xr.DataArray:
    """Load monthly water masks and convert to yearly wet fraction.

    Parameters
    ----------
    mask_glob:
        Glob pattern or list of files for monthly binary masks. Files must be
        readable by :func:`xarray.open_mfdataset` with ``engine='rasterio'`` and
        contain a ``time`` dimension.
    chunks: dict, optional
        Dask chunk sizes to apply when reading each raster. Pass a mapping
        like ``{"x": 512, "y": 512}`` to enable lazy loading and progress
        reporting during downstream computations.

    Returns
    -------
    xr.DataArray
        DataArray of yearly water fraction with dimensions (time, y, x).
    """
    datasets = []
    for p in paths:
        da = rxr.open_rasterio(p, masked=True)
        if chunks:
            da = da.chunk(chunks)
        datasets.append(da)

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
    if np.isnan(ts).all() or np.nanstd(ts) == 0:
        return np.nan

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


def pixel_trend(
    wet_year: xr.DataArray,
    progress: bool = True,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Calculate per‑pixel Theil–Sen slope and MK p‑value.

    Parameters
    ----------
    wet_year : xr.DataArray
        Yearly fraction of wet months with dimensions ``(time, y, x)``.
    progress : bool, optional
        If ``True``, display a progress bar while computing the trend.
    """
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


def save_trend_results(
    slope: xr.DataArray,
    pval: xr.DataArray,
    output_stem: str | Path,
) -> tuple[Path, Path]:
    """Save slope and p-value arrays to GeoTIFF and NumPy files.

    Parameters
    ----------
    slope, pval : xr.DataArray
        Arrays returned by :func:`pixel_trend`.
    output_stem : str or Path
        Path stem for output files (without extension). ``_slope`` and
        ``_pval`` plus the extensions ``.tif`` and ``.npy`` will be appended.

    Returns
    -------
    tuple[Path, Path]
        Paths to the generated GeoTIFF files for ``slope`` and ``pval``.
    """

    output_stem = Path(output_stem)

    slope_tif = output_stem.with_name(output_stem.name + "_slope.tif")
    pval_tif = output_stem.with_name(output_stem.name + "_pval.tif")
    slope_npy = output_stem.with_name(output_stem.name + "_slope.npy")
    pval_npy = output_stem.with_name(output_stem.name + "_pval.npy")

    # Ensure arrays are loaded before writing to disk
    slope = slope.load()
    pval = pval.load()

    try:
        slope.rio.to_raster(slope_tif)
        pval.rio.to_raster(pval_tif)
    except Exception:
        # Fall back to netCDF if GeoTIFF writing fails (e.g., missing CRS)
        slope.to_netcdf(slope_tif.with_suffix(".nc"))
        pval.to_netcdf(pval_tif.with_suffix(".nc"))

    np.save(slope_npy, slope.values)
    np.save(pval_npy, pval.values)

    return slope_tif, pval_tif
