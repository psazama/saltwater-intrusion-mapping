"""Analytical helpers for deriving water salinity products from imagery."""

import logging
import os
import shutil
from pathlib import Path
from typing import Sequence
from urllib.request import urlopen

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.windows import Window
from tqdm import tqdm

from swmaps.config import data_path
from swmaps.core.indices import compute_ndwi


def _safe_normalized_difference(
    numerator: np.ndarray, denominator: np.ndarray
) -> np.ndarray:
    """Compute a normalized difference index with zero-division protection.

    Args:
        numerator (np.ndarray): Numerator band.
        denominator (np.ndarray): Denominator band.

    Returns:
        np.ndarray: Normalised difference array with NaNs avoided.
    """

    return np.divide(
        numerator - denominator,
        numerator + denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=(numerator + denominator) != 0,
    ).astype(np.float32)


def _default_wod_example() -> Path:
    """Return the default World Ocean Database example profile path.

    Args:
        None

    Returns:
        Path: Absolute path to the bundled sample profile.
    """

    return data_path("salinity_labels", "WOD", "WOD_CAS_T_S_2020_7.nc")


def download_salinity_datasets(
    listing_file,
    destination="salinity_labels/codc",
    base_url="http://www.ocean.iap.ac.cn/ftp/cheng/CODCv2.1_Insitu_T_S_database/nc/",
):
    """Download CODC salinity NetCDF files listed in a text file.

    Args:
        listing_file (str | Path): Text file with one filename per line.
            Lines starting with '#' or blank lines are ignored.
        destination (str | Path): Directory to save downloaded files.
        base_url (str): Base URL hosting the CODC NetCDF files.

    Returns:
        list[Path]: Paths to the downloaded (or existing) files.
    """

    listing_path = Path(listing_file)
    if not listing_path.exists():
        raise FileNotFoundError(f"Listing file not found: {listing_file}")

    target_dir = Path(destination)
    target_dir.mkdir(parents=True, exist_ok=True)

    with listing_path.open("r", encoding="utf-8") as f:
        filenames = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

    downloaded_paths = []
    for fname in tqdm(filenames, desc="Downloading salinity datasets"):
        dest = target_dir / fname
        if dest.exists():
            downloaded_paths.append(dest)
            continue

        url = base_url + fname
        try:
            with urlopen(url) as r, open(dest, "wb") as out:
                shutil.copyfileobj(r, out)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            continue

        downloaded_paths.append(dest)

    return downloaded_paths


def build_salinity_truth(
    dataset_files: Sequence[str | Path] | None = None,
    output_csv: str | Path | None = None,
    depth: float = 1.0,
    prof_limit: int | None = None,
) -> None:
    """Extract near-surface salinity observations into a flat CSV dataset.
    Data source:
    Zhang, B., Cheng, L., Tan, Z. et al.
    CODC-v1: a quality-controlled and bias-corrected ocean temperature profile database from 1940–2023.
    http://www.ocean.iap.ac.cn/ftp/cheng/CODCv2.1_Insitu_T_S_database/

    Args:
        dataset_files (Sequence[str | Path] | None): Collection of NetCDF
            profile files. If ``None``, use the default sample dataset.
        output_csv (str | Path | None): Destination for the combined CSV.
        depth (float): Maximum sampling depth (metres) considered near
            surface.
        prof_limit (int | None): Optional limit on the number of profiles to
            process from each file.

    Returns:
        None: Data are written to ``output_csv``.
    """

    dataset_files = (
        list(dataset_files) if dataset_files is not None else [_default_wod_example()]
    )
    output_csv = (
        Path(output_csv)
        if output_csv
        else data_path("salinity_labels", "codc_salinity_profiles.csv")
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    lat_idx = 4
    lon_idx = 5
    year_idx = 1
    month_idx = 2
    day_idx = 3

    header_written = False

    for dataset_file in tqdm(dataset_files):
        print(".")
        try:
            ds = xr.open_dataset(dataset_file)
        except Exception:
            logging.warning(f"Error opening dataset: {dataset_file}")
            continue

        # Use full range if no limit provided
        if prof_limit is None:
            prof_limit = ds.sizes["N_PROF"]

        prof_slice = slice(0, prof_limit)

        print("..")
        # Extract salinity and depth for those profiles
        sal = ds["Salinity_origin"].isel(N_PROF=prof_slice)
        dep = ds["Depth_origin"].isel(N_PROF=prof_slice)
        lats = (
            ds["Profile_info_record_all"]
            .isel(STRINGS18=lat_idx, N_PROF=prof_slice)
            .values
        )
        lons = (
            ds["Profile_info_record_all"]
            .isel(STRINGS18=lon_idx, N_PROF=prof_slice)
            .values
        )
        years = (
            ds["Profile_info_record_all"]
            .isel(STRINGS18=year_idx, N_PROF=prof_slice)
            .values.astype(int)
        )
        months = (
            ds["Profile_info_record_all"]
            .isel(STRINGS18=month_idx, N_PROF=prof_slice)
            .values.astype(int)
        )
        days = (
            ds["Profile_info_record_all"]
            .isel(STRINGS18=day_idx, N_PROF=prof_slice)
            .values.astype(int)
        )

        times = [
            f"{y:04d}-{m:02d}-{d:02d}" if y > 0 else None
            for y, m, d in zip(years, months, days)
        ]

        print("...")
        valid_profile_mask = (dep <= depth).any(dim="N_LEVELS").values
        valid_indices = np.where(valid_profile_mask)[0]

        print("....")
        records = []

        for i in tqdm(valid_indices):
            try:
                sal_i = sal.isel(N_PROF=i).values
                dep_i = dep.isel(N_PROF=i).values

                valid = dep_i <= depth
                if not np.any(valid):
                    continue

                surface_val = sal_i[valid][0]
                if np.isnan(surface_val):
                    continue

                records.append(
                    {
                        "salinity": surface_val,
                        "latitude": lats[i],
                        "longitude": lons[i],
                        "date": times[i],
                        "source_file": os.path.basename(dataset_file),
                    }
                )

            except Exception as e:
                print(f"Profile {i} in {dataset_file} failed: {e}")
                continue

        df = pd.DataFrame(records)

        print(".....")
        if not header_written:
            df.to_csv(output_csv, mode="w", index=False)
            header_written = True
        else:
            df.to_csv(output_csv, mode="a", header=False, index=False)

        print("......")


def load_salinity_truth(truth_file: str | Path | None = None) -> pd.DataFrame:
    """Load the prepared salinity truth table from disk and drop missing rows.

    Args:
        truth_file (str | Path | None): Path to the CSV generated by
            :func:`build_salinity_truth`.

    Returns:
        pandas.DataFrame: Cleaned truth table without missing values.
    """

    truth_file = (
        Path(truth_file)
        if truth_file
        else data_path("salinity_labels", "codc_salinity_profiles.csv")
    )
    df = pd.read_csv(truth_file)
    df_clean = df.dropna()
    return df_clean


def process_salinity_features_chunk(
    src: rasterio.io.DatasetReader,
    win: Window,
    band_index: dict[str, int],
    src_lbl: rasterio.io.DatasetReader | None = None,
    dst_y: rasterio.io.DatasetWriter | None = None,
    dst_y_win: Window | None = None,
    water_threshold: float = 0.2,
    profile: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute salinity features and water mask for a window.

    Args:
        src (rasterio.io.DatasetReader): Open raster mosaic reader.
        win (Window): Window describing the area to read.
        band_index (dict[str, int]): Mapping of spectral band names to
            raster band numbers.
        src_lbl (rasterio.io.DatasetReader | None): Optional label raster
            reader.
        dst_y (rasterio.io.DatasetWriter | None): Optional label writer for
            water-masked salinity values.
        dst_y_win (Window | None): Window in the destination label raster.
        water_threshold (float): NDWI threshold distinguishing water pixels.
        profile (dict | None): Optional raster profile override.

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature stack and water mask arrays for
        the specified window.
    """
    blue = src.read(band_index["blue"], window=win)
    green = src.read(band_index["green"], window=win)
    red = src.read(band_index["red"], window=win)
    nir = src.read(band_index["nir08"], window=win)
    swir1 = src.read(band_index["swir16"], window=win)
    swir2 = src.read(band_index["swir22"], window=win)

    # Compute indices
    ndti = np.divide(
        green - blue, green + blue, out=np.zeros_like(green), where=(green + blue) != 0
    )
    turbidity = np.divide(red, green, out=np.zeros_like(red), where=green != 0)
    chlorophyll_proxy = np.divide(
        blue, green, out=np.zeros_like(blue), where=green != 0
    )
    salinity_proxy = swir1 + swir2
    ndvi = np.divide(
        nir - red, nir + red, out=np.zeros_like(nir), where=(nir + red) != 0
    )
    # Compute NDWI + mask
    if profile is None:
        profile = src.profile.copy()
        transform = src.window_transform(win)
        profile.update(
            {"height": win.height, "width": win.width, "transform": transform}
        )

    ndwi_mask = compute_ndwi(
        green, nir, profile, out_path=None, display=False, threshold=water_threshold
    )
    water_mask = ndwi_mask.astype(bool)

    # Stack and mask
    feat_stack = np.stack([ndti, turbidity, chlorophyll_proxy, salinity_proxy, ndvi])
    feat_stack = np.nan_to_num(feat_stack, nan=0.0)
    feat_stack *= water_mask

    # Optionally write masked label
    if src_lbl and dst_y and dst_y_win:
        label_data = src_lbl.read(1, window=win)
        masked_label = (label_data * water_mask).astype("float32")
        dst_y.write(masked_label, 1, window=dst_y_win)

    return feat_stack.astype("float32"), ndwi_mask.astype("float32")


def extract_salinity_features_from_mosaic(
    mosaic_path: str | Path,
    mission_band_index: dict[str, int],
    output_feature_path: str | Path,
    output_mask_path: str | Path,
    label_path: str | Path | None = None,
    output_label_path: str | Path | None = None,
    chunk_size: int = 512,
    water_threshold: float = 0.2,
) -> None:
    """Extract features across a mosaic and persist arrays to disk.

    Args:
        mosaic_path (str | Path): Path to the input multispectral GeoTIFF.
        mission_band_index (dict[str, int]): Band index mapping for the
            mission.
        output_feature_path (str | Path): Path for the multi-band feature
            GeoTIFF.
        output_mask_path (str | Path): Path for the single-band water mask.
        label_path (str | Path | None): Optional salinity label raster to
            sample.
        output_label_path (str | Path | None): Optional destination for the
            masked label raster.
        chunk_size (int): Window size used when iterating over the mosaic.
        water_threshold (float): NDWI threshold distinguishing water pixels.

    Returns:
        None: Outputs are written to disk.
    """

    with rasterio.open(mosaic_path) as src:
        profile = src.profile.copy()
        width, height = src.width, src.height

        # Update profiles
        profile.update(count=5, dtype="float32", compress="lzw", BIGTIFF="YES")
        with rasterio.open(output_feature_path, "w", **profile) as dst_feat:
            mask_profile = profile.copy()
            mask_profile.update(count=1)
            with rasterio.open(output_mask_path, "w", **mask_profile) as dst_mask:
                if label_path and output_label_path:
                    with rasterio.open(label_path) as src_lbl:
                        lbl_profile = src_lbl.profile.copy()
                    lbl_profile.update(dtype="float32", compress="lzw", BIGTIFF="YES")
                    dst_y = rasterio.open(output_label_path, "w", **lbl_profile)
                else:
                    src_lbl = None
                    dst_y = None
                    dst_y_win = None

                for i in range(0, height, chunk_size):
                    for j in range(0, width, chunk_size):
                        win = Window(
                            j,
                            i,
                            min(chunk_size, width - j),
                            min(chunk_size, height - i),
                        )

                        feat_stack, water_mask = process_salinity_features_chunk(
                            src,
                            win,
                            mission_band_index,
                            src_lbl=src_lbl,
                            dst_y=dst_y,
                            dst_y_win=dst_y_win,
                            water_threshold=water_threshold,
                        )

                        for band_idx in range(feat_stack.shape[0]):
                            dst_feat.write(
                                feat_stack[band_idx], band_idx + 1, window=win
                            )

                        dst_mask.write(water_mask, 1, window=win)
                if label_path and output_label_path:
                    dst_y.close()
