"""Logic for downloading and preparing salinity ground truth datasets"""

import logging
import os
import shutil
from pathlib import Path
from typing import Sequence
from urllib.request import urlopen

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from swmaps.config import data_path


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
    if output_csv.exists() and output_csv.stat().st_size > 0:
        logging.info(
            "Salinity truth CSV already exists at %s; skipping rebuild", output_csv
        )
        return
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
