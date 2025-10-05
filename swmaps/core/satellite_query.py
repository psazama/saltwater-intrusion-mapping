"""Helpers for querying STAC catalogs and downloading satellite imagery."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pystac
import rasterio
from pystac_client import Client
from rasterio.crs import CRS
from rasterio.session import AWSSession
from rasterio.transform import Affine
from tqdm import tqdm

from swmaps.config import data_path

from .missions import get_mission
from .raster_utils import downsample_to_landsat


def query_satellite_items(
    mission: str = "sentinel-2",
    bbox: list[float] | None = None,
    date_range: str | None = None,
    max_items: int | None = None,
    debug: bool = False,
) -> tuple[list[pystac.Item], dict[str, str]]:
    """Query the AWS Earth Search STAC API for imagery matching the criteria.

    Args:
        mission (str): Mission slug understood by :func:`get_mission`.
        bbox (list[float] | None): Bounding box ``[minx, miny, maxx, maxy]``
            in WGS84 coordinates.
        date_range (str | None): ISO-8601 date or range string.
        max_items (int | None): Maximum number of STAC items to fetch.
        debug (bool): If ``True``, print identifiers for the retrieved items.

    Returns:
        tuple[list[pystac.Item], dict[str, str]]: Matching STAC items and the
        band mapping for the mission.

    Raises:
        ValueError: When no matching items are found.
    """

    mission_specs = get_mission(mission)
    bands = mission_specs["bands"]
    collection = mission_specs["collection"]
    query_filter = mission_specs["query_filter"]

    catalog = Client.open("https://earth-search.aws.element84.com/v1")
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=date_range,
        query=query_filter,
        max_items=max_items,
    )

    items = list(search.items())
    if not items:
        raise ValueError("No items found for your search.")

    if debug:
        for item in items:
            print(f"Found item: {item.id}")

    return items, bands


def download_satellite_bands_from_item(
    item: pystac.Item,
    bands: dict[str, str],
    to_disk: bool = True,
    data_dir: str | Path | None = None,
    debug: bool = False,
    mission: str | None = None,
    downsample_to_landsat_res: bool = False,
    target_resolution: float = 30,
) -> list[tuple[int, np.ndarray, Affine, CRS]]:
    """Download selected bands for a STAC item.

    Args:
        item (pystac.Item): STAC item describing the asset locations.
        bands (dict[str, str]): Mapping of STAC asset keys to file suffixes.
        to_disk (bool): When ``True``, persist downloaded bands to ``data_dir``.
        data_dir (str | Path | None): Directory for saved rasters.
        debug (bool): If ``True``, emit debug logging for each band.
        mission (str | None): Mission slug, used to determine resampling.
        downsample_to_landsat_res (bool): If ``True`` and mission is
            Sentinel-2, resample to Landsat resolution.
        target_resolution (float): Target resolution used when resampling.

    Returns:
        list[tuple[int, np.ndarray, Affine, CRS]]: Tuples containing the
        one-based band index, data array, transform, and CRS for each band.
    """

    session = AWSSession(requester_pays=True)
    data_dir = Path(data_dir) if data_dir else data_path()
    if to_disk:
        data_dir.mkdir(parents=True, exist_ok=True)
    out_path = None

    def _fetch_one(k_n):
        """Download a single band asset and replace nodata values with ``NaN``.

        Args:
            k_n (tuple[str, str]): Pair of the STAC asset key and the desired
                filename stem for the downloaded band.

        Returns:
            tuple[str, str, np.ndarray, Affine, CRS]: Asset key, friendly band
            name, array data, affine transform, and CRS captured from the
            source dataset.
        """
        key, name = k_n
        href = item.assets[key].href
        with rasterio.Env(session):
            with rasterio.open(href) as src:
                arr = src.read(1).astype(np.float32)
                nod = src.nodata or 0
                arr[arr == nod] = np.nan
                return key, name, arr, src.transform, src.crs

    band_order = list(bands.keys())
    band_data: list[tuple[int, np.ndarray, Affine, str]] = []

    with ThreadPoolExecutor(max_workers=len(bands)) as pool:
        for key, name, data, trf, crs in pool.map(_fetch_one, bands.items()):
            if downsample_to_landsat_res and mission == "sentinel-2":
                data, trf = downsample_to_landsat(data, trf, crs, target_resolution)

            band_index = band_order.index(key) + 1
            band_data.append((band_index, data, trf, crs))

            if to_disk:
                out_path = data_dir / f"{item.id}_{name}.tif"
                profile = {
                    "driver": "GTiff",
                    "dtype": "float32",
                    "nodata": np.nan,
                    "crs": crs,
                    "transform": trf,
                    "height": data.shape[0],
                    "width": data.shape[1],
                }
                with rasterio.open(out_path, "w", **profile, BIGTIFF="YES") as dst:
                    dst.write(data, 1)

            if debug:
                nan_pct = np.isnan(data).sum() / data.size
                logging.debug(f"{item.id} – {name}: {data.shape}  NaNs={nan_pct:.2%}")

    return band_data


def _stack_bands(
    item: pystac.Item,
    bands: dict[str, str],
) -> tuple[np.ndarray, Affine, rasterio.crs.CRS]:
    """Fetch and stack all requested bands for a STAC item into a single array.

    Args:
        item (pystac.Item): STAC item holding band assets.
        bands (dict[str, str]): Mapping of band identifiers to asset keys.

    Returns:
        tuple[np.ndarray, Affine, rasterio.crs.CRS]: Stacked band array,
        affine transform, and CRS associated with the imagery.
    """
    session = AWSSession(requester_pays=True)
    arrays = []
    transforms = None
    crs = None
    for key in bands.keys():
        href = item.assets[key].href
        with rasterio.Env(session), rasterio.open(href) as src:
            arr = src.read(1).astype(np.float32)
            if transforms is None:
                transforms, crs = src.transform, src.crs
            arrays.append(arr)
    return np.stack(arrays), transforms, crs


def find_satellite_coverage(
    df: pd.DataFrame,
    missions: list[str] = ["sentinel-2", "landsat-5", "landsat-7"],
    buffer_km: float = 5,
) -> pd.DataFrame:
    """Annotate salinity observations with missions that have nearby imagery.

    Args:
        df (pandas.DataFrame): Observation table containing ``latitude``,
            ``longitude``, and ``date`` columns.
        missions (list[str]): Mission slugs to consider.
        buffer_km (float): Spatial buffer radius around each observation.

    Returns:
        pandas.DataFrame: Input frame with an added ``covered_by`` column
        listing missions that provide imagery.
    """
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        lat, lon, date = row["latitude"], row["longitude"], row["date"]
        try:
            # Buffer in degrees (approx)
            buffer_deg = buffer_km / 111.0
            bbox = [
                lon - buffer_deg,
                lat - buffer_deg,
                lon + buffer_deg,
                lat + buffer_deg,
            ]

            # ±1 day range
            dt = datetime.strptime(date, "%Y-%m-%d")
            date_range = (
                f"{(dt - timedelta(days=1)).date()}/{(dt + timedelta(days=1)).date()}"
            )

            covered_by = []
            for mission in missions:
                try:
                    mission_info = get_mission(mission)
                    start_date, end_date = mission_info["valid_date_range"]
                    if start_date and dt < datetime.strptime(start_date, "%Y-%m-%d"):
                        continue
                    if end_date and dt > datetime.strptime(end_date, "%Y-%m-%d"):
                        continue

                    items, _ = query_satellite_items(
                        mission=mission, bbox=bbox, date_range=date_range, max_items=1
                    )
                    if items:
                        covered_by.append(mission)
                except Exception:
                    pass

            results.append(covered_by)
        except Exception as e:
            print(f"Failed on row {row}: {e}")
            results.append([])

    df["covered_by"] = results
    return df


def download_matching_images(
    df: pd.DataFrame,
    missions: list[str] = ["sentinel-2", "landsat-5", "landsat-7"],
    buffer_km: float = 0.1,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Download imagery for each observation's matched missions.

    Args:
        df (pandas.DataFrame): Observation table with ``covered_by`` details.
        missions (list[str]): Mission slugs considered for download.
        buffer_km (float): Buffer radius to define the query bounding box.
        output_dir (str | Path | None): Destination directory for downloads.

    Returns:
        pandas.DataFrame: DataFrame with an additional ``downloaded_files``
        column listing downloaded multiband rasters per observation.
    """
    output_dir = Path(output_dir) if output_dir else data_path("matched_downloads")
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        lat, lon, date = row["latitude"], row["longitude"], row["date"]
        dt = datetime.strptime(date, "%Y-%m-%d")
        date_range = (
            f"{(dt - timedelta(days=1)).date()}/{(dt + timedelta(days=1)).date()}"
        )
        buffer_deg = buffer_km / 111.0
        bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]

        row_downloads = []

        for mission in row.get("covered_by", []):
            try:
                items, bands = query_satellite_items(
                    mission=mission, bbox=bbox, date_range=date_range, max_items=1
                )
                if not items:
                    continue

                item = items[0]
                date_tag = dt.strftime("%Y%m%d")
                download_dir = os.path.join(
                    output_dir, f"{mission}_{date_tag}_{lat:.4f}_{lon:.4f}"
                )
                os.makedirs(download_dir, exist_ok=True)

                band_files = []
                for band_key in bands.values():
                    path = os.path.join(download_dir, f"{item.id}_{band_key}.tif")
                    band_files.append(path)

                multi_band_path = os.path.join(download_dir, f"{item.id}_multiband.tif")

                if os.path.exists(multi_band_path):
                    print(f"[SKIP] Multiband file already exists: {multi_band_path}")
                    row_downloads.append(multi_band_path)
                    continue

                # Download each band
                download_satellite_bands_from_item(
                    item,
                    bands,
                    to_disk=True,
                    data_dir=download_dir,
                    mission=mission,
                    downsample_to_landsat_res=(mission == "sentinel-2"),
                )

                # Read all bands and stack
                band_arrays = []
                first_profile = None
                for path in band_files:
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"Expected band missing: {path}")
                    with rasterio.open(path) as src:
                        data = src.read(1)
                        band_arrays.append(data)
                        if first_profile is None:
                            first_profile = src.profile.copy()

                # Update profile for multiband
                first_profile.update(count=len(band_arrays))
                with rasterio.open(multi_band_path, "w", **first_profile) as dst:
                    for i, band_array in enumerate(band_arrays, start=1):
                        dst.write(band_array, i)

                row_downloads.append(multi_band_path)

            except Exception as e:
                print(
                    f"[ERROR] Could not download or combine {mission} for {lat},{lon} on {date}: {e}"
                )

        downloaded_paths.append(row_downloads)

    df = df.copy()
    df["downloaded_files"] = downloaded_paths
    return df
