import os
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import xgboost as xgb
from rasterio.windows import Window
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from swmaps.config import data_path
from swmaps.core.download_tools import compute_ndwi


def _safe_normalized_difference(
    numerator: np.ndarray, denominator: np.ndarray
) -> np.ndarray:
    """Compute a normalized difference index with zero-division protection."""

    return np.divide(
        numerator - denominator,
        numerator + denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=(numerator + denominator) != 0,
    ).astype(np.float32)


def _default_wod_example() -> Path:
    return data_path("salinity_labels", "WOD", "WOD_CAS_T_S_2020_1.nc")


def estimate_salinity_level(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    swir2: np.ndarray,
    *,
    reflectance_scale: float | None = 10000.0,
    water_threshold: float = 0.2,
    salinity_proxy_scale: float = 1.2,
    salinity_proxy_threshold: float = 0.35,
    chlorophyll_reference: float = 2.0,
) -> dict[str, np.ndarray]:
    """Estimate water salinity categories from multispectral bands.

    The heuristic combines the remote sensing proxies commonly cited for
    differentiating freshwater from saline or hypersaline water bodies:

    - Water detection from NDWI/MNDWI (green vs. NIR/SWIR)
    - Turbidity and Normalised Difference Turbidity Index (red vs. green/blue)
    - Chlorophyll proxies (green vs. blue)
    - Salinity proxy index from short-wave infrared reflectance (SWIR1/2)
    - Vegetation stress indicator via NDVI around the water pixel

    Inputs are expected to be surface reflectance bands from Sentinel-2, Landsat,
    or similar sensors. When the data are scaled (e.g., Sentinel-2 L2A stored as
    integers 0–10,000), ``reflectance_scale`` rescales the input into the
    0–1 range before the indices are computed.

    Parameters
    ----------
    blue, green, red, nir, swir1, swir2:
        Arrays representing the corresponding spectral bands. All arrays must
        share the same shape.
    reflectance_scale:
        If provided, each band is divided by this value to convert to
        reflectance. Set to ``None`` to skip rescaling.
    water_threshold:
        Threshold applied to NDWI/MNDWI to declare a pixel water-covered.
    salinity_proxy_scale:
        Normalising constant for the SWIR salinity proxy ``swir1 + swir2``.
    salinity_proxy_threshold:
        Pixels with a normalised salinity proxy above this value are also
        considered water (useful for bright saline pans with low NDWI).
    chlorophyll_reference:
        Reference ratio for the chlorophyll proxy. Values above this reference
        are treated as healthy (low salinity), whereas lower values indicate a
        potential salinity signal.

    Returns
    -------
    dict
        ``{"score", "class_map", "water_mask", "indices"}`` where

        - ``score`` is a float32 array (0–1) salinity intensity estimate with
          NaNs where water is not detected.
        - ``class_map`` is a string array with labels ``{"land", "fresh",
          "brackish", "saline"}``.
        - ``water_mask`` is a boolean array marking detected water pixels.
        - ``indices`` is a dictionary of the intermediate proxies used in the
          computation for transparency/debugging.
    """

    bands = [
        np.asarray(arr, dtype=np.float32, order="C") for arr in (blue, green, red, nir, swir1, swir2)
    ]

    if reflectance_scale:
        bands = [band / reflectance_scale for band in bands]

    blue_r, green_r, red_r, nir_r, swir1_r, swir2_r = bands

    ndwi = _safe_normalized_difference(green_r, nir_r)
    mndwi = _safe_normalized_difference(green_r, swir1_r)
    ndvi = _safe_normalized_difference(nir_r, red_r)
    ndti = _safe_normalized_difference(green_r, blue_r)

    turbidity_ratio = np.divide(
        red_r,
        green_r,
        out=np.zeros_like(red_r, dtype=np.float32),
        where=green_r != 0,
    ).astype(np.float32)
    chlorophyll_ratio = np.divide(
        green_r,
        blue_r,
        out=np.zeros_like(green_r, dtype=np.float32),
        where=blue_r != 0,
    ).astype(np.float32)

    salinity_proxy = np.clip(swir1_r + swir2_r, a_min=0.0, a_max=None).astype(np.float32)
    salinity_proxy_norm = np.clip(
        np.divide(
            salinity_proxy,
            salinity_proxy_scale,
            out=np.zeros_like(salinity_proxy, dtype=np.float32),
            where=salinity_proxy_scale != 0,
        ),
        0.0,
        1.0,
    )

    chlorophyll_norm = np.clip(
        np.divide(
            chlorophyll_ratio,
            chlorophyll_reference,
            out=np.zeros_like(chlorophyll_ratio, dtype=np.float32),
            where=chlorophyll_reference != 0,
        ),
        0.0,
        1.0,
    )
    turbidity_norm = np.clip(turbidity_ratio / 2.0, 0.0, 1.0)

    ndwi_scaled = np.clip((ndwi + 1.0) / 2.0, 0.0, 1.0)
    mndwi_scaled = np.clip((mndwi + 1.0) / 2.0, 0.0, 1.0)

    # Base water detection
    water_mask = (ndwi > water_threshold) | (mndwi > water_threshold)

    # Define weak water evidence
    weak_water = (ndwi > 0.0) | (mndwi > 0.0)

    # Allow salinity proxy to expand mask only where weak water evidence exists
    salinity_override = (salinity_proxy_norm > salinity_proxy_threshold) & weak_water
    water_mask = water_mask | salinity_override

    dryness_component = 1.0 - ndwi_scaled
    saline_surface_component = 1.0 - mndwi_scaled
    chlorophyll_deficit = 1.0 - chlorophyll_norm

    score = (
        0.2 * dryness_component
        + 0.15 * saline_surface_component
        + 0.45 * salinity_proxy_norm
        + 0.1 * turbidity_norm
        + 0.1 * chlorophyll_deficit
    )
    score = np.clip(score, 0.0, 1.0).astype(np.float32)

    score = np.where(water_mask, score, np.nan)
    score = score.astype(np.float32, copy=False)

    class_map = np.full(score.shape, "land", dtype="<U8")
    score_filled = np.nan_to_num(score, nan=0.0)
    class_map = np.where(
        water_mask,
        np.select(
            [score_filled < 0.35, score_filled < 0.6],
            ["fresh", "brackish"],
            default="saline",
        ),
        "land",
    )

    indices = {
        "ndwi": ndwi,
        "mndwi": mndwi,
        "ndvi": ndvi,
        "ndti": ndti,
        "turbidity_ratio": turbidity_ratio,
        "chlorophyll_ratio": chlorophyll_ratio,
        "salinity_proxy": salinity_proxy,
        "salinity_proxy_norm": salinity_proxy_norm,
    }

    return {
        "score": score,
        "class_map": class_map,
        "water_mask": water_mask,
        "indices": indices,
    }


def build_salinity_truth(
    dataset_files: Sequence[str | Path] | None = None,
    output_csv: str | Path | None = None,
    depth: float = 1.0,
    prof_limit: int | None = None,
) -> None:
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

    for dataset_file in tqdm(dataset_files[::3]):
        print(".")
        ds = xr.open_dataset(dataset_file)

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
    """Read bands and compute feature stack, water mask, and optionally write labels for a given window."""
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
    """
    Windowed salinity feature extraction and disk-based writing.

    Args:
        mosaic_path (str): Path to input multispectral TIFF.
        mission_band_index (dict): Band index mapping.
        output_feature_path (str): Path to write X features as a 5-band float32 TIFF.
        output_mask_path (str): Path to write water mask as 1-band float32 TIFF.
        label_path (str, optional): Input salinity label map (same shape as mosaic).
        output_label_path (str, optional): Output label raster path (same shape as mask).
        chunk_size (int): Size of processing window in pixels.
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


def train_salinity_deng(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    save_model_path: str | Path | None = None,
) -> tuple[xgb.XGBRegressor, dict[str, float]]:
    """
    Train an XGBoost regressor on salinity feature data.:
    Deng et al., 2024. Monitoring Salinity in Inner Mongolian Lakes Based on Sentinel‑2 Images and Machine Learning.
    https://doi.org/10.3390/rs16203881

    Parameters:
        X (np.ndarray): Feature matrix (n_samples, n_features)
        y (np.ndarray): Target salinity values (n_samples,)
        test_size (float): Fraction of data for testing
        random_state (int): Seed for reproducibility
        save_model_path (str or None): Optional path to save trained model (.joblib)

    Returns:
        model: Trained XGBoost model
        metrics (dict): RMSE and R² scores
    """
    if save_model_path is not None:
        save_model_path = Path(save_model_path)
    else:
        save_model_path = data_path("models", "xgb_deng.joblib")
        save_model_path.parent.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"XGBoost trained — RMSE: {rmse:.2f}, R²: {r2:.3f}")

    if save_model_path:
        joblib.dump(model, save_model_path)
        print(f"Model saved to {save_model_path}")

    return model, {"rmse": rmse, "r2": r2}
