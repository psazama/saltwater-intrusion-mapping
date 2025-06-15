import os

import joblib
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import xgboost as xgb
from rasterio.windows import Window
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from download_tools import compute_ndwi


def salinity_truth(
    dataset_files=["data/salinity_labels/WOD/WOD_CAS_T_S_2020_1.nc"],
    output_csv="data/salinity_labels/codc_salinity_profiles.csv",
    depth=1.0,
    prof_limit=None,
):
    lat_idx = 4
    lon_idx = 5
    year_idx = 1
    month_idx = 2
    day_idx = 3

    all_dfs = []

    for dataset_file in dataset_files:
        ds = xr.open_dataset(dataset_file)

        # Use full range if no limit provided
        if prof_limit is None:
            prof_limit = ds.sizes["N_PROF"]

        prof_slice = slice(0, prof_limit)

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

        # Surface filter: only depths ≤ 1.0m
        surface_mask = dep <= depth
        surface_sal = sal.where(surface_mask)

        # Convert to DataFrame
        df = surface_sal.to_dataframe(name="salinity").reset_index()
        df = df.dropna(subset=["salinity"])
        df["latitude"] = df["N_PROF"].map(dict(zip(range(prof_limit), lats)))
        df["longitude"] = df["N_PROF"].map(dict(zip(range(prof_limit), lons)))
        df["date"] = df["N_PROF"].map(dict(zip(range(prof_limit), times)))
        df["source_file"] = os.path.basename(dataset_file)

        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    print(
        f"Extracted {len(final_df)} valid salinity measurements from {len(dataset_files)} file(s)"
    )

    # Save
    final_df.to_csv(output_csv, index=False)
    return final_df


def process_salinity_features_chunk(
    src,
    win,
    band_index,
    src_lbl=None,
    dst_y=None,
    dst_y_win=None,
    water_threshold=0.2,
    profile=None,
):
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
    mosaic_path,
    mission_band_index,
    output_feature_path,
    output_mask_path,
    label_path=None,
    output_label_path=None,
    chunk_size=512,
    water_threshold=0.2,
):
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


def train_salinity_deng(X, y, test_size=0.2, random_state=42, save_model_path=None):
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
