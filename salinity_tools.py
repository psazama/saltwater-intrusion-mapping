import joblib
import numpy as np
import rasterio
import xarray as xr
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def salinity_truth():
    ds = xr.open_dataset("data/salinity_labels/WOD_CAS_T_S_2020_1.nc")
    df = ds.to_dataframe().reset_index()
    df.to_csv("codc_salinity_profiles.csv", index=False)

    df = df[df["instrument"] == "CTD"]
    df = df[df["depth"] <= 1.0]


def extract_salinity_features_from_mosaic(
    mosaic_path, mission_band_index, label_path=None
):
    """
    Loads stacked Sentinel/Landsat mosaic and computes model features for water pixels.

    Parameters:
        mosaic_path (str): Path to stacked multispectral GeoTIFF.
        band_index (dict): Mapping of band names to indices (1-based).
        label_path (str, optional): Optional path to salinity label GeoTIFF (same shape as mosaic).

    Returns:
        X (np.ndarray): Feature matrix (n_samples, n_features) for water pixels.
        y (np.ndarray or None): Target salinity values (n_samples,) if label_path is provided, else None.
        water_mask (np.ndarray): Boolean mask of selected water pixels (for reuse or mapping).
    """
    with rasterio.open(mosaic_path) as src:
        blue = src.read(mission_band_index["blue"])
        green = src.read(mission_band_index["green"])
        red = src.read(mission_band_index["red"])
        nir = src.read(mission_band_index["nir08"])
        swir1 = src.read(mission_band_index["swir16"])
        swir2 = src.read(mission_band_index["swir22"])

    # NDWI for water masking
    ndwi = (green - nir) / (green + nir + 1e-10)
    water_mask = ndwi > 0.2

    # Compute features
    ndti = (green - blue) / (green + blue + 1e-10)
    turbidity = red / (green + 1e-10)
    chlorophyll_proxy = blue / green
    salinity_proxy = swir1 + swir2
    ndvi = (nir - red) / (nir + red + 1e-10)

    # Stack features and apply mask
    stacked = np.stack(
        [ndti, turbidity, chlorophyll_proxy, salinity_proxy, ndvi], axis=-1
    )
    X = stacked[water_mask]

    y = None
    if label_path:
        with rasterio.open(label_path) as src:
            salinity_map = src.read(1)
        y = salinity_map[water_mask]
        y = y.astype(np.float32)

    return X, y, water_mask


def calc_salinity_deng(X, y, test_size=0.2, random_state=42, save_model_path=None):
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

    print(f"✅ XGBoost trained — RMSE: {rmse:.2f}, R²: {r2:.3f}")

    if save_model_path:
        joblib.dump(model, save_model_path)
        print(f"📦 Model saved to {save_model_path}")

    return model, {"rmse": rmse, "r2": r2}
