from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from swmaps.config import data_path


def train_salinity_deng(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    save_model_path: str | Path | None = None,
) -> tuple[xgb.XGBRegressor, dict[str, float]]:
    """Train an XGBoost regressor on salinity feature data.

    Based on Deng et al. (2024), "Monitoring Salinity in Inner Mongolian Lakes
    Based on Sentinel-2 Images and Machine Learning".

    Args:
        X (np.ndarray): Feature matrix shaped ``(n_samples, n_features)``.
        y (np.ndarray): Target salinity values.
        test_size (float): Fraction of data reserved for testing.
        random_state (int): Random seed for splitting and model initialisation.
        save_model_path (str | Path | None): Optional path where the trained
            model is saved in ``.joblib`` format.

    Returns:
        tuple[xgb.XGBRegressor, dict[str, float]]: Trained model and a metric
        dictionary containing RMSE and R² scores.
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
