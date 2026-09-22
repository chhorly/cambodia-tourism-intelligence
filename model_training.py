"""Train and serialize the Cambodia tourism demand model.

The model learns on log1p arrivals to reduce the influence of high-volume
months. Predictions are converted back to passenger counts with expm1 by the
serving API.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Cambodia_Tourism_Enriched.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "cambodia_tourism_lgbm.pkl"
TARGET = "International_Tourist_Arrivals"
FEATURES = [
    "Lag_1",
    "Lag_3",
    "Lag_12",
    "Rolling_3_Mean",
    "Rolling_12_Mean",
    "Lag_1_to_Rolling_3",
    "Lag_1_to_Lag_12",
    "Rolling_3_to_12",
    "Average_Temperature",
    "Rainfall",
    "Cambodia_Holiday_Days",
    "COVID_Indicator",
    "Sin_Month",
    "Cos_Month",
    "Total_Origin_Holidays",
    "China_Holidays",
]


def prepare_training_data(data_path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build the same feature contract used by the prediction API."""
    df = pd.read_csv(data_path, parse_dates=["Date"]).sort_values("Date").copy()

    df["Lag_1_to_Rolling_3"] = df["Lag_1"] / (df["Rolling_3_Mean"] + 1e-5)
    df["Lag_1_to_Lag_12"] = df["Lag_1"] / (df["Lag_12"] + 1e-5)
    df["Rolling_3_to_12"] = df["Rolling_3_Mean"] / (df["Rolling_12_Mean"] + 1e-5)
    df["Sin_Month"] = np.sin(2 * np.pi * df["Month_Number"] / 12)
    df["Cos_Month"] = np.cos(2 * np.pi * df["Month_Number"] / 12)

    valid = df.dropna(subset=FEATURES + [TARGET])
    years_from_anchor = 2025 - valid["Date"].dt.year
    sample_weights = np.exp(-0.15 * np.maximum(0, years_from_anchor))
    sample_weights = np.where(
        valid["Date"].dt.year.isin([2020, 2021]),
        sample_weights * 0.4,
        sample_weights,
    )
    return valid[FEATURES], valid[TARGET], pd.Series(sample_weights, index=valid.index)


def build_model() -> LGBMRegressor:
    """Create the base estimator used by the time-series grid search."""
    return LGBMRegressor(
        objective="regression_l1",
        random_state=42,
        verbose=-1,
    )


def evaluate_holdout(
    features: pd.DataFrame,
    target: pd.Series,
    sample_weights: pd.Series,
    holdout_months: int = 12,
) -> dict[str, float | int]:
    """Evaluate the final chronological months without look-ahead leakage."""
    if len(features) <= holdout_months:
        raise ValueError("Not enough records for the requested holdout horizon.")

    split_at = len(features) - holdout_months
    train_features = features.iloc[:split_at]
    train_target = target.iloc[:split_at]
    test_features = features.iloc[split_at:]
    test_target = target.iloc[split_at:]

    grid = GridSearchCV(
        estimator=build_model(),
        param_grid={
            "n_estimators": [60, 100, 140],
            "learning_rate": [0.03, 0.06, 0.1],
            "num_leaves": [8, 15, 25],
            "max_depth": [3, 5, -1],
            "min_child_samples": [4, 8, 12],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    grid.fit(
        train_features,
        np.log1p(train_target),
        sample_weight=sample_weights.iloc[:split_at],
    )
    predictions = np.expm1(grid.best_estimator_.predict(test_features))
    actuals = test_target
    return {
        "holdout_months": holdout_months,
        "training_observations": split_at,
        "testing_observations": holdout_months,
        "total_observations": len(features),
        "total_features": len(features.columns),
        "holdout_mae": float(mean_absolute_error(actuals, predictions)),
        "holdout_rmse": float(mean_squared_error(actuals, predictions) ** 0.5),
        "holdout_mape": float((np.abs(actuals - predictions) / actuals).mean()),
        "holdout_r2": float(r2_score(actuals, predictions)),
        "best_params": grid.best_params_,
    }


def train_and_save(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> Path:
    """Train on log1p arrivals and save a versioned serving artifact."""
    features, target, sample_weights = prepare_training_data(data_path)
    metrics = evaluate_holdout(features, target, sample_weights)
    model = LGBMRegressor(
        objective="regression_l1",
        random_state=42,
        verbose=-1,
        **metrics["best_params"],
    )
    model.fit(features, np.log1p(target), sample_weight=sample_weights)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "target_transform": "log1p",
            "inverse_transform": "expm1",
            "sample_weighting": "recency_with_covid_downweighting",
            "metrics": metrics,
        },
        model_path,
    )
    return model_path


if __name__ == "__main__":
    output = train_and_save()
    print(f"Saved log-transformed model to {output}")
