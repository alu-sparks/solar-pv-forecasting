from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("outputs/week4_solar/pvdaq_system10_2008_hourly.csv")
OUTPUT_DIR = Path("outputs/week7_solar")
K_VALUES = [5, 15, 50]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "lag_5",
    "lag_6",
    "lag_24",
    "rolling_3",
    "rolling_6",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]


def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    data = hourly.sort_values("measured_on").copy()
    target = data["pv_generation_w"]
    for lag in [1, 2, 3, 4, 5, 6, 24]:
        data[f"lag_{lag}"] = target.shift(lag)
    data["rolling_3"] = target.shift(1).rolling(3).mean()
    data["rolling_6"] = target.shift(1).rolling(6).mean()

    hour = data["measured_on"].dt.hour
    day_of_year = data["measured_on"].dt.dayofyear
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366)
    data["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366)
    data["scheduled_daytime"] = hour.between(6, 18)
    return data.dropna(subset=FEATURES).reset_index(drop=True)


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    return center, scale


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def knn_predict_many_k(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    k_values: list[int],
    chunk_size: int = 128,
) -> dict[int, np.ndarray]:
    max_k = max(k_values)
    predictions = {k: np.empty(len(test_x), dtype=float) for k in k_values}
    train_norm = np.sum(train_x * train_x, axis=1)

    for start in range(0, len(test_x), chunk_size):
        stop = min(start + chunk_size, len(test_x))
        query = test_x[start:stop]
        distances = (
            np.sum(query * query, axis=1)[:, None]
            + train_norm[None, :]
            - 2 * query @ train_x.T
        )
        distances = np.maximum(distances, 0.0)
        candidates = np.argpartition(distances, max_k - 1, axis=1)[:, :max_k]
        candidate_distances = np.take_along_axis(distances, candidates, axis=1)
        order = np.argsort(candidate_distances, axis=1)
        neighbors = np.take_along_axis(candidates, order, axis=1)
        neighbor_targets = train_y[neighbors]
        cumulative = np.cumsum(neighbor_targets, axis=1)
        for k in k_values:
            predictions[k][start:stop] = cumulative[:, k - 1] / k
    return predictions


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = add_intercept(x)
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + alpha * penalty, design.T @ y)


def ridge_predict(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return add_intercept(x) @ coefficients


def tune_ridge(
    outer_train: pd.DataFrame, test_start: pd.Timestamp
) -> float:
    validation_start = test_start - pd.Timedelta(days=30)
    inner_train = outer_train.loc[outer_train["measured_on"] < validation_start]
    inner_validation = outer_train.loc[
        (outer_train["measured_on"] >= validation_start)
        & outer_train["scheduled_daytime"]
    ]
    train_x = inner_train[FEATURES].to_numpy(float)
    train_y = inner_train["pv_generation_w"].to_numpy(float)
    validation_x = inner_validation[FEATURES].to_numpy(float)
    validation_y = inner_validation["pv_generation_w"].to_numpy(float)
    center, scale = fit_scaler(train_x)
    train_x = (train_x - center) / scale
    validation_x = (validation_x - center) / scale
    scores = {}
    for alpha in RIDGE_ALPHAS:
        coefficients = fit_ridge(train_x, train_y, alpha)
        scores[alpha] = mae(validation_y, ridge_predict(validation_x, coefficients))
    return float(min(scores, key=scores.get))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly = pd.read_csv(DATA_PATH, parse_dates=["measured_on"])
    data = build_features(hourly)
    rows: list[dict[str, float | str]] = []

    for month in range(4, 10):
        test_start = pd.Timestamp(2008, month, 1)
        test_end = test_start + pd.offsets.MonthBegin(1)
        train = data.loc[data["measured_on"] < test_start]
        test = data.loc[
            (data["measured_on"] >= test_start)
            & (data["measured_on"] < test_end)
            & data["scheduled_daytime"]
        ]

        train_x = train[FEATURES].to_numpy(float)
        train_y = train["pv_generation_w"].to_numpy(float)
        test_x = test[FEATURES].to_numpy(float)
        test_y = test["pv_generation_w"].to_numpy(float)
        center, scale = fit_scaler(train_x)
        train_x_scaled = (train_x - center) / scale
        test_x_scaled = (test_x - center) / scale

        knn_predictions = knn_predict_many_k(
            train_x_scaled, train_y, test_x_scaled, K_VALUES
        )
        predictions: dict[str, np.ndarray] = {
            f"kNN (k={k})": knn_predictions[k] for k in K_VALUES
        }

        best_alpha = tune_ridge(train, test_start)
        ridge_coefficients = fit_ridge(train_x_scaled, train_y, best_alpha)
        predictions["Ridge"] = ridge_predict(test_x_scaled, ridge_coefficients)
        predictions["Persistence"] = test["lag_1"].to_numpy(float)

        mean_actual = float(np.mean(test_y))
        for method, predicted in predictions.items():
            error = mae(test_y, predicted)
            rows.append(
                {
                    "test_month": test_start.strftime("%Y-%m"),
                    "method": method,
                    "mae_w": error,
                    "normalized_mae_percent": 100 * error / mean_actual,
                    "ridge_alpha": best_alpha if method == "Ridge" else np.nan,
                }
            )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby("method", as_index=False)
        .agg(
            mean_mae_w=("mae_w", "mean"),
            sd_mae_w=("mae_w", "std"),
            mean_normalized_mae_percent=("normalized_mae_percent", "mean"),
        )
        .sort_values("mean_mae_w")
    )
    best_knn = summary.loc[summary["method"].str.startswith("kNN")].iloc[0]
    ridge = summary.loc[summary["method"] == "Ridge"].iloc[0]
    details = {
        "features": FEATURES,
        "folds": 6,
        "best_knn": best_knn.to_dict(),
        "ridge": ridge.to_dict(),
        "best_knn_minus_ridge_mae_w": float(
            best_knn["mean_mae_w"] - ridge["mean_mae_w"]
        ),
    }

    results.to_csv(OUTPUT_DIR / "knn_monthly_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "knn_summary.csv", index=False)
    (OUTPUT_DIR / "knn_summary.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
