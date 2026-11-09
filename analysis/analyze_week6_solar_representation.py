from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("outputs/week4_solar/pvdaq_system10_2008_hourly.csv")
OUTPUT_DIR = Path("outputs/week6_solar")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

REDUCED_FEATURES = [
    "lag_1",
    "lag_24",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

FULL_FEATURES = [
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

    # Both rolling summaries use only observations available before the target hour.
    data["rolling_3"] = target.shift(1).rolling(3).mean()
    data["rolling_6"] = target.shift(1).rolling(6).mean()

    hour = data["measured_on"].dt.hour
    day_of_year = data["measured_on"].dt.dayofyear
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366)
    data["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366)
    data["scheduled_daytime"] = hour.between(6, 18)

    return data.dropna(subset=FULL_FEATURES).reset_index(drop=True)


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    return mean, scale


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = add_intercept(x)
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + alpha * penalty, design.T @ y)


def predict_ridge(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return add_intercept(x) @ coefficients


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def tune_and_evaluate(
    data: pd.DataFrame,
    features: list[str],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> dict[str, float]:
    outer_train = data.loc[data["measured_on"] < test_start].copy()
    outer_test = data.loc[
        (data["measured_on"] >= test_start)
        & (data["measured_on"] < test_end)
        & data["scheduled_daytime"]
    ].copy()

    inner_cutoff = test_start - pd.Timedelta(days=30)
    inner_train = outer_train.loc[outer_train["measured_on"] < inner_cutoff]
    inner_validation = outer_train.loc[
        (outer_train["measured_on"] >= inner_cutoff)
        & outer_train["scheduled_daytime"]
    ]

    inner_x = inner_train[features].to_numpy(dtype=float)
    inner_y = inner_train["pv_generation_w"].to_numpy(dtype=float)
    validation_x = inner_validation[features].to_numpy(dtype=float)
    validation_y = inner_validation["pv_generation_w"].to_numpy(dtype=float)

    inner_mean, inner_scale = standardize_fit(inner_x)
    inner_x_scaled = (inner_x - inner_mean) / inner_scale
    validation_x_scaled = (validation_x - inner_mean) / inner_scale

    validation_scores: dict[float, float] = {}
    for alpha in ALPHAS:
        coefficients = fit_ridge(inner_x_scaled, inner_y, alpha)
        validation_scores[alpha] = mae(
            validation_y, predict_ridge(validation_x_scaled, coefficients)
        )

    best_alpha = min(validation_scores, key=validation_scores.get)

    train_x = outer_train[features].to_numpy(dtype=float)
    train_y = outer_train["pv_generation_w"].to_numpy(dtype=float)
    test_x = outer_test[features].to_numpy(dtype=float)
    test_y = outer_test["pv_generation_w"].to_numpy(dtype=float)

    train_mean, train_scale = standardize_fit(train_x)
    train_x_scaled = (train_x - train_mean) / train_scale
    test_x_scaled = (test_x - train_mean) / train_scale
    coefficients = fit_ridge(train_x_scaled, train_y, best_alpha)
    predictions = predict_ridge(test_x_scaled, coefficients)

    return {
        "alpha": float(best_alpha),
        "mae_w": mae(test_y, predictions),
        "normalized_mae_percent": float(
            100 * mae(test_y, predictions) / np.mean(test_y)
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly = pd.read_csv(DATA_PATH, parse_dates=["measured_on"])
    data = build_features(hourly)

    rows: list[dict[str, object]] = []
    for month in range(4, 10):
        test_start = pd.Timestamp(year=2008, month=month, day=1)
        test_end = pd.Timestamp(year=2008, month=month + 1, day=1)
        reduced = tune_and_evaluate(data, REDUCED_FEATURES, test_start, test_end)
        full = tune_and_evaluate(data, FULL_FEATURES, test_start, test_end)
        rows.append(
            {
                "test_month": test_start.strftime("%Y-%m"),
                "reduced_alpha": reduced["alpha"],
                "reduced_mae_w": reduced["mae_w"],
                "reduced_nmae_percent": reduced["normalized_mae_percent"],
                "full_alpha": full["alpha"],
                "full_mae_w": full["mae_w"],
                "full_nmae_percent": full["normalized_mae_percent"],
                "full_minus_reduced_mae_w": full["mae_w"]
                - reduced["mae_w"],
            }
        )

    results = pd.DataFrame(rows)
    summary = {
        "reduced_features": REDUCED_FEATURES,
        "full_features": FULL_FEATURES,
        "folds": int(len(results)),
        "reduced_mae_mean_w": float(results["reduced_mae_w"].mean()),
        "reduced_mae_sd_w": float(results["reduced_mae_w"].std(ddof=1)),
        "full_mae_mean_w": float(results["full_mae_w"].mean()),
        "full_mae_sd_w": float(results["full_mae_w"].std(ddof=1)),
        "mean_full_minus_reduced_mae_w": float(
            results["full_minus_reduced_mae_w"].mean()
        ),
        "folds_where_full_is_better": int(
            (results["full_minus_reduced_mae_w"] < 0).sum()
        ),
        "reduced_nmae_mean_percent": float(
            results["reduced_nmae_percent"].mean()
        ),
        "full_nmae_mean_percent": float(results["full_nmae_percent"].mean()),
    }

    results.to_csv(OUTPUT_DIR / "representation_results.csv", index=False)
    (OUTPUT_DIR / "representation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(results.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
