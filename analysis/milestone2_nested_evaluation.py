from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 530
CAPACITY_W = 1120.0
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
    data["pv_generation_w"] = data["pv_generation_w"].clip(lower=0)
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


def validate_time_and_feature_lineage(
    hourly: pd.DataFrame, featured: pd.DataFrame
) -> dict[str, object]:
    """Execute timestamp, DST-treatment, and predictor-lineage leakage checks."""
    raw = hourly.sort_values("measured_on").copy()
    raw["pv_generation_w"] = raw["pv_generation_w"].clip(lower=0)
    timestamps = raw["measured_on"]
    one_hour = np.timedelta64(1, "h")
    differences = timestamps.diff().dropna()

    checks: dict[str, object] = {
        "timestamps_strictly_increasing": bool(timestamps.is_monotonic_increasing),
        "timestamps_unique": bool(not timestamps.duplicated().any()),
        "all_timestamp_steps_one_hour": bool((differences == one_hour).all()),
        "timestamp_timezone_encoded": bool(timestamps.dt.tz is not None),
        "dst_treatment": (
            "The source timestamps are timezone-naive and form a continuous fixed hourly "
            "clock. No timezone localization or daylight-saving conversion is applied; "
            "the recorded hour is preserved for reproducibility."
        ),
    }

    for label, start in {
        "spring_dst_date_regular_hourly_sequence": pd.Timestamp("2008-03-09 00:00"),
        "fall_dst_date_regular_hourly_sequence": pd.Timestamp("2008-11-02 00:00"),
    }.items():
        window = timestamps.loc[timestamps.between(start, start + pd.DateOffset(hours=6))]
        checks[label] = bool(
            len(window) == 7 and (window.diff().dropna() == one_hour).all()
        )

    indexed_target = raw.set_index("measured_on")["pv_generation_w"]
    observed = featured.set_index("measured_on")
    lineage_checks: dict[str, bool] = {}
    for lag in [1, 2, 3, 4, 5, 6, 24]:
        expected = indexed_target.shift(lag).reindex(observed.index)
        lineage_checks[f"lag_{lag}_uses_only_prior_values"] = bool(
            np.allclose(observed[f"lag_{lag}"], expected, equal_nan=False)
        )
    for window in [3, 6]:
        expected = indexed_target.shift(1).rolling(window).mean().reindex(observed.index)
        lineage_checks[f"rolling_{window}_is_shifted_before_aggregation"] = bool(
            np.allclose(observed[f"rolling_{window}"], expected, equal_nan=False)
        )
    checks["feature_lineage_checks"] = lineage_checks
    checks["all_feature_lineage_checks_passed"] = bool(all(lineage_checks.values()))

    required = [
        checks["timestamps_strictly_increasing"],
        checks["timestamps_unique"],
        checks["all_timestamp_steps_one_hour"],
        checks["spring_dst_date_regular_hourly_sequence"],
        checks["fall_dst_date_regular_hourly_sequence"],
        checks["all_feature_lineage_checks_passed"],
    ]
    checks["all_static_leakage_checks_passed"] = bool(all(required))
    if not checks["all_static_leakage_checks_passed"]:
        raise AssertionError("A timestamp or feature-lineage leakage check failed.")
    return checks


def candidate_models(seed: int) -> dict[str, list[tuple[dict, object]]]:
    ridge = [
        (
            {"alpha": alpha},
            Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))]),
        )
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    ]
    knn = []
    for k in [5, 15, 50]:
        for weights in ["uniform", "distance"]:
            knn.append(
                (
                    {"n_neighbors": k, "weights": weights},
                    Pipeline(
                        [
                            ("scale", StandardScaler()),
                            (
                                "model",
                                KNeighborsRegressor(
                                    n_neighbors=k,
                                    weights=weights,
                                    p=2,
                                    n_jobs=-1,
                                ),
                            ),
                        ]
                    ),
                )
            )
    forest_specs = [
        {"max_depth": 6, "min_samples_leaf": 10, "max_features": 0.5},
        {"max_depth": 6, "min_samples_leaf": 20, "max_features": 0.5},
        {"max_depth": 9, "min_samples_leaf": 10, "max_features": 0.5},
        {"max_depth": 9, "min_samples_leaf": 20, "max_features": 0.75},
    ]
    forest = [
        (
            spec,
            RandomForestRegressor(
                n_estimators=240,
                random_state=seed,
                n_jobs=-1,
                **spec,
            ),
        )
        for spec in forest_specs
    ]
    return {"Ridge": ridge, "kNN": knn, "Random forest": forest}


def tune_model(
    name: str,
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    seed: int,
) -> tuple[object, dict, float]:
    x_train = inner_train[FEATURES]
    y_train = inner_train["pv_generation_w"]
    x_valid = inner_valid[FEATURES]
    y_valid = inner_valid["pv_generation_w"]
    best_model = None
    best_params = None
    best_mae = np.inf
    for params, model in candidate_models(seed)[name]:
        model.fit(x_train, y_train)
        pred = np.clip(model.predict(x_valid), 0, None)
        score = mean_absolute_error(y_valid, pred)
        if score < best_mae:
            best_model = model
            best_params = params
            best_mae = float(score)
    assert best_model is not None and best_params is not None
    return best_model, best_params, best_mae


def metric_row(
    month: str,
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | str | int]:
    residual = y_pred - y_true
    over = residual > 0
    return {
        "test_month": month,
        "method": method,
        "n_test": int(len(y_true)),
        "mae_w": float(mean_absolute_error(y_true, y_pred)),
        "capacity_nmae_percent": float(
            100 * mean_absolute_error(y_true, y_pred) / CAPACITY_W
        ),
        "rmse_w": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mean_signed_error_w": float(np.mean(residual)),
        "overprediction_rate_percent": float(100 * np.mean(over)),
        "mean_overprediction_w": float(np.mean(residual[over])) if over.any() else 0.0,
    }


def block_bootstrap_difference(
    prediction_rows: pd.DataFrame,
    model: str,
    baseline: str = "Persistence",
    block_length: int = 13,
    repeats: int = 2000,
    seed: int = SEED,
) -> dict[str, float | int | str]:
    wide = prediction_rows.pivot_table(
        index=["measured_on", "actual_w"], columns="method", values="predicted_w"
    ).reset_index()
    difference = np.abs(wide["actual_w"] - wide[model]) - np.abs(
        wide["actual_w"] - wide[baseline]
    )
    values = difference.to_numpy(float)
    n = len(values)
    starts = np.arange(0, n - block_length + 1)
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats)
    blocks_needed = int(np.ceil(n / block_length))
    for i in range(repeats):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate(
            [values[s : s + block_length] for s in sampled_starts]
        )[:n]
        boot[i] = sample.mean()
    lower, upper = np.quantile(boot, [0.025, 0.975])
    return {
        "comparison": f"{model} minus {baseline}",
        "mean_mae_difference_w": float(values.mean()),
        "ci_lower_w": float(lower),
        "ci_upper_w": float(upper),
        "block_length_daytime_hours": block_length,
        "bootstrap_repeats": repeats,
    }


def make_figures(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    order = ["Persistence", "Ridge", "kNN", "Random forest"]
    for method in order:
        part = metrics.loc[metrics["method"] == method]
        ax.plot(part["test_month"], part["mae_w"], marker="o", label=method)
    ax.set(
        xlabel="Outer test month",
        ylabel="Daytime MAE (W)",
        title="Leakage-safe outer-fold error by month",
    )
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(figures / "monthly_mae.png", dpi=200)
    plt.close(fig)

    imp = (
        importance.groupby("feature", as_index=False)["importance_mae_increase_w"]
        .mean()
        .sort_values("importance_mae_increase_w", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.barh(imp["feature"], imp["importance_mae_increase_w"], color="#355C7D")
    ax.set(
        xlabel="Mean increase in test MAE after permutation (W)",
        ylabel="Feature",
        title="Random-forest outer-test permutation importance",
    )
    fig.tight_layout()
    fig.savefig(figures / "permutation_importance.png", dpi=200)
    plt.close(fig)

    july = predictions.loc[
        (predictions["test_month"] == "2008-07")
        & (predictions["measured_on"] < pd.Timestamp("2008-07-08"))
        & predictions["method"].isin(["Actual", "Persistence", "Random forest"])
    ].copy()
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    for method, style in [
        ("Actual", {"linewidth": 2.0, "color": "black"}),
        ("Persistence", {"linewidth": 1.3, "color": "#C44E52"}),
        ("Random forest", {"linewidth": 1.5, "color": "#4C72B0"}),
    ]:
        part = july.loc[july["method"] == method]
        ax.plot(part["measured_on"], part["predicted_w"], label=method, **style)
    ax.set(
        xlabel="Forecast target hour",
        ylabel="PV power (W)",
        title="Illustrative July daytime forecasts",
    )
    ax.legend(frameon=True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures / "july_forecast_example.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data, parse_dates=["measured_on"])
    expected = pd.date_range(raw["measured_on"].min(), raw["measured_on"].max(), freq="h")
    audit = {
        "raw_rows": int(len(raw)),
        "duplicate_timestamps": int(raw["measured_on"].duplicated().sum()),
        "missing_expected_hours": int(len(expected.difference(raw["measured_on"]))),
        "missing_power_values": int(raw["pv_generation_w"].isna().sum()),
        "negative_power_values_before_clip": int((raw["pv_generation_w"] < 0).sum()),
        "zero_generation_hours": int((raw["pv_generation_w"].clip(lower=0) == 0).sum()),
    }
    data = build_features(raw)
    audit["eligible_feature_rows"] = int(len(data))
    audit["rows_removed_for_lag_initialization"] = int(len(raw) - len(data))
    leakage_checks = validate_time_and_feature_lineage(raw, data)
    fold_leakage_checks: list[dict[str, object]] = []

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    parameter_rows: list[dict] = []
    importance_rows: list[dict] = []

    for fold, month in enumerate(range(4, 10), start=1):
        test_start = pd.Timestamp(2008, month, 1)
        test_end = test_start + pd.offsets.MonthBegin(1)
        outer_train = data.loc[data["measured_on"] < test_start].copy()
        outer_test = data.loc[
            (data["measured_on"] >= test_start)
            & (data["measured_on"] < test_end)
            & data["scheduled_daytime"]
        ].copy()
        validation_start = test_start - pd.DateOffset(days=30)
        inner_train = outer_train.loc[outer_train["measured_on"] < validation_start]
        inner_valid = outer_train.loc[
            (outer_train["measured_on"] >= validation_start)
            & outer_train["scheduled_daytime"]
        ]
        month_label = test_start.strftime("%Y-%m")
        fold_checks = {
            "test_month": month_label,
            "outer_train_ends_before_outer_test": bool(
                outer_train["measured_on"].max() < outer_test["measured_on"].min()
            ),
            "inner_train_ends_before_inner_validation": bool(
                inner_train["measured_on"].max() < inner_valid["measured_on"].min()
            ),
            "inner_validation_ends_before_outer_test": bool(
                inner_valid["measured_on"].max() < outer_test["measured_on"].min()
            ),
        }
        fold_checks["all_fold_checks_passed"] = bool(
            all(value for key, value in fold_checks.items() if key != "test_month")
        )
        if not fold_checks["all_fold_checks_passed"]:
            raise AssertionError(f"Chronological split check failed for {month_label}.")
        fold_leakage_checks.append(fold_checks)
        y_test = outer_test["pv_generation_w"].to_numpy(float)
        x_test = outer_test[FEATURES]

        predictions: dict[str, np.ndarray] = {
            "Persistence": outer_test["lag_1"].to_numpy(float)
        }
        fitted_models: dict[str, object] = {}

        for method in ["Ridge", "kNN", "Random forest"]:
            _, best_params, inner_mae = tune_model(
                method, inner_train, inner_valid, SEED + fold
            )
            if method == "Ridge":
                final_model = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("model", Ridge(alpha=best_params["alpha"])),
                    ]
                )
            elif method == "kNN":
                final_model = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            KNeighborsRegressor(
                                n_neighbors=best_params["n_neighbors"],
                                weights=best_params["weights"],
                                p=2,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                )
            else:
                final_model = RandomForestRegressor(
                    n_estimators=240,
                    random_state=SEED + fold,
                    n_jobs=-1,
                    **best_params,
                )
            final_model.fit(outer_train[FEATURES], outer_train["pv_generation_w"])
            fitted_models[method] = final_model
            predictions[method] = np.clip(final_model.predict(x_test), 0, None)
            parameter_rows.append(
                {
                    "test_month": month_label,
                    "method": method,
                    "inner_validation_mae_w": inner_mae,
                    "selected_parameters": json.dumps(best_params, sort_keys=True),
                }
            )

        for method, predicted in predictions.items():
            metric_rows.append(metric_row(month_label, method, y_test, predicted))
            for timestamp, actual, pred in zip(
                outer_test["measured_on"], y_test, predicted
            ):
                prediction_rows.append(
                    {
                        "test_month": month_label,
                        "measured_on": timestamp,
                        "method": method,
                        "actual_w": actual,
                        "predicted_w": float(pred),
                    }
                )
        for timestamp, actual in zip(outer_test["measured_on"], y_test):
            prediction_rows.append(
                {
                    "test_month": month_label,
                    "measured_on": timestamp,
                    "method": "Actual",
                    "actual_w": actual,
                    "predicted_w": actual,
                }
            )

        forest = fitted_models["Random forest"]
        perm = permutation_importance(
            forest,
            x_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=12,
            random_state=SEED + fold,
            n_jobs=-1,
        )
        for feature, mean, sd in zip(
            FEATURES, perm.importances_mean, perm.importances_std
        ):
            importance_rows.append(
                {
                    "test_month": month_label,
                    "feature": feature,
                    "importance_mae_increase_w": float(mean),
                    "importance_sd_w": float(sd),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df["measured_on"] = pd.to_datetime(predictions_df["measured_on"])
    parameters = pd.DataFrame(parameter_rows)
    importance = pd.DataFrame(importance_rows)
    summary = (
        metrics.groupby("method", as_index=False)
        .agg(
            mean_mae_w=("mae_w", "mean"),
            sd_mae_w=("mae_w", "std"),
            mean_capacity_nmae_percent=("capacity_nmae_percent", "mean"),
            mean_rmse_w=("rmse_w", "mean"),
            mean_signed_error_w=("mean_signed_error_w", "mean"),
            mean_overprediction_rate_percent=("overprediction_rate_percent", "mean"),
            mean_overprediction_w=("mean_overprediction_w", "mean"),
        )
        .sort_values("mean_mae_w")
    )
    baseline_mae = float(
        summary.loc[summary["method"] == "Persistence", "mean_mae_w"].iloc[0]
    )
    summary["mae_improvement_vs_persistence_percent"] = (
        100 * (baseline_mae - summary["mean_mae_w"]) / baseline_mae
    )
    bootstrap = block_bootstrap_difference(predictions_df, "Random forest")
    leakage_checks["fold_checks"] = fold_leakage_checks
    leakage_checks["all_fold_checks_passed"] = bool(
        all(row["all_fold_checks_passed"] for row in fold_leakage_checks)
    )
    leakage_checks["scaling_fitted_inside_ridge_and_knn_training_pipelines"] = True
    leakage_checks["outer_test_never_used_for_hyperparameter_selection"] = True

    metrics.to_csv(args.output / "outer_fold_metrics.csv", index=False)
    summary.to_csv(args.output / "model_summary.csv", index=False)
    parameters.to_csv(args.output / "selected_hyperparameters.csv", index=False)
    importance.to_csv(args.output / "permutation_importance_by_fold.csv", index=False)
    predictions_df.to_csv(args.output / "outer_fold_predictions.csv", index=False)
    (args.output / "data_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (args.output / "block_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    (args.output / "leakage_checks.json").write_text(
        json.dumps(leakage_checks, indent=2), encoding="utf-8"
    )
    make_figures(metrics, predictions_df, importance, args.output)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(json.dumps(bootstrap, indent=2))


if __name__ == "__main__":
    main()
