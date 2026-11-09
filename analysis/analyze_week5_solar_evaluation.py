from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATA_PATH = Path("outputs/week4_solar/pvdaq_system10_2008_hourly.csv")
OUTPUT_DIR = Path("outputs/week5_solar")
DAY_START_HOUR = 6
DAY_END_HOUR = 18


def add_persistence_forecast(data: pd.DataFrame) -> pd.DataFrame:
    result = data.sort_values("measured_on").copy()
    result["persistence_forecast_w"] = result["pv_generation_w"].shift(1)
    result["absolute_error_w"] = (
        result["pv_generation_w"] - result["persistence_forecast_w"]
    ).abs()
    result["scheduled_daytime"] = result["measured_on"].dt.hour.between(
        DAY_START_HOUR, DAY_END_HOUR
    )
    return result.dropna(subset=["persistence_forecast_w"])


def daytime_mae(data: pd.DataFrame) -> float:
    daytime = data.loc[data["scheduled_daytime"]]
    return float(daytime["absolute_error_w"].mean())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hourly = pd.read_csv(DATA_PATH, parse_dates=["measured_on"])
    evaluated = add_persistence_forecast(hourly)

    split_index = int(len(evaluated) * 0.80)
    holdout = evaluated.iloc[split_index:]
    holdout_daytime = holdout.loc[holdout["scheduled_daytime"]]

    # Expanding-origin monthly evaluations within the development period: the
    # history available before each month is treated as training information,
    # and the final 20% holdout remains untouched by these resamples.
    monthly_rows: list[dict[str, object]] = []
    holdout_start = holdout["measured_on"].min()
    for month in range(4, 13):
        test_start = pd.Timestamp(year=2008, month=month, day=1)
        if month == 12:
            test_end = pd.Timestamp(year=2009, month=1, day=1)
        else:
            test_end = pd.Timestamp(year=2008, month=month + 1, day=1)

        if test_end > holdout_start:
            break

        train = evaluated.loc[evaluated["measured_on"] < test_start]
        test = evaluated.loc[
            (evaluated["measured_on"] >= test_start)
            & (evaluated["measured_on"] < test_end)
        ]
        test_daytime = test.loc[test["scheduled_daytime"]]
        monthly_rows.append(
            {
                "fold": len(monthly_rows) + 1,
                "training_through": str(train["measured_on"].max()),
                "test_month": test_start.strftime("%Y-%m"),
                "daytime_test_hours": int(len(test_daytime)),
                "daytime_mae_w": float(test_daytime["absolute_error_w"].mean()),
                "mean_daytime_actual_w": float(
                    test_daytime["pv_generation_w"].mean()
                ),
                "normalized_daytime_mae_percent": float(
                    100
                    * test_daytime["absolute_error_w"].mean()
                    / test_daytime["pv_generation_w"].mean()
                ),
                "all_hour_mae_w": float(test["absolute_error_w"].mean()),
            }
        )

    folds = pd.DataFrame(monthly_rows)
    fold_mae = folds["daytime_mae_w"]
    summary = {
        "procedure": "one-hour persistence forecast",
        "daytime_window": f"{DAY_START_HOUR:02d}:00-{DAY_END_HOUR:02d}:59",
        "single_80_20_holdout_start": str(holdout["measured_on"].min()),
        "single_80_20_holdout_end": str(holdout["measured_on"].max()),
        "single_80_20_daytime_hours": int(len(holdout_daytime)),
        "single_80_20_daytime_mae_w": daytime_mae(holdout),
        "single_80_20_mean_daytime_actual_w": float(
            holdout_daytime["pv_generation_w"].mean()
        ),
        "single_80_20_normalized_daytime_mae_percent": float(
            100
            * holdout_daytime["absolute_error_w"].mean()
            / holdout_daytime["pv_generation_w"].mean()
        ),
        "single_80_20_all_hour_mae_w": float(holdout["absolute_error_w"].mean()),
        "rolling_folds": int(len(folds)),
        "rolling_daytime_mae_mean_w": float(fold_mae.mean()),
        "rolling_daytime_mae_sd_w": float(fold_mae.std(ddof=1)),
        "rolling_daytime_mae_min_w": float(fold_mae.min()),
        "rolling_daytime_mae_max_w": float(fold_mae.max()),
        "rolling_daytime_mae_range_w": float(fold_mae.max() - fold_mae.min()),
        "rolling_normalized_daytime_mae_mean_percent": float(
            folds["normalized_daytime_mae_percent"].mean()
        ),
        "rolling_normalized_daytime_mae_sd_percent": float(
            folds["normalized_daytime_mae_percent"].std(ddof=1)
        ),
        "rolling_normalized_daytime_mae_min_percent": float(
            folds["normalized_daytime_mae_percent"].min()
        ),
        "rolling_normalized_daytime_mae_max_percent": float(
            folds["normalized_daytime_mae_percent"].max()
        ),
    }

    folds.to_csv(OUTPUT_DIR / "rolling_origin_results.csv", index=False)
    (OUTPUT_DIR / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(folds.to_string(index=False, float_format=lambda value: f"{value:.1f}"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
