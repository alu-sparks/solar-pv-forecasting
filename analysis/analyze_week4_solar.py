from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SYSTEM_ID = 10
YEAR = 2008
BUCKET_URL = "https://oedi-data-lake.s3.amazonaws.com"
PREFIX = f"pvdaq/csv/pvdata/system_id={SYSTEM_ID}/year={YEAR}/"
OUTPUT_DIR = Path("outputs/week4_solar")


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ANLY-530-course-analysis/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def list_csv_keys() -> list[str]:
    query = urllib.parse.urlencode(
        {"list-type": "2", "prefix": PREFIX, "max-keys": "1000"}
    )
    root = ET.fromstring(fetch_bytes(f"{BUCKET_URL}/?{query}"))
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = [
        item.findtext("s3:Key", namespaces=namespace)
        for item in root.findall("s3:Contents", namespace)
    ]
    return sorted(key for key in keys if key and key.endswith(".csv"))


def read_remote_csv(key: str) -> pd.DataFrame:
    encoded_key = urllib.parse.quote(key, safe="/=")
    return pd.read_csv(io.BytesIO(fetch_bytes(f"{BUCKET_URL}/{encoded_key}")))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    keys = list_csv_keys()
    if not keys:
        raise RuntimeError("No PVDAQ CSV files found for the selected system and year.")

    with ThreadPoolExecutor(max_workers=12) as pool:
        daily_frames = list(pool.map(read_remote_csv, keys))

    raw = pd.concat(daily_frames, ignore_index=True)
    raw["measured_on"] = pd.to_datetime(raw["measured_on"], errors="coerce")
    raw["ac_power_w"] = pd.to_numeric(raw["ac_power__423"], errors="coerce")
    raw.loc[raw["ac_power_w"] <= -99990, "ac_power_w"] = np.nan
    raw = raw.dropna(subset=["measured_on", "ac_power_w"]).sort_values("measured_on")

    # Negative nighttime readings represent system consumption, not PV generation.
    raw["pv_generation_w"] = raw["ac_power_w"].clip(lower=0)
    hourly = (
        raw.set_index("measured_on")["pv_generation_w"]
        .resample("1h")
        .mean()
        .dropna()
        .rename("pv_generation_w")
        .to_frame()
        .reset_index()
    )

    split_index = int(len(hourly) * 0.80)
    train = hourly.iloc[:split_index]
    test = hourly.iloc[split_index:]
    baseline_w = float(train["pv_generation_w"].mean())
    baseline_mae_w = float((test["pv_generation_w"] - baseline_w).abs().mean())

    target = hourly["pv_generation_w"]
    summary = {
        "system_id": SYSTEM_ID,
        "year": YEAR,
        "daily_files": len(keys),
        "raw_valid_15_minute_observations": int(len(raw)),
        "hourly_observations": int(len(hourly)),
        "missing_expected_hours": int(8784 - len(hourly)),
        "zero_generation_hours": int((target == 0).sum()),
        "zero_generation_percent": float((target == 0).mean() * 100),
        "minimum_w": float(target.min()),
        "median_w": float(target.median()),
        "mean_w": float(target.mean()),
        "q75_w": float(target.quantile(0.75)),
        "maximum_w": float(target.max()),
        "training_mean_baseline_w": baseline_w,
        "test_mae_of_training_mean_baseline_w": baseline_mae_w,
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "first_timestamp": str(hourly["measured_on"].min()),
        "last_timestamp": str(hourly["measured_on"].max()),
    }

    hourly.to_csv(OUTPUT_DIR / "pvdaq_system10_2008_hourly.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.hist(target, bins=35, color="#176B63", edgecolor="white", linewidth=0.5)
    ax.axvline(
        target.mean(),
        color="#D79500",
        linewidth=2,
        linestyle="--",
        label=f"Mean = {target.mean():.0f} W",
    )
    ax.set_title("Distribution of Hourly Solar PV Generation")
    ax.set_xlabel("Average AC generation during the hour (W)")
    ax.set_ylabel("Number of hourly observations")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "target_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
