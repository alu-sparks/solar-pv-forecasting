# One Hour Ahead Solar PV Forecasting

This repository supports the ANLY 530 KA1 Milestone 1 research protocol, **Leakage Aware One Hour Ahead Solar Photovoltaic Power Forecasting**.

## Research question

Under a leakage-safe nested rolling-origin design, how much do ridge regression, k-nearest-neighbors regression, and a constrained random forest improve daytime one-hour-ahead PV power forecasts over persistence, and are the improvements stable across months?

## Data

The study uses the public NREL PVDAQ System 10 record for 2008. The raw data are not committed to this repository. Retrieve the data through the [NREL PVDAQ public datasets](https://doi.org/10.25984/1846021), then place the hourly file at:

```text
data/pvdaq_system10_2008_hourly.csv
```

The exploratory scripts expect a chronologically sorted hourly dataset and create their own leakage-safe lagged and rolling predictors.

## Repository structure

```text
analysis/        Exploratory and model-comparison scripts from Weeks 4–9
results/         Compact CSV and JSON summaries used to establish feasibility
requirements.txt
README.md
```

## Reproducibility

1. Create a Python 3.11 or newer environment.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Retrieve the PVDAQ data and place it in `data/` as described above.
4. Run the analysis scripts in numerical order. Each script documents its inputs and output folder.

The confirmatory study proposed in the protocol will use nested rolling-origin validation. The included Week 4–9 scripts are exploratory evidence of feasibility and should not be interpreted as an untouched final test.

## Prediction boundary

Only information available at the forecast origin is permitted. Lagged PV, shifted rolling summaries, and calendar variables are allowed. Future observed weather and any statistic fitted using outer-test observations are prohibited.

## Ethics and scope

The public engineering dataset contains no personal information. Results apply only to the studied PV system and year unless externally validated. The analysis does not support causal, household-level, or environmental-justice claims.

## Author

Alamelu Chandrasekaran — individual project.
