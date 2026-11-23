# One-Hour-Ahead Solar PV Forecasting

This repository supports the ANLY 530 KA1 Milestone 2 progress report, **Leakage-Aware One-Hour-Ahead Solar Photovoltaic Power Forecasting**.

## Research question

Under a leakage-safe nested rolling-origin design, how much do ridge regression, k-nearest-neighbors regression, and a constrained random forest improve daytime one-hour-ahead PV forecasts over persistence, and are the improvements stable across months?

## Current finding

Across six outer monthly test folds (April–September 2008), the constrained random forest produced the lowest mean daytime MAE: 84.28 W, compared with 100.67 W for kNN, 105.82 W for ridge, and 148.21 W for persistence. A moving-block bootstrap estimated the forest-minus-persistence MAE difference at -63.86 W (95% interval: -68.32 to -58.70 W). These findings are preliminary and specific to one PV system and year.

## Data

The study uses the public NREL PVDAQ System 10 record for 2008. Raw data are not committed. Retrieve the data from the [NREL PVDAQ public datasets](https://doi.org/10.25984/1846021) and place the hourly file at:

```text
data/pvdaq_system10_2008_hourly.csv
```

The expected columns are `measured_on` and `pv_generation_w`. The analysis script creates lagged, shifted rolling, and cyclical calendar predictors without using future observations.

## Repository structure

```text
analysis/
  milestone2_nested_evaluation.py
data/
  README.md
results/
  milestone2/
    figures/
    block_bootstrap.json
    data_audit.json
    model_summary.csv
    outer_fold_metrics.csv
    outer_fold_predictions.csv
    permutation_importance_by_fold.csv
    selected_hyperparameters.csv
  [earlier exploratory summaries retained for provenance]
requirements.txt
README.md
```

## Reproduce the analysis

1. Use Python 3.11 or newer.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Retrieve the data and place it in `data/` as described above.
4. Run:

   ```bash
   python analysis/milestone2_nested_evaluation.py \
     --data data/pvdaq_system10_2008_hourly.csv \
     --output results/milestone2
   ```

The script fixes random seeds, records all outer-fold metrics and selected hyperparameters, and recreates the figures used in the report. The earlier Week 4–9 files remain in the repository as exploratory provenance; the `milestone2_nested_evaluation.py` script and `results/milestone2/` directory contain the current leakage-safe research execution.

## Validation and prediction boundary

April through September are six consecutive outer test months. At each fold, all training observations occur before the test month. Hyperparameters are selected on a 30-day validation window contained entirely within the outer training history. Ridge and kNN scaling is fitted inside training pipelines. Only lagged PV, shifted rolling summaries, and calendar variables available at the forecast origin are used. Future observed weather and statistics fitted on outer-test data are prohibited.

## Scope

The data contain no personal information. Results apply only to the studied PV system and year unless externally validated. The analysis does not support causal, household-level, or environmental-justice claims.

## Author

Alamelu Chandrasekaran — individual project.
