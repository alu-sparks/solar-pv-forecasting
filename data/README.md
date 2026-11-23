# Data retrieval

Raw PVDAQ observations are not included in this repository.

Download the public NREL PVDAQ System 10 record for 2008 from https://doi.org/10.25984/1846021 and prepare an hourly CSV named:

```text
pvdaq_system10_2008_hourly.csv
```

The file must contain:

- `measured_on`: hourly timestamp
- `pv_generation_w`: hourly mean AC photovoltaic power in watts

The analysis script validates chronological completeness and creates all predictors from past observations only.
