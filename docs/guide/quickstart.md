# Getting Started

## Input data

rucola expects two DataFrames (or CSV files / DuckDB tables):

| Table | Required columns |
|---|---|
| `stations` | `station_id`, `latitude`, `longitude` |
| `values` | `station_id`, `date`, `value`, `parameter` |

Values must be pre-aggregated to **annual resolution** and pre-filtered to a **single parameter** before passing to `Rucola`.

## Loading data

### From Polars DataFrames

```python
import polars as pl
import rucola

values = pl.read_parquet("values.parquet").filter(
    pl.col("parameter") == "precipitation_height"
)
stations = pl.read_parquet("stations.parquet")

r = rucola.Rucola.from_polars(values, stations)
```

### From CSV files

```python
r = rucola.Rucola.from_csv("values.csv", "stations.csv")
```

### From a DuckDB file

Requires `pip install rucola[duckdb]`.

```python
r = rucola.Rucola.from_duckdb("climate.duckdb")
```

## Running the procedure

```python
from rucola import RunConfig

# Precipitation — multiplicative ratio correction
detection = r.run(RunConfig(mode="ratio"))

# Temperature — additive difference correction
detection = r.run(RunConfig(mode="difference"))
```

`run()` returns a `DetectionResult` containing raw breakpoint data for every station across all six steps.

## Applying corrections

```python
result = detection.normalize()

# Summary table: one row per station
print(result.summary)

# All applied corrections
print(result.corrections)

# Corrected annual series for one station
sid = next(iter(result.station_results))
print(result.station_results[sid].annual_corrected)
```

## Customising tests

By default only the SNHT is used. Pass multiple tests for consensus detection:

```python
detection = r.run(
    RunConfig(
        tests=[
            rucola.SNHTTest(),
            rucola.BuishandTest(),
            rucola.PettittTest(),
        ],
        mode="ratio",
    )
)
```

## Normalisation options

```python
from rucola import NormalizationConfig

result = detection.normalize(
    NormalizationConfig(
        consensus="majority",       # require >50 % of tests to agree
        tiebreak="strongest_signal",
        break_window_years=3,
        min_correction_magnitude=0.02,
    )
)
```

## Break predicates

Predicates let you filter which detected breaks are actually applied during normalization. Combine them with `&` (AND), `|` (OR), and `~` (NOT):

```python
from rucola import (
    NormalizationConfig,
    YearBetween, StationIn, StepIn,
    MagnitudeAbove, SignalAbove,
    NSignificantAbove, NeighborCountAbove,
)

# Only apply corrections within a historical window
result = detection.normalize(
    NormalizationConfig(predicate=YearBetween(min=1960, max=2010))
)

# Require strong statistical evidence and a minimum correction size
result = detection.normalize(
    NormalizationConfig(
        predicate=SignalAbove(1.5) & MagnitudeAbove(threshold=0.05)
    )
)

# At least 3 tests agree and the reference pool had enough stations
result = detection.normalize(
    NormalizationConfig(
        predicate=NSignificantAbove(3) & NeighborCountAbove(4)
    )
)

# Restrict to a known-problematic subset of stations
result = detection.normalize(
    NormalizationConfig(predicate=StationIn({"S1", "S3"}))
)
```

See the [normalization guide](normalization.md) for a full explanation of all predicates and how they interact with consensus, tiebreak, and overrides.

## Saving and loading results

```python
# Save
detection.to_json("detection.json")
result.to_json("result.json")

# Load
detection = rucola.DetectionResult.from_json("detection.json")
result = rucola.HomogenizationResult.from_json("result.json")
```

## Markdown report

```python
print(detection.to_markdown())
```
