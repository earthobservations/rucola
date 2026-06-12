<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/8/8d/Rukola.JPG" alt="rucola" width="400"></p>

# rucola

Climate station data homogenization implementing the six-step procedure from [González-Rouco et al. (2001)](https://journals.ametsoc.org/view/journals/clim/14/5/1520-0442_2001_014_0964_qcahop_2.0.co_2.xml), with six pluggable breakpoint tests.

[![CI status](https://github.com/earthobservations/rucola/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/earthobservations/rucola/actions/workflows/tests.yml)
[![Docs status](https://github.com/earthobservations/rucola/actions/workflows/docs.yml/badge.svg?branch=main)](https://earthobservations.github.io/rucola/)
[![Code coverage](https://codecov.io/gh/earthobservations/rucola/branch/main/graph/badge.svg)](https://codecov.io/gh/earthobservations/rucola)
[![PyPI version](https://img.shields.io/pypi/v/rucola.svg)](https://pypi.org/project/rucola/)
[![PyPI status](https://img.shields.io/pypi/status/rucola.svg)](https://pypi.org/project/rucola/)
[![PyPI downloads/month](https://static.pepy.tech/personalized-badge/rucola?period=month&units=international_system&left_color=grey&right_color=blue&left_text=PyPI%20downloads/month)](https://pepy.tech/project/rucola)
[![Python versions](https://img.shields.io/pypi/pyversions/rucola.svg)](https://pypi.org/project/rucola/)
[![License](https://img.shields.io/github/license/earthobservations/rucola)](https://github.com/earthobservations/rucola/blob/main/LICENSE.md)

**[Documentation](https://earthobservations.github.io/rucola/)**

> **Beta:** rucola is under active development. The API may change between minor versions before a stable 1.0 release.

## Overview

Long climate records from ground stations are frequently affected by non-climatic discontinuities — station relocations, instrument replacements, changes in observation practice. rucola detects and corrects these breakpoints using an iterative reference-station approach:

1. Build a normalized Q-series for each candidate station relative to its neighbors
2. Apply one or more statistical breakpoint tests to the Q-series
3. Correct detected breaks and refine the reference pool across six steps

Six tests are available: **SNHT** (Alexandersson 1986), **Buishand range** (Buishand 1982), **Pettitt** (Pettitt 1979), **Worsley likelihood ratio** (Worsley 1979), **Easterling–Peterson two-phase regression** (Easterling & Peterson 1995), and **STARS** sequential regime-shift test (Rodionov 2004). Tests can be run individually or in consensus combinations.

Both **ratio** (multiplicative, for precipitation) and **difference** (additive, for temperature) correction modes are supported.

## Installation

```bash
pip install rucola           # core (Polars + CSV)
pip install rucola[duckdb]   # with DuckDB support
```

## Quick start

```python
import rucola

r = rucola.Rucola.from_csv("values.csv", "stations.csv")

# Run the six-step procedure
detection = r.run(rucola.RunConfig(mode="ratio"))

# Apply corrections
result = detection.normalize()

print(result.summary)
print(result.corrections)
```

### Input format

| Table | Required columns |
|---|---|
| `stations` | `station_id`, `latitude`, `longitude` |
| `values` | `station_id`, `date`, `value`, `parameter` |

Values must be at **annual resolution** and pre-filtered to a **single parameter**.

### Loaders

```python
# from Polars DataFrames
rucola.Rucola.from_polars(values_df, stations_df)

# from CSV files
rucola.Rucola.from_csv("values.csv", "stations.csv")

# from a DuckDB file  (requires: pip install rucola[duckdb])
rucola.Rucola.from_duckdb("climate.duckdb")
```

### Multiple tests with consensus detection

```python
detection = r.run(
    rucola.RunConfig(
        tests=[
            rucola.SNHTTest(),
            rucola.BuishandTest(),
            rucola.PettittTest(),
            rucola.StarsTest(l=10),   # sequential regime-shift test
        ],
        mode="ratio",
    )
)
```

### Normalization options

```python
from rucola import NormalizationConfig

result = detection.normalize(
    NormalizationConfig(
        consensus="majority",          # require >50 % of tests to agree
        tiebreak="strongest_signal",
        break_window_years=3,
        min_correction_magnitude=0.02,
        min_relative_signal=1.2,       # require signal 1.2× the critical value
        min_years_from_end=5,          # reject edge-effect artefacts (Hawkins 1977)
    )
)
```

### Break predicates

Use composable predicates to filter which detected breaks are applied. Combine them with `&`, `|`, and `~`:

```python
from rucola import (
    NormalizationConfig,
    YearBetween, StationIn, StepIn,
    MagnitudeAbove, SignalAbove,
    NSignificantAbove, NeighborCountAbove,
)

# Trusted year window + minimum correction size
result = detection.normalize(
    NormalizationConfig(
        predicate=YearBetween(min=1960, max=2010) & MagnitudeAbove(threshold=0.05)
    )
)

# Only correct a specific set of stations
result = detection.normalize(
    NormalizationConfig(predicate=StationIn({"S1", "S3", "S7"}))
)

# Require strong evidence: signal 1.5× critical value, at least 3 tests agree,
# detected from a solid reference pool, skip early unreliable steps
result = detection.normalize(
    NormalizationConfig(
        predicate=SignalAbove(1.5) & NSignificantAbove(3)
                  & NeighborCountAbove(4) & ~StepIn({1, 2})
    )
)
```

Predicates are fully serializable via `to_dict()` / `BreakPredicate.from_dict()`.

### Saving and loading results

```python
detection.to_json("detection.json")
result.to_json("result.json")

detection = rucola.DetectionResult.from_json("detection.json")
result    = rucola.HomogenizationResult.from_json("result.json")
```

## Development

```bash
git clone https://github.com/earthobservations/rucola
cd rucola
uv sync --all-groups

# code quality
poe format            # auto-format with ruff
poe lint              # check formatting and linting
poe type              # type-check with ty

# testing
poe test              # fast unit + constructor tests (default)
poe test-unit         # unit tests only, verbose
poe test-slow         # full pipeline tests, verbose
poe test-integration  # DWD integration tests (requires network)
poe test-all          # everything, verbose
poe coverage          # run tests and generate coverage.xml for Codecov

# docs
poe docs-serve        # live preview at localhost:8000
poe docs-build        # build static site

# security / hygiene
poe audit             # scan dependencies for vulnerabilities
poe deptry            # check for unused/missing dependencies
poe zizmor            # audit GitHub Actions workflows

# all-in-one
poe check             # lint + type + audit + test
```

## References

- González-Rouco et al. (2001), *J. Climate* 14(5):964–978. [doi:10.1175/1520-0442(2001)014<0964:QCAHOP>2.0.CO;2](https://doi.org/10.1175/1520-0442(2001)014<0964:QCAHOP>2.0.CO;2)
- Hawkins (1977), *Biometrika* 64(2):279–288. [doi:10.1093/biomet/64.2.279](https://doi.org/10.1093/biomet/64.2.279)
- Alexandersson (1986), *Int. J. Climatol.* 6(6):661–675. [doi:10.1002/joc.3370060607](https://doi.org/10.1002/joc.3370060607)
- Alexandersson & Moberg (1997), *Int. J. Climatol.* 17(1):25–34. [doi:10.1002/(SICI)1097-0088(199701)17:1<25::AID-JOC103>3.0.CO;2-J](https://doi.org/10.1002/(SICI)1097-0088(199701)17:1<25::AID-JOC103>3.0.CO;2-J)
- Buishand (1982), *J. Hydrol.* 58(1–2):11–29. [doi:10.1016/0022-1694(82)90066-X](https://doi.org/10.1016/0022-1694(82)90066-X)
- Pettitt (1979), *Appl. Stat.* 28(2):126–135. [doi:10.2307/2346729](https://doi.org/10.2307/2346729)
- Worsley (1979), *J. Amer. Statist. Assoc.* 74(366):365–367. [doi:10.1080/01621459.1979.10482519](https://doi.org/10.1080/01621459.1979.10482519)
- Easterling & Peterson (1995), *Int. J. Climatol.* 15(4):369–377. [doi:10.1002/joc.3370150403](https://doi.org/10.1002/joc.3370150403)
- Rodionov (2004), *Geophys. Res. Lett.* 31, L09204. [doi:10.1029/2004GL019448](https://doi.org/10.1029/2004GL019448)

## Authors

rucola was created by [Benjamin Gutzmann](mailto:benjamin@eobs.org), with the majority of the implementation written by [Claude](https://claude.ai) (Anthropic).
