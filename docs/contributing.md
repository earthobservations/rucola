# Contributing

## Setup

```bash
git clone https://github.com/earthobservations/rucola
cd rucola
uv sync --all-groups
```

## Development workflow

```bash
poe format            # auto-format with ruff
poe lint              # check formatting and linting
poe type              # type-check with ty
poe test              # fast unit tests (default)
poe test-slow         # full pipeline tests
poe test-all          # everything including DWD integration tests
poe check             # lint + type + audit + test
```

## Running specific test groups

```bash
poe test-unit         # pure function tests, no I/O
poe test-slow         # tests that run the full 6-step procedure
poe test-integration  # tests that call the DWD API (requires network)
```

## Security and hygiene

```bash
poe audit             # scan dependencies for vulnerabilities (uv audit)
poe deptry            # check for unused/missing dependencies
poe zizmor            # audit GitHub Actions workflows
```

## Project layout

```
src/rucola/
├── __init__.py          # Rucola, RunConfig — public API and 6-step procedure
├── _algorithms.py       # haversine, Q-series, neighbor selection, correction
├── _homogeneity.py      # SNHT, Buishand, Pettitt, Worsley, Easterling-Peterson
├── _normalization.py    # NormalizationConfig, Normalizer
├── _preprocessing.py    # winsorize_outliers, compute_annual_totals/means
└── _results.py          # DetectionResult, HomogenizationResult, serialization
```

## Pull requests

- Target the `main` branch
- All CI checks must pass (lint, type, tests)
- Add tests for new behaviour — aim to keep coverage above 70 %
- Update `CHANGELOG.md` under `[Unreleased]`
