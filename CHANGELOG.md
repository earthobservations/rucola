# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-17

### Added

- Six-step SNHT homogenization pipeline (`Rucola.run()` / `.normalize()`) implementing González-Rouco et al. (2001), including the **step 6c diagnostic re-test** of the doubly-corrected series. `HCC6` is assigned only when consensus confirms the series is homogeneous after the two González-Rouco corrections; otherwise the station is classified `ICC6`. The re-test result is stored under `detections_by_step[63]` for inspection and never applied as a correction.
- Six statistical breakpoint tests: SNHT (Alexandersson 1986), Buishand range (Buishand 1982), Pettitt (Pettitt 1979), Worsley likelihood ratio (Worsley 1979), Easterling–Peterson two-phase regression (Easterling & Peterson 1995), and STARS sequential regime-shift (Rodionov 2004). STARS is configurable via `l` (minimum regime length / RSI confirmation window).
- `ratio` and `difference` correction modes for precipitation and temperature respectively.
- Constructors: `Rucola.from_polars`, `Rucola.from_csv`, `Rucola.from_duckdb`. `Rucola.__init__` validates inputs for duplicate dates and multiple parameters with fail-fast `ValueError`s.
- `NormalizationConfig` with `consensus`, `tiebreak`, `break_window_years`, `min_correction_magnitude`, `min_relative_signal`, `min_years_from_end`, `max_corrections_per_station`, `predicate`, and `overrides` controls. `min_relative_signal` rejects test results below a configurable ratio of test statistic to critical value, even when significant; mirrored on `HomogenizationTest`.
- Composable break predicates: `YearBetween`, `StationIn`, `StepIn`, `MagnitudeAbove`, `TestSignificant`, `SignalAbove`, `NSignificantAbove`, `NeighborCountAbove`, combinable with `&` / `|` / `~` and fully serializable via `to_dict` / `BreakPredicate.from_dict`.
- `DetectionResult` and `HomogenizationResult` result types with `to_json` / `from_json` round-trip serialization. `DetectionRecord.break_year` is typed `int | None` and is `None` when no test qualified at that step.
- Winsorization and annual aggregation helpers (`winsorize_outliers`, `compute_annual_totals`, `compute_annual_means`).
- DWD integration tests for Saxony precipitation (380 stations) and temperature (58 stations) with locked expected values.
- pytest marker scheme: `unit`, `slow`, `integration`.
- `poe` tasks: `test`, `test-unit`, `test-slow`, `test-integration`, `test-all`, `check`, `lint`, `format`, `type`, `docs-serve`, `docs-build`, `audit`, `deptry`, `zizmor`, `coverage`.
- Interactive `notebooks/saxony_station_homogenization.py` (marimo) with colour-coded group stat cards, group-breakdown stacked-bar chart, colour-coded category column on the detection summary table, colour emoji prefixes on station-picker labels, and a "Only stations with corrections" toggle for the Step 4 chart picker.

[Unreleased]: https://github.com/gutzbenj/rucola/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gutzbenj/rucola/releases/tag/v0.1.0
