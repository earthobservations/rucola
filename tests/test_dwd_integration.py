"""Integration tests — full homogenization pipeline on DWD annual data for Saxony.

Locked expected values from a verified run on 2026-06-11. Update if DWD substantially
revises historical records or the procedure changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
import pytest

import rucola

if TYPE_CHECKING:
    from rucola._algorithms import CorrectionMode

_SQL = "state = 'Sachsen'"
_MIN_BREAK_YEAR = 1800
_GROUP_TOLERANCE = 5
_CORRECTION_TOLERANCE = 10


@dataclass(frozen=True)
class _Scenario:
    """Configuration and expected results for one parameter's integration test."""

    wd_parameter: tuple[str, str, str]  # (resolution, dataset, parameter_name)
    mode: CorrectionMode
    n_stations: int
    expected_groups: dict[str, int]
    expected_corrections: int
    min_factor: float
    max_factor: float


_PRECIPITATION = _Scenario(
    wd_parameter=("annual", "more_precip", "precipitation_height"),
    mode="ratio",
    n_stations=380,
    expected_groups={
        "H4": 240,  # homogeneous (includes stations always found clean)
        "HC4": 46,  # corrected once at step 4
        "HC5": 13,  # corrected at step 5
        "HCC6": 12,  # corrected twice
        "INSUFFICIENT_DATA": 69,
        "UNTESTABLE": 0,
    },
    expected_corrections=203,
    min_factor=0.5,  # multiplicative; < 0.5 or > 2.0 is physically implausible
    max_factor=2.0,
)

_TEMPERATURE = _Scenario(
    wd_parameter=("annual", "climate_summary", "temperature_air_mean_2m"),
    mode="difference",
    n_stations=58,
    expected_groups={
        "H4": 4,
        "HC4": 5,
        "HC5": 3,
        "HCC6": 39,  # most 150-year temperature records have two breaks
        "INSUFFICIENT_DATA": 7,
        "UNTESTABLE": 0,
    },
    expected_corrections=142,
    min_factor=-10.0,  # additive °C offset; ±10°C is the physical plausibility bound
    max_factor=10.0,
)


def _run_scenario(scenario: _Scenario) -> None:
    from wetterdienst.provider.dwd.observation import DwdObservationRequest  # noqa: PLC0415

    req = DwdObservationRequest(parameters=[scenario.wd_parameter])
    station_result = req.filter_by_sql(_SQL)

    stations = station_result.df.select("station_id", "latitude", "longitude")
    assert len(stations) == scenario.n_stations, f"Expected {scenario.n_stations} stations, got {len(stations)}"

    frames = [
        r.df.select(
            pl.col("station_id"),
            pl.col("date").dt.replace_time_zone(None).cast(pl.Date).alias("date"),
            pl.col("value"),
            pl.col("parameter"),
        )
        for r in station_result.values.query()
    ]
    assert frames, "No value frames returned from wetterdienst"
    values = pl.concat(frames).sort("station_id", "date")

    r = rucola.Rucola.from_polars(values, stations, parameter=scenario.wd_parameter[2])
    result = r.run(rucola.RunConfig(mode=scenario.mode))

    actual_groups: dict[str, int] = dict(result.summary.group_by("group").len().sort("group").rows())
    for group, expected in scenario.expected_groups.items():
        actual = actual_groups.get(group, 0)
        assert abs(actual - expected) <= _GROUP_TOLERANCE, f"Group {group!r}: expected ~{expected}, got {actual}"

    total_corrections = sum(len(d.corrections) for d in result.station_detections.values())
    assert abs(total_corrections - scenario.expected_corrections) <= _CORRECTION_TOLERANCE, (
        f"Expected ~{scenario.expected_corrections} corrections, got {total_corrections}"
    )

    max_year: int = values.select(pl.col("date").dt.year().max()).item()
    for sid, det in result.station_detections.items():
        for corr in det.corrections:
            assert scenario.min_factor <= corr.factor <= scenario.max_factor, (
                f"Station {sid}: factor {corr.factor:.3f} outside [{scenario.min_factor}, {scenario.max_factor}]"
            )
            assert _MIN_BREAK_YEAR <= corr.break_year <= max_year, (
                f"Station {sid}: break year {corr.break_year} outside [{_MIN_BREAK_YEAR}, {max_year}]"
            )


@pytest.mark.integration
@pytest.mark.slow
def test_dwd_saxony_precipitation() -> None:
    """Full homogenization pipeline on 380 DWD annual precipitation stations in Saxony."""
    _run_scenario(_PRECIPITATION)


@pytest.mark.integration
@pytest.mark.slow
def test_dwd_saxony_temperature() -> None:
    """Full homogenization pipeline on 58 DWD annual mean temperature stations in Saxony."""
    _run_scenario(_TEMPERATURE)
