"""Tests for the Rucola class — validation, run(), and normalize()."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

import rucola
from rucola._homogeneity import SNHTTest
from tests.conftest import BREAK_FACTOR

pytestmark = pytest.mark.slow

_CORRECTION_REL_TOLERANCE = 0.15

# ── construction validation ─────────────────────────────────────────────────


def test_missing_stations_column(values_df: pl.DataFrame) -> None:
    """Missing required column in stations raises ValueError."""
    bad = pl.DataFrame({"station_id": ["S1"], "latitude": [51.0]})
    with pytest.raises(ValueError, match="longitude"):
        rucola.Rucola(values_df, bad)


def test_missing_values_column(stations_df: pl.DataFrame) -> None:
    """Missing required column in values raises ValueError."""
    bad = pl.DataFrame({"station_id": ["S1"], "date": [date(2000, 1, 1)]})
    with pytest.raises(ValueError, match="value"):
        rucola.Rucola(bad, stations_df)


def test_station_coverage_check(values_df: pl.DataFrame, stations_df: pl.DataFrame) -> None:
    """Station ID in values without a matching stations entry raises ValueError."""
    extra = values_df.vstack(
        pl.DataFrame({"station_id": ["GHOST"], "date": [date(2000, 1, 1)], "value": [1.0], "parameter": ["p"]})
    )
    with pytest.raises(ValueError, match="GHOST"):
        rucola.Rucola(extra, stations_df)


def test_unsorted_dates_raises(stations_df: pl.DataFrame) -> None:
    """Values not sorted ascending by date raises ValueError."""
    bad = pl.DataFrame(
        {
            "station_id": ["S1", "S1"],
            "date": [date(2001, 1, 1), date(2000, 1, 1)],
            "value": [100.0, 100.0],
            "parameter": ["p", "p"],
        }
    )
    with pytest.raises(ValueError, match="sorted"):
        rucola.Rucola(bad, stations_df.filter(pl.col("station_id") == "S1"))


def test_sub_annual_resolution_raises(stations_df: pl.DataFrame) -> None:
    """Sub-annual data (multiple records per station-year) raises ValueError on run()."""
    bad = pl.DataFrame(
        {
            "station_id": ["S1", "S1"],
            "date": [date(2000, 1, 1), date(2000, 6, 1)],
            "value": [100.0, 200.0],
            "parameter": ["p", "p"],
        }
    )
    r = rucola.Rucola(bad, stations_df.filter(pl.col("station_id") == "S1"))
    with pytest.raises(ValueError, match="sub-annual"):
        r.run()


# ── run() parameter validation ──────────────────────────────────────────────


def test_min_series_years_consistency_raises(rucola_instance: rucola.Rucola) -> None:
    """min_series_years below 2*min_years_from_end+1 raises ValueError."""
    with pytest.raises(ValueError, match="min_series_years"):
        rucola_instance.run(rucola.RunConfig(tests=[SNHTTest(min_years_from_end=10)], min_series_years=5))


# ── run() output ────────────────────────────────────────────────────────────


def test_run_returns_detection_result(rucola_instance: rucola.Rucola) -> None:
    """run() returns a DetectionResult containing all five stations."""
    result = rucola_instance.run()
    assert isinstance(result, rucola.DetectionResult)
    assert set(result.station_detections) == {"S1", "S2", "S3", "S4", "S5"}


def test_run_station_ids_restricts_output(rucola_instance: rucola.Rucola) -> None:
    """station_ids restricts analysis to the specified subset."""
    result = rucola_instance.run(rucola.RunConfig(station_ids=["S1", "S2"]))
    assert set(result.station_detections) == {"S1", "S2"}


def test_run_corrects_broken_station(rucola_instance: rucola.Rucola) -> None:
    """Station S1 with an injected break receives at least one correction."""
    result = rucola_instance.run()
    s1 = result.station_detections["S1"]
    assert len(s1.corrections) >= 1
    f = s1.corrections[0].factor
    assert pytest.approx(1 / BREAK_FACTOR, rel=_CORRECTION_REL_TOLERANCE) == f


def test_run_insufficient_data(stations_df: pl.DataFrame) -> None:
    """Station with fewer years than min_series_years is labelled INSUFFICIENT_DATA."""
    short = pl.DataFrame(
        {
            "station_id": ["S1"] * 3,
            "date": [date(2000, 1, 1), date(2001, 1, 1), date(2002, 1, 1)],
            "value": [100.0, 110.0, 90.0],
            "parameter": ["p"] * 3,
        }
    )
    r = rucola.Rucola(short, stations_df.filter(pl.col("station_id") == "S1"))
    result = r.run()
    assert result.station_detections["S1"].group == "INSUFFICIENT_DATA"


def test_normalize_returns_homogenization_result(rucola_instance: rucola.Rucola) -> None:
    """normalize() returns a HomogenizationResult with S1 included."""
    result = rucola_instance.run().normalize()
    assert isinstance(result, rucola.HomogenizationResult)
    assert "S1" in result.station_results


# ── Rucola repr and properties ───────────────────────────────────────────────


def test_rucola_repr(rucola_instance: rucola.Rucola) -> None:
    """Rucola repr contains the class name, parameter, and station count."""
    r = repr(rucola_instance)
    assert "Rucola(" in r
    assert "precip" in r
    assert "stations=5" in r


def test_rucola_date_range(rucola_instance: rucola.Rucola) -> None:
    """date_range returns two ISO date strings spanning the values table."""
    lo, hi = rucola_instance.date_range
    assert lo < hi
    assert lo.startswith("1990")
    assert hi.startswith("2021")
