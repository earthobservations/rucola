"""Tests for rucola.preprocessing — winsorization and annual aggregation."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from rucola._preprocessing import compute_annual_means, compute_annual_totals, winsorize_outliers

pytestmark = pytest.mark.unit

_OUTLIER = 1_000_000.0


def test_winsorize_clips_extreme_outlier() -> None:
    """An extreme outlier is clipped to the P_out threshold."""
    values = pl.DataFrame({"station_id": ["S1"] * 10, "value": [100.0] * 9 + [_OUTLIER]})
    result = winsorize_outliers(values)
    max_val = result["value"].max()
    assert isinstance(max_val, float)
    assert max_val < _OUTLIER


def test_winsorize_preserves_normal_values() -> None:
    """Values within the normal range are unchanged by winsorization."""
    values = pl.DataFrame({"station_id": ["S1"] * 4, "value": [80.0, 90.0, 100.0, 110.0]})
    result = winsorize_outliers(values)
    assert result["value"].to_list() == pytest.approx([80.0, 90.0, 100.0, 110.0])


def test_compute_annual_totals_sums_within_year() -> None:
    """Two observations in the same year are summed into a single annual total."""
    values = pl.DataFrame(
        {
            "station_id": ["S1"] * 3,
            "date": [date(2000, 1, 1), date(2000, 7, 1), date(2001, 1, 1)],
            "value": [100.0, 200.0, 50.0],
            "parameter": ["p"] * 3,
        }
    )
    result = compute_annual_totals(values, min_coverage=0.0).sort("date")
    assert result["value"][0] == pytest.approx(300.0)
    assert result["value"][1] == pytest.approx(50.0)


def test_compute_annual_totals_nulls_insufficient_coverage() -> None:
    """A year with fewer observations than min_coverage * 365 is set to null."""
    values = pl.DataFrame(
        {
            "station_id": ["S1"],
            "date": [date(2000, 6, 1)],
            "value": [500.0],
            "parameter": ["p"],
        }
    )
    result = compute_annual_totals(values, min_coverage=0.8)
    assert result["value"][0] is None


def test_compute_annual_totals_date_is_jan1() -> None:
    """Annual totals are stamped with January 1 of the aggregated year."""
    values = pl.DataFrame(
        {
            "station_id": ["S1"],
            "date": [date(2000, 6, 1)],
            "value": [500.0],
            "parameter": ["p"],
        }
    )
    result = compute_annual_totals(values, min_coverage=0.0)
    assert result["date"][0] == date(2000, 1, 1)


def test_compute_annual_means_averages_within_year() -> None:
    """Two observations in the same year are averaged into a single annual mean."""
    values = pl.DataFrame(
        {
            "station_id": ["S1"] * 2,
            "date": [date(2000, 1, 1), date(2000, 7, 1)],
            "value": [100.0, 200.0],
            "parameter": ["p"] * 2,
        }
    )
    result = compute_annual_means(values, min_coverage=0.0)
    assert result["value"][0] == pytest.approx(150.0)
