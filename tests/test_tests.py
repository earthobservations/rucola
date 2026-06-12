"""Tests for rucola._homogeneity — statistical homogeneity tests."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import polars as pl
import pytest

from rucola._homogeneity import (
    _SNHT_CRIT,
    BuishandTest,
    EasterlingPetersonTest,
    PettittTest,
    SNHTTest,
    StarsTest,
    TestResult,
    WorsleyTest,
    _interpolate_crit,
)

if TYPE_CHECKING:
    from rucola._homogeneity import HomogenizationTest

pytestmark = pytest.mark.unit

_ALL_TESTS = [SNHTTest, BuishandTest, PettittTest, WorsleyTest, EasterlingPetersonTest, StarsTest]

N = 40
BREAK_POS = 20
_YEARS = list(range(1980, 1980 + N))
_BREAK_YEAR_TOLERANCE = 2
_MIN_SIGNAL_RATIO = 1.0
_SNHT_CRIT_N10 = 8.45  # tabulated 0.05 critical value for n=10


def _homogeneous() -> tuple[pl.Series, pl.Series]:
    return pl.Series("q", [1.0] * N), pl.Series("year", _YEARS)


def _with_break(shift: float = 3.0) -> tuple[pl.Series, pl.Series]:
    vals = [1.0 + shift if i < BREAK_POS else 1.0 for i in range(N)]
    return pl.Series("q", vals), pl.Series("year", _YEARS)


@pytest.mark.parametrize("test_cls", _ALL_TESTS)
def test_homogeneous_not_significant(test_cls: type[HomogenizationTest]) -> None:
    """Homogeneous series is not flagged as significant."""
    q, years = _homogeneous()
    assert not test_cls().detect(q, years).is_significant


@pytest.mark.parametrize("test_cls", _ALL_TESTS)
def test_clear_break_significant(test_cls: type[HomogenizationTest]) -> None:
    """Series with a clear step change is flagged as significant."""
    q, years = _with_break()
    assert test_cls().detect(q, years).is_significant


@pytest.mark.parametrize("test_cls", _ALL_TESTS)
def test_break_year_within_two_of_truth(test_cls: type[HomogenizationTest]) -> None:
    """Detected break year is within two years of the true break position."""
    q, years = _with_break()
    result = test_cls().detect(q, years)
    assert abs(result.break_year - (1980 + BREAK_POS)) <= _BREAK_YEAR_TOLERANCE


@pytest.mark.parametrize("test_cls", _ALL_TESTS)
def test_relative_signal_above_one_when_significant(test_cls: type[HomogenizationTest]) -> None:
    """Relative signal exceeds 1.0 when a break is detected."""
    q, years = _with_break()
    result = test_cls().detect(q, years)
    assert result.relative_signal > _MIN_SIGNAL_RATIO


def test_is_inhomogeneous_edge_guard_rejects_near_end() -> None:
    """Break within min_years_from_end of the series end is rejected."""
    test = SNHTTest(min_years_from_end=5)
    q, years = _with_break()
    result = SNHTTest().detect(q, years)
    near_end = replace(result, break_year=result.segment_end - 2, is_significant=True)
    assert not test.is_inhomogeneous(near_end)


def test_is_inhomogeneous_accepts_central_break() -> None:
    """Break well away from both ends is accepted."""
    test = SNHTTest(min_years_from_end=5)
    q, years = _with_break()
    result = SNHTTest().detect(q, years)
    assert result.is_significant
    assert test.is_inhomogeneous(result)


def test_snht_repr() -> None:
    """SNHTTest repr contains the class name, alpha, edge guard, and signal threshold."""
    r = repr(SNHTTest(alpha=0.05))
    assert "SNHTTest" in r
    assert "alpha=0.05" in r
    assert "min_years_from_end=5" in r
    assert "min_relative_signal=1.0" in r


def test_min_relative_signal_rejects_weak_break() -> None:
    """A break below min_relative_signal is rejected even when statistically significant."""
    q, years = _with_break(shift=0.8)  # weak but still significant
    result = SNHTTest().detect(q, years)
    assert result.is_significant
    strict = SNHTTest(min_relative_signal=result.relative_signal + 0.1)
    assert not strict.is_inhomogeneous(result)


_EP_BREAK_POS = 20  # break position in the EP-specific trended test series


def test_easterling_peterson_detects_break_in_trended_series() -> None:
    """EP test detects a break even when a linear trend is present."""
    vals = [float(i) + (3.0 if i >= _EP_BREAK_POS else 0.0) for i in range(40)]
    q = pl.Series("q", vals)
    years = pl.Series("year", list(range(1980, 2020)))
    result = EasterlingPetersonTest().detect(q, years)
    assert result.is_significant
    expected_break = 1980 + _EP_BREAK_POS
    assert abs(result.break_year - expected_break) <= _BREAK_YEAR_TOLERANCE


def test_min_relative_signal_accepts_strong_break() -> None:
    """A break well above min_relative_signal is accepted."""
    q, years = _with_break()
    result = SNHTTest().detect(q, years)
    lenient = SNHTTest(min_relative_signal=result.relative_signal - 0.5)
    assert lenient.is_inhomogeneous(result)


# ── _interpolate_crit boundary conditions ───────────────────────────────────


def test_snht_small_n_clamps_to_minimum_table_entry() -> None:
    """Series shorter than the smallest table entry uses the minimum critical value."""
    q = pl.Series("q", [1.0] * 3 + [3.0] * 3)
    years = pl.Series("year", list(range(2000, 2006)))
    result = SNHTTest().detect(q, years)
    assert result.critical_value == pytest.approx(_SNHT_CRIT_N10)


def test_snht_large_n_clamps_to_maximum_table_entry() -> None:
    """Series longer than the largest table entry uses the maximum critical value."""
    q = pl.Series("q", [1.0] * 2000)
    years = pl.Series("year", list(range(1000, 3000)))
    result = SNHTTest().detect(q, years)
    assert result.critical_value == pytest.approx(_SNHT_CRIT[0.05][1000])


def test_too_short_series_returns_null_result() -> None:
    """Series below minimum length returns a non-significant result with zero statistic."""
    q = pl.Series("q", [1.0, 2.0, 3.0])
    years = pl.Series("year", [2000, 2001, 2002])
    result = SNHTTest().detect(q, years)
    assert not result.is_significant
    assert result.test_statistic == pytest.approx(0.0)


# ── TestResult serialisation ─────────────────────────────────────────────────


def test_test_result_to_dict_from_dict_roundtrip() -> None:
    """TestResult survives a to_dict / from_dict round-trip."""
    q, years = _with_break()
    result = SNHTTest().detect(q, years)
    loaded = TestResult.from_dict(result.to_dict())
    assert loaded.test_name == result.test_name
    assert loaded.is_significant == result.is_significant
    assert loaded.break_year == result.break_year
    assert loaded.test_statistic == pytest.approx(result.test_statistic)
    assert loaded.critical_value == pytest.approx(result.critical_value)
    assert loaded.series == pytest.approx(result.series)
    assert loaded.years_tested == result.years_tested


# ── relative_signal edge case ────────────────────────────────────────────────


def test_test_result_relative_signal_zero_critical_value() -> None:
    """relative_signal returns 0.0 when critical_value is at or near zero."""
    result = TestResult(
        test_name="snht",
        is_significant=False,
        break_year=2000,
        test_statistic=5.0,
        critical_value=0.0,
        n_years=10,
        segment_start=2000,
        segment_end=2009,
    )
    assert result.relative_signal == pytest.approx(0.0)


# ── invalid alpha ────────────────────────────────────────────────────────────


def test_snht_invalid_alpha_raises() -> None:
    """SNHTTest raises ValueError for an alpha not in its critical-value table."""
    q, years = _with_break()
    with pytest.raises(ValueError, match="alpha"):
        SNHTTest(alpha=0.07).detect(q, years)


def test_buishand_invalid_alpha_raises() -> None:
    """BuishandTest raises ValueError for an alpha not in its critical-value table."""
    q, years = _with_break()
    with pytest.raises(ValueError, match="alpha"):
        BuishandTest(alpha=0.07).detect(q, years)


# ── _interpolate_crit mid-range ──────────────────────────────────────────────


def test_interpolate_crit_midpoint() -> None:
    """Linear interpolation at the midpoint between two table entries equals their mean."""
    table = _SNHT_CRIT[0.05]
    # n=25 is halfway between n=20 (9.56) and n=30 (9.83)
    result = _interpolate_crit(table, 25)
    assert result == pytest.approx((table[20] + table[30]) / 2, rel=1e-6)


# ── start-edge guard ─────────────────────────────────────────────────────────


def test_is_inhomogeneous_edge_guard_rejects_near_start() -> None:
    """Break within min_years_from_end of the series start is rejected."""
    test = SNHTTest(min_years_from_end=5)
    q, years = _with_break()
    result = SNHTTest().detect(q, years)
    near_start = replace(result, break_year=result.segment_start + 2, is_significant=True)
    assert not test.is_inhomogeneous(near_start)


# ── all-null Q-series ────────────────────────────────────────────────────────


@pytest.mark.parametrize("test_cls", _ALL_TESTS)
def test_all_null_series_returns_non_significant(test_cls: type[HomogenizationTest]) -> None:
    """Q-series of all nulls produces a non-significant result."""
    q = pl.Series("q", [None] * N, dtype=pl.Float64)
    years = pl.Series("year", _YEARS)
    result = test_cls().detect(q, years)
    assert not result.is_significant


# ── stricter alpha ───────────────────────────────────────────────────────────


def test_snht_alpha_01_critical_value_stricter_than_alpha_05() -> None:
    """SNHT at alpha=0.01 uses a larger critical value than at alpha=0.05 for the same series."""
    q, years = _with_break()
    crit_05 = SNHTTest(alpha=0.05).detect(q, years).critical_value
    crit_01 = SNHTTest(alpha=0.01).detect(q, years).critical_value
    assert crit_01 > crit_05


# ── StarsTest-specific ────────────────────────────────────────────────────────


def test_stars_repr_includes_l() -> None:
    """StarsTest repr contains the class name, l, alpha, edge guard, and signal threshold."""
    r = repr(StarsTest(l=15, alpha=0.05))
    assert "StarsTest" in r
    assert "l=15" in r
    assert "alpha=0.05" in r
    assert "min_years_from_end=5" in r
    assert "min_relative_signal=1.0" in r


def test_stars_invalid_alpha_raises() -> None:
    """StarsTest raises ValueError for an alpha not in its critical-value table."""
    q, years = _with_break()
    with pytest.raises(ValueError, match="alpha"):
        StarsTest(alpha=0.07).detect(q, years)


def test_stars_too_short_series_returns_null() -> None:
    """Series shorter than 2*l returns a non-significant null result."""
    cut_off = 10
    q = pl.Series("q", [1.0] * (2 * cut_off - 1))  # one observation short
    years = pl.Series("year", list(range(2000, 2000 + len(q))))
    result = StarsTest(l=cut_off).detect(q, years)
    assert not result.is_significant
    assert result.test_statistic == pytest.approx(0.0)


def test_stars_alpha_01_critical_value_stricter_than_alpha_05() -> None:
    """StarsTest at alpha=0.01 uses a larger critical value than at alpha=0.05."""
    q, years = _with_break()
    crit_05 = StarsTest(alpha=0.05).detect(q, years).critical_value
    crit_01 = StarsTest(alpha=0.01).detect(q, years).critical_value
    assert crit_01 > crit_05


def test_stars_larger_l_larger_critical_value() -> None:
    """Larger cut-off length l means fewer degrees of freedom and a stricter critical value."""
    q, years = _with_break()
    crit_l5 = StarsTest(l=5).detect(q, years).critical_value
    crit_l10 = StarsTest(l=10).detect(q, years).critical_value
    assert crit_l5 > crit_l10


_STARS_DOUBLE_BREAK_UPPER_BOUND = 1995


def test_stars_minimum_regime_length_prevents_double_detection() -> None:
    """Two breaks closer than l years apart: only the first is detected."""
    cut_off = 10
    # Break at index 10 (value shifts from 0 to 3) and again at index 15 (back to 0).
    # Both breaks are within cut_off=10 of each other, so STARS should only confirm the first.
    vals = [0.0] * 10 + [3.0] * 5 + [0.0] * 15
    q = pl.Series("q", vals)
    years = pl.Series("year", list(range(1980, 1980 + len(vals))))
    result = StarsTest(l=cut_off).detect(q, years)
    # At most one break confirmed; second shift (back to 0) falls within next_check window
    assert result.is_significant
    assert result.break_year < _STARS_DOUBLE_BREAK_UPPER_BOUND  # first break, not the reversion
