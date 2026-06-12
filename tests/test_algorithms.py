"""Tests for rucola._algorithms — pure functions for distance, Q-series, correction, and neighbors."""

from __future__ import annotations

import polars as pl
import pytest

from rucola._algorithms import (
    NeighborInfo,
    apply_correction,
    build_correlation_cache,
    build_reference_series,
    compute_correction_factor,
    compute_q_series,
    haversine_km,
    select_neighbors,
)

pytestmark = pytest.mark.unit

_PARIS_LONDON_MIN_KM = 330.0
_PARIS_LONDON_MAX_KM = 350.0
_ONE_NEIGHBOR = 1


def test_haversine_paris_london() -> None:
    """Haversine distance between Paris and London is approximately 341 km."""
    d = haversine_km(48.85, 2.35, 51.51, -0.13)
    assert _PARIS_LONDON_MIN_KM < d < _PARIS_LONDON_MAX_KM


def test_haversine_same_point() -> None:
    """Distance from a point to itself is zero."""
    assert haversine_km(51.0, 13.0, 51.0, 13.0) == pytest.approx(0.0)


def test_haversine_symmetry() -> None:
    """Haversine distance is symmetric."""
    assert haversine_km(48.0, 10.0, 52.0, 14.0) == pytest.approx(haversine_km(52.0, 14.0, 48.0, 10.0))


# ── compute_q_series ────────────────────────────────────────────────────────


def test_q_series_ratio_homogeneous() -> None:
    """Constant candidate normalised by a unit reference yields all-ones Q-series."""
    q = compute_q_series(pl.Series("c", [100.0] * 20), pl.Series("r", [1.0] * 20), mode="ratio")
    assert q.drop_nulls().to_list() == pytest.approx([1.0] * 20)


def test_q_series_difference() -> None:
    """Difference mode subtracts reference from candidate element-wise."""
    q = compute_q_series(pl.Series("c", [10.0, 20.0, 30.0]), pl.Series("r", [8.0, 18.0, 28.0]), mode="difference")
    assert q.to_list() == pytest.approx([2.0, 2.0, 2.0])


def test_q_series_null_candidate_propagates() -> None:
    """Null in candidate propagates to null in the Q-series."""
    q = compute_q_series(pl.Series("c", [None, 100.0, 100.0]), pl.Series("r", [1.0, 1.0, 1.0]), mode="ratio")
    assert q[0] is None


def test_q_series_null_reference_propagates() -> None:
    """Null in reference propagates to null in the Q-series."""
    q = compute_q_series(pl.Series("c", [100.0, 100.0, 100.0]), pl.Series("r", [1.0, None, 1.0]), mode="ratio")
    assert q[1] is None


# ── compute_correction_factor ───────────────────────────────────────────────


def test_correction_factor_ratio() -> None:
    """Ratio correction factor equals the post-break mean divided by the pre-break mean."""
    q = pl.Series("q", [0.8] * 10 + [1.2] * 10)
    years = pl.Series("year", list(range(1990, 2010)))
    f = compute_correction_factor(q, years, break_year=2000, mode="ratio")
    assert f == pytest.approx(1.2 / 0.8, rel=1e-4)


def test_correction_factor_difference() -> None:
    """Difference correction factor equals the post-break mean minus the pre-break mean."""
    q = pl.Series("q", [1.0] * 10 + [3.0] * 10)
    years = pl.Series("year", list(range(1990, 2010)))
    f = compute_correction_factor(q, years, break_year=2000, mode="difference")
    assert f == pytest.approx(2.0)


def test_correction_factor_neutral_when_empty_segment() -> None:
    """Break year at the start of the series returns the neutral factor (1.0 for ratio)."""
    q = pl.Series("q", [1.0] * 5)
    years = pl.Series("year", list(range(2000, 2005)))
    f = compute_correction_factor(q, years, break_year=2000, mode="ratio")
    assert f == pytest.approx(1.0)


# ── apply_correction ────────────────────────────────────────────────────────


def test_apply_correction_ratio() -> None:
    """Ratio correction multiplies all pre-break values by the given factor."""
    vals = pl.Series("s", [100.0, 100.0, 100.0, 100.0])
    years = pl.Series("year", [1990, 1991, 1992, 1993])
    corrected = apply_correction(vals, years, break_year=1992, factor=2.0, mode="ratio")
    assert corrected.to_list() == pytest.approx([200.0, 200.0, 100.0, 100.0])


def test_apply_correction_difference() -> None:
    """Difference correction adds the factor to all pre-break values."""
    vals = pl.Series("s", [10.0, 10.0, 10.0, 10.0])
    years = pl.Series("year", [1990, 1991, 1992, 1993])
    corrected = apply_correction(vals, years, break_year=1992, factor=5.0, mode="difference")
    assert corrected.to_list() == pytest.approx([15.0, 15.0, 10.0, 10.0])


def test_apply_correction_preserves_nulls() -> None:
    """Null values in the series remain null after correction."""
    vals = pl.Series("s", [None, 100.0, 100.0])
    years = pl.Series("year", [1990, 1991, 1992])
    corrected = apply_correction(vals, years, break_year=1992, factor=2.0, mode="ratio")
    assert corrected[0] is None
    assert corrected[1] == pytest.approx(200.0)
    assert corrected[2] == pytest.approx(100.0)


# ── build_correlation_cache edge cases ──────────────────────────────────────


def test_build_correlation_cache_empty_wide() -> None:
    """Wide DataFrame with no station columns returns an empty cache."""
    wide = pl.DataFrame({"year": [2000, 2001]})
    assert build_correlation_cache(wide) == {}


def test_build_correlation_cache_single_station() -> None:
    """Single-station DataFrame returns a cache entry with no correlations."""
    wide = pl.DataFrame({"year": [2000, 2001, 2002], "S1": [1.0, 2.0, 3.0]})
    cache = build_correlation_cache(wide)
    assert cache["S1"] == {}


# ── build_reference_series edge cases ───────────────────────────────────────


def test_build_reference_series_no_valid_neighbors() -> None:
    """Empty neighbor list produces an all-null reference series."""
    wide = pl.DataFrame({"year": [2000, 2001], "S1": [1.0, 2.0], "S2": [3.0, 4.0]})
    ref = build_reference_series(wide, [], mode="ratio")
    assert all(v is None for v in ref.to_list())


def test_build_reference_series_ratio_zero_mean_excluded() -> None:
    """Neighbor with zero long-term mean is excluded in ratio mode, yielding all-null reference."""
    wide = pl.DataFrame({"year": [2000, 2001], "S2": [0.0, 0.0]})
    n = NeighborInfo(station_id="S2", distance_km=10.0, correlation=0.9, weight=0.81)
    ref = build_reference_series(wide, [n], mode="ratio")
    assert all(v is None for v in ref.to_list())


# ── select_neighbors slow path (no cache) ───────────────────────────────────


def test_select_neighbors_slow_path_finds_neighbor() -> None:
    """select_neighbors without cache arguments finds a highly-correlated neighbor."""
    stations = pl.DataFrame(
        {
            "station_id": ["S1", "S2"],
            "latitude": [51.0, 51.1],
            "longitude": [13.0, 13.1],
        }
    )
    wide = pl.DataFrame(
        {
            "year": list(range(2000, 2015)),
            "S1": [float(i) for i in range(15)],
            "S2": [float(i) * 1.01 for i in range(15)],
        }
    )
    nbrs = select_neighbors("S1", stations, wide, min_overlap_years=5)
    assert len(nbrs) == _ONE_NEIGHBOR
    assert nbrs[0].station_id == "S2"


def test_select_neighbors_candidate_not_in_wide() -> None:
    """select_neighbors returns empty list when candidate is absent from the wide DataFrame."""
    stations = pl.DataFrame(
        {
            "station_id": ["S1"],
            "latitude": [51.0],
            "longitude": [13.0],
        }
    )
    wide = pl.DataFrame({"year": [2000], "S2": [1.0]})
    nbrs = select_neighbors("S1", stations, wide)
    assert nbrs == []
