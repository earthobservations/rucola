"""Tests for rucola._normalization — consensus modes, tiebreaks, and magnitude filters."""

from __future__ import annotations

import polars as pl
import pytest

from rucola._algorithms import NeighborInfo
from rucola._homogeneity import TestResult
from rucola._normalization import (
    BreakInfo,
    BreakPredicate,
    MagnitudeAbove,
    NeighborCountAbove,
    NormalizationConfig,
    NSignificantAbove,
    SignalAbove,
    StationIn,
    StepIn,
    TestSignificant,
    YearBetween,
)
from rucola._results import DetectionRecord, DetectionResult, StationDetection

pytestmark = pytest.mark.unit

_ONE_CORRECTION = 1
_NO_CORRECTIONS = 0
_SMALL_FACTOR = 1.02
_MIN_MAGNITUDE = 0.05
_YEAR_WINDOW = 2


def _tr(*, sig: bool = True, break_year: int = 2005, rel: float = 2.0) -> TestResult:
    return TestResult(
        test_name="snht",
        is_significant=sig,
        break_year=break_year,
        test_statistic=rel,
        critical_value=1.0,
        n_years=20,
        segment_start=1985,
        segment_end=2020,
    )


def _det_result(
    test_results: list[TestResult],
    factor: float = 1.2,
    mode: str = "ratio",
    n_neighbors: int = 5,
) -> DetectionResult:
    years = pl.Series("year", list(range(1990, 2010)))
    rec = DetectionRecord(step=1, break_year=2005, factor=factor, test_results=test_results, was_applied=True)
    nbrs = [
        NeighborInfo(station_id=f"N{i}", distance_km=None, correlation=0.8, weight=0.64) for i in range(n_neighbors)
    ]
    sd = StationDetection(
        station_id="S1",
        group="I1",
        annual_original=pl.Series("S1", [100.0] * 20),
        annual_corrected=pl.Series("S1", [100.0] * 20),
        years=years,
        detections_by_step={1: rec},
        neighbors_by_step={1: nbrs},
        corrections=[],
    )
    return DetectionResult(station_detections={"S1": sd}, parameter="test", mode=mode)


# ── consensus modes ─────────────────────────────────────────────────────────


def test_consensus_any_accepts_single_significant() -> None:
    """consensus='any' applies a correction when at least one test is significant."""
    dr = _det_result([_tr()])
    result = dr.normalize(NormalizationConfig(consensus="any"))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


def test_consensus_unanimous_requires_all_significant() -> None:
    """consensus='unanimous' skips correction when one of two tests is not significant."""
    dr = _det_result([_tr(), _tr(sig=False)])
    result = dr.normalize(NormalizationConfig(consensus="unanimous", tiebreak="skip"))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


def test_consensus_unanimous_passes_when_all_agree() -> None:
    """consensus='unanimous' applies correction when all tests agree."""
    dr = _det_result([_tr(), _tr()])
    result = dr.normalize(NormalizationConfig(consensus="unanimous"))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


def test_consensus_majority_requires_more_than_half() -> None:
    """consensus='majority' skips correction when only one of three tests is significant."""
    dr = _det_result([_tr(), _tr(sig=False), _tr(sig=False)])
    result = dr.normalize(NormalizationConfig(consensus="majority", tiebreak="skip"))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


def test_consensus_majority_passes_with_majority() -> None:
    """consensus='majority' applies correction when two of three tests are significant."""
    dr = _det_result([_tr(), _tr(), _tr(sig=False)])
    result = dr.normalize(NormalizationConfig(consensus="majority"))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


def test_default_normalization_config_consensus_is_majority() -> None:
    """NormalizationConfig() defaults to consensus='majority', not 'any'.

    One significant test out of three passes 'any' but fails 'majority'.
    Without tiebreak fallback this must produce no correction.
    """
    dr = _det_result([_tr(), _tr(sig=False), _tr(sig=False)])
    result = dr.normalize(NormalizationConfig(tiebreak="skip"))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


def test_consensus_strongest_signal_uses_most_significant() -> None:
    """consensus='strongest_signal' applies correction using the most significant test."""
    dr = _det_result([_tr(rel=3.0), _tr(sig=False, rel=0.5)])
    result = dr.normalize(NormalizationConfig(consensus="strongest_signal"))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


def test_consensus_strongest_signal_no_correction_when_none_significant() -> None:
    """consensus='strongest_signal' skips correction when no test is significant."""
    dr = _det_result([_tr(sig=False), _tr(sig=False)])
    result = dr.normalize(NormalizationConfig(consensus="strongest_signal"))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


# ── tiebreak ────────────────────────────────────────────────────────────────


def test_tiebreak_skip_applies_no_correction() -> None:
    """tiebreak='skip' produces no correction when consensus is not met."""
    dr = _det_result([_tr(), _tr(sig=False)])
    result = dr.normalize(NormalizationConfig(consensus="unanimous", tiebreak="skip"))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


def test_tiebreak_strongest_signal_picks_best_when_years_disagree() -> None:
    """tiebreak='strongest_signal' resolves year disagreement by picking the highest signal."""
    dr = _det_result([_tr(break_year=2000, rel=3.0), _tr(break_year=2010, rel=1.5)])
    result = dr.normalize(
        NormalizationConfig(consensus="any", break_window_years=_YEAR_WINDOW, tiebreak="strongest_signal")
    )
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


def test_tiebreak_skip_when_years_disagree() -> None:
    """tiebreak='skip' produces no correction when tests disagree on break year."""
    dr = _det_result([_tr(break_year=2000), _tr(break_year=2010)])
    result = dr.normalize(NormalizationConfig(consensus="any", break_window_years=_YEAR_WINDOW, tiebreak="skip"))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


# ── other paths ─────────────────────────────────────────────────────────────


def test_min_correction_magnitude_filters_small_factor() -> None:
    """Correction with magnitude below min_correction_magnitude is not applied."""
    dr = _det_result([_tr()], factor=_SMALL_FACTOR)
    result = dr.normalize(NormalizationConfig(consensus="any", min_correction_magnitude=_MIN_MAGNITUDE))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


def test_empty_test_results_produces_no_correction() -> None:
    """DetectionRecord with no test results produces no correction."""
    dr = _det_result([])
    result = dr.normalize(NormalizationConfig(consensus="any"))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


def test_difference_mode_neutral_is_zero() -> None:
    """In difference mode, a non-significant detection leaves the series unchanged."""
    dr = _det_result([_tr(sig=False)], factor=0.0, mode="difference")
    result = dr.normalize(NormalizationConfig(consensus="any"))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


# ── year range filter (via YearBetween predicate) ────────────────────────────


def test_year_between_rejects_early_break() -> None:
    """YearBetween(min=2000) blocks a break at year 1995."""
    dr = _det_result([_tr(break_year=1995)])
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=YearBetween(min=2000)))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


def test_year_between_accepts_break_on_lower_boundary() -> None:
    """YearBetween(min=2000) accepts a break exactly at year 2000."""
    dr = _det_result([_tr(break_year=2000)])
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=YearBetween(min=2000)))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


def test_year_between_rejects_late_break() -> None:
    """YearBetween(max=2000) blocks a break at year 2008."""
    dr = _det_result([_tr(break_year=2008)])
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=YearBetween(max=2000)))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


def test_year_between_accepts_break_on_upper_boundary() -> None:
    """YearBetween(max=2000) accepts a break exactly at year 2000."""
    dr = _det_result([_tr(break_year=2000)])
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=YearBetween(max=2000)))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


# ── station filter (via StationIn predicate) ─────────────────────────────────


def test_station_in_skips_unlisted_station() -> None:
    """StationIn({"S2"}) blocks corrections for station S1."""
    dr = _det_result([_tr()])
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=StationIn({"S2"})))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


def test_station_in_corrects_listed_station() -> None:
    """StationIn({"S1"}) allows corrections for station S1."""
    dr = _det_result([_tr()])
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=StationIn({"S1"})))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


# ── predicate composition ─────────────────────────────────────────────────────


def test_predicate_blocks_correction() -> None:
    """A predicate that never matches prevents every correction."""
    dr = _det_result([_tr()])
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=YearBetween(min=9999)))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


def test_predicate_receives_correct_break_info() -> None:
    """Predicate receives a BreakInfo with the correct fields."""

    class _CapturePredicate(BreakPredicate):
        def __init__(self) -> None:
            self.calls: list[BreakInfo] = []

        def __call__(self, info: BreakInfo) -> bool:
            self.calls.append(info)
            return True

        def to_dict(self) -> dict:
            return {"type": "capture"}

        @classmethod
        def _from_dict(cls, d: dict) -> _CapturePredicate:  # noqa: ARG003
            return cls()

    capture = _CapturePredicate()
    dr = _det_result([_tr(break_year=2005)], factor=1.2)
    dr.normalize(NormalizationConfig(consensus="any", predicate=capture))
    assert len(capture.calls) == 1
    info = capture.calls[0]
    assert info.station_id == "S1"
    assert info.break_year == 2005  # noqa: PLR2004
    assert info.factor == pytest.approx(1.2)
    assert info.step == 1
    assert len(info.test_results) == 1


def test_predicate_and_composition() -> None:
    """AND-composed predicate requires both conditions."""
    dr = _det_result([_tr(break_year=2005)])
    # YearBetween passes (2005 in [2000,2010]) but StationIn blocks (S1 not in {S2})
    pred = YearBetween(min=2000, max=2010) & StationIn({"S2"})
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=pred))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


def test_predicate_or_composition() -> None:
    """OR-composed predicate passes when either condition holds."""
    dr = _det_result([_tr(break_year=2005)])
    # StationIn passes (S1 in {S1}), so OR passes regardless of year filter
    pred = YearBetween(min=9999) | StationIn({"S1"})
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=pred))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


def test_predicate_not_composition() -> None:
    """NOT-composed predicate inverts the result."""
    dr = _det_result([_tr(break_year=2005)])
    # ~StationIn({"S1"}) blocks S1
    pred = ~StationIn({"S1"})
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=pred))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


# ── overrides ────────────────────────────────────────────────────────────────


# ── StepIn predicate ─────────────────────────────────────────────────────────


def test_step_in_blocks_correction_from_excluded_step() -> None:
    """StepIn({2}) blocks a break detected at step 1."""
    dr = _det_result([_tr()])
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=StepIn({2})))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


def test_step_in_accepts_correction_from_included_step() -> None:
    """StepIn({1}) accepts a break detected at step 1."""
    dr = _det_result([_tr()])
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=StepIn({1})))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


# ── MagnitudeAbove predicate ──────────────────────────────────────────────────


def test_magnitude_above_blocks_small_correction() -> None:
    """MagnitudeAbove(0.5) blocks a factor=1.2 correction (magnitude 0.2)."""
    dr = _det_result([_tr()], factor=1.2)
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=MagnitudeAbove(threshold=0.5)))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


def test_magnitude_above_accepts_large_correction() -> None:
    """MagnitudeAbove(0.1) accepts a factor=1.2 correction (magnitude 0.2)."""
    dr = _det_result([_tr()], factor=1.2)
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=MagnitudeAbove(threshold=0.1)))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


def test_magnitude_above_difference_mode() -> None:
    """MagnitudeAbove uses |factor| in difference mode."""
    dr = _det_result([_tr()], factor=-0.8, mode="difference")
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=MagnitudeAbove(threshold=0.5)))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


# ── TestSignificant predicate ─────────────────────────────────────────────────


def test_test_significant_accepts_when_named_test_is_significant() -> None:
    """TestSignificant('snht') accepts when the snht test result is significant."""
    dr = _det_result([_tr(sig=True)])
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=TestSignificant("snht")))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


def test_test_significant_blocks_when_named_test_not_significant() -> None:
    """TestSignificant('snht') blocks when the snht test result is not significant."""
    dr = _det_result([_tr(sig=False)])
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=TestSignificant("snht")))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


def test_test_significant_blocks_when_test_name_does_not_match() -> None:
    """TestSignificant('buishand') blocks when only the snht result is present."""
    dr = _det_result([_tr(sig=True)])
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=TestSignificant("buishand")))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


# ── SignalAbove predicate ─────────────────────────────────────────────────────


def test_signal_above_accepts_when_signal_exceeds_threshold() -> None:
    """SignalAbove(1.5) accepts a break with relative_signal=2.0."""
    dr = _det_result([_tr(rel=2.0)])  # relative_signal = 2.0/1.0 = 2.0
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=SignalAbove(threshold=1.5)))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


def test_signal_above_blocks_when_signal_below_threshold() -> None:
    """SignalAbove(3.0) blocks a break with relative_signal=2.0."""
    dr = _det_result([_tr(rel=2.0)])
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=SignalAbove(threshold=3.0)))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


def test_signal_above_uses_max_across_tests() -> None:
    """SignalAbove uses the highest signal across all test results."""
    dr = _det_result([_tr(rel=1.2), _tr(rel=3.5)])  # max = 3.5
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=SignalAbove(threshold=3.0)))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


def test_signal_above_blocks_empty_test_results() -> None:
    """SignalAbove blocks when there are no test results."""
    dr = _det_result([])
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=SignalAbove(threshold=0.0)))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


# ── NSignificantAbove predicate ───────────────────────────────────────────────


def test_n_significant_above_accepts_when_count_met() -> None:
    """NSignificantAbove(2) accepts when two tests are significant."""
    dr = _det_result([_tr(sig=True), _tr(sig=True), _tr(sig=False)])
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=NSignificantAbove(n=2)))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


def test_n_significant_above_blocks_when_count_not_met() -> None:
    """NSignificantAbove(2) blocks when only one test is significant."""
    dr = _det_result([_tr(sig=True), _tr(sig=False), _tr(sig=False)])
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=NSignificantAbove(n=2)))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


# ── NeighborCountAbove predicate ──────────────────────────────────────────────


def test_neighbor_count_above_accepts_when_count_met() -> None:
    """NeighborCountAbove(3) accepts when 5 neighbors were used."""
    dr = _det_result([_tr()], n_neighbors=5)
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=NeighborCountAbove(n=3)))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


def test_neighbor_count_above_blocks_when_count_not_met() -> None:
    """NeighborCountAbove(6) blocks when only 5 neighbors were used."""
    dr = _det_result([_tr()], n_neighbors=5)
    result = dr.normalize(NormalizationConfig(consensus="any", predicate=NeighborCountAbove(n=6)))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


# ── overrides ────────────────────────────────────────────────────────────────


def test_overrides_replace_algorithmic_detections() -> None:
    """Overrides dict replaces algorithmic detections for the given station."""
    dr = _det_result([_tr(sig=False)])  # algorithm would apply no correction
    result = dr.normalize(NormalizationConfig(overrides={"S1": [(2003, 1.15)]}))
    corrs = result.station_results["S1"].corrections
    assert len(corrs) == _ONE_CORRECTION
    _override_year = 2003
    _override_factor = 1.15
    assert corrs[0].break_year == _override_year
    assert corrs[0].factor == pytest.approx(_override_factor)


def test_overrides_do_not_affect_other_stations() -> None:
    """Overrides only applies to the named station; others use algorithmic detections."""
    dr = _det_result([_tr()])
    result = dr.normalize(NormalizationConfig(consensus="any", overrides={"S99": [(2000, 1.1)]}))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


# ── max_corrections_per_station ──────────────────────────────────────────────


def _det_result_two_breaks(factor: float = 1.2) -> DetectionResult:
    """DetectionResult with two independent detected breaks at steps 1 and 2."""
    years = pl.Series("year", list(range(1990, 2010)))
    rec1 = DetectionRecord(
        step=1, break_year=1998, factor=factor, test_results=[_tr(break_year=1998)], was_applied=True
    )
    rec2 = DetectionRecord(
        step=2, break_year=2005, factor=factor, test_results=[_tr(break_year=2005)], was_applied=True
    )
    nbrs = [NeighborInfo(station_id="N0", distance_km=None, correlation=0.8, weight=0.64)]
    sd = StationDetection(
        station_id="S1",
        group="HCC6",
        annual_original=pl.Series("S1", [100.0] * 20),
        annual_corrected=pl.Series("S1", [100.0] * 20),
        years=years,
        detections_by_step={1: rec1, 2: rec2},
        neighbors_by_step={1: nbrs, 2: nbrs},
        corrections=[],
    )
    return DetectionResult(station_detections={"S1": sd}, parameter="test")


def test_max_corrections_per_station_cap() -> None:
    """max_corrections_per_station=1 limits output to a single correction even when two are detected."""
    dr = _det_result_two_breaks()
    result = dr.normalize(NormalizationConfig(consensus="any", max_corrections_per_station=1))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


def test_max_corrections_per_station_default_allows_two() -> None:
    """Default max_corrections_per_station=2 allows both breaks through."""
    dr = _det_result_two_breaks()
    result = dr.normalize(NormalizationConfig(consensus="any"))
    assert len(result.station_results["S1"].corrections) == 2  # noqa: PLR2004


# ── edge-effect guard (min_years_from_end) ───────────────────────────────────


def _tr_edge(*, break_year: int, segment_start: int, segment_end: int, sig: bool = True) -> TestResult:
    """TestResult with explicit segment boundaries for edge-guard testing."""
    return TestResult(
        test_name="snht",
        is_significant=sig,
        break_year=break_year,
        test_statistic=2.0,
        critical_value=1.0,
        n_years=segment_end - segment_start + 1,
        segment_start=segment_start,
        segment_end=segment_end,
    )


def _det_result_edge(tr: TestResult) -> DetectionResult:
    years = pl.Series("year", list(range(tr.segment_start, tr.segment_end + 1)))
    rec = DetectionRecord(step=1, break_year=tr.break_year, factor=1.2, test_results=[tr], was_applied=True)
    nbrs = [NeighborInfo(station_id="N0", distance_km=None, correlation=0.8, weight=0.64)]
    sd = StationDetection(
        station_id="S1",
        group="I1",
        annual_original=pl.Series("S1", [100.0] * len(years)),
        annual_corrected=pl.Series("S1", [100.0] * len(years)),
        years=years,
        detections_by_step={1: rec},
        neighbors_by_step={1: nbrs},
        corrections=[],
    )
    return DetectionResult(station_detections={"S1": sd}, parameter="test")


def test_edge_guard_rejects_break_too_close_to_start() -> None:
    """Break within min_years_from_end of segment_start is rejected as an edge artefact."""
    tr = _tr_edge(break_year=1994, segment_start=1990, segment_end=2020)
    # 1994 - 1990 = 4 < min_years_from_end=5 → rejected
    dr = _det_result_edge(tr)
    result = dr.normalize(NormalizationConfig(consensus="any", min_years_from_end=5))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


def test_edge_guard_rejects_break_too_close_to_end() -> None:
    """Break within min_years_from_end of segment_end is rejected as an edge artefact."""
    tr = _tr_edge(break_year=2017, segment_start=1990, segment_end=2020)
    # 2020 - 2017 + 1 = 4 < min_years_from_end=5 → rejected
    dr = _det_result_edge(tr)
    result = dr.normalize(NormalizationConfig(consensus="any", min_years_from_end=5))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


def test_edge_guard_accepts_break_exactly_at_min_distance() -> None:
    """Break exactly min_years_from_end away from the boundary is accepted."""
    tr = _tr_edge(break_year=1995, segment_start=1990, segment_end=2020)
    # 1995 - 1990 = 5 >= 5, and 2020 - 1995 + 1 = 26 >= 5 → accepted
    dr = _det_result_edge(tr)
    result = dr.normalize(NormalizationConfig(consensus="any", min_years_from_end=5))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


def test_edge_guard_zero_disables_guard() -> None:
    """min_years_from_end=0 disables the edge guard and accepts any break year."""
    tr = _tr_edge(break_year=1990, segment_start=1990, segment_end=2000)
    dr = _det_result_edge(tr)
    result = dr.normalize(NormalizationConfig(consensus="any", min_years_from_end=0))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


def test_edge_guard_applies_to_strongest_signal_consensus() -> None:
    """Edge guard is enforced even when consensus='strongest_signal'."""
    tr = _tr_edge(break_year=1991, segment_start=1990, segment_end=2020)
    # 1991 - 1990 = 1 < 5 → rejected regardless of consensus mode
    dr = _det_result_edge(tr)
    result = dr.normalize(NormalizationConfig(consensus="strongest_signal", min_years_from_end=5))
    assert len(result.station_results["S1"].corrections) == _NO_CORRECTIONS


# ── tiebreak with some-but-not-enough significant tests ──────────────────────


def test_tiebreak_strongest_signal_fires_on_partial_consensus() -> None:
    """tiebreak='strongest_signal' uses the best significant test when consensus fails."""
    # 3 tests, 1 significant — majority requires 2, so consensus fails
    dr = _det_result([_tr(rel=3.0), _tr(sig=False, rel=0.5), _tr(sig=False, rel=0.5)])
    result = dr.normalize(NormalizationConfig(consensus="majority", tiebreak="strongest_signal"))
    assert len(result.station_results["S1"].corrections) == _ONE_CORRECTION


# ── predicate serialization round-trips ──────────────────────────────────────


def test_year_between_serialization_roundtrip() -> None:
    """YearBetween survives a to_dict / from_dict round-trip."""
    p = YearBetween(min=1960, max=2010)
    assert BreakPredicate.from_dict(p.to_dict()) == p


def test_year_between_no_bounds_roundtrip() -> None:
    """YearBetween with no bounds survives a to_dict / from_dict round-trip."""
    p = YearBetween()
    assert BreakPredicate.from_dict(p.to_dict()) == p


def test_station_in_serialization_roundtrip() -> None:
    """StationIn survives a to_dict / from_dict round-trip."""
    p = StationIn({"S1", "S3"})
    assert BreakPredicate.from_dict(p.to_dict()) == p


def test_step_in_serialization_roundtrip() -> None:
    """StepIn survives a to_dict / from_dict round-trip."""
    p = StepIn({1, 3, 5})
    assert BreakPredicate.from_dict(p.to_dict()) == p


def test_test_significant_serialization_roundtrip() -> None:
    """TestSignificant survives a to_dict / from_dict round-trip."""
    p = TestSignificant("snht")
    assert BreakPredicate.from_dict(p.to_dict()) == p


def test_magnitude_above_serialization_roundtrip() -> None:
    """MagnitudeAbove survives a to_dict / from_dict round-trip."""
    p = MagnitudeAbove(threshold=0.05)
    assert BreakPredicate.from_dict(p.to_dict()) == p


def test_signal_above_serialization_roundtrip() -> None:
    """SignalAbove survives a to_dict / from_dict round-trip."""
    p = SignalAbove(threshold=1.5)
    assert BreakPredicate.from_dict(p.to_dict()) == p


def test_signal_above_direct_empty_test_results_returns_false() -> None:
    """SignalAbove returns False when called directly with no test results."""
    info = BreakInfo(station_id="S", break_year=2000, factor=1.2, step=1, mode="ratio", test_results=[])
    assert SignalAbove(threshold=0.0)(info) is False


def test_n_significant_above_serialization_roundtrip() -> None:
    """NSignificantAbove survives a to_dict / from_dict round-trip."""
    p = NSignificantAbove(n=3)
    assert BreakPredicate.from_dict(p.to_dict()) == p


def test_neighbor_count_above_serialization_roundtrip() -> None:
    """NeighborCountAbove survives a to_dict / from_dict round-trip."""
    p = NeighborCountAbove(n=4)
    assert BreakPredicate.from_dict(p.to_dict()) == p


def test_and_predicate_serialization_roundtrip() -> None:
    """AND composite predicate survives a to_dict / from_dict round-trip."""
    p = YearBetween(min=1960) & SignalAbove(threshold=1.5)
    p2 = BreakPredicate.from_dict(p.to_dict())
    info = BreakInfo(station_id="S", break_year=1980, factor=1.2, step=1, mode="ratio", test_results=[_tr(rel=2.0)])
    assert p2(info) == p(info)


def test_or_predicate_serialization_roundtrip() -> None:
    """OR composite predicate survives a to_dict / from_dict round-trip."""
    p = YearBetween(max=1960) | YearBetween(min=2000)
    p2 = BreakPredicate.from_dict(p.to_dict())
    info_early = BreakInfo(station_id="S", break_year=1950, factor=1.2, step=1, mode="ratio", test_results=[])
    info_mid = BreakInfo(station_id="S", break_year=1980, factor=1.2, step=1, mode="ratio", test_results=[])
    assert p2(info_early) is True
    assert p2(info_mid) is False


def test_not_predicate_serialization_roundtrip() -> None:
    """NOT composite predicate survives a to_dict / from_dict round-trip."""
    p = ~StepIn({1, 2})
    p2 = BreakPredicate.from_dict(p.to_dict())
    info_step1 = BreakInfo(station_id="S", break_year=2000, factor=1.2, step=1, mode="ratio", test_results=[])
    info_step3 = BreakInfo(station_id="S", break_year=2000, factor=1.2, step=3, mode="ratio", test_results=[])
    assert p2(info_step1) is False
    assert p2(info_step3) is True


def test_from_dict_unknown_type_raises() -> None:
    """from_dict raises ValueError for an unregistered predicate type."""
    with pytest.raises(ValueError, match="Unknown predicate type"):
        BreakPredicate.from_dict({"type": "does_not_exist"})
