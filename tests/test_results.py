"""Tests for result dataclasses — StationResult, HomogenizationResult, DetectionResult."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from rucola._homogeneity import TestResult
from rucola._results import (
    CorrectionRecord,
    DetectionRecord,
    DetectionResult,
    HomogenizationResult,
    StationDetection,
    StationResult,
)

if TYPE_CHECKING:
    import rucola

pytestmark = pytest.mark.slow

_N_STATIONS = 5
_EXPECTED_COLUMNS_CORRECTIONS = {"station_id", "step", "break_year", "factor"}

# ── StationResult properties ────────────────────────────────────────────────


def test_station_result_n_corrections(rucola_instance: rucola.Rucola) -> None:
    """n_corrections equals the length of the corrections list."""
    result = rucola_instance.run().normalize()
    s1 = result.station_results["S1"]
    assert s1.n_corrections == len(s1.corrections)


def test_station_result_is_homogeneous(rucola_instance: rucola.Rucola) -> None:
    """is_homogeneous returns True iff the group label starts with H."""
    result = rucola_instance.run().normalize()
    s1 = result.station_results["S1"]
    assert s1.is_homogeneous == s1.group.startswith("H")


def test_station_result_break_years(rucola_instance: rucola.Rucola) -> None:
    """break_years has one entry per correction."""
    result = rucola_instance.run().normalize()
    s1 = result.station_results["S1"]
    assert isinstance(s1.break_years, list)
    assert len(s1.break_years) == s1.n_corrections


def test_station_result_repr(rucola_instance: rucola.Rucola) -> None:
    """StationResult repr contains the class name and station ID."""
    result = rucola_instance.run().normalize()
    r = repr(result.station_results["S1"])
    assert "StationResult" in r
    assert "S1" in r


def test_station_result_eq_roundtrip(rucola_instance: rucola.Rucola) -> None:
    """StationResult eq works via series_equal and survives a dict round-trip."""
    result = rucola_instance.run().normalize()
    s1 = result.station_results["S1"]
    assert s1 == StationResult.from_dict(s1.to_dict())


def test_station_detection_repr(rucola_instance: rucola.Rucola) -> None:
    """StationDetection repr is compact and contains the key fields."""
    result = rucola_instance.run()
    r = repr(result.station_detections["S1"])
    assert "StationDetection" in r
    assert "S1" in r
    assert "years=" in r
    assert "steps_tested=" in r


def test_station_detection_eq_roundtrip(rucola_instance: rucola.Rucola) -> None:
    """StationDetection eq works via series_equal and survives a dict round-trip."""
    result = rucola_instance.run()
    s1 = result.station_detections["S1"]
    assert s1 == StationDetection.from_dict(s1.to_dict())


# ── HomogenizationResult properties ────────────────────────────────────────


def test_homogenization_corrections_dataframe(rucola_instance: rucola.Rucola) -> None:
    """Corrections property returns a DataFrame with the expected columns."""
    result = rucola_instance.run().normalize()
    df = result.corrections
    assert set(df.columns) >= _EXPECTED_COLUMNS_CORRECTIONS


def test_homogenization_corrections_empty() -> None:
    """Corrections returns an empty but correctly-schemed DataFrame when no corrections exist."""
    sr = StationResult(
        station_id="X",
        group="H1",
        corrections=[],
        neighbors_by_step={},
        annual_original=pl.Series("X", [1.0]),
        annual_corrected=pl.Series("X", [1.0]),
        years=pl.Series("year", [2000]),
    )
    hr = HomogenizationResult(station_results={"X": sr}, parameter="p")
    df = hr.corrections
    assert len(df) == 0
    assert "station_id" in df.columns


def test_homogenization_summary(rucola_instance: rucola.Rucola) -> None:
    """Summary returns one row per station with n_corrections column."""
    result = rucola_instance.run().normalize()
    summary = result.summary
    assert "station_id" in summary.columns
    assert "n_corrections" in summary.columns
    assert len(summary) == _N_STATIONS


def test_homogenization_group_counts(rucola_instance: rucola.Rucola) -> None:
    """group_counts totals across all stations."""
    result = rucola_instance.run().normalize()
    gc = result.group_counts
    assert "group" in gc.columns
    assert gc["n_stations"].sum() == _N_STATIONS


def test_homogenization_repr(rucola_instance: rucola.Rucola) -> None:
    """HomogenizationResult repr contains the class name and parameter."""
    result = rucola_instance.run().normalize()
    r = repr(result)
    assert "HomogenizationResult" in r
    assert "precip" in r


# ── DetectionResult properties ──────────────────────────────────────────────


def test_detection_result_summary(rucola_instance: rucola.Rucola) -> None:
    """Summary returns one row per station with n_significant column."""
    result = rucola_instance.run()
    summary = result.summary
    assert "station_id" in summary.columns
    assert "n_significant" in summary.columns
    assert len(summary) == _N_STATIONS


def test_detection_result_repr(rucola_instance: rucola.Rucola) -> None:
    """DetectionResult repr contains the class name and parameter."""
    result = rucola_instance.run()
    r = repr(result)
    assert "DetectionResult" in r
    assert "precip" in r


def test_detection_result_to_markdown(rucola_instance: rucola.Rucola) -> None:
    """to_markdown returns a string with a Summary header and station rows."""
    result = rucola_instance.run()
    md = result.to_markdown()
    assert "## Summary" in md
    assert "S1" in md
    assert "| station_id" in md


def test_detection_result_to_dict_from_dict(rucola_instance: rucola.Rucola) -> None:
    """DetectionResult survives a to_dict / from_dict round-trip."""
    result = rucola_instance.run()
    d = result.to_dict()
    loaded = DetectionResult.from_dict(d)
    assert loaded.parameter == result.parameter
    assert set(loaded.station_detections) == set(result.station_detections)


def test_homogenization_result_to_dict_from_dict(rucola_instance: rucola.Rucola) -> None:
    """HomogenizationResult survives a to_dict / from_dict round-trip."""
    result = rucola_instance.run().normalize()
    d = result.to_dict()
    loaded = HomogenizationResult.from_dict(d)
    assert loaded.parameter == result.parameter
    assert set(loaded.station_results) == set(result.station_results)
    s1_orig = result.station_results["S1"]
    s1_loaded = loaded.station_results["S1"]
    assert s1_loaded.group == s1_orig.group
    assert s1_loaded.break_years == s1_orig.break_years


# ── DetectionRecord serialisation ────────────────────────────────────────────


def test_detection_record_to_dict_from_dict_roundtrip() -> None:
    """DetectionRecord survives a to_dict / from_dict round-trip."""
    tr = TestResult(
        test_name="snht",
        is_significant=True,
        break_year=2005,
        test_statistic=12.0,
        critical_value=10.0,
        n_years=20,
        segment_start=1990,
        segment_end=2009,
    )
    rec = DetectionRecord(step=4, break_year=2005, factor=1.15, test_results=[tr], was_applied=True)
    loaded = DetectionRecord.from_dict(rec.to_dict())
    assert loaded.step == rec.step
    assert loaded.break_year == rec.break_year
    assert loaded.factor == pytest.approx(rec.factor)
    assert loaded.was_applied == rec.was_applied
    assert len(loaded.test_results) == 1
    assert loaded.test_results[0].test_name == "snht"
    assert loaded.test_results[0].is_significant


# ── to_markdown ───────────────────────────────────────────────────────────────

# Helpers for constructing minimal DetectionResult objects without the pipeline.


def _tr(*, sig: bool = True, break_year: int = 2005, rel: float = 2.0) -> TestResult:
    return TestResult(
        test_name="snht",
        is_significant=sig,
        break_year=break_year,
        test_statistic=rel * 10.0,
        critical_value=10.0,
        n_years=20,
        segment_start=1990,
        segment_end=2009,
    )


def _make_detection_result(
    station_id: str = "X",
    group: str = "HC4",
    detections_by_step: dict | None = None,
    corrections: list | None = None,
    parameter: str = "my_param",
) -> DetectionResult:
    years = pl.Series("year", list(range(1990, 2010)))
    sd = StationDetection(
        station_id=station_id,
        group=group,
        annual_original=pl.Series(station_id, [1.0] * 20),
        annual_corrected=pl.Series(station_id, [1.0] * 20),
        years=years,
        detections_by_step=detections_by_step or {},
        corrections=corrections or [],
    )
    return DetectionResult(station_detections={station_id: sd}, parameter=parameter)


def test_to_markdown_heading_contains_parameter() -> None:
    """H1 heading contains the parameter name."""
    dr = _make_detection_result(parameter="precip_height")
    assert "# Detection Results — precip_height" in dr.to_markdown()


def test_to_markdown_per_step_detail_headers() -> None:
    """Per-step detail table has the expected column headers."""
    rec = DetectionRecord(step=1, break_year=2005, factor=1.1, test_results=[_tr()], was_applied=True)
    dr = _make_detection_result(
        detections_by_step={1: rec},
        corrections=[CorrectionRecord(step=1, break_year=2005, factor=1.1)],
    )
    md = dr.to_markdown()
    assert "| test | significant | break_year | relative_signal |" in md


def test_to_markdown_applied_shows_yes() -> None:
    """A step with was_applied=True shows 'yes' in the step overview table."""
    rec = DetectionRecord(step=1, break_year=2005, factor=1.1, test_results=[_tr()], was_applied=True)
    dr = _make_detection_result(
        detections_by_step={1: rec},
        corrections=[CorrectionRecord(step=1, break_year=2005, factor=1.1)],
    )
    assert "| 1 | 2005 | yes |" in dr.to_markdown()


def test_to_markdown_not_applied_shows_no() -> None:
    """A step with was_applied=False shows 'no' in the step overview table."""
    rec = DetectionRecord(step=2, break_year=2005, factor=1.0, test_results=[_tr(sig=False)], was_applied=False)
    dr = _make_detection_result(detections_by_step={2: rec})
    assert "| 2 | 2005 | no |" in dr.to_markdown()


def test_to_markdown_step_6a_6b_labels() -> None:
    """Steps 61 and 62 are rendered as '6a' and '6b' respectively."""
    rec_6a = DetectionRecord(step=61, break_year=2005, factor=1.1, test_results=[_tr()], was_applied=True)
    rec_6b = DetectionRecord(
        step=62, break_year=1998, factor=1.05, test_results=[_tr(break_year=1998)], was_applied=True
    )
    years = pl.Series("year", list(range(1990, 2010)))
    sd = StationDetection(
        station_id="X",
        group="HCC6",
        annual_original=pl.Series("X", [1.0] * 20),
        annual_corrected=pl.Series("X", [1.0] * 20),
        years=years,
        detections_by_step={61: rec_6a, 62: rec_6b},
        corrections=[
            CorrectionRecord(step=6, break_year=2005, factor=1.1),
            CorrectionRecord(step=6, break_year=1998, factor=1.05),
        ],
    )
    dr = DetectionResult(station_detections={"X": sd}, parameter="p")
    md = dr.to_markdown()
    assert "6a" in md
    assert "6b" in md


def test_to_markdown_year_disagreement_warning() -> None:
    """Heading includes a year-disagreement warning when two significant tests name different years."""
    tr_snht = _tr(break_year=2000)
    tr_buishand = TestResult(
        test_name="buishand",
        is_significant=True,
        break_year=2008,
        test_statistic=20.0,
        critical_value=10.0,
        n_years=20,
        segment_start=1990,
        segment_end=2009,
    )
    rec = DetectionRecord(step=1, break_year=2000, factor=1.1, test_results=[tr_snht, tr_buishand], was_applied=True)
    dr = _make_detection_result(
        detections_by_step={1: rec},
        corrections=[CorrectionRecord(step=1, break_year=2000, factor=1.1)],
    )
    assert "⚠ year disagreement" in dr.to_markdown()


def test_to_markdown_bold_when_significant() -> None:
    """A significant test result is rendered with bold **yes** in the detail table."""
    rec = DetectionRecord(step=1, break_year=2005, factor=1.1, test_results=[_tr(sig=True)], was_applied=True)
    dr = _make_detection_result(
        detections_by_step={1: rec},
        corrections=[CorrectionRecord(step=1, break_year=2005, factor=1.1)],
    )
    assert "**yes**" in dr.to_markdown()


def test_to_markdown_station_without_detections_skipped() -> None:
    """Stations with no detections_by_step are excluded from the per-station detail section."""
    dr = _make_detection_result(station_id="BARE", group="INSUFFICIENT_DATA", detections_by_step={})
    md = dr.to_markdown()
    assert "## BARE" not in md
