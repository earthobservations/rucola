"""Tests for DetectionResult and HomogenizationResult JSON serialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

import rucola
from rucola._algorithms import NeighborInfo
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
    from pathlib import Path

# ── unit-level fixtures (no pipeline) ────────────────────────────────────────

_YEARS = list(range(1990, 2010))
_N = len(_YEARS)


def _minimal_detection_result() -> DetectionResult:
    tr = TestResult(
        test_name="snht",
        is_significant=True,
        break_year=2005,
        test_statistic=12.0,
        critical_value=10.0,
        n_years=_N,
        segment_start=1990,
        segment_end=2009,
    )
    rec = DetectionRecord(step=1, break_year=2005, factor=1.15, test_results=[tr], was_applied=True)
    nbrs = [NeighborInfo(station_id="N1", distance_km=50.0, correlation=0.9, weight=0.81)]
    sd = StationDetection(
        station_id="S1",
        group="HC3",
        annual_original=pl.Series("S1", [100.0] * _N),
        annual_corrected=pl.Series("S1", [115.0] * 15 + [100.0] * 5),
        years=pl.Series("year", _YEARS),
        detections_by_step={1: rec},
        neighbors_by_step={1: nbrs},
        corrections=[CorrectionRecord(step=1, break_year=2005, factor=1.15)],
    )
    return DetectionResult(station_detections={"S1": sd}, parameter="precip", mode="ratio")


def _minimal_homogenization_result() -> HomogenizationResult:
    nbrs = [NeighborInfo(station_id="N1", distance_km=50.0, correlation=0.9, weight=0.81)]
    sr = StationResult(
        station_id="S1",
        group="HC3",
        corrections=[CorrectionRecord(step=1, break_year=2005, factor=1.15)],
        neighbors_by_step={1: nbrs},
        annual_original=pl.Series("S1", [100.0] * _N),
        annual_corrected=pl.Series("S1", [115.0] * 15 + [100.0] * 5),
        years=pl.Series("year", _YEARS),
    )
    return HomogenizationResult(station_results={"S1": sr}, parameter="precip")


# ── unit: to_json / from_json ─────────────────────────────────────────────────


@pytest.mark.unit
def test_detection_result_to_json_from_json_unit(tmp_path: Path) -> None:
    """DetectionResult.to_json / from_json round-trip without running the pipeline."""
    dr = _minimal_detection_result()
    path = tmp_path / "detection.json"
    dr.to_json(path)
    assert path.exists()
    loaded = DetectionResult.from_json(path)
    assert loaded.parameter == dr.parameter
    assert loaded.mode == dr.mode
    assert set(loaded.station_detections) == {"S1"}
    s = loaded.station_detections["S1"]
    assert s.group == "HC3"
    assert s.annual_original.to_list() == pytest.approx([100.0] * _N)
    assert len(s.corrections) == 1
    assert s.corrections[0].break_year == 2005  # noqa: PLR2004


@pytest.mark.unit
def test_homogenization_result_to_json_from_json_unit(tmp_path: Path) -> None:
    """HomogenizationResult.to_json / from_json round-trip without running the pipeline."""
    hr = _minimal_homogenization_result()
    path = tmp_path / "homogenization.json"
    hr.to_json(path)
    assert path.exists()
    loaded = HomogenizationResult.from_json(path)
    assert loaded.parameter == hr.parameter
    assert set(loaded.station_results) == {"S1"}
    s = loaded.station_results["S1"]
    assert s.group == "HC3"
    assert s.annual_corrected.to_list() == pytest.approx([115.0] * 15 + [100.0] * 5)
    assert s.corrections[0].factor == pytest.approx(1.15)


@pytest.mark.unit
def test_detection_result_to_json_file_is_valid_json(tmp_path: Path) -> None:
    """to_json writes parseable JSON (no binary blobs, no truncation)."""
    import json  # noqa: PLC0415

    dr = _minimal_detection_result()
    path = tmp_path / "detection.json"
    dr.to_json(path)
    data = json.loads(path.read_text())
    assert "station_detections" in data
    assert "parameter" in data


@pytest.mark.unit
def test_homogenization_result_neighbors_survive_roundtrip(tmp_path: Path) -> None:
    """Neighbor metadata (correlation, distance) is preserved through to_json / from_json."""
    hr = _minimal_homogenization_result()
    path = tmp_path / "hr.json"
    hr.to_json(path)
    loaded = HomogenizationResult.from_json(path)
    nbrs = loaded.station_results["S1"].neighbors_by_step[1]
    assert len(nbrs) == 1
    assert nbrs[0].station_id == "N1"
    assert nbrs[0].correlation == pytest.approx(0.9)
    assert nbrs[0].distance_km == pytest.approx(50.0)


@pytest.mark.unit
def test_detection_result_neighbors_survive_roundtrip(tmp_path: Path) -> None:
    """Neighbor metadata is preserved through DetectionResult to_json / from_json."""
    dr = _minimal_detection_result()
    path = tmp_path / "dr_nbrs.json"
    dr.to_json(path)
    loaded = DetectionResult.from_json(path)
    nbrs = loaded.station_detections["S1"].neighbors_by_step[1]
    assert len(nbrs) == 1
    assert nbrs[0].station_id == "N1"
    assert nbrs[0].correlation == pytest.approx(0.9)
    assert nbrs[0].distance_km == pytest.approx(50.0)


# ── unit: to_markdown ────────────────────────────────────────────────────────

# Rich fixture: two stations, two tests per step, disagreement, applied/not-applied.
#
# S1 (HC3, 1 correction):
#   Step 1 applied — snht significant (2005, rel=1.500), buishand not (2003, rel=0.800)
#
# S2 (H2, 0 corrections):
#   Step 1 not applied — snht significant (1998, rel=1.200), buishand significant (2004, rel=1.100)
#   Both tests significant but disagree on year → year-disagreement warning.


def _tr(*, name: str, sig: bool, break_year: int, stat: float, crit: float = 10.0) -> TestResult:
    return TestResult(
        test_name=name,
        is_significant=sig,
        break_year=break_year,
        test_statistic=stat,
        critical_value=crit,
        n_years=_N,
        segment_start=_YEARS[0],
        segment_end=_YEARS[-1],
    )


def _rich_detection_result() -> DetectionResult:
    nbrs = [NeighborInfo(station_id="N1", distance_km=50.0, correlation=0.9, weight=0.81)]

    sd_s1 = StationDetection(
        station_id="S1",
        group="HC3",
        annual_original=pl.Series("S1", [100.0] * _N),
        annual_corrected=pl.Series("S1", [115.0] * 15 + [100.0] * 5),
        years=pl.Series("year", _YEARS),
        detections_by_step={
            1: DetectionRecord(
                step=1,
                break_year=2005,
                factor=1.15,
                was_applied=True,
                test_results=[
                    _tr(name="snht",     sig=True,  break_year=2005, stat=15.0),
                    _tr(name="buishand", sig=False, break_year=2003, stat=8.0),
                ],
            ),
        },
        neighbors_by_step={1: nbrs},
        corrections=[CorrectionRecord(step=1, break_year=2005, factor=1.15)],
    )

    sd_s2 = StationDetection(
        station_id="S2",
        group="H2",
        annual_original=pl.Series("S2", [100.0] * _N),
        annual_corrected=pl.Series("S2", [100.0] * _N),
        years=pl.Series("year", _YEARS),
        detections_by_step={
            1: DetectionRecord(
                step=1,
                break_year=1998,
                factor=1.0,
                was_applied=False,
                test_results=[
                    _tr(name="snht",     sig=True, break_year=1998, stat=12.0),
                    _tr(name="buishand", sig=True, break_year=2004, stat=11.0),
                ],
            ),
        },
        neighbors_by_step={1: nbrs},
        corrections=[],
    )

    return DetectionResult(
        station_detections={"S1": sd_s1, "S2": sd_s2},
        parameter="precip",
        mode="ratio",
    )


@pytest.mark.unit
def test_to_markdown_heading_contains_parameter() -> None:
    """H1 heading names the parameter."""
    assert "# Detection Results — precip" in _rich_detection_result().to_markdown()


@pytest.mark.unit
def test_to_markdown_summary_lists_both_stations() -> None:
    """Summary table contains a row for every station."""
    md = _rich_detection_result().to_markdown()
    assert "## Summary" in md
    assert "| S1 |" in md
    assert "| S2 |" in md


@pytest.mark.unit
def test_to_markdown_summary_ordering_corrections_descending() -> None:
    """Station with corrections appears before the homogeneous station in the summary."""
    md = _rich_detection_result().to_markdown()
    assert md.index("| S1 |") < md.index("| S2 |")


@pytest.mark.unit
def test_to_markdown_applied_step_shows_yes() -> None:
    """Step applied for S1 shows 'yes'."""
    assert "| 1 | 2005 | yes |" in _rich_detection_result().to_markdown()


@pytest.mark.unit
def test_to_markdown_not_applied_step_shows_no() -> None:
    """Step not applied for S2 shows 'no'."""
    md = _rich_detection_result().to_markdown()
    assert "| 1 | 1998 | no |" in md


@pytest.mark.unit
def test_to_markdown_per_step_detail_columns() -> None:
    """Per-step detail table has the four expected column headers."""
    assert "| test | significant | break_year | relative_signal |" in _rich_detection_result().to_markdown()


@pytest.mark.unit
def test_to_markdown_significant_test_is_bold() -> None:
    """A significant test is rendered **yes**; a non-significant one is plain 'no'."""
    md = _rich_detection_result().to_markdown()
    assert "**yes**" in md
    assert "| buishand | no |" in md


@pytest.mark.unit
def test_to_markdown_relative_signal_formatted() -> None:
    """Relative signal is formatted to three decimal places."""
    md = _rich_detection_result().to_markdown()
    assert "1.500" in md   # snht at S1: 15.0 / 10.0
    assert "0.800" in md   # buishand at S1: 8.0 / 10.0


@pytest.mark.unit
def test_to_markdown_year_disagreement_warning() -> None:
    """Step heading includes a year-disagreement warning when two significant tests disagree."""
    md = _rich_detection_result().to_markdown()
    assert "⚠ year disagreement" in md


@pytest.mark.unit
def test_to_markdown_station_section_uses_group_label() -> None:
    """Per-station heading includes the group label for both stations."""
    md = _rich_detection_result().to_markdown()
    assert "## S1  ·  HC3" in md
    assert "## S2  ·  H2" in md


@pytest.mark.unit
def test_to_markdown_step_6a_6b_labels() -> None:
    """Steps 61 and 62 are rendered as 6a and 6b."""
    dr = DetectionResult(
        station_detections={
            "X": StationDetection(
                station_id="X",
                group="HCC6",
                annual_original=pl.Series("X", [1.0] * _N),
                annual_corrected=pl.Series("X", [1.0] * _N),
                years=pl.Series("year", _YEARS),
                detections_by_step={
                    61: DetectionRecord(step=61, break_year=2005, factor=1.1, test_results=[], was_applied=True),
                    62: DetectionRecord(step=62, break_year=1998, factor=1.05, test_results=[], was_applied=True),
                },
                corrections=[
                    CorrectionRecord(step=6, break_year=2005, factor=1.1),
                    CorrectionRecord(step=6, break_year=1998, factor=1.05),
                ],
            )
        },
        parameter="p",
    )
    md = dr.to_markdown()
    assert "| 6a |" in md
    assert "| 6b |" in md


@pytest.mark.unit
def test_to_markdown_station_without_detections_excluded() -> None:
    """Stations with no detections_by_step have no per-station section."""
    dr = DetectionResult(
        station_detections={
            "BARE": StationDetection(
                station_id="BARE",
                group="INSUFFICIENT_DATA",
                annual_original=pl.Series("BARE", [1.0] * _N),
                annual_corrected=pl.Series("BARE", [1.0] * _N),
                years=pl.Series("year", _YEARS),
            )
        },
        parameter="p",
    )
    assert "## BARE" not in dr.to_markdown()


# ── slow: full-pipeline round-trips ───────────────────────────────────────────


@pytest.mark.slow
def test_detection_result_roundtrip(rucola_instance: rucola.Rucola, tmp_path: Path) -> None:
    """DetectionResult survives a to_json / from_json round-trip."""
    result = rucola_instance.run()
    path = tmp_path / "detection.json"
    result.to_json(path)
    loaded = rucola.DetectionResult.from_json(path)

    assert loaded.parameter == result.parameter
    assert loaded.mode == result.mode
    assert set(loaded.station_detections) == set(result.station_detections)
    for sid, orig in result.station_detections.items():
        restored = loaded.station_detections[sid]
        assert restored.group == orig.group
        assert len(restored.corrections) == len(orig.corrections)
        assert restored.annual_original.to_list() == orig.annual_original.to_list()


def test_homogenization_result_roundtrip(rucola_instance: rucola.Rucola, tmp_path: Path) -> None:
    """HomogenizationResult survives a to_json / from_json round-trip."""
    result = rucola_instance.run().normalize()
    path = tmp_path / "homogenization.json"
    result.to_json(path)
    loaded = rucola.HomogenizationResult.from_json(path)

    assert loaded.parameter == result.parameter
    assert set(loaded.station_results) == set(result.station_results)
    for sid, orig in result.station_results.items():
        restored = loaded.station_results[sid]
        assert restored.group == orig.group
        assert len(restored.corrections) == len(orig.corrections)
        assert restored.annual_corrected.to_list() == orig.annual_corrected.to_list()
