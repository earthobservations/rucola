from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from rucola._algorithms import CorrectionMode, GroupLabel, NeighborInfo
    from rucola._homogeneity import TestResult
    from rucola._normalization import NormalizationConfig


@dataclass
class CorrectionRecord:
    """One correction applied to a station's series."""

    step: int  # procedure step (1–6) at which the correction was made
    break_year: int  # first year of the post-break (corrected-to) segment
    factor: float  # multiplicative factor applied to pre-break values


@dataclass
class StationResult:
    """Full homogenization result for one station."""

    station_id: str
    group: GroupLabel  # final classification, e.g. "H2", "HC3", "HCC6"
    corrections: list[CorrectionRecord]
    neighbors_by_step: dict[int, list[NeighborInfo]]
    annual_original: pl.Series
    annual_corrected: pl.Series
    years: pl.Series

    @property
    def n_corrections(self) -> int:
        return len(self.corrections)

    @property
    def is_homogeneous(self) -> bool:
        return self.group.startswith("H")

    @property
    def break_years(self) -> list[int]:
        return [c.break_year for c in self.corrections]

    __hash__ = None  # mutable dataclass — not hashable

    def __repr__(self) -> str:
        return f"StationResult({self.station_id!r}, group={self.group!r}, corrections={self.n_corrections})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StationResult):
            return NotImplemented
        return (
            self.station_id == other.station_id
            and self.group == other.group
            and self.corrections == other.corrections
            and self.neighbors_by_step == other.neighbors_by_step
            and self.annual_original.equals(other.annual_original)
            and self.annual_corrected.equals(other.annual_corrected)
            and self.years.equals(other.years)
        )

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "station_id": self.station_id,
            "group": self.group,
            "corrections": [{"step": c.step, "break_year": c.break_year, "factor": c.factor} for c in self.corrections],
            "neighbors_by_step": {
                str(k): [
                    {
                        "station_id": n.station_id,
                        "distance_km": n.distance_km,
                        "correlation": n.correlation,
                        "weight": n.weight,
                    }
                    for n in v
                ]
                for k, v in self.neighbors_by_step.items()
            },
            "annual_original": self.annual_original.to_list(),
            "annual_corrected": self.annual_corrected.to_list(),
            "years": self.years.to_list(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> StationResult:
        """Deserialise from a dict produced by to_dict."""
        from rucola._algorithms import NeighborInfo  # noqa: PLC0415

        sid = d["station_id"]
        return cls(
            station_id=sid,
            group=d["group"],
            corrections=[CorrectionRecord(**c) for c in d.get("corrections", [])],
            neighbors_by_step={
                int(k): [NeighborInfo(**n) for n in v] for k, v in d.get("neighbors_by_step", {}).items()
            },
            annual_original=pl.Series(sid, d["annual_original"]),
            annual_corrected=pl.Series(sid, d["annual_corrected"]),
            years=pl.Series("year", d["years"]),
        )


@dataclass
class HomogenizationResult:
    """Full 6-step homogenization result for all tested stations."""

    station_results: dict[str, StationResult]
    parameter: str

    @property
    def corrections(self) -> pl.DataFrame:
        """All applied corrections across all stations, flat DataFrame.

        Columns: station_id, step, break_year, factor.
        """
        records = [
            {
                "station_id": sid,
                "step": c.step,
                "break_year": c.break_year,
                "factor": round(c.factor, 4),
            }
            for sid, res in self.station_results.items()
            for c in res.corrections
        ]
        if not records:
            return pl.DataFrame(
                schema={
                    "station_id": pl.String,
                    "step": pl.Int32,
                    "break_year": pl.Int32,
                    "factor": pl.Float64,
                },
            )
        return pl.DataFrame(records).sort("station_id", "step", "break_year")

    @property
    def summary(self) -> pl.DataFrame:
        """One row per tested station.

        Sorted by n_corrections descending. Columns: station_id, group,
        n_corrections, break_years, n_neighbors (from last step used).
        """
        records = [
            {
                "station_id": sid,
                "group": res.group,
                "n_corrections": res.n_corrections,
                "break_years": res.break_years,
                "n_neighbors": len(res.neighbors_by_step[max(res.neighbors_by_step)]) if res.neighbors_by_step else 0,
            }
            for sid, res in self.station_results.items()
        ]
        if not records:
            return pl.DataFrame(
                schema={
                    "station_id": pl.String,
                    "group": pl.String,
                    "n_corrections": pl.Int32,
                    "break_years": pl.List(pl.Int32),
                    "n_neighbors": pl.Int32,
                },
            )
        return pl.DataFrame(records).sort("n_corrections", "station_id", descending=[True, False])

    @property
    def group_counts(self) -> pl.DataFrame:
        """Count of stations per final group label."""
        return self.summary.group_by("group").len().rename({"len": "n_stations"}).sort("group")

    def __repr__(self) -> str:
        n = len(self.station_results)
        n_h = sum(1 for r in self.station_results.values() if r.is_homogeneous)
        n_corr = sum(r.n_corrections for r in self.station_results.values())
        return (
            f"HomogenizationResult("
            f"parameter={self.parameter!r}, "
            f"{n} stations: {n_h} homogeneous, "
            f"{n - n_h} corrected/remaining, "
            f"{n_corr} total corrections)"
        )

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "_version": 1,
            "parameter": self.parameter,
            "station_results": {sid: sr.to_dict() for sid, sr in self.station_results.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> HomogenizationResult:
        """Deserialise from a dict produced by to_dict."""
        return cls(
            parameter=d["parameter"],
            station_results={sid: StationResult.from_dict(sr) for sid, sr in d["station_results"].items()},
        )

    def to_json(self, path: str | Path) -> None:
        """Write to a JSON file at path."""
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_json(cls, path: str | Path) -> HomogenizationResult:
        """Load from a JSON file written by to_json."""
        return cls.from_dict(json.loads(Path(path).read_text()))


# ── Detection-phase result types ─────────────────────────────────────────────


@dataclass
class DetectionRecord:
    """All test results for one station at one procedure step."""

    step: int
    break_year: int | None
    factor: float
    test_results: list[TestResult] = field(default_factory=list)
    was_applied: bool = False

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "step": self.step,
            "break_year": self.break_year,
            "factor": self.factor,
            "test_results": [r.to_dict() for r in self.test_results],
            "was_applied": self.was_applied,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DetectionRecord:
        """Deserialise from a dict produced by to_dict."""
        from rucola._homogeneity import TestResult  # noqa: PLC0415

        return cls(
            step=d["step"],
            break_year=d["break_year"],
            factor=d["factor"],
            test_results=[TestResult.from_dict(r) for r in d.get("test_results", [])],
            was_applied=d.get("was_applied", False),
        )


@dataclass
class StationDetection:
    """Raw detection data for one station across all procedure steps."""

    station_id: str
    group: GroupLabel
    annual_original: pl.Series
    annual_corrected: pl.Series
    years: pl.Series
    detections_by_step: dict[int, DetectionRecord] = field(default_factory=dict)
    neighbors_by_step: dict[int, list[NeighborInfo]] = field(default_factory=dict)
    corrections: list[CorrectionRecord] = field(default_factory=list)

    __hash__ = None  # mutable dataclass — not hashable

    def __repr__(self) -> str:
        n_years = self.annual_original.drop_nulls().len()
        steps = sorted(self.neighbors_by_step)
        return (
            f"StationDetection({self.station_id!r}, group={self.group!r}, "
            f"years={n_years}, corrections={len(self.corrections)}, steps_tested={steps})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StationDetection):
            return NotImplemented
        return (
            self.station_id == other.station_id
            and self.group == other.group
            and self.detections_by_step == other.detections_by_step
            and self.corrections == other.corrections
            and self.annual_original.equals(other.annual_original)
            and self.annual_corrected.equals(other.annual_corrected)
            and self.years.equals(other.years)
        )

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "station_id": self.station_id,
            "group": self.group,
            "annual_original": self.annual_original.to_list(),
            "annual_corrected": self.annual_corrected.to_list(),
            "years": self.years.to_list(),
            "detections_by_step": {str(k): v.to_dict() for k, v in self.detections_by_step.items()},
            "neighbors_by_step": {
                str(k): [
                    {"station_id": n.station_id, "distance_km": n.distance_km,
                     "correlation": n.correlation, "weight": n.weight}
                    for n in v
                ]
                for k, v in self.neighbors_by_step.items()
            },
            "corrections": [{"step": c.step, "break_year": c.break_year, "factor": c.factor} for c in self.corrections],
        }

    @classmethod
    def from_dict(cls, d: dict) -> StationDetection:
        """Deserialise from a dict produced by to_dict."""
        from rucola._algorithms import NeighborInfo  # noqa: PLC0415

        return cls(
            station_id=d["station_id"],
            group=d["group"],
            annual_original=pl.Series(d["station_id"], d["annual_original"]),
            annual_corrected=pl.Series(d["station_id"], d["annual_corrected"]),
            years=pl.Series("year", d["years"]),
            detections_by_step={
                int(k): DetectionRecord.from_dict(v) for k, v in d.get("detections_by_step", {}).items()
            },
            neighbors_by_step={
                int(k): [NeighborInfo(**n) for n in v] for k, v in d.get("neighbors_by_step", {}).items()
            },
            corrections=[CorrectionRecord(**c) for c in d.get("corrections", [])],
        )


@dataclass
class DetectionResult:
    """Raw output of Rucola.run() — all test results before normalization."""

    station_detections: dict[str, StationDetection]
    parameter: str
    mode: CorrectionMode = "ratio"

    def normalize(self, config: NormalizationConfig | None = None) -> HomogenizationResult:
        """Apply corrections based on config and return a HomogenizationResult.

        Parameters
        ----------
        config :
            Normalization configuration. Defaults to NormalizationConfig() if not provided.

        """
        from rucola._normalization import NormalizationConfig as _Cfg  # noqa: PLC0415
        from rucola._normalization import Normalizer  # noqa: PLC0415

        return Normalizer(config or _Cfg()).apply(self)

    @property
    def summary(self) -> pl.DataFrame:
        """Per-station detection summary.

        Columns: station_id, group, n_steps_tested, n_significant, n_applied.
        Sorted by n_significant descending.
        """
        records = []
        for sid, det in self.station_detections.items():
            n_significant = sum(
                1 for rec in det.detections_by_step.values() for tr in rec.test_results if tr.is_significant
            )
            records.append(
                {
                    "station_id": sid,
                    "group": det.group,
                    "n_steps_tested": len(det.detections_by_step),
                    "n_significant": n_significant,
                    "n_applied": len(det.corrections),
                }
            )
        if not records:
            return pl.DataFrame(
                schema={
                    "station_id": pl.String,
                    "group": pl.String,
                    "n_steps_tested": pl.Int32,
                    "n_significant": pl.Int32,
                    "n_applied": pl.Int32,
                }
            )
        return pl.DataFrame(records).sort("n_significant", "station_id", descending=[True, False])

    def to_markdown(self) -> str:  # noqa: C901
        """Render detection results as a Markdown document.

        Produces a summary table followed by a per-station section. Each
        station section contains a step overview and — when multiple tests are
        run or tests disagree on the break year — a per-step detail table
        showing each test's result individually.
        """

        def _step_label(step: int) -> str:
            return {61: "6a", 62: "6b", 63: "6c"}.get(step, str(step))

        def _row(*cells: str) -> str:
            return "| " + " | ".join(cells) + " |"

        lines: list[str] = [
            f"# Detection Results — {self.parameter}",
            "",
            "## Summary",
            "",
            _row("station_id", "group", "steps_tested", "n_significant", "breaks_applied"),
            _row(*["---"] * 5),
        ]

        ordered = sorted(
            self.station_detections.items(),
            key=lambda x: len(x[1].corrections),
            reverse=True,
        )

        for sid, det in ordered:
            n_sig = sum(1 for rec in det.detections_by_step.values() for tr in rec.test_results if tr.is_significant)
            lines.append(
                _row(
                    sid,
                    det.group or "—",
                    str(len(det.detections_by_step)),
                    str(n_sig),
                    str(len(det.corrections)),
                )
            )

        lines += ["", "---", ""]

        for sid, det in ordered:
            if not det.detections_by_step:
                continue

            lines.append(f"## {sid}  ·  {det.group or '—'}")
            lines.append("")
            lines += [
                _row("step", "break_year", "applied"),
                _row(*["---"] * 3),
            ]
            for step, rec in sorted(det.detections_by_step.items()):
                year_str = "—" if rec.break_year is None else str(rec.break_year)
                lines.append(_row(_step_label(step), year_str, "yes" if rec.was_applied else "no"))
            lines.append("")

            for step, rec in sorted(det.detections_by_step.items()):
                if not rec.test_results:
                    continue

                break_years = {tr.break_year for tr in rec.test_results if tr.is_significant}
                status = "applied" if rec.was_applied else "not applied"
                heading = f"### Step {_step_label(step)} · {status}"
                if len(break_years) > 1:
                    heading += f" · ⚠ year disagreement {sorted(break_years)}"
                lines += [heading, ""]
                lines += [
                    _row("test", "significant", "break_year", "relative_signal"),
                    _row(*["---"] * 4),
                ]
                for tr in rec.test_results:
                    sig = "**yes**" if tr.is_significant else "no"
                    lines.append(_row(tr.test_name, sig, str(tr.break_year), f"{tr.relative_signal:.3f}"))
                lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "_version": 1,
            "parameter": self.parameter,
            "mode": self.mode,
            "station_detections": {sid: sd.to_dict() for sid, sd in self.station_detections.items()},
        }

    def to_json(self, path: str | Path) -> None:
        """Write to a JSON file at path."""
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_dict(cls, d: dict) -> DetectionResult:
        """Deserialise from a dict produced by to_dict."""
        return cls(
            parameter=d["parameter"],
            mode=d.get("mode", "ratio"),
            station_detections={sid: StationDetection.from_dict(sd) for sid, sd in d["station_detections"].items()},
        )

    @classmethod
    def from_json(cls, path: str | Path) -> DetectionResult:
        """Load from a JSON file written by to_json."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __repr__(self) -> str:
        """Return short summary string."""
        n = len(self.station_detections)
        total = sum(len(sd.corrections) for sd in self.station_detections.values())
        return f"DetectionResult(parameter={self.parameter!r}, stations={n}, corrections_applied={total})"
