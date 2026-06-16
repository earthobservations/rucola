"""Normalization config, break predicates, and correction logic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from rucola._algorithms import CorrectionMode, apply_correction
from rucola._results import CorrectionRecord, DetectionRecord, HomogenizationResult, StationResult

if TYPE_CHECKING:
    from rucola._algorithms import NeighborInfo
    from rucola._homogeneity import TestResult
    from rucola._results import DetectionResult

ConsensusRule = Literal["unanimous", "majority", "any", "strongest_signal"]
Tiebreak = Literal["strongest_signal", "skip"]

_STEP_6C_DIAGNOSTIC = 63  # post-correction re-test; never applied as a correction

# ---------------------------------------------------------------------------
# BreakInfo — context passed to predicates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BreakInfo:
    """Context passed to a :class:`BreakPredicate` for each candidate correction.

    Parameters
    ----------
    station_id :
        The station being evaluated.
    break_year :
        Consensus break year resolved from the test results.
    factor :
        Correction factor (ratio: multiplicative; difference: additive).
    step :
        Procedure step (1–6) at which this break was detected.
    mode :
        Correction mode — ``"ratio"`` or ``"difference"``.
    test_results :
        Individual results from each homogeneity test, including each
        test's suggested break year and test statistic.
    n_neighbors :
        Number of reference stations used to build the reference series at
        this step. Used by :class:`NeighborCountAbove`.

    """

    station_id: str
    break_year: int
    factor: float
    step: int
    mode: CorrectionMode
    test_results: list[TestResult]
    n_neighbors: int = 0


# ---------------------------------------------------------------------------
# Predicate base and registry
# ---------------------------------------------------------------------------

_PRED_REGISTRY: dict[str, type[BreakPredicate]] = {}


def _register(name: str):  # noqa: ANN202
    def _decorator(cls: type) -> type:
        _PRED_REGISTRY[name] = cls
        cls._pred_type = name  # type: ignore[attr-defined]
        return cls

    return _decorator


class BreakPredicate(ABC):
    """Abstract base for composable, serializable break predicates.

    Combine predicates with ``&`` (AND), ``|`` (OR), and ``~`` (NOT):

    >>> from rucola import YearBetween, StationIn
    >>> p = YearBetween(min=1960, max=2010) & StationIn({"S1", "S3"})
    >>> p2 = YearBetween(min=1950, max=1960) | YearBetween(min=2000, max=2010)
    >>> p3 = ~StepIn({1, 2})

    """

    @abstractmethod
    def __call__(self, info: BreakInfo) -> bool:
        """Return True to accept the break, False to discard it."""

    def __and__(self, other: BreakPredicate) -> BreakPredicate:
        return _AndPredicate(self, other)

    def __or__(self, other: BreakPredicate) -> BreakPredicate:
        return _OrPredicate(self, other)

    def __invert__(self) -> BreakPredicate:
        return _NotPredicate(self)

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""

    @classmethod
    def from_dict(cls, d: dict) -> BreakPredicate:
        """Deserialise from a dict produced by to_dict."""
        pred_type = d["type"]
        if pred_type not in _PRED_REGISTRY:
            msg = f"Unknown predicate type {pred_type!r}. Known: {sorted(_PRED_REGISTRY)}"
            raise ValueError(msg)
        return _PRED_REGISTRY[pred_type]._from_dict(d)  # type: ignore[attr-defined]  # noqa: SLF001

    @classmethod
    def _from_dict(cls, d: dict) -> BreakPredicate:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Leaf predicates
# ---------------------------------------------------------------------------


@_register("year_between")
@dataclass(frozen=True)
class YearBetween(BreakPredicate):
    """Accept breaks whose year falls within [min, max] (inclusive, both optional).

    Parameters
    ----------
    min :
        Earliest acceptable break year.
    max :
        Latest acceptable break year.

    """

    min: int | None = None
    max: int | None = None

    def __call__(self, info: BreakInfo) -> bool:
        if self.min is not None and info.break_year < self.min:
            return False
        return self.max is None or info.break_year <= self.max

    def to_dict(self) -> dict:
        return {"type": "year_between", "min": self.min, "max": self.max}

    @classmethod
    def _from_dict(cls, d: dict) -> YearBetween:
        return cls(min=d.get("min"), max=d.get("max"))


@_register("station_in")
@dataclass(frozen=True)
class StationIn(BreakPredicate):
    """Accept breaks only for stations in the given set.

    Parameters
    ----------
    station_ids :
        Whitelist of station IDs to correct. All others are left uncorrected.

    """

    station_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.station_ids, frozenset):
            object.__setattr__(self, "station_ids", frozenset(self.station_ids))

    def __call__(self, info: BreakInfo) -> bool:
        return info.station_id in self.station_ids

    def to_dict(self) -> dict:
        return {"type": "station_in", "station_ids": sorted(self.station_ids)}

    @classmethod
    def _from_dict(cls, d: dict) -> StationIn:
        return cls(station_ids=frozenset(d["station_ids"]))


@_register("step_in")
@dataclass(frozen=True)
class StepIn(BreakPredicate):
    """Accept breaks detected in specific procedure steps.

    Parameters
    ----------
    steps :
        Set of step numbers to accept (1–6; step 6 is represented as 61/62 internally).

    """

    steps: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.steps, frozenset):
            object.__setattr__(self, "steps", frozenset(self.steps))

    def __call__(self, info: BreakInfo) -> bool:
        return info.step in self.steps

    def to_dict(self) -> dict:
        return {"type": "step_in", "steps": sorted(self.steps)}

    @classmethod
    def _from_dict(cls, d: dict) -> StepIn:
        return cls(steps=frozenset(d["steps"]))


@_register("test_significant")
@dataclass(frozen=True)
class TestSignificant(BreakPredicate):
    """Accept breaks where a specific test flagged the series as significant.

    Parameters
    ----------
    test_name :
        Test name to require (e.g. ``"snht"``, ``"buishand"``, ``"pettitt"``).

    """

    test_name: str = ""

    def __call__(self, info: BreakInfo) -> bool:
        return any(tr.test_name == self.test_name and tr.is_significant for tr in info.test_results)

    def to_dict(self) -> dict:
        return {"type": "test_significant", "test_name": self.test_name}

    @classmethod
    def _from_dict(cls, d: dict) -> TestSignificant:
        return cls(test_name=d["test_name"])


@_register("magnitude_above")
@dataclass(frozen=True)
class MagnitudeAbove(BreakPredicate):
    """Accept breaks whose correction magnitude exceeds a threshold.

    Magnitude is ``|factor - 1|`` in ratio mode and ``|factor|`` in difference mode.

    Parameters
    ----------
    threshold :
        Minimum absolute correction magnitude.

    """

    threshold: float = 0.0

    def __call__(self, info: BreakInfo) -> bool:
        magnitude = abs(info.factor - 1.0) if info.mode == "ratio" else abs(info.factor)
        return magnitude > self.threshold

    def to_dict(self) -> dict:
        return {"type": "magnitude_above", "threshold": self.threshold}

    @classmethod
    def _from_dict(cls, d: dict) -> MagnitudeAbove:
        return cls(threshold=d["threshold"])


@_register("signal_above")
@dataclass(frozen=True)
class SignalAbove(BreakPredicate):
    """Accept breaks where the maximum relative signal exceeds a threshold.

    Relative signal is ``test_statistic / critical_value``; values above 1.0
    are significant. Use this to require a stronger signal than the test's own
    95 % threshold without changing the critical value.

    Parameters
    ----------
    threshold :
        Minimum relative signal required across any test result.

    """

    threshold: float = 1.0

    def __call__(self, info: BreakInfo) -> bool:
        if not info.test_results:
            return False
        return max(tr.relative_signal for tr in info.test_results) > self.threshold

    def to_dict(self) -> dict:
        return {"type": "signal_above", "threshold": self.threshold}

    @classmethod
    def _from_dict(cls, d: dict) -> SignalAbove:
        return cls(threshold=d["threshold"])


@_register("n_significant_above")
@dataclass(frozen=True)
class NSignificantAbove(BreakPredicate):
    """Accept breaks where at least ``n`` tests are significant.

    Provides finer control than the ``consensus`` rule on
    :class:`NormalizationConfig`, which uses relative thresholds
    (any / majority / unanimous). Use this when you want an absolute count
    independent of how many tests were run.

    Parameters
    ----------
    n :
        Minimum number of significant tests required.

    """

    n: int = 1

    def __call__(self, info: BreakInfo) -> bool:
        return sum(1 for tr in info.test_results if tr.is_significant) >= self.n

    def to_dict(self) -> dict:
        return {"type": "n_significant_above", "n": self.n}

    @classmethod
    def _from_dict(cls, d: dict) -> NSignificantAbove:
        return cls(n=d["n"])


@_register("neighbor_count_above")
@dataclass(frozen=True)
class NeighborCountAbove(BreakPredicate):
    """Accept breaks detected using at least ``n`` reference stations.

    Breaks built from very few neighbors are less reliable. This predicate
    lets you discard detections that lacked a solid reference pool.

    Parameters
    ----------
    n :
        Minimum number of neighbors required.

    """

    n: int = 1

    def __call__(self, info: BreakInfo) -> bool:
        return info.n_neighbors >= self.n

    def to_dict(self) -> dict:
        return {"type": "neighbor_count_above", "n": self.n}

    @classmethod
    def _from_dict(cls, d: dict) -> NeighborCountAbove:
        return cls(n=d["n"])


# ---------------------------------------------------------------------------
# Composite predicates
# ---------------------------------------------------------------------------


@_register("and")
@dataclass(frozen=True)
class _AndPredicate(BreakPredicate):
    left: BreakPredicate
    right: BreakPredicate

    def __call__(self, info: BreakInfo) -> bool:
        return self.left(info) and self.right(info)

    def to_dict(self) -> dict:
        return {"type": "and", "left": self.left.to_dict(), "right": self.right.to_dict()}

    @classmethod
    def _from_dict(cls, d: dict) -> _AndPredicate:
        return cls(
            left=BreakPredicate.from_dict(d["left"]),
            right=BreakPredicate.from_dict(d["right"]),
        )


@_register("or")
@dataclass(frozen=True)
class _OrPredicate(BreakPredicate):
    left: BreakPredicate
    right: BreakPredicate

    def __call__(self, info: BreakInfo) -> bool:
        return self.left(info) or self.right(info)

    def to_dict(self) -> dict:
        return {"type": "or", "left": self.left.to_dict(), "right": self.right.to_dict()}

    @classmethod
    def _from_dict(cls, d: dict) -> _OrPredicate:
        return cls(
            left=BreakPredicate.from_dict(d["left"]),
            right=BreakPredicate.from_dict(d["right"]),
        )


@_register("not")
@dataclass(frozen=True)
class _NotPredicate(BreakPredicate):
    operand: BreakPredicate

    def __call__(self, info: BreakInfo) -> bool:
        return not self.operand(info)

    def to_dict(self) -> dict:
        return {"type": "not", "operand": self.operand.to_dict()}

    @classmethod
    def _from_dict(cls, d: dict) -> _NotPredicate:
        return cls(operand=BreakPredicate.from_dict(d["operand"]))


def _edge_safe(tr: TestResult, min_years: int) -> bool:
    """Return True if break_year is at least min_years from either segment boundary."""
    return tr.break_year - tr.segment_start >= min_years and tr.segment_end - tr.break_year + 1 >= min_years


# ---------------------------------------------------------------------------
# NormalizationConfig
# ---------------------------------------------------------------------------


@dataclass
class NormalizationConfig:
    """Configuration for how detections are turned into corrections.

    Parameters
    ----------
    consensus :
        How many tests must agree to accept a break.
        ``"unanimous"`` — all tests; ``"majority"`` — more than half;
        ``"any"`` — at least one; ``"strongest_signal"`` — always use
        the test with the highest relative signal, ignoring the others.
    break_window_years :
        Tolerance (±years) within which two tests are considered to agree
        on the same break year.
    tiebreak :
        What to do when no consensus is reached.
        ``"strongest_signal"`` — use the detection with the highest
        relative signal; ``"skip"`` — do not correct this station/step.
    min_correction_magnitude :
        Ignore corrections below this absolute magnitude
        (ratio: |factor − 1|; difference: |factor|).
    min_relative_signal :
        Minimum ratio of test statistic to critical value required to accept
        a break (default: 1.0). Should match the value used on the
        :class:`HomogenizationTest` instances during detection.
    max_corrections_per_station :
        Cap on the number of breaks applied per station.
    predicate :
        Optional :class:`BreakPredicate` (or composed predicate) to filter
        which detected breaks are accepted. Fully serializable.

        Example::

            from rucola import YearBetween, StationIn, StepIn
            NormalizationConfig(
                predicate=YearBetween(min=1960, max=2010) & ~StepIn({1, 2})
            )

    min_years_from_end :
        Breaks detected within this many years of either series boundary are
        rejected as edge-effect artefacts (Hawkins 1977). Must match the value
        used by the homogeneity tests during detection (default: 5).
    overrides :
        Per-station manual corrections that *replace* the algorithmically
        detected breaks entirely. Dict mapping station_id to a list of
        ``(break_year, factor)`` tuples.

    """

    consensus: ConsensusRule = "majority"
    break_window_years: int = 2
    tiebreak: Tiebreak = "strongest_signal"
    min_correction_magnitude: float = 0.0
    min_relative_signal: float = 1.0
    max_corrections_per_station: int = 2
    min_years_from_end: int = 5
    predicate: BreakPredicate | None = None
    overrides: dict[str, list[tuple[int, float]]] | None = None


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


class Normalizer:
    """Re-applies detected corrections to the original annual series."""

    def __init__(self, config: NormalizationConfig) -> None:
        """Initialise with a NormalizationConfig."""
        self._cfg = config

    def apply(self, result: DetectionResult) -> HomogenizationResult:
        """Apply corrections from result according to config and return HomogenizationResult."""
        mode = result.mode
        station_results: dict[str, StationResult] = {}

        for sid, det in result.station_detections.items():
            annual = det.annual_original
            years = det.years
            corrections: list[CorrectionRecord] = []

            if self._cfg.overrides is not None and sid in self._cfg.overrides:
                accepted = list(self._cfg.overrides[sid])
            else:
                accepted = self._accepted_corrections(det.detections_by_step, det.neighbors_by_step, mode, sid)

            accepted.sort(key=lambda x: x[0])

            for break_year, factor in accepted[: self._cfg.max_corrections_per_station]:
                annual = apply_correction(annual, years, break_year, factor, mode)
                corrections.append(CorrectionRecord(step=0, break_year=break_year, factor=factor))

            station_results[sid] = StationResult(
                station_id=sid,
                group=det.group,
                corrections=corrections,
                neighbors_by_step=det.neighbors_by_step,
                annual_original=det.annual_original,
                annual_corrected=annual,
                years=years,
            )

        return HomogenizationResult(station_results=station_results, parameter=result.parameter)

    def _accepted_corrections(
        self,
        detections_by_step: dict[int, DetectionRecord],
        neighbors_by_step: dict[int, list[NeighborInfo]],
        mode: CorrectionMode,
        station_id: str,
    ) -> list[tuple[int, float]]:
        """Return list of (break_year, factor) that pass all filters."""
        out: list[tuple[int, float]] = []
        for step in sorted(detections_by_step):
            if step == _STEP_6C_DIAGNOSTIC:
                continue
            rec = detections_by_step[step]
            break_year, factor = self._resolve(rec, mode)
            if break_year is None:
                continue
            magnitude = abs(factor - 1.0) if mode == "ratio" else abs(factor)
            if magnitude < self._cfg.min_correction_magnitude:
                continue
            if self._cfg.predicate is not None:
                info = BreakInfo(
                    station_id=station_id,
                    break_year=break_year,
                    factor=factor,
                    step=step,
                    mode=mode,
                    test_results=rec.test_results,
                    n_neighbors=len(neighbors_by_step.get(step, [])),
                )
                if not self._cfg.predicate(info):
                    continue
            out.append((break_year, factor))
        return out

    def _resolve(self, rec: DetectionRecord, mode: CorrectionMode) -> tuple[int | None, float]:  # noqa: PLR0911
        """Apply consensus + tiebreak to one DetectionRecord."""
        test_results = rec.test_results
        if not test_results:
            return None, self._neutral(mode)

        cfg = self._cfg

        min_edge = cfg.min_years_from_end

        if cfg.consensus == "strongest_signal":
            best = max(test_results, key=lambda r: r.relative_signal)
            if best.is_significant and _edge_safe(best, min_edge) and best.relative_signal >= cfg.min_relative_signal:
                return best.break_year, rec.factor
            return None, self._neutral(mode)

        significant = [
            r
            for r in test_results
            if r.is_significant and _edge_safe(r, min_edge) and r.relative_signal >= cfg.min_relative_signal
        ]
        n_total = len(test_results)

        if cfg.consensus == "unanimous":
            required = n_total
        elif cfg.consensus == "majority":
            required = n_total // 2 + 1
        else:  # "any"
            required = 1

        if len(significant) < required:
            if cfg.tiebreak == "strongest_signal" and significant:
                best = max(significant, key=lambda r: r.relative_signal)
                return best.break_year, rec.factor
            return None, self._neutral(mode)

        agreed = self._year_consensus(significant, cfg.break_window_years)
        if agreed is not None:
            return agreed, rec.factor

        if cfg.tiebreak == "strongest_signal":
            best = max(significant, key=lambda r: r.relative_signal)
            return best.break_year, rec.factor
        return None, self._neutral(mode)

    @staticmethod
    def _year_consensus(results: list[TestResult], window: int) -> int | None:
        """Return median break year if all results agree within window, else None."""
        years = [r.break_year for r in results]
        if max(years) - min(years) <= window:
            return sorted(years)[len(years) // 2]
        return None

    def _neutral(self, mode: CorrectionMode) -> float:
        return 1.0 if mode == "ratio" else 0.0
