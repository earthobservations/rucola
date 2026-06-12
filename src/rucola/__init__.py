"""Homogenization toolbox for climate station data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

import polars as pl

from rucola._algorithms import (
    _VALID_GROUPS,
    CorrectionMode,
    GroupLabel,
    NeighborInfo,
    apply_correction,
    build_correlation_cache,
    build_distance_cache,
    build_reference_series,
    compute_correction_factor,
    compute_q_series,
    select_neighbors,
)
from rucola._homogeneity import (
    BuishandTest,  # noqa: F401
    EasterlingPetersonTest,  # noqa: F401
    HomogenizationTest,
    PettittTest,  # noqa: F401
    SNHTTest,
    StarsTest,  # noqa: F401
    TestResult,
    WorsleyTest,  # noqa: F401
)
from rucola._normalization import (  # noqa: F401, TC001
    BreakInfo,
    BreakPredicate,
    ConsensusRule,
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
from rucola._results import (
    CorrectionRecord,
    DetectionRecord,
    DetectionResult,
    HomogenizationResult,  # noqa: F401
    StationDetection,
    StationResult,  # noqa: F401
)

_STATION_ID_SAMPLE_SIZE = 5
_LAT_MIN, _LAT_MAX = -90.0, 90.0
_LON_MIN, _LON_MAX = -180.0, 180.0

# ---------------------------------------------------------------------------
# Internal state tracking across the 6-step procedure
# ---------------------------------------------------------------------------


@dataclass
class _StationState:
    station_id: str
    annual_original: pl.Series
    annual_current: pl.Series  # updated after each correction
    years: pl.Series
    _group: GroupLabel = field(default="", init=False, repr=False)
    corrections: list[CorrectionRecord] = field(default_factory=list)
    detections_by_step: dict[int, DetectionRecord] = field(default_factory=dict)
    neighbors_by_step: dict[int, list[NeighborInfo]] = field(default_factory=dict)

    @property
    def group(self) -> GroupLabel:
        return self._group

    @group.setter
    def group(self, value: GroupLabel) -> None:
        if value not in _VALID_GROUPS:
            msg = f"Invalid group {value!r}. Must be one of {sorted(_VALID_GROUPS - {''})}"
            raise ValueError(msg)
        self._group = value


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Configuration for the González-Rouco six-step detection procedure.

    Parameters
    ----------
    tests :
        Homogeneity tests to run at each step. Defaults to ``[SNHTTest()]``.
        Pass multiple tests for consensus detection.
    mode :
        ``"ratio"`` (multiplicative, precipitation) or ``"difference"``
        (additive, temperature).
    run_consensus :
        How many tests must agree to classify a station as inhomogeneous
        during the six-step procedure. ``"majority"`` (default),
        ``"any"`` (most sensitive), or ``"unanimous"``.
        ``"strongest_signal"`` is accepted but treated as ``"any"`` at
        detection time; its special tiebreak behaviour only applies in
        :class:`NormalizationConfig`.
    min_series_years :
        Minimum non-null annual values required to process a station (default: 20).
    max_gap_years :
        Stations with a consecutive null gap exceeding this are excluded.
    max_neighbors :
        Maximum number of reference stations (default: 10).
    min_correlation :
        Minimum Pearson correlation to include a neighbor (default: 0.5).
    max_distance_km :
        Search radius for neighbors in km. ``None`` disables the filter.
    station_ids :
        Restrict the run to this subset of station IDs.

    """

    tests: list[HomogenizationTest] | None = None
    mode: CorrectionMode = "ratio"
    run_consensus: ConsensusRule = "majority"
    min_series_years: int = 20
    max_gap_years: int | None = None
    max_neighbors: int = 10
    min_correlation: float = 0.5
    max_distance_km: float | None = None
    station_ids: list[str] | None = None
    progress: bool = False
    on_step: Callable[[int, str], None] | None = None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class Rucola:
    """Homogenization toolbox for climate station data.

    Implements the six-step quality control and homogenization procedure from
    González-Rouco et al. (2001), with six pluggable breakpoint tests
    (SNHT, Buishand, Pettitt, Worsley, Easterling–Peterson, STARS) and an
    iteratively refined reference pool.

    Each instance represents one parameter (e.g. precipitation_height).
    Use the ``from_*`` class methods to load data from different sources.

    Minimum required columns
    ------------------------
    stations : station_id (str), latitude (float), longitude (float)
    values   : station_id (str), date (date/datetime), value (float),
               parameter (str)

    References
    ----------
    González-Rouco et al. (2001), J. Climate 14(5):964–978.
        https://doi.org/10.1175/1520-0442(2001)014<0964:QCAHOP>2.0.CO;2
    Alexandersson (1986), Int. J. Climatol. 6(6):661–675.
    Alexandersson & Moberg (1997), Int. J. Climatol. 17(1):25–34.
    Hanssen-Bauer & Førland (1994), J. Climate 7(7):1001–1013.

    """

    STATIONS_REQUIRED: frozenset[str] = frozenset({"station_id", "latitude", "longitude"})
    VALUES_REQUIRED: frozenset[str] = frozenset({"station_id", "date", "value"})

    def __init__(
        self,
        values: pl.DataFrame,
        stations: pl.DataFrame | None = None,
        parameter: str | None = None,
    ) -> None:
        """Initialise directly from pre-loaded DataFrames. Prefer ``from_*`` class methods."""
        self._check_columns(values, self.VALUES_REQUIRED, "values")
        self._check_value_dtype(values)
        self._check_date_order(values)
        self._check_duplicate_dates(values)
        self._check_single_parameter(values)

        if stations is not None:
            self._check_columns(stations, self.STATIONS_REQUIRED, "stations")
            self._check_station_coverage(values, stations)
            self._check_stations(stations)
            self.stations: pl.DataFrame = stations
        else:
            station_ids = values["station_id"].unique().sort().to_list()
            self.stations = pl.DataFrame({"station_id": station_ids})

        self.values = values
        self.parameter = parameter or "unspecified"

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_duckdb(
        cls,
        path: str | Path,
        parameter: str | None = None,
        stations_table: str | None = "stations",
        values_table: str = "values",
    ) -> Self:
        """Load from a DuckDB file.

        Parameters
        ----------
        path :
            Path to the .duckdb file.
        parameter :
            Label for the parameter stored in this instance (e.g.
            ``"precipitation_height"``). Pure metadata — filter the table to a
            single parameter before calling this method.
        stations_table :
            Name of the stations table (default: ``"stations"``). Pass ``None``
            to run without station metadata.
        values_table :
            Name of the values table (default: ``"values"``).

        """
        try:
            import duckdb  # noqa: PLC0415
        except ImportError as e:
            msg = "DuckDB is required for from_duckdb(). Install it with: pip install rucola[duckdb]"
            raise ImportError(msg) from e
        with duckdb.connect(str(path), read_only=True) as con:
            stations = con.execute(f"SELECT * FROM {stations_table}").pl() if stations_table else None  # noqa: S608
            values = con.execute(f"SELECT * FROM {values_table}").pl()  # noqa: S608
        return cls._cast(values, stations, parameter=parameter)

    @classmethod
    def from_csv(
        cls,
        values_path: str | Path,
        stations_path: str | Path | None = None,
        parameter: str | None = None,
    ) -> Self:
        """Load from CSV files.

        Pre-filter ``values`` to a single parameter before calling this method.
        ``stations_path`` is optional; omit it to run without station metadata.
        """
        values = pl.read_csv(str(values_path), try_parse_dates=True)
        stations = pl.read_csv(str(stations_path), try_parse_dates=True) if stations_path else None
        return cls._cast(values, stations, parameter=parameter)

    @classmethod
    def from_polars(
        cls,
        values: pl.DataFrame,
        stations: pl.DataFrame | None = None,
        parameter: str | None = None,
    ) -> Self:
        """Load from Polars DataFrames.

        Pre-filter ``values`` to a single parameter before calling this method.
        ``stations`` is optional; omit it to run without station metadata.
        """
        return cls._cast(values, stations, parameter=parameter)

    # ------------------------------------------------------------------
    # Core 6-step procedure
    # ------------------------------------------------------------------

    def run(  # noqa: C901, PLR0912, PLR0915
        self,
        config: RunConfig | None = None,
    ) -> DetectionResult:
        """Run the González-Rouco (2001) six-step homogenization procedure.

        Overview
        --------
        Step 0 (pre-processing): Winsorize daily values to P_out = q_0.75 + 3·IQR
                                  per station, then aggregate to annual totals.

        Steps 1–6 iteratively refine the reference station pool and apply
        corrections. Each step tests a subset of candidate stations using a
        progressively more reliable set of reference stations:

          Step 1 – First assessment.  ALL vs ALL.
                   → groups H1 (homogeneous) and I1 (inhomogeneous).
                   Corrections applied to I1.

          Step 2 – Adjusted references.  ALL vs (H1 + corrected I1).
                   → groups H2, I2.
                   Corrections applied to I2.

          Step 3 – Test corrected I2 series.  corr(I2) vs (H2 + corr(I2)).
                   → groups HC3 (corrected, now homogeneous), IC3 (still inhomogeneous).

          Step 4 – Only homogeneous references.  (H2 + corr(I2)) vs (H2 + HC3).
                   → groups H4, HC4, I4, IC4.
                   Corrections applied to I4.

          Step 5 – Last single-break corrections.  I4 vs (H4 + HC4).
                   → groups HC5, IC5.

          Step 6 – Double-break correction.  (IC4 + IC5) vs (H4 + HC4 + HC5).
                   Two-pass: first correct the later break on the post-first-break
                   sub-series, then correct the earlier break on the full series.
                   → groups HCC6 or ICC6.

        Parameters
        ----------
        config :
            Run configuration. Defaults to ``RunConfig()`` if not provided.

        Returns
        -------
        DetectionResult
            Raw detection data for all stations. Call ``.normalize()`` to
            obtain a ``HomogenizationResult`` with corrections applied.

        """
        cfg = config or RunConfig()

        _pbar: Any = None
        if cfg.progress:
            try:
                from tqdm import tqdm  # noqa: PLC0415
                _pbar = tqdm(total=6, desc="step 1/6", unit="step", leave=True)
            except ImportError:
                import warnings  # noqa: PLC0415
                warnings.warn("tqdm is not installed; install it with: pip install rucola[tqdm]", stacklevel=2)

        if cfg.max_distance_km is not None and "latitude" not in self.stations.columns:
            msg = "max_distance_km requires a stations DataFrame with latitude/longitude columns."
            raise ValueError(msg)

        # ── Resolve tests and derive min_years_from_end ───────────────────
        _tests = cfg.tests or [SNHTTest()]
        min_years_from_end = max(t.min_years_from_end for t in _tests)

        # ── Parameter consistency check ───────────────────────────────────
        min_detectable = 2 * min_years_from_end + 1
        if cfg.min_series_years < min_detectable:
            msg = (
                f"min_series_years={cfg.min_series_years} is below the minimum detectable series length "
                f"({min_detectable} = 2 * min_years_from_end + 1 = 2 * {min_years_from_end} + 1). "
                "No break could ever pass the edge-effect guard. "
                f"Set min_series_years >= {min_detectable} or lower min_years_from_end."
            )
            raise ValueError(msg)

        # ── Step 0: validate resolution and pivot to wide ────────────────
        self._check_annual_resolution(self.values)
        annual_wide_base = (
            self.values.with_columns(pl.col("date").dt.year().alias("year"))
            .select("station_id", "year", "value")
            .pivot(on="station_id", index="year", values="value", aggregate_function="first")
            .sort("year")
        )
        years = annual_wide_base["year"]

        candidate_ids = [sid for sid in self.stations["station_id"].to_list() if sid in annual_wide_base.columns]
        if cfg.station_ids is not None:
            allowed = set(cfg.station_ids)
            candidate_ids = [s for s in candidate_ids if s in allowed]

        # ── Pre-filter: series length and gap checks ──────────────────────
        def _max_gap(sid: str) -> int:
            best, cur = 0, 0
            for v in annual_wide_base[sid].to_list():
                cur = cur + 1 if v is None else 0
                best = max(best, cur)
            return best

        insufficient_ids: list[str] = []
        filtered: list[str] = []
        for sid in candidate_ids:
            col = annual_wide_base[sid]
            too_short = col.drop_nulls().len() < cfg.min_series_years
            gapped = cfg.max_gap_years is not None and _max_gap(sid) > cfg.max_gap_years
            if too_short or gapped:
                insufficient_ids.append(sid)
            else:
                filtered.append(sid)
        candidate_ids = filtered

        # Initialise per-station state
        states: dict[str, _StationState] = {
            sid: _StationState(
                station_id=sid,
                annual_original=annual_wide_base[sid],
                annual_current=annual_wide_base[sid],
                years=years,
            )
            for sid in candidate_ids
        }

        # ── helpers ──────────────────────────────────────────────────────

        def build_wide() -> pl.DataFrame:
            data: dict = {"year": years.to_list()}
            for sid in candidate_ids:
                data[sid] = states[sid].annual_current.to_list()
            return pl.DataFrame(data)

        _dist_cache = build_distance_cache(self.stations) if cfg.max_distance_km is not None else None

        def _advance(step: int, desc: str) -> None:
            if _pbar is not None:
                _pbar.set_description(desc)
                _pbar.update(1)
            if cfg.on_step is not None:
                cfg.on_step(step, desc)

        def _meets_consensus(tr_list: list[TestResult]) -> bool:
            sig = sum(1 for t, tr in zip(_tests, tr_list, strict=False) if t.is_inhomogeneous(tr))
            n = len(_tests)
            if cfg.run_consensus == "unanimous":
                return sig == n
            if cfg.run_consensus == "majority":
                return sig > n // 2
            # "any" / "strongest_signal" — at detection time both require at least one inhomogeneous test
            return sig >= 1

        def _resolve_break(tr_list: list[TestResult]) -> int | None:
            sig = [tr for t, tr in zip(_tests, tr_list, strict=False) if t.is_inhomogeneous(tr)]
            if not sig:
                return None
            return max(sig, key=lambda r: r.relative_signal).break_year

        def run_tests_for(
            test_ids: list[str],
            ref_ids: set[str],
            step: int,
        ) -> dict[str, tuple[list[TestResult], list[NeighborInfo], pl.Series]]:
            """Run all tests for each station; store results in state."""
            wide = build_wide()
            _corr_cache = build_correlation_cache(wide)
            results: dict[str, tuple[list[TestResult], list[NeighborInfo], pl.Series]] = {}
            for cid in test_ids:
                if cid not in wide.columns:
                    continue
                nbrs = select_neighbors(
                    cid,
                    self.stations,
                    wide,
                    max_neighbors=cfg.max_neighbors,
                    min_correlation=cfg.min_correlation,
                    max_distance_km=cfg.max_distance_km,
                    allowed_ids=ref_ids,
                    dist_cache=_dist_cache,
                    corr_cache=_corr_cache,
                )
                if not nbrs:
                    continue
                ref = build_reference_series(wide, nbrs, cfg.mode)
                q = compute_q_series(wide[cid], ref, cfg.mode)
                tr_list = [t.detect(q, years) for t in _tests]
                states[cid].neighbors_by_step[step] = nbrs
                results[cid] = (tr_list, nbrs, q)
            return results

        def correct_stations(
            test_results: dict[str, tuple[list[TestResult], list[NeighborInfo], pl.Series]],
            ids_to_correct: set[str],
            step: int,
        ) -> None:
            """Apply correction if any test detects inhomogeneity; store DetectionRecord."""
            for cid in ids_to_correct:
                if cid not in test_results:
                    continue
                tr_list, _, q = test_results[cid]
                is_inh = _meets_consensus(tr_list)
                break_year = _resolve_break(tr_list)
                neutral = 1.0 if cfg.mode == "ratio" else 0.0
                f = compute_correction_factor(q, years, break_year, cfg.mode) if is_inh else neutral
                if is_inh:
                    states[cid].annual_current = apply_correction(
                        states[cid].annual_current, years, break_year, f, cfg.mode
                    )
                    states[cid].corrections.append(CorrectionRecord(step=step, break_year=break_year, factor=f))
                states[cid].detections_by_step[step] = DetectionRecord(
                    step=step,
                    break_year=break_year,
                    factor=f,
                    test_results=tr_list,
                    was_applied=is_inh,
                )

        # ── Step 1 ────────────────────────────────────────────────────────
        all_ids = set(candidate_ids)
        r1 = run_tests_for(candidate_ids, all_ids, step=1)

        h1: set[str] = set()
        i1: set[str] = set()
        for sid in candidate_ids:
            if sid in r1 and _meets_consensus(r1[sid][0]):
                i1.add(sid)
                states[sid].group = "I1"
            else:
                h1.add(sid)
                states[sid].group = "H1"

        correct_stations(r1, i1, step=1)
        _advance(2, f"step 2/6 — {len(candidate_ids)} stations vs corrected pool")

        # ── Step 2 ────────────────────────────────────────────────────────
        ref2 = h1 | i1  # i1 now corrected in states
        r2 = run_tests_for(candidate_ids, ref2, step=2)

        h2: set[str] = set()
        i2: set[str] = set()
        for sid in candidate_ids:
            if sid in r2 and _meets_consensus(r2[sid][0]):
                i2.add(sid)
                states[sid].group = "I2"
            else:
                h2.add(sid)
                states[sid].group = "H2"

        correct_stations(r2, i2, step=2)
        _advance(3, f"step 3/6 — {len(i2)} corrected stations re-tested")

        # ── Step 3 ────────────────────────────────────────────────────────
        corrected_i2 = i2
        ref3 = h2 | corrected_i2
        r3 = run_tests_for(list(corrected_i2), ref3, step=3)

        hc3: set[str] = set()
        ic3: set[str] = set()
        for sid in corrected_i2:
            if sid in r3 and _meets_consensus(r3[sid][0]):
                ic3.add(sid)
                states[sid].group = "IC3"
            else:
                hc3.add(sid)
                states[sid].group = "HC3"

        # No corrections in step 3 (classification only, corrections already done in step 2)
        _advance(4, f"step 4/6 — {len(h2) + len(corrected_i2)} stations vs homogeneous references")

        # ── Step 4 ────────────────────────────────────────────────────────
        test4 = list(h2 | corrected_i2)
        ref4 = h2 | hc3
        r4 = run_tests_for(test4, ref4, step=4)

        h4: set[str] = set()
        i4: set[str] = set()
        hc4: set[str] = set()
        ic4: set[str] = set()
        for sid in h2:
            if sid in r4 and _meets_consensus(r4[sid][0]):
                i4.add(sid)
                states[sid].group = "I4"
            else:
                h4.add(sid)
                states[sid].group = "H4"
        for sid in corrected_i2:
            if sid in r4 and _meets_consensus(r4[sid][0]):
                ic4.add(sid)
                states[sid].group = "IC4"
            else:
                hc4.add(sid)
                states[sid].group = "HC4"

        correct_stations(r4, i4, step=4)
        _advance(5, f"step 5/6 — {len(i4)} remaining inhomogeneous stations")

        # ── Step 5 ────────────────────────────────────────────────────────
        ref5 = h4 | hc4
        r5 = run_tests_for(list(i4), ref5, step=5)

        hc5: set[str] = set()
        ic5: set[str] = set()
        for sid in i4:
            if sid in r5 and _meets_consensus(r5[sid][0]):
                ic5.add(sid)
                states[sid].group = "IC5"
            else:
                hc5.add(sid)
                states[sid].group = "HC5"

        # Only i4 stations meeting consensus get a correction; IC5 (failed consensus) do not.
        correct_stations(r5, i4, step=5)
        _advance(6, f"step 6/6 — double-break correction for {len(ic4 | ic5)} stations")

        # ── Step 6 ────────────────────────────────────────────────────────
        # For IC4 and IC5: two-break correction.
        # Pass 6a: omit data before first known break, test post-break sub-series.
        # Pass 6b: with the second break corrected, test full series for first break.
        _corr_cache_6b = build_correlation_cache(build_wide())

        double_break_ids = ic4 | ic5
        ref6 = h4 | hc4 | hc5

        years_list = years.to_list()

        for cid in double_break_ids:
            state = states[cid]
            if not state.corrections:
                continue
            first_break = state.corrections[0].break_year

            # find the index in years where the post-first-break segment starts
            start_idx = next((i for i, y in enumerate(years_list) if y >= first_break), None)
            if start_idx is None or (len(years_list) - start_idx) < 2 * min_years_from_end:
                continue

            # --- 6a: test only the post-first-break portion ---
            _wide_cols = set(build_wide().columns)
            partial_wide = pl.DataFrame(
                {"year": years_list[start_idx:]}
                | {
                    sid: states[sid].annual_current[start_idx:].to_list()
                    for sid in candidate_ids
                    if sid in _wide_cols
                },
            )
            nbrs6a = select_neighbors(
                cid,
                self.stations,
                partial_wide,
                max_neighbors=cfg.max_neighbors,
                min_correlation=cfg.min_correlation,
                max_distance_km=cfg.max_distance_km,
                allowed_ids=ref6,
                dist_cache=_dist_cache,
                corr_cache=build_correlation_cache(partial_wide),
            )
            if nbrs6a:
                ref_6a = build_reference_series(partial_wide, nbrs6a, cfg.mode)
                q_6a = compute_q_series(partial_wide[cid], ref_6a, cfg.mode)
                tr_6a = [t.detect(q_6a, years[start_idx:]) for t in _tests]
                states[cid].neighbors_by_step[61] = nbrs6a
                inh_6a = _meets_consensus(tr_6a)
                br_6a = _resolve_break(tr_6a)
                neutral = 1.0 if cfg.mode == "ratio" else 0.0
                f_6a = compute_correction_factor(q_6a, years[start_idx:], br_6a, cfg.mode) if inh_6a else neutral
                if inh_6a:
                    states[cid].annual_current = apply_correction(
                        states[cid].annual_current,
                        years,
                        br_6a,
                        f_6a,
                        cfg.mode,
                    )
                    states[cid].corrections.append(CorrectionRecord(step=6, break_year=br_6a, factor=f_6a))
                states[cid].detections_by_step[61] = DetectionRecord(
                    step=61,
                    break_year=br_6a,
                    factor=f_6a,
                    test_results=tr_6a,
                    was_applied=inh_6a,
                )

            # --- 6b: test full series to correct the first break ---
            wide6b = build_wide()
            nbrs6b = select_neighbors(
                cid,
                self.stations,
                wide6b,
                max_neighbors=cfg.max_neighbors,
                min_correlation=cfg.min_correlation,
                max_distance_km=cfg.max_distance_km,
                allowed_ids=ref6,
                dist_cache=_dist_cache,
                corr_cache=_corr_cache_6b,
            )
            if not nbrs6b:
                states[cid].group = "ICC6"
                continue

            ref_6b = build_reference_series(wide6b, nbrs6b, cfg.mode)
            q_6b = compute_q_series(wide6b[cid], ref_6b, cfg.mode)
            tr_6b = [t.detect(q_6b, years) for t in _tests]
            states[cid].neighbors_by_step[62] = nbrs6b
            inh_6b = _meets_consensus(tr_6b)
            br_6b = _resolve_break(tr_6b)
            neutral = 1.0 if cfg.mode == "ratio" else 0.0
            f_6b = compute_correction_factor(q_6b, years, br_6b, cfg.mode) if inh_6b else neutral
            states[cid].detections_by_step[62] = DetectionRecord(
                step=62,
                break_year=br_6b,
                factor=f_6b,
                test_results=tr_6b,
                was_applied=inh_6b,
            )
            if inh_6b:
                states[cid].annual_current = apply_correction(
                    states[cid].annual_current,
                    years,
                    br_6b,
                    f_6b,
                    cfg.mode,
                )
                states[cid].corrections.append(CorrectionRecord(step=6, break_year=br_6b, factor=f_6b))

            # --- 6c: re-test the doubly-corrected series — González-Rouco (2001) §3.b.
            # HCC6 means "homogeneous after two corrections"; ICC6 means a residual
            # inhomogeneity (e.g. a third break) survives. Without this re-test every
            # IC4/IC5 station would silently become HCC6.
            wide6c = build_wide()
            ref_6c = build_reference_series(wide6c, nbrs6b, cfg.mode)
            q_6c = compute_q_series(wide6c[cid], ref_6c, cfg.mode)
            tr_6c = [t.detect(q_6c, years) for t in _tests]
            still_inh = _meets_consensus(tr_6c)
            states[cid].detections_by_step[63] = DetectionRecord(
                step=63,
                break_year=_resolve_break(tr_6c),
                factor=1.0 if cfg.mode == "ratio" else 0.0,
                test_results=tr_6c,
                was_applied=False,
            )
            states[cid].group = "ICC6" if still_inh else "HCC6"

        # mark any double-break stations we couldn't fully correct
        for cid in double_break_ids:
            if states[cid].group in ("IC4", "IC5"):
                states[cid].group = "ICC6"

        _advance(7, "done")
        if _pbar is not None:
            _pbar.close()

        # ── Build DetectionResult ─────────────────────────────────────────
        # Stations that never found a neighbor at any step → UNTESTABLE.
        # Use neighbors_by_step (set for all tested stations) rather than
        # detections_by_step (only set for stations that were inhomogeneous),
        # so that stations tested and found homogeneous keep their H* group.
        for sid in candidate_ids:
            if not states[sid].neighbors_by_step:
                states[sid].group = "UNTESTABLE"

        station_detections: dict[str, StationDetection] = {
            sid: StationDetection(
                station_id=sid,
                group=states[sid].group,
                annual_original=states[sid].annual_original,
                annual_corrected=states[sid].annual_current,
                years=years,
                detections_by_step=states[sid].detections_by_step,
                neighbors_by_step=states[sid].neighbors_by_step,
                corrections=states[sid].corrections,
            )
            for sid in candidate_ids
        }

        # Add pre-filtered stations with INSUFFICIENT_DATA label
        for sid in insufficient_ids:
            annual = annual_wide_base[sid]
            station_detections[sid] = StationDetection(
                station_id=sid,
                group="INSUFFICIENT_DATA",
                annual_original=annual,
                annual_corrected=annual,
                years=years,
            )

        return DetectionResult(
            station_detections=station_detections,
            parameter=self.parameter,
            mode=cfg.mode,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_stations(self) -> int:
        """Number of stations in the dataset."""
        return len(self.stations)

    @property
    def date_range(self) -> tuple[str, str]:
        """Earliest and latest date in the values table as ISO strings."""
        d = self.values["date"]
        return str(d.min()), str(d.max())

    def __repr__(self) -> str:
        """Return a short summary string."""
        lo, hi = self.date_range
        return (
            f"Rucola(parameter={self.parameter!r}, "
            f"stations={self.n_stations}, "
            f"records={len(self.values):,}, "
            f"period={lo} – {hi})"  # noqa: RUF001
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_date_order(values: pl.DataFrame) -> None:
        """Raise if any station's dates are not sorted ascending."""
        unsorted = (
            values.group_by("station_id")
            .agg((pl.col("date") == pl.col("date").cum_max()).all().alias("sorted"))
            .filter(~pl.col("sorted"))["station_id"]
            .to_list()
        )
        if unsorted:
            n = len(unsorted)
            sample = sorted(unsorted)[:_STATION_ID_SAMPLE_SIZE]
            suffix = " ..." if n > _STATION_ID_SAMPLE_SIZE else ""
            msg = f"{n} station(s) in `values` have dates not sorted ascending: {sample}{suffix}"
            raise ValueError(msg)

    @staticmethod
    def _check_duplicate_dates(values: pl.DataFrame) -> None:
        """Raise if any station has two records for the same date."""
        dup_ids = (
            values.group_by("station_id", "date")
            .len()
            .filter(pl.col("len") > 1)["station_id"]
            .unique()
            .to_list()
        )
        if dup_ids:
            n = len(dup_ids)
            sample = sorted(dup_ids)[:_STATION_ID_SAMPLE_SIZE]
            suffix = " ..." if n > _STATION_ID_SAMPLE_SIZE else ""
            msg = f"{n} station(s) in `values` have duplicate dates: {sample}{suffix}"
            raise ValueError(msg)

    @staticmethod
    def _check_single_parameter(values: pl.DataFrame) -> None:
        """Raise if `values` contains more than one parameter value."""
        if "parameter" not in values.columns:
            return
        params = values["parameter"].drop_nulls().unique().to_list()
        if len(params) > 1:
            msg = (
                f"`values` contains multiple parameters: {sorted(params)!r}. "
                "Filter to a single parameter before loading."
            )
            raise ValueError(msg)

    @staticmethod
    def _check_annual_resolution(values: pl.DataFrame) -> None:
        """Raise if values contains more than one record per station per year."""
        max_per: int = (
            values.with_columns(pl.col("date").dt.year().alias("_year"))
            .group_by("station_id", "_year")
            .len()
            .select(pl.col("len").max().cast(pl.Int64))
            .item()
        )
        if max_per > 1:
            msg = (
                f"values has sub-annual resolution ({max_per} records per station-year). "
                "Pre-aggregate to annual using rucola._preprocessing.compute_annual_totals "
                "or compute_annual_means before passing to Rucola."
            )
            raise ValueError(msg)

    @staticmethod
    def _check_value_dtype(values: pl.DataFrame) -> None:
        """Raise if the `value` column is not numeric."""
        dtype = values["value"].dtype
        if not dtype.is_numeric():
            msg = f"`values.value` must be numeric, got {dtype}. Cast to Float64 before loading."
            raise TypeError(msg)

    @staticmethod
    def _check_stations(stations: pl.DataFrame) -> None:
        """Raise on duplicate station IDs, null coordinates, or out-of-range coordinates."""
        dupes = stations.group_by("station_id").len().filter(pl.col("len") > 1)["station_id"].to_list()
        if dupes:
            msg = f"`stations` has duplicate station_id(s): {sorted(dupes)}"
            raise ValueError(msg)

        null_coords = stations.filter(pl.col("latitude").is_null() | pl.col("longitude").is_null())[
            "station_id"
        ].to_list()
        if null_coords:
            msg = f"`stations` has null latitude/longitude for station_id(s): {sorted(null_coords)}"
            raise ValueError(msg)

        bad_lat = stations.filter((pl.col("latitude") < _LAT_MIN) | (pl.col("latitude") > _LAT_MAX))[
            "station_id"
        ].to_list()
        if bad_lat:
            msg = f"`stations` has latitude outside [{_LAT_MIN}, {_LAT_MAX}] for station_id(s): {sorted(bad_lat)}"
            raise ValueError(msg)

        bad_lon = stations.filter((pl.col("longitude") < _LON_MIN) | (pl.col("longitude") > _LON_MAX))[
            "station_id"
        ].to_list()
        if bad_lon:
            msg = f"`stations` has longitude outside [{_LON_MIN}, {_LON_MAX}] for station_id(s): {sorted(bad_lon)}"
            raise ValueError(msg)

    @staticmethod
    def _check_columns(df: pl.DataFrame, required: frozenset[str], name: str) -> None:
        missing = required - set(df.columns)
        if missing:
            msg = f"`{name}` is missing required column(s): {sorted(missing)}"
            raise ValueError(msg)

    @staticmethod
    def _check_station_coverage(values: pl.DataFrame, stations: pl.DataFrame) -> None:
        """All station_ids in values must have a matching entry in stations."""
        value_ids = set(values["station_id"].unique().to_list())
        station_ids = set(stations["station_id"].to_list())
        missing = value_ids - station_ids
        if missing:
            n = len(missing)
            sample = sorted(missing)[:_STATION_ID_SAMPLE_SIZE]
            suffix = " ..." if n > _STATION_ID_SAMPLE_SIZE else ""
            msg = f"{n} station_id(s) in `values` have no entry in `stations`: {sample}{suffix}"
            raise ValueError(msg)

    @classmethod
    def _cast(
        cls,
        values: pl.DataFrame,
        stations: pl.DataFrame | None = None,
        parameter: str | None = None,
    ) -> Self:
        """Normalise column types and construct the instance."""
        values = values.with_columns(pl.col("station_id").cast(pl.String))
        if stations is not None:
            stations = stations.with_columns(pl.col("station_id").cast(pl.String))
        if "date" in values.columns and values["date"].dtype not in (pl.Date, pl.Datetime):
            values = values.with_columns(pl.col("date").cast(pl.Date))
        return cls(values=values, stations=stations, parameter=parameter)
