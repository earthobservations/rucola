# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "altair>=6.2.1,<7.0.0",
#   "duckdb>=1.5.3,<2.0.0",
#   "marimo>=0.23.9,<1",
#   "polars>=1.41.2,<2.0.0",
#   "rucola",
#   "wetterdienst>=0.121.1,<1.0.0",
# ]
# ///
import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Homogenization — Saxony")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Climate Station Homogenization — Saxony

        This notebook applies the [González-Rouco et al. (2001)](https://doi.org/10.1175/1520-0442(2001)014<0964:QCAHOP>2.0.CO;2)
        six-step homogenization procedure to annual climate records from all DWD stations
        in the German federal state of Saxony. Five breakpoint tests are run in consensus:
        SNHT, Buishand, Pettitt, Worsley likelihood ratio, and Easterling–Peterson.

        The workflow has three stages:

        1. **Load & detect** — fetch data from the DWD open data API and run the homogenization procedure
        2. **Evaluate** — review the detected breaks and decide on normalization settings
        3. **Correct** — apply the corrections and inspect the homogenized series
        """
    )
    return


@app.cell
def _():
    import polars as pl
    import rucola
    from rucola import (
        BreakInfo,
        BreakPredicate,
        BuishandTest,
        EasterlingPetersonTest,
        NormalizationConfig,
        PettittTest,
        RunConfig,
        SNHTTest,
        StarsTest,
        WorsleyTest,
    )
    from wetterdienst.provider.dwd.observation import DwdObservationRequest

    return (
        BreakInfo,
        BreakPredicate,
        BuishandTest,
        DwdObservationRequest,
        EasterlingPetersonTest,
        NormalizationConfig,
        PettittTest,
        RunConfig,
        SNHTTest,
        StarsTest,
        WorsleyTest,
        pl,
        rucola,
    )


@app.cell
def _(mo):
    parameter_picker = mo.ui.dropdown(
        options={
            "Precipitation (precipitation_height)": "precipitation_height",
            "Temperature 2m mean (temperature_air_mean_2m)": "temperature_air_mean_2m",
        },
        value="Precipitation (precipitation_height)",
        label="Parameter",
    )
    parameter_picker
    return (parameter_picker,)


@app.cell
def _(mo):
    _ALL_TESTS = [
        "SNHT",
        "Buishand range",
        "Pettitt",
        "Worsley likelihood ratio",
        "Easterling–Peterson",
        "STARS",
    ]
    test_picker = mo.ui.multiselect(
        options=_ALL_TESTS,
        value=_ALL_TESTS,
        label="Homogeneity tests",
    )
    test_picker
    return (test_picker,)


@app.cell
def _(mo, test_picker):
    _warning = (
        mo.callout(mo.md("Select **at least 2 tests** for consensus detection to be meaningful."), kind="warn")
        if len(test_picker.value) < 2
        else mo.md("")
    )
    _warning
    return


@app.cell
def _(mo):
    consensus_picker = mo.ui.dropdown(
        options=["majority", "any", "unanimous"],
        value="majority",
        label="Consensus (detection & normalization)",
    )
    consensus_picker
    return (consensus_picker,)


@app.cell
def _(DwdObservationRequest, mo, parameter_picker, pl):
    _DATASET = {
        "precipitation_height": "more_precip",
        "temperature_air_mean_2m": "climate_summary",
    }
    _param = parameter_picker.value
    _req = DwdObservationRequest(parameters=[("annual", _DATASET[_param], _param)])
    _station_result = _req.filter_by_sql("state = 'Sachsen'")
    stations = _station_result.df.select("station_id", "latitude", "longitude")

    _frames = []
    with mo.status.progress_bar(total=len(stations), title="Fetching station data …") as _pbar:
        for _r in _station_result.values.query():
            _frames.append(
                _r.df.select(
                    pl.col("station_id"),
                    pl.col("date").dt.replace_time_zone(None).cast(pl.Date).alias("date"),
                    pl.col("value"),
                    pl.col("parameter"),
                )
            )
            _pbar.update(subtitle=_r.df["station_id"][0])
    values = pl.concat(_frames).sort("station_id", "date")

    mo.callout(
        mo.md(
            f"Loaded **{len(stations)}** stations · "
            f"**{len(values):,}** annual records · "
            f"period {values['date'].min().year}–{values['date'].max().year}"
        ),
        kind="success",
    )
    return stations, values


@app.cell
def _(DwdObservationRequest, mo, parameter_picker, stations):
    _DATASET = {
        "precipitation_height": "more_precip",
        "temperature_air_mean_2m": "climate_summary",
    }
    _DEVICE_KEYWORD = {
        "precipitation_height": "Niederschlag",
        "temperature_air_mean_2m": "Temperatur",
    }
    _param = parameter_picker.value
    _kw = _DEVICE_KEYWORD[_param]
    _req = DwdObservationRequest(parameters=[("annual", _DATASET[_param], _param)])

    station_histories: dict[str, set[int]] = {}
    _n_with_records = 0
    _sids = stations["station_id"].to_list()
    with mo.status.progress_bar(total=len(_sids), title="Fetching station histories …") as _pbar:
        for _sid in _sids:
            try:
                _item = next(_req.filter_by_station_id(_sid).history.query(), None)
            except (UnicodeDecodeError, StopIteration, OSError):
                _item = None
            if _item is not None:
                _hist = _item.history
                _event_years = set(
                    [g.start_date.year for g in _hist.geography if g.start_date]
                    + [d.start_date.year for d in _hist.device if d.start_date and _kw in (d.method or "")]
                    + [o.start_date.year for o in _hist.name.operator if o.start_date]
                )
                station_histories[_sid] = _event_years
                if _event_years:
                    _n_with_records += 1
            _pbar.update(subtitle=_sid)

    mo.callout(
        mo.md(f"Loaded history for **{_n_with_records}** of **{len(_sids)}** stations"),
        kind="success",
    )
    return (station_histories,)


@app.cell
def _(
    BuishandTest,
    EasterlingPetersonTest,
    PettittTest,
    RunConfig,
    SNHTTest,
    StarsTest,
    WorsleyTest,
    consensus_picker,
    mo,
    parameter_picker,
    pl,
    rucola,
    stations,
    test_picker,
    values,
):
    _MODE = {
        "precipitation_height": "ratio",
        "temperature_air_mean_2m": "difference",
    }
    _TEST_MAP = {
        "SNHT": SNHTTest,
        "Buishand range": BuishandTest,
        "Pettitt": PettittTest,
        "Worsley likelihood ratio": WorsleyTest,
        "Easterling–Peterson": EasterlingPetersonTest,
        "STARS": StarsTest,
    }
    _param = parameter_picker.value
    _tests = [_TEST_MAP[t]() for t in test_picker.value] or [SNHTTest()]

    with mo.status.progress_bar(total=6, title="step 1/6 — all stations vs all") as _pbar:
        _r = rucola.Rucola.from_polars(values, stations, parameter=_param)
        detection = _r.run(
            RunConfig(
                tests=_tests,
                mode=_MODE[_param],
                run_consensus=consensus_picker.value,
                on_step=lambda _, desc: _pbar.update(title=desc),
            )
        )

    _s = detection.summary
    _grp = pl.col("group")
    _n_homogeneous = len(_s.filter(_grp.str.starts_with("H") & ~_grp.str.starts_with("HC")))
    _n_corrected = len(_s.filter(_grp.str.starts_with("HC")))
    _n_inhomogeneous = len(_s.filter(_grp.str.starts_with("IC")))
    _n_skipped = len(_s.filter(_grp.is_in(["INSUFFICIENT_DATA", "UNTESTABLE"])))
    _n_corrections = int(_s["n_applied"].sum())

    mo.hstack(
        [
            mo.stat(str(len(_s)), label="Stations processed", bordered=True),
            mo.stat(str(_n_homogeneous), label="🟢 Homogeneous", caption="H*", bordered=True),
            mo.stat(str(_n_corrected), label="🔵 Corrected", caption="HC* / HCC*", bordered=True),
            mo.stat(str(_n_inhomogeneous), label="🔴 Inhomogeneous", caption="IC* / ICC*", bordered=True),
            mo.stat(str(_n_skipped), label="⚪ Skipped", caption="missing / no neighbors", bordered=True),
            mo.stat(str(_n_corrections), label="Corrections", caption="applied during detection", bordered=True),
        ],
        justify="start",
        wrap=True,
    )
    return (detection,)


@app.cell
def _(mo):
    mo.md("## Step 1 — Evaluate detection results")
    return


@app.cell
def _(detection, mo):
    mo.callout(
        mo.md(
            "Review the detected breaks below before applying corrections. "
            "The **group** column shows the final classification: "
            "`H*` = homogeneous, `HC*`/`HCC*` = corrected, `IC*`/`ICC*` = still inhomogeneous, "
            "`INSUFFICIENT_DATA` = too short, `UNTESTABLE` = no neighbors found."
        ),
        kind="info",
    )
    return


@app.cell
def _(detection, pl):
    import altair as _alt  # noqa: ICN001

    _GROUP_PALETTE = {
        "🟢 Homogeneous": "#2ca02c",
        "🔵 Corrected": "#1f77b4",
        "🔴 Inhomogeneous": "#d62728",
        "⚪ Skipped": "#9e9e9e",
    }

    def category_for(g: str) -> str:
        if g.startswith("HC"):
            return "🔵 Corrected"
        if g.startswith("H"):
            return "🟢 Homogeneous"
        if g.startswith("IC"):
            return "🔴 Inhomogeneous"
        return "⚪ Skipped"

    _summary_categorised = detection.summary.with_columns(
        pl.col("group").map_elements(category_for, return_dtype=pl.String).alias("category"),
    )
    _counts = (
        _summary_categorised.group_by("category")
        .agg(pl.len().alias("count"))
        .with_columns(pl.lit("stations").alias("axis"))
    )

    _chart = (
        _alt.Chart(_counts)
        .mark_bar()
        .encode(
            x=_alt.X("count:Q", stack="normalize", axis=_alt.Axis(format=".0%", title="Share of stations")),
            y=_alt.Y("axis:N", axis=None),
            color=_alt.Color(
                "category:N",
                scale=_alt.Scale(domain=list(_GROUP_PALETTE), range=list(_GROUP_PALETTE.values())),
                legend=_alt.Legend(title=None, orient="bottom"),
            ),
            order=_alt.Order("category:N"),
            tooltip=["category:N", "count:Q"],
        )
        .properties(height=46)
    )
    _chart
    return (category_for,)


@app.cell
def _(category_for, detection, mo, pl):
    _summary = detection.summary.with_columns(
        pl.col("group").map_elements(category_for, return_dtype=pl.String).alias("category"),
    ).select("station_id", "category", "group", "n_steps_tested", "n_significant", "n_applied")
    mo.ui.table(
        _summary,
        selection=None,
        label=f"Detection summary — {len(_summary)} stations",
    )
    return


@app.cell
def _(detection, station_histories):
    _MATCH_WINDOW = 3
    history_matches: dict[str, set[int]] = {
        sid: {
            y
            for y in (rec.break_year for rec in det.detections_by_step.values() if rec.break_year is not None)
            if any(abs(y - ey) <= _MATCH_WINDOW for ey in station_histories.get(sid, set()))
        }
        for sid, det in detection.station_detections.items()
    }
    return (history_matches,)


@app.cell
def _(mo):
    group_picker = mo.ui.radio(
        options=["All", "🔴 Inhomogeneous (IC*)", "🔵 Corrected (HC*)", "🟢 Homogeneous (H*)", "⚪ Skipped"],
        value="All",
        label="Filter by group",
        inline=True,
    )
    group_picker
    return (group_picker,)


@app.cell
def _(detection, group_picker, history_matches, mo):
    # Collect all detected break years per station (across all steps)
    _station_breaks: dict[str, set[int]] = {}
    for _sid, _sdet in detection.station_detections.items():
        _yrs: set[int] = set()
        for _rec in _sdet.detections_by_step.values():
            if _rec.break_year is not None:
                _yrs.add(_rec.break_year)
        _station_breaks[_sid] = _yrs

    def _in_group(g: str) -> bool:
        sel = group_picker.value
        if sel == "All":
            return True
        if "Inhomogeneous" in sel:
            return g.startswith("IC")
        if "Corrected" in sel:
            return g.startswith("HC")
        if "Homogeneous" in sel:
            return g.startswith("H") and not g.startswith("HC")
        if "Skipped" in sel:
            return g in ("INSUFFICIENT_DATA", "UNTESTABLE")
        return True

    def _sort_key(sid: str) -> tuple:
        g = detection.station_detections[sid].group
        if g.startswith("IC"):
            return (0, sid)
        if g.startswith("HC"):
            return (1, sid)
        if g.startswith("H"):
            return (2, sid)
        return (3, sid)

    def _label(sid: str) -> str:
        g = detection.station_detections[sid].group
        if g.startswith("IC"):
            sym = "🔴"
        elif g.startswith("HC"):
            sym = "🔵"
        elif g.startswith("H"):
            sym = "🟢"
        else:
            sym = "⚪"
        matched = history_matches.get(sid, set())
        breaks = sorted(_station_breaks[sid])
        if breaks:
            break_str = " ".join(f"{y}★" if y in matched else str(y) for y in breaks)
            return f"{sym} {sid}  [{g}]  ·  {break_str}"
        return f"{sym} {sid}  [{g}]"

    _sorted_ids = sorted(
        (sid for sid, sdet in detection.station_detections.items() if _in_group(sdet.group)),
        key=_sort_key,
    )
    _options = {_label(sid): sid for sid in _sorted_ids}
    _first_key = next(iter(_options)) if _options else None
    station_picker = mo.ui.dropdown(
        options=_options or {"(no stations in group)": None},
        value=_first_key,
        label="Inspect station",
    )
    mo.vstack(
        [
            station_picker,
            mo.md("*★ = break year confirmed in DWD station records (±3 yr)*"),
        ]
    )
    return (station_picker,)


@app.cell
def _(detection, mo, station_picker):
    _sid = station_picker.value
    if _sid is None:
        mo.md("*No stations in selected group.*")
    else:
        _det = detection.station_detections[_sid]
        _n = len(_det.corrections)
        mo.md(f"**Station {_sid}** · group `{_det.group}` · {_n} correction(s) applied during detection")
    return


@app.cell
def _(detection, mo, pl, station_picker):
    _sid = station_picker.value
    if _sid is not None:
        _det = detection.station_detections[_sid]
        _rows = []
        for _step, _rec in sorted(_det.detections_by_step.items()):
            for _tr in _rec.test_results:
                _rows.append(
                    {
                        "step": _step,
                        "test": _tr.test_name,
                        "break_year": _tr.break_year,
                        "statistic": round(_tr.test_statistic, 3),
                        "critical": round(_tr.critical_value, 3),
                        "significant": _tr.is_significant,
                        "applied": _rec.was_applied,
                    }
                )
        _df = pl.DataFrame(_rows) if _rows else pl.DataFrame()
        if len(_df) > 0:
            mo.ui.table(_df, selection=None, label=f"Test results for {_sid}")
        else:
            mo.md(f"*No test results for {_sid}*")
    return


@app.cell
def _(DwdObservationRequest, detection, mo, parameter_picker, pl, station_picker):
    _sid = station_picker.value
    _out = mo.md("")
    if _sid is not None:
        _MATCH_WINDOW = 3  # ± years
        _DATASET = {
            "precipitation_height": "more_precip",
            "temperature_air_mean_2m": "climate_summary",
        }
        _DEVICE_KEYWORD = {
            "precipitation_height": "Niederschlag",
            "temperature_air_mean_2m": "Temperatur",
        }
        _DEVICE_LABEL = {
            "precipitation_height": "Precipitation instruments",
            "temperature_air_mean_2m": "Temperature instruments",
        }

        _param = parameter_picker.value
        _det = detection.station_detections.get(_sid)
        _break_years = {rec.break_year for rec in _det.detections_by_step.values()} if _det else set()

        def _match(event_year: int | None):
            from datetime import date

            if event_year is None or not _break_years:
                return None
            candidates = [y for y in _break_years if abs(y - event_year) <= _MATCH_WINDOW]
            matched = min(candidates, key=lambda y: abs(y - event_year)) if candidates else None
            return date(matched, 1, 1) if matched is not None else None

        with mo.status.spinner(f"Fetching station history for {_sid} …"):
            _req = DwdObservationRequest(parameters=[("annual", _DATASET[_param], _param)])
            _sr = _req.filter_by_station_id(_sid)
            try:
                _hist_item = next(_sr.history.query(), None)
            except UnicodeDecodeError:
                _hist_item = None
                _decode_error = True
            else:
                _decode_error = False

        if _decode_error:
            _out = mo.callout(
                mo.md(
                    f"Station history unavailable for **{_sid}** — encoding error in DWD source file"
                    " (known wetterdienst issue for stations with umlauts in their name)."
                ),
                kind="warn",
            )
        elif _hist_item is None:
            _out = mo.md("*No station history available.*")
        else:
            _hist = _hist_item.history

            _geo_rows = [
                {
                    "from": g.start_date.date() if g.start_date else None,
                    "to": g.end_date.date() if g.end_date else None,
                    "latitude": round(g.latitude, 4),
                    "longitude": round(g.longitude, 4),
                    "height_m": g.station_height,
                    "break_match": _match(g.start_date.year if g.start_date else None),
                }
                for g in _hist.geography
            ]

            _kw = _DEVICE_KEYWORD[_param]
            _dev_rows = [
                {
                    "from": d.start_date.date() if d.start_date else None,
                    "to": d.end_date.date() if d.end_date else None,
                    "device": d.device_type,
                    "method": d.method,
                    "break_match": _match(d.start_date.year if d.start_date else None),
                }
                for d in _hist.device
                if _kw in (d.method or "")
            ]

            _op_rows = [
                {
                    "from": o.start_date.date() if o.start_date else None,
                    "to": o.end_date.date() if o.end_date else None,
                    "operator": o.operator_name,
                    "break_match": _match(o.start_date.year if o.start_date else None),
                }
                for o in _hist.name.operator
            ]

            _all_rows = _geo_rows + _dev_rows + _op_rows
            _matched_breaks = {r["break_match"] for r in _all_rows if r["break_match"] is not None}
            _n_matched = len(_matched_breaks)
            _n_total = len(_break_years)

            if _n_total == 0:
                _callout_text = "No breaks detected for this station."
                _callout_kind = "neutral"
            elif _n_matched == _n_total:
                _callout_text = (
                    f"All **{_n_total}** detected break(s) match a station history event (±{_MATCH_WINDOW} yr)."
                )
                _callout_kind = "success"
            elif _n_matched > 0:
                _callout_text = (
                    f"**{_n_matched} of {_n_total}** detected breaks"
                    f" match a station history event (±{_MATCH_WINDOW} yr)."
                )
                _callout_kind = "warn"
            else:
                _callout_text = (
                    f"None of the **{_n_total}** detected break(s) match a station history event (±{_MATCH_WINDOW} yr)."
                )
                _callout_kind = "danger"

            _matched_years = {r["break_match"].year for r in _all_rows if r["break_match"] is not None}
            _break_years_str = (
                ", ".join(f"{y}★" if y in _matched_years else str(y) for y in sorted(_break_years))
                if _break_years
                else "—"
            )
            _out = mo.vstack(
                [
                    mo.md("### Station history"),
                    mo.md(f"**Detected break years:** {_break_years_str}  *(★ = confirmed in station records)*"),
                    mo.callout(mo.md(_callout_text), kind=_callout_kind),
                    mo.ui.table(pl.DataFrame(_geo_rows), selection=None, label="Relocations"),
                    mo.ui.table(pl.DataFrame(_dev_rows), selection=None, label=_DEVICE_LABEL[_param]),
                    mo.ui.table(pl.DataFrame(_op_rows), selection=None, label="Operators"),
                ]
            )

    _out
    return


@app.cell
def _(mo):
    mo.md("## Step 2 — Configure normalization")
    return


@app.cell
def _(mo):
    tiebreak = mo.ui.dropdown(
        options=["strongest_signal", "skip"],
        value="strongest_signal",
        label="Tiebreak",
    )
    break_window = mo.ui.slider(start=1, stop=5, value=2, label="Break window ± years")
    min_magnitude = mo.ui.slider(start=0.0, stop=0.20, step=0.01, value=0.0, label="Min correction magnitude")
    max_corrections = mo.ui.slider(start=1, stop=5, value=2, label="Max corrections per station")
    history_only = mo.ui.checkbox(label="Only correct breaks confirmed in station records (★)", value=True)
    mo.vstack(
        [
            mo.hstack([tiebreak], justify="start"),
            mo.hstack([break_window, min_magnitude, max_corrections], justify="start"),
            mo.hstack([history_only], justify="start"),
        ]
    )
    return break_window, history_only, max_corrections, min_magnitude, tiebreak


@app.cell
def _(
    BreakInfo,
    BreakPredicate,
    NormalizationConfig,
    break_window,
    consensus_picker,
    detection,
    history_matches,
    history_only,
    max_corrections,
    min_magnitude,
    tiebreak,
):
    class _HistoryConfirmed(BreakPredicate):
        def __call__(self, info: BreakInfo) -> bool:
            return info.break_year in history_matches.get(info.station_id, set())

        def to_dict(self) -> dict:
            return {"type": "_history_confirmed"}

    _predicate = _HistoryConfirmed() if history_only.value else None
    result = detection.normalize(
        NormalizationConfig(
            consensus=consensus_picker.value,
            tiebreak=tiebreak.value,
            break_window_years=break_window.value,
            min_correction_magnitude=min_magnitude.value,
            max_corrections_per_station=max_corrections.value,
            predicate=_predicate,
        )
    )
    return (result,)


@app.cell
def _(mo):
    mo.accordion(
        {
            "Alternative normalisation recipes — predicates": mo.md(
                """
Replace the widget-driven `result = detection.normalize(...)` cell above with one of
these in **edit mode**. Add the imports and variable names shown to the cell's parameter list.

```python
from rucola import (
    MagnitudeAbove, NeighborCountAbove, NormalizationConfig,
    NSignificantAbove, SignalAbove, StationIn, StepIn, YearBetween,
)

# 1. Strict evidence — strong signal, ≥2 tests agree, solid reference pool
result = detection.normalize(
    NormalizationConfig(
        predicate=SignalAbove(1.5) & NSignificantAbove(2) & NeighborCountAbove(4),
    )
)

# 2. Trusted year window — restrict corrections to a known reliable period
result = detection.normalize(
    NormalizationConfig(predicate=YearBetween(min=1950, max=2010))
)

# 3. Skip early steps (fewer neighbors, less stable) + minimum magnitude
result = detection.normalize(
    NormalizationConfig(
        predicate=~StepIn({1, 2}) & MagnitudeAbove(threshold=0.03),
    )
)

# 4. Apply corrections only to a specific subset of stations
result = detection.normalize(
    NormalizationConfig(predicate=StationIn({"01051", "01612"}))
)

# 5. Combine consensus setting with a predicate
result = detection.normalize(
    NormalizationConfig(
        consensus="unanimous",
        predicate=SignalAbove(1.5) & YearBetween(min=1960),
    )
)
```
"""
            ),
            "Alternative normalisation recipes — manual overrides": mo.md(
                """
Use when the station history tables above confirm a break that the algorithm missed,
placed in the wrong year, or got the wrong factor for. Overrides bypass detection and
all predicates for the named station. Add `detection, NormalizationConfig` to the cell's
parameter list.

```python
# Single station — one confirmed relocation
result = detection.normalize(
    NormalizationConfig(
        overrides={
            "01051": [(1977, 1.08)],
        }
    )
)

# Multiple stations, multiple events
result = detection.normalize(
    NormalizationConfig(
        overrides={
            "01051": [(1977, 1.08)],
            "01612": [(1965, 0.95), (1990, 1.07)],
        }
    )
)

# Combined: predicate-filtered algorithm + manual overrides for edge cases
# (overrides bypass the predicate; algorithmic breaks must still pass it)
result = detection.normalize(
    NormalizationConfig(
        consensus="majority",
        predicate=SignalAbove(1.5) & NSignificantAbove(2),
        overrides={
            "01051": [(1977, 1.08)],
        },
    )
)
```
"""
            ),
        }
    )


@app.cell
def _(mo):
    mo.md("## Step 3 — Corrected results")
    return


@app.cell
def _(mo, pl, result):
    _corr = result.corrections
    _summary = result.summary
    _n_corrected = len(_summary.filter(pl.col("n_corrections") > 0))
    _n_homogeneous = len(_summary.filter(pl.col("n_corrections") == 0))

    mo.hstack(
        [
            mo.stat(str(_n_homogeneous), label="Stations unchanged"),
            mo.stat(str(_n_corrected), label="Stations corrected"),
            mo.stat(str(len(_corr)), label="Total corrections"),
            mo.stat(
                f"{_corr['factor'].mean():.3f}" if len(_corr) > 0 else "—",
                label="Mean correction factor",
            ),
        ]
    )
    return


@app.cell
def _(mo, result):
    mo.ui.table(
        result.corrections,
        selection=None,
        label="Applied corrections",
    )
    return


@app.cell
def _(mo, result):
    mo.ui.table(
        result.summary,
        selection=None,
        label="Homogenization summary",
    )
    return


@app.cell
def _(mo):
    mo.md("## Step 4 — Homogenized series")
    return


@app.cell
def _(mo):
    only_corrected = mo.ui.checkbox(label="Only stations with corrections", value=True)
    return (only_corrected,)


@app.cell
def _(detection, mo, only_corrected, result):
    def _sort_key(sid):
        g = result.station_results[sid].group if sid in result.station_results else "ZZZ"
        if g.startswith("IC"):
            return (0, sid)
        if g.startswith("HC"):
            return (1, sid)
        if g.startswith("H"):
            return (2, sid)
        return (3, sid)

    def _label(sid):
        sr = result.station_results.get(sid)
        g = sr.group if sr else "—"
        n = sr.n_corrections if sr else 0
        if n > 0:
            symbol = "🔵"
        elif g.startswith("IC"):
            symbol = "🔴"
        elif g.startswith("H"):
            symbol = "🟢"
        else:
            symbol = "⚪"
        corr = f"  · {n} correction{'s' if n != 1 else ''}" if n > 0 else ""
        return f"{symbol} {sid}  [{g}]{corr}"

    def _has_corrections(sid: str) -> bool:
        sr = result.station_results.get(sid)
        return bool(sr and sr.n_corrections > 0)

    _ids = detection.station_detections.keys()
    if only_corrected.value:
        _ids = [sid for sid in _ids if _has_corrections(sid)]
    _sorted_ids = sorted(_ids, key=_sort_key)
    _options = {_label(sid): sid for sid in _sorted_ids}
    _first_key = next(iter(_options)) if _options else None
    chart_station_picker = mo.ui.dropdown(
        options=_options or {"(no stations with corrections)": None},
        value=_first_key,
        label="Plot station",
    )
    mo.vstack([only_corrected, chart_station_picker])
    return (chart_station_picker,)


@app.cell
def _(mo):
    show_trendlines = mo.ui.checkbox(label="Show trendlines", value=True)
    show_trendlines
    return (show_trendlines,)


@app.cell
def _(chart_station_picker, mo, parameter_picker, pl, result, show_trendlines):
    import altair as alt

    _sid = chart_station_picker.value
    _sr = result.station_results.get(_sid)

    if _sr is None:
        _chart_out = mo.md(f"*No homogenization result for station {_sid}.*")
    else:
        _y_label = {
            "precipitation_height": "Annual precipitation (mm)",
            "temperature_air_mean_2m": "Mean annual temperature (°C)",
        }[parameter_picker.value]

        _years = _sr.years.cast(pl.Int32).to_list()
        _valid_corrections = _sr.corrections
        _corrected = bool(_valid_corrections)
        _x = alt.X("year:Q", title="Year", axis=alt.Axis(format="d", tickMinStep=1))
        _y = alt.Y("value:Q", title=_y_label)

        if _corrected:
            _df = pl.DataFrame(
                {
                    "year": _years,
                    "Original": _sr.annual_original.cast(pl.Float64).to_list(),
                    "Homogenized": _sr.annual_corrected.cast(pl.Float64).to_list(),
                }
            ).unpivot(index="year", variable_name="series", value_name="value")

            _lines = (
                alt.Chart(_df)
                .mark_line()
                .encode(
                    x=_x,
                    y=_y,
                    color=alt.Color(
                        "series:N",
                        scale=alt.Scale(
                            domain=["Original", "Homogenized"],
                            range=["#bbbbbb", "#1f77b4"],
                        ),
                        legend=alt.Legend(title=None, orient="top-left"),
                    ),
                    strokeWidth=alt.condition(alt.datum.series == "Homogenized", alt.value(2), alt.value(1)),
                    opacity=alt.condition(alt.datum.series == "Homogenized", alt.value(1.0), alt.value(0.7)),
                )
            )

            _break_df = pl.DataFrame(
                {
                    "break_year": [c.break_year for c in _valid_corrections],
                    "factor": [f"×{c.factor:.3f}" for c in _valid_corrections],
                }
            )
            _rules = (
                alt.Chart(_break_df)
                .mark_rule(color="#d62728", strokeDash=[4, 3], strokeWidth=1.5)
                .encode(x="break_year:Q")
            )
            _labels = (
                alt.Chart(_break_df)
                .mark_text(align="left", dx=4, dy=-8, color="#d62728", fontSize=11)
                .encode(x="break_year:Q", y=alt.value(16), text="factor:N")
            )
            if show_trendlines.value:
                _trends = (
                    alt.Chart(_df)
                    .transform_regression("year", "value", groupby=["series"])
                    .mark_line(strokeDash=[6, 3], strokeWidth=1.5)
                    .encode(
                        x=_x,
                        y=_y,
                        color=alt.Color(
                            "series:N",
                            scale=alt.Scale(
                                domain=["Original", "Homogenized"],
                                range=["#bbbbbb", "#1f77b4"],
                            ),
                            legend=None,
                        ),
                        opacity=alt.condition(alt.datum.series == "Homogenized", alt.value(1.0), alt.value(0.7)),
                    )
                )
                _spec = _lines + _trends + _rules + _labels
            else:
                _spec = _lines + _rules + _labels

        else:
            _df = pl.DataFrame(
                {
                    "year": _years,
                    "value": _sr.annual_original.cast(pl.Float64).to_list(),
                }
            )
            _line = alt.Chart(_df).mark_line(color="#1f77b4", strokeWidth=2).encode(x=_x, y=_y)
            if show_trendlines.value:
                _trend = (
                    alt.Chart(_df)
                    .transform_regression("year", "value")
                    .mark_line(color="#1f77b4", strokeWidth=1.5, strokeDash=[6, 3])
                    .encode(x=_x, y=_y)
                )
                _spec = _line + _trend
            else:
                _spec = _line

        _no_corr_reason = {
            "INSUFFICIENT_DATA": "insufficient data for testing",
            "UNTESTABLE": "no reference neighbors found",
        }.get(_sr.group, "no corrections applied")
        _subtitle = (
            [f"{len(_valid_corrections)} correction(s) applied"]
            if _corrected
            else [f"Original series only — {_no_corr_reason}"]
        )

        _chart_out = mo.ui.altair_chart(
            _spec.properties(
                title=alt.TitleParams(
                    f"Station {_sid} — group {_sr.group}",
                    subtitle=_subtitle,
                ),
                width=620,
                height=300,
            ).interactive()
        )

    _chart_out
    return


if __name__ == "__main__":
    app.run()
