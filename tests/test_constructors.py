"""Tests for Rucola constructors — from_csv, from_duckdb, and data filters."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import duckdb
import polars as pl
import pytest

import rucola

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

_N_STATIONS = 5
_GAP_START = 2000
_GAP_END = 2010
_GAP_THRESHOLD = 5
_SINGLE_NULL_YEAR = 2005


def test_from_csv(values_df: pl.DataFrame, stations_df: pl.DataFrame, tmp_path: Path) -> None:
    """Rucola.from_csv loads data correctly from two CSV files."""
    vp = tmp_path / "values.csv"
    sp = tmp_path / "stations.csv"
    values_df.write_csv(str(vp))
    stations_df.write_csv(str(sp))
    r = rucola.Rucola.from_csv(vp, sp, parameter="precip")
    assert r.parameter == "precip"
    assert r.n_stations == _N_STATIONS


def test_from_duckdb(values_df: pl.DataFrame, stations_df: pl.DataFrame, tmp_path: Path) -> None:
    """Rucola.from_duckdb loads data correctly from default table names."""
    db_path = str(tmp_path / "data.duckdb")
    with duckdb.connect(db_path) as con:
        con.register("stations_df", stations_df)
        con.register("values_df", values_df)
        con.execute("CREATE TABLE stations AS SELECT * FROM stations_df")
        con.execute("CREATE TABLE values AS SELECT * FROM values_df")
    r = rucola.Rucola.from_duckdb(db_path, parameter="precip")
    assert r.parameter == "precip"
    assert r.n_stations == _N_STATIONS


def test_from_duckdb_custom_table_names(values_df: pl.DataFrame, stations_df: pl.DataFrame, tmp_path: Path) -> None:
    """Rucola.from_duckdb accepts custom table names via keyword arguments."""
    db_path = str(tmp_path / "data.duckdb")
    with duckdb.connect(db_path) as con:
        con.register("stations_df", stations_df)
        con.register("values_df", values_df)
        con.execute("CREATE TABLE my_stations AS SELECT * FROM stations_df")
        con.execute("CREATE TABLE my_values AS SELECT * FROM values_df")
    r = rucola.Rucola.from_duckdb(db_path, stations_table="my_stations", values_table="my_values")
    assert r.n_stations == _N_STATIONS


def test_max_gap_years_rejects_gapped_station(stations_df: pl.DataFrame) -> None:
    """Station with a consecutive null gap exceeding max_gap_years is labelled INSUFFICIENT_DATA."""
    years = list(range(1990, 2022))
    rows = [
        {
            "station_id": "S1",
            "date": date(y, 1, 1),
            "value": None if _GAP_START <= y <= _GAP_END else 100.0,
            "parameter": "p",
        }
        for y in years
    ]
    values = pl.DataFrame(rows).sort("station_id", "date")
    r = rucola.Rucola(values, stations_df.filter(pl.col("station_id") == "S1"))
    result = r.run(rucola.RunConfig(max_gap_years=_GAP_THRESHOLD))
    assert result.station_detections["S1"].group == "INSUFFICIENT_DATA"


def test_max_gap_years_accepts_small_gap(stations_df: pl.DataFrame) -> None:
    """Station with a gap smaller than max_gap_years passes the filter."""
    years = list(range(1990, 2022))
    rows = [
        {
            "station_id": "S1",
            "date": date(y, 1, 1),
            "value": None if y == _SINGLE_NULL_YEAR else 100.0,
            "parameter": "p",
        }
        for y in years
    ]
    values = pl.DataFrame(rows).sort("station_id", "date")
    r = rucola.Rucola(values, stations_df.filter(pl.col("station_id") == "S1"))
    result = r.run(rucola.RunConfig(max_gap_years=_GAP_THRESHOLD))
    assert result.station_detections["S1"].group != "INSUFFICIENT_DATA"


# ── input validation ─────────────────────────────────────────────────────────


def test_non_numeric_value_column_raises(stations_df: pl.DataFrame) -> None:
    """Non-numeric value column raises TypeError."""
    values = pl.DataFrame({"station_id": ["S1"], "date": [date(2000, 1, 1)], "value": ["abc"], "parameter": ["p"]})
    with pytest.raises(TypeError, match="numeric"):
        rucola.Rucola(values, stations_df.filter(pl.col("station_id") == "S1"))


def test_duplicate_station_id_raises(values_df: pl.DataFrame) -> None:
    """Duplicate station_id in stations raises ValueError."""
    stations = pl.DataFrame({"station_id": ["S1", "S1"], "latitude": [51.0, 51.0], "longitude": [13.0, 13.0]})
    with pytest.raises(ValueError, match="duplicate"):
        rucola.Rucola(values_df.filter(pl.col("station_id") == "S1"), stations)


def test_null_latitude_raises(values_df: pl.DataFrame) -> None:
    """Null latitude in stations raises ValueError."""
    stations = pl.DataFrame({"station_id": ["S1"], "latitude": [None], "longitude": [13.0]})
    with pytest.raises(ValueError, match="null"):
        rucola.Rucola(values_df.filter(pl.col("station_id") == "S1"), stations)


def test_latitude_out_of_range_raises(values_df: pl.DataFrame) -> None:
    """Latitude outside [-90, 90] raises ValueError."""
    stations = pl.DataFrame({"station_id": ["S1"], "latitude": [91.0], "longitude": [13.0]})
    with pytest.raises(ValueError, match="latitude"):
        rucola.Rucola(values_df.filter(pl.col("station_id") == "S1"), stations)


def test_longitude_out_of_range_raises(values_df: pl.DataFrame) -> None:
    """Longitude outside [-180, 180] raises ValueError."""
    stations = pl.DataFrame({"station_id": ["S1"], "latitude": [51.0], "longitude": [181.0]})
    with pytest.raises(ValueError, match="longitude"):
        rucola.Rucola(values_df.filter(pl.col("station_id") == "S1"), stations)


def test_no_stations_runs_successfully(values_df: pl.DataFrame) -> None:
    """Rucola works without a stations table, using correlation-only neighbour selection."""
    r = rucola.Rucola(values_df)
    result = r.run()
    assert isinstance(result, rucola.DetectionResult)


def test_no_stations_distance_filter_raises(values_df: pl.DataFrame) -> None:
    """Requesting max_distance_km without station coordinates raises ValueError."""
    r = rucola.Rucola(values_df)
    with pytest.raises(ValueError, match="max_distance_km"):
        r.run(rucola.RunConfig(max_distance_km=500.0))


def test_no_stations_distance_km_is_none(values_df: pl.DataFrame) -> None:
    """NeighborInfo.distance_km is None when no station coordinates are provided."""
    r = rucola.Rucola(values_df)
    result = r.run()
    for det in result.station_detections.values():
        for neighbors in det.neighbors_by_step.values():
            for n in neighbors:
                assert n.distance_km is None


def test_from_csv_without_stations(values_df: pl.DataFrame, tmp_path: Path) -> None:
    """from_csv works when stations_path is omitted."""
    vp = tmp_path / "values.csv"
    values_df.write_csv(str(vp))
    r = rucola.Rucola.from_csv(vp)
    assert isinstance(r, rucola.Rucola)


def test_duplicate_dates_raises() -> None:
    """Two records for the same station and date raise ValueError."""
    values = pl.DataFrame(
        {
            "station_id": ["S1", "S1"],
            "date": [date(2000, 1, 1), date(2000, 1, 1)],
            "value": [10.0, 11.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate dates"):
        rucola.Rucola(values)


def test_multiple_parameters_raises() -> None:
    """Values with more than one parameter value raise ValueError."""
    values = pl.DataFrame(
        {
            "station_id": ["S1", "S1"],
            "date": [date(2000, 1, 1), date(2001, 1, 1)],
            "value": [10.0, 11.0],
            "parameter": ["precip", "temp"],
        }
    )
    with pytest.raises(ValueError, match="multiple parameters"):
        rucola.Rucola(values)
