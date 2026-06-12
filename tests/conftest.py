"""Shared fixtures for rucola tests."""

from __future__ import annotations

import math
import random
from datetime import date

import polars as pl
import pytest

import rucola

YEARS = list(range(1990, 2022))  # 32 years, even split around break
BREAK_YEAR = 2006
BREAK_FACTOR = 0.85  # 15% drop pre-break — detectable but keeps neighbor correlation above 0.5

_STATIONS = {
    "S1": (51.00, 13.00),
    "S2": (51.10, 13.10),
    "S3": (51.20, 12.90),
    "S4": (50.90, 13.20),
    "S5": (51.05, 13.05),
}


def _synthetic_values(station_id: str, seed: int) -> list[float]:
    rng = random.Random(seed)
    vals = [500.0 + 50.0 * math.sin(2 * math.pi * i / 10) + rng.gauss(0, 15) for i in range(len(YEARS))]
    if station_id == "S1":
        vals = [v * BREAK_FACTOR if y < BREAK_YEAR else v for v, y in zip(vals, YEARS, strict=True)]
    return vals


@pytest.fixture(scope="session")
def stations_df() -> pl.DataFrame:
    """DataFrame of synthetic station metadata."""
    return pl.DataFrame(
        {
            "station_id": list(_STATIONS),
            "latitude": [c[0] for c in _STATIONS.values()],
            "longitude": [c[1] for c in _STATIONS.values()],
        }
    )


@pytest.fixture(scope="session")
def values_df() -> pl.DataFrame:
    """DataFrame of synthetic annual values with a break injected in S1."""
    rows = []
    for i, sid in enumerate(_STATIONS):
        for year, val in zip(YEARS, _synthetic_values(sid, seed=i), strict=True):
            rows.append({"station_id": sid, "date": date(year, 1, 1), "value": val, "parameter": "precip"})
    return pl.DataFrame(rows).sort("station_id", "date")


@pytest.fixture(scope="session")
def rucola_instance(stations_df: pl.DataFrame, values_df: pl.DataFrame) -> rucola.Rucola:
    """Rucola instance built from the synthetic 5-station dataset."""
    return rucola.Rucola.from_polars(values_df, stations_df, parameter="precip")
