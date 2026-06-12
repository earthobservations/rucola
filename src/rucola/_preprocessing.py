"""Pre-processing utilities for preparing daily data before homogenization."""

from __future__ import annotations

import polars as pl

_DAYS_PER_YEAR = 365.0


def winsorize_outliers(values: pl.DataFrame) -> pl.DataFrame:
    """Trim daily values above P_out = q_0.75 + 3·IQR per station.

    Implements Eq. 1 from González-Rouco et al. (2001). The extreme value is
    replaced by P_out rather than removed, preserving the event while reducing
    its influence on non-resistant statistics. Requires ``station_id`` and
    ``value`` columns.
    """
    thresholds = (
        values.group_by("station_id")
        .agg(
            pl.col("value").quantile(0.75).alias("q75"),
            (pl.col("value").quantile(0.75) - pl.col("value").quantile(0.25)).alias("iqr"),
        )
        .with_columns((pl.col("q75") + 3 * pl.col("iqr")).alias("p_out"))
        .select("station_id", "p_out")
    )
    return (
        values.join(thresholds, on="station_id", how="left")
        .with_columns(
            pl.when(pl.col("value") > pl.col("p_out")).then(pl.col("p_out")).otherwise(pl.col("value")).alias("value"),
        )
        .drop("p_out")
    )


def compute_annual_totals(
    values: pl.DataFrame,
    min_coverage: float = 0.8,
) -> pl.DataFrame:
    """Aggregate daily values to annual totals (e.g. precipitation).

    Years with fewer than ``min_coverage × 365`` valid observations are set to
    null. Returns one row per (station_id, year) with a ``date`` column set to
    Jan 1 of each year, making the result directly compatible with ``Rucola``.
    """
    return (
        values.with_columns(pl.col("date").dt.year().alias("_year"))
        .group_by("station_id", "_year", "parameter")
        .agg(
            pl.col("value").count().alias("n_obs"),
            pl.col("value").sum().alias("value"),
        )
        .with_columns(
            pl.when(pl.col("n_obs") >= min_coverage * _DAYS_PER_YEAR)
            .then(pl.col("value"))
            .otherwise(None)
            .alias("value"),
            pl.date(pl.col("_year"), 1, 1).alias("date"),
        )
        .drop("_year", "n_obs")
        .sort("station_id", "date")
    )


def compute_annual_means(
    values: pl.DataFrame,
    min_coverage: float = 0.8,
) -> pl.DataFrame:
    """Aggregate daily values to annual means (e.g. temperature).

    Years with fewer than ``min_coverage × 365`` valid observations are set to
    null. Returns one row per (station_id, year) compatible with ``Rucola``.
    """
    return (
        values.with_columns(pl.col("date").dt.year().alias("_year"))
        .group_by("station_id", "_year", "parameter")
        .agg(
            pl.col("value").count().alias("n_obs"),
            pl.col("value").mean().alias("value"),
        )
        .with_columns(
            pl.when(pl.col("n_obs") >= min_coverage * _DAYS_PER_YEAR)
            .then(pl.col("value"))
            .otherwise(None)
            .alias("value"),
            pl.date(pl.col("_year"), 1, 1).alias("date"),
        )
        .drop("_year", "n_obs")
        .sort("station_id", "date")
    )
