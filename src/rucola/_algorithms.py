import math
from dataclasses import dataclass
from typing import Literal, get_args

import polars as pl

_NEAR_ZERO = 1e-12

# All valid group labels produced by the 6-step procedure.
GroupLabel = Literal[
    "",
    "H1",
    "I1",
    "H2",
    "I2",
    "HC3",
    "IC3",
    "H4",
    "I4",
    "HC4",
    "IC4",
    "HC5",
    "IC5",
    "HCC6",
    "ICC6",
    "UNTESTABLE",  # no neighbors found at any step
    "INSUFFICIENT_DATA",  # series too short or gap too large
]
_VALID_GROUPS: frozenset[str] = frozenset(get_args(GroupLabel))

CorrectionMode = Literal["ratio", "difference"]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return earth_radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class NeighborInfo:
    station_id: str
    distance_km: float | None  # None when distance filtering is disabled
    correlation: float
    weight: float  # = r²  (square of Pearson correlation, per paper)


def build_distance_cache(stations: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Precompute all pairwise haversine distances between stations.

    Call once per ``Rucola`` instance and reuse across all procedure steps.
    """
    rows = {r["station_id"]: r for r in stations.select("station_id", "latitude", "longitude").iter_rows(named=True)}
    cache: dict[str, dict[str, float]] = {sid: {} for sid in rows}
    ids = list(rows)
    for i, sid_a in enumerate(ids):
        ra = rows[sid_a]
        for sid_b in ids[i + 1 :]:
            rb = rows[sid_b]
            d = haversine_km(ra["latitude"], ra["longitude"], rb["latitude"], rb["longitude"])
            cache[sid_a][sid_b] = d
            cache[sid_b][sid_a] = d
    return cache


def build_correlation_cache(
    annual_wide: pl.DataFrame,
    min_overlap_years: int = 10,
) -> dict[str, dict[str, float]]:
    """Precompute all valid pairwise Pearson correlations from the wide annual DataFrame.

    Only pairs with at least ``min_overlap_years`` jointly non-null observations
    are included. Uses ``numpy`` for the overlap count matrix multiply.
    Call once per step (the corrected series changes each step).
    """
    station_cols = [c for c in annual_wide.columns if c != "year"]
    if not station_cols:
        return {}

    sub = annual_wide.select(station_cols)
    cache: dict[str, dict[str, float]] = {}

    for sid_a in station_cols:
        others = [s for s in station_cols if s != sid_a]
        if not others:
            cache[sid_a] = {}
            continue

        # Compute correlations and overlap counts for all partners in one pass
        col_a_valid = pl.col(sid_a).is_not_null()
        row = sub.select(
            [pl.corr(sid_a, s).alias(f"c_{s}") for s in others]
            + [(col_a_valid & pl.col(s).is_not_null()).sum().alias(f"o_{s}") for s in others]
        ).row(0, named=True)

        cache[sid_a] = {
            s: float(v)
            for s in others
            if (v := row[f"c_{s}"]) is not None and not math.isnan(float(v)) and row[f"o_{s}"] >= min_overlap_years
        }

    return cache


def select_neighbors(  # noqa: C901, PLR0912, PLR0913
    candidate_id: str,
    stations: pl.DataFrame,
    annual_wide: pl.DataFrame,
    max_neighbors: int = 10,
    min_correlation: float = 0.5,
    max_distance_km: float | None = None,
    min_overlap_years: int = 10,
    allowed_ids: set[str] | None = None,
    *,
    dist_cache: dict[str, dict[str, float]] | None = None,
    corr_cache: dict[str, dict[str, float]] | None = None,
) -> list[NeighborInfo]:
    """Select reference neighbors for a candidate station.

    Computes Pearson correlation on overlapping non-null years and filters to
    min_correlation. Weights = r² (square of correlation). If max_distance_km is
    given, stations beyond that radius are excluded first. allowed_ids restricts
    the reference pool (used in steps 2–6 of the procedure).

    Pass ``dist_cache`` and ``corr_cache`` (from ``build_distance_cache`` /
    ``build_correlation_cache``) for a large speedup on repeated calls.
    """
    if candidate_id not in annual_wide.columns:
        return []

    neighbors: list[NeighborInfo] = []

    if corr_cache is not None:
        # Fast path: O(1) dict lookups instead of per-pair DataFrame ops
        cand_dists = dist_cache.get(candidate_id, {}) if dist_cache is not None else {}
        cand_corrs = corr_cache.get(candidate_id, {})
        candidates = [
            sid
            for sid in annual_wide.columns
            if sid not in {"year", candidate_id} and (allowed_ids is None or sid in allowed_ids)
        ]
        for sid in candidates:
            dist: float | None = cand_dists.get(sid)
            if max_distance_km is not None:
                if dist is None:
                    continue
                if dist > max_distance_km:
                    continue
            corr = cand_corrs.get(sid)
            if corr is None or corr < min_correlation:
                continue
            neighbors.append(NeighborInfo(sid, dist, corr, weight=corr**2))
    else:
        # Slow path: original per-pair computation (fallback / no-cache case)
        cand = stations.filter(pl.col("station_id") == candidate_id).row(0, named=True)
        for row in stations.filter(pl.col("station_id") != candidate_id).iter_rows(named=True):
            sid = row["station_id"]
            if sid not in annual_wide.columns:
                continue
            if allowed_ids is not None and sid not in allowed_ids:
                continue
            dist = (
                haversine_km(cand["latitude"], cand["longitude"], row["latitude"], row["longitude"])
                if max_distance_km is not None
                else None
            )
            if max_distance_km is not None and dist is not None and dist > max_distance_km:
                continue
            overlap = annual_wide.select("year", candidate_id, sid).drop_nulls()
            if len(overlap) < min_overlap_years:
                continue
            corr = overlap.select(pl.corr(candidate_id, sid)).item()
            if corr is None or math.isnan(corr) or corr < min_correlation:
                continue
            neighbors.append(NeighborInfo(sid, dist, corr, weight=corr**2))

    neighbors.sort(key=lambda n: n.correlation, reverse=True)
    return neighbors[:max_neighbors]


def build_reference_series(
    annual_wide: pl.DataFrame,
    neighbors: list[NeighborInfo],
    mode: CorrectionMode = "ratio",
) -> pl.Series:
    """Build the weighted reference series G_i.

    ratio mode (precipitation, Eq. 3 González-Rouco 2001): each neighbor is
    normalized by its long-term mean before weighting, producing a dimensionless
    series. G_i = Σ_j( r_j² · Q_ij/Q̄_j ) / Σ_j( r_j² )

    difference mode (temperature): weighted mean of raw neighbor values.
    G_i = Σ_j( r_j² · T_ij ) / Σ_j( r_j² )
    """
    neighbor_means: dict[str, float] = {}
    valid_station_ids: set[str] = set()

    for n in neighbors:
        vals = annual_wide[n.station_id].drop_nulls().cast(pl.Float64)
        if len(vals) == 0:
            continue
        if mode == "ratio":
            raw = vals.mean()
            if isinstance(raw, float) and raw > 0.0:
                neighbor_means[n.station_id] = raw
                valid_station_ids.add(n.station_id)
        else:
            valid_station_ids.add(n.station_id)

    valid_neighbors = [n for n in neighbors if n.station_id in valid_station_ids]
    if not valid_neighbors:
        return pl.Series("reference", [None] * len(annual_wide), dtype=pl.Float64)

    ref: list[float | None] = []
    for i in range(len(annual_wide)):
        total_w = 0.0
        total_v = 0.0
        for n in valid_neighbors:
            val = annual_wide[n.station_id][i]
            if val is not None:
                weight_val = val / neighbor_means[n.station_id] if mode == "ratio" else val
                total_v += n.weight * weight_val
                total_w += n.weight
        ref.append(total_v / total_w if total_w > 0 else None)

    return pl.Series("reference", ref)


def compute_q_series(
    candidate: pl.Series,
    reference: pl.Series,
    mode: CorrectionMode = "ratio",
) -> pl.Series:
    """Compute the Q-series used as input to the SNHT.

    ratio mode (precipitation, Eq. 2 González-Rouco 2001):
        q_i = (P_i / P̄) / G_i  — null where candidate, reference, or means are zero.

    difference mode (temperature):
        q_i = T_i - G_i  — null where either value is null.
    """
    cand_vals = candidate.to_list()
    ref_vals = reference.to_list()

    q: list[float | None] = []
    if mode == "ratio":
        valid_cand = [v for v in cand_vals if v is not None]
        p_mean = sum(valid_cand) / len(valid_cand) if valid_cand else None
        for p, g in zip(cand_vals, ref_vals, strict=True):
            if p is None or g is None or p_mean is None or p_mean == 0 or g == 0:
                q.append(None)
            else:
                q.append((p / p_mean) / g)
    else:
        for p, g in zip(cand_vals, ref_vals, strict=True):
            q.append(p - g if p is not None and g is not None else None)

    return pl.Series("q", q)


def compute_correction_factor(
    q_series: pl.Series,
    years: pl.Series,
    break_year: int,
    mode: CorrectionMode = "ratio",
) -> float:
    """Compute the correction factor to align the pre-break segment with post-break.

    ratio mode (precipitation, Eq. 5 González-Rouco 2001):
        f = q̄_after / q̄_before  — multiplicative factor, neutral value 1.0.

    difference mode (temperature):
        f = q̄_after - q̄_before  — additive offset, neutral value 0.0.
    """
    q_vals = q_series.to_list()
    y_vals = years.to_list()

    before = [q for q, y in zip(q_vals, y_vals, strict=True) if y < break_year and q is not None]
    after = [q for q, y in zip(q_vals, y_vals, strict=True) if y >= break_year and q is not None]

    if not before or not after:
        return 1.0 if mode == "ratio" else 0.0

    q_before = sum(before) / len(before)
    q_after = sum(after) / len(after)

    if mode == "ratio":
        return q_after / q_before if q_before > _NEAR_ZERO else 1.0
    return q_after - q_before


def apply_correction(
    annual_series: pl.Series,
    years: pl.Series,
    break_year: int,
    factor: float,
    mode: CorrectionMode = "ratio",
) -> pl.Series:
    """Apply correction to all annual values before break_year.

    ratio mode: multiplies pre-break values by factor.
    difference mode: adds factor to pre-break values.
    Null values and post-break values are left unchanged.
    """
    corrected: list[float | None] = []
    for val, year in zip(annual_series.to_list(), years.to_list(), strict=True):
        if val is None:
            corrected.append(None)
        elif year < break_year:
            corrected.append(val * factor if mode == "ratio" else val + factor)
        else:
            corrected.append(val)
    return pl.Series(annual_series.name, corrected)
