"""Statistical tests for detecting single breakpoints in climate series."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from statistics import NormalDist as _NormalDist
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

_NEAR_ZERO = 1e-12
_MIN_YEARS = 4  # minimum valid observations for any test

# ── Two-tailed t-distribution critical values ─────────────────────────────────
# Used by StarsTest (Rodionov 2004).  Keyed by alpha then degrees of freedom.
_T_CRIT: dict[float, dict[int, float]] = {
    0.05: {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        25: 2.060,
        30: 2.042,
        40: 2.021,
        50: 2.009,
        60: 2.000,
        80: 1.990,
        100: 1.984,
        120: 1.980,
    },
    0.01: {
        1: 63.657,
        2: 9.925,
        3: 5.841,
        4: 4.604,
        5: 4.032,
        6: 3.707,
        7: 3.499,
        8: 3.355,
        9: 3.250,
        10: 3.169,
        11: 3.106,
        12: 3.055,
        13: 3.012,
        14: 2.977,
        15: 2.947,
        16: 2.921,
        17: 2.898,
        18: 2.878,
        19: 2.861,
        20: 2.845,
        25: 2.787,
        30: 2.750,
        40: 2.704,
        50: 2.678,
        60: 2.660,
        80: 2.639,
        100: 2.626,
        120: 2.617,
    },
    0.10: {
        1: 6.314,
        2: 2.920,
        3: 2.353,
        4: 2.132,
        5: 2.015,
        6: 1.943,
        7: 1.895,
        8: 1.860,
        9: 1.833,
        10: 1.812,
        11: 1.796,
        12: 1.782,
        13: 1.771,
        14: 1.761,
        15: 1.753,
        16: 1.746,
        17: 1.740,
        18: 1.734,
        19: 1.729,
        20: 1.725,
        25: 1.708,
        30: 1.697,
        40: 1.684,
        50: 1.676,
        60: 1.671,
        80: 1.664,
        100: 1.660,
        120: 1.658,
    },
}

# ── SNHT critical values (Alexandersson & Moberg 1997, Table A1) ─────────────
_SNHT_CRIT: dict[float, dict[int, float]] = {
    0.05: {
        10: 8.45,
        20: 9.56,
        30: 9.83,
        40: 10.00,
        50: 10.07,
        60: 10.13,
        70: 10.18,
        80: 10.20,
        90: 10.23,
        100: 10.25,
        150: 10.35,
        200: 10.40,
        300: 10.45,
        400: 10.48,
        500: 10.50,
        1000: 10.57,
    },
    0.01: {
        10: 11.37,
        20: 12.26,
        30: 12.55,
        40: 12.70,
        50: 12.80,
        60: 12.87,
        70: 12.92,
        80: 12.96,
        90: 12.99,
        100: 13.01,
        150: 13.11,
        200: 13.17,
        300: 13.24,
        400: 13.28,
        500: 13.30,
        1000: 13.39,
    },
    0.10: {
        10: 6.55,
        20: 7.65,
        30: 7.94,
        40: 8.10,
        50: 8.17,
        60: 8.22,
        70: 8.26,
        80: 8.28,
        90: 8.30,
        100: 8.31,
        150: 8.37,
        200: 8.40,
        300: 8.44,
        400: 8.46,
        500: 8.47,
        1000: 8.51,
    },
}

# ── Buishand R critical values (Buishand 1982, Table 1) ──────────────────────
# R = (max S_k − min S_k) / (√n · σ),  σ = population std
# n > 100 values are linearly interpolated on 1/√n toward the asymptotic limit.
_BUISHAND_CRIT: dict[float, dict[int, float]] = {
    0.05: {10: 1.28, 20: 1.43, 30: 1.50, 40: 1.53, 50: 1.55, 100: 1.62, 200: 1.66, 500: 1.69, 1000: 1.71},
    0.01: {10: 1.38, 20: 1.60, 30: 1.70, 40: 1.74, 50: 1.78, 100: 1.86, 200: 1.90, 500: 1.94, 1000: 1.96},
    0.10: {10: 1.21, 20: 1.34, 30: 1.40, 40: 1.42, 50: 1.44, 100: 1.50, 200: 1.54, 500: 1.57, 1000: 1.58},
}


def _interpolate_crit(table: dict[int, float], n: int) -> float:
    ns = sorted(table)
    if n <= ns[0]:
        return table[ns[0]]
    if n >= ns[-1]:
        return table[ns[-1]]
    for i in range(len(ns) - 1):
        if ns[i] <= n <= ns[i + 1]:
            t0, t1 = table[ns[i]], table[ns[i + 1]]
            return t0 + (t1 - t0) * (n - ns[i]) / (ns[i + 1] - ns[i])
    return table[ns[-1]]


@dataclass
class TestResult:
    """Result of one homogeneity test on a Q-series."""

    test_name: str
    is_significant: bool
    break_year: int
    test_statistic: float
    critical_value: float
    n_years: int
    segment_start: int
    segment_end: int
    series: list[float] = field(default_factory=list)
    years_tested: list[int] = field(default_factory=list)

    @property
    def relative_signal(self) -> float:
        """Test statistic divided by critical value; > 1.0 means significant."""
        return self.test_statistic / self.critical_value if self.critical_value > _NEAR_ZERO else 0.0

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "test_name": self.test_name,
            "is_significant": self.is_significant,
            "break_year": self.break_year,
            "test_statistic": self.test_statistic,
            "critical_value": self.critical_value,
            "n_years": self.n_years,
            "segment_start": self.segment_start,
            "segment_end": self.segment_end,
            "series": self.series,
            "years_tested": self.years_tested,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TestResult:
        return cls(**d)


class HomogenizationTest(ABC):
    """Abstract base for single-breakpoint homogeneity tests.

    All tests operate on a pre-computed Q-series (output of compute_q_series).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_years_from_end: int = 5,
        min_relative_signal: float = 1.0,
    ) -> None:
        """Initialise with significance level, edge-effect guard, and signal threshold.

        Parameters
        ----------
        alpha :
            Significance level for the hypothesis test (default: 0.05).
        min_years_from_end :
            Breaks within this many years of either series end are rejected
            (Hawkins 1977 edge effect, default: 5).
        min_relative_signal :
            Minimum ratio of test statistic to critical value required to accept
            a break (default: 1.0, i.e. any significant result). Raise above 1.0
            to require a stronger signal and reduce false positives.

        """
        self.alpha = alpha
        self.min_years_from_end = min_years_from_end
        self.min_relative_signal = min_relative_signal

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in result records."""

    @abstractmethod
    def detect(self, q_series: pl.Series, years: pl.Series) -> TestResult:
        """Run the test and return a result."""

    def is_inhomogeneous(self, result: TestResult) -> bool:
        """Return True if result passes significance, edge-effect, and signal-strength checks."""
        if not result.is_significant:
            return False
        if result.relative_signal < self.min_relative_signal:
            return False
        years_from_start = result.break_year - result.segment_start
        years_from_end = result.segment_end - result.break_year + 1
        return years_from_start >= self.min_years_from_end and years_from_end >= self.min_years_from_end

    def __repr__(self) -> str:
        """Return short summary string."""
        return (
            f"{self.__class__.__name__}("
            f"alpha={self.alpha}, "
            f"min_years_from_end={self.min_years_from_end}, "
            f"min_relative_signal={self.min_relative_signal})"
        )


class SNHTTest(HomogenizationTest):
    """Standard Normal Homogeneity Test (Alexandersson 1986).

    Uses tabulated critical values from Alexandersson & Moberg (1997) at
    alpha ∈ {0.01, 0.05, 0.10}.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_years_from_end: int = 5,
        min_relative_signal: float = 1.0,
    ) -> None:
        """Initialise; raise immediately if alpha is not in the critical-value table."""
        super().__init__(alpha=alpha, min_years_from_end=min_years_from_end, min_relative_signal=min_relative_signal)
        if alpha not in _SNHT_CRIT:
            msg = f"SNHT: alpha must be one of {sorted(_SNHT_CRIT)}, got {alpha}"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        """Return test name."""
        return "snht"

    def detect(self, q_series: pl.Series, years: pl.Series) -> TestResult:
        """Apply SNHT to q_series and return a TestResult."""
        rows = [(q, int(y)) for q, y in zip(q_series.to_list(), years.to_list(), strict=True) if q is not None]
        n = len(rows)
        seg_start = rows[0][1] if rows else (int(years[0]) if len(years) > 0 else 0)
        seg_end = rows[-1][1] if rows else (int(years[-1]) if len(years) > 0 else 0)
        crit = _interpolate_crit(_SNHT_CRIT[self.alpha], max(n, 10))

        null = TestResult(
            test_name=self.name,
            is_significant=False,
            break_year=seg_start,
            test_statistic=0.0,
            critical_value=crit,
            n_years=n,
            segment_start=seg_start,
            segment_end=seg_end,
        )
        if n < _MIN_YEARS:
            return null

        q_vals = [r[0] for r in rows]
        y_vals = [r[1] for r in rows]

        q_mean = sum(q_vals) / n
        q_std = math.sqrt(sum((q - q_mean) ** 2 for q in q_vals) / n)
        if q_std < _NEAR_ZERO:
            return null

        z = [(q - q_mean) / q_std for q in q_vals]
        cumsum = [0.0] * (n + 1)
        for i, zi in enumerate(z):
            cumsum[i + 1] = cumsum[i] + zi

        t_vals: list[float] = []
        for m in range(1, n):
            z1 = cumsum[m] / m
            z2 = (cumsum[n] - cumsum[m]) / (n - m)
            t_vals.append(m * z1 * z1 + (n - m) * z2 * z2)

        t0 = max(t_vals)
        m_opt = t_vals.index(t0) + 1
        break_year = y_vals[m_opt]

        return TestResult(
            test_name=self.name,
            is_significant=t0 > crit,
            break_year=break_year,
            test_statistic=t0,
            critical_value=crit,
            n_years=n,
            segment_start=seg_start,
            segment_end=seg_end,
            series=t_vals,
            years_tested=y_vals,
        )


class BuishandTest(HomogenizationTest):
    """Buishand range test (Buishand 1982).

    Statistic R = (max S_k − min S_k) / (√n · σ) where S_k is the
    cumulative deviation from the mean and σ is the population std.
    Uses tabulated critical values at alpha ∈ {0.01, 0.05, 0.10}.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_years_from_end: int = 5,
        min_relative_signal: float = 1.0,
    ) -> None:
        """Initialise; raise immediately if alpha is not in the critical-value table."""
        super().__init__(alpha=alpha, min_years_from_end=min_years_from_end, min_relative_signal=min_relative_signal)
        if alpha not in _BUISHAND_CRIT:
            msg = f"Buishand: alpha must be one of {sorted(_BUISHAND_CRIT)}, got {alpha}"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        """Return test name."""
        return "buishand"

    def detect(self, q_series: pl.Series, years: pl.Series) -> TestResult:
        """Apply Buishand range test to q_series and return a TestResult."""
        rows = [(q, int(y)) for q, y in zip(q_series.to_list(), years.to_list(), strict=True) if q is not None]
        n = len(rows)
        seg_start = rows[0][1] if rows else (int(years[0]) if len(years) > 0 else 0)
        seg_end = rows[-1][1] if rows else (int(years[-1]) if len(years) > 0 else 0)
        crit = _interpolate_crit(_BUISHAND_CRIT[self.alpha], n)

        null = TestResult(
            test_name=self.name,
            is_significant=False,
            break_year=seg_start,
            test_statistic=0.0,
            critical_value=crit,
            n_years=n,
            segment_start=seg_start,
            segment_end=seg_end,
        )
        if n < _MIN_YEARS:
            return null

        q_vals = [r[0] for r in rows]
        y_vals = [r[1] for r in rows]

        q_mean = sum(q_vals) / n
        q_std = math.sqrt(sum((q - q_mean) ** 2 for q in q_vals) / n)
        if q_std < _NEAR_ZERO:
            return null

        # cumulative deviations S_k for k = 1, …, n
        s_vals: list[float] = []
        running = 0.0
        for q in q_vals:
            running += q - q_mean
            s_vals.append(running)

        r_stat = (max(s_vals) - min(s_vals)) / (math.sqrt(n) * q_std)

        # break point: last year of pre-break segment → first year of post-break
        k_star = max(range(n), key=lambda k: abs(s_vals[k]))
        break_year = y_vals[k_star + 1] if k_star + 1 < n else y_vals[-1]

        return TestResult(
            test_name=self.name,
            is_significant=r_stat > crit,
            break_year=break_year,
            test_statistic=r_stat,
            critical_value=crit,
            n_years=n,
            segment_start=seg_start,
            segment_end=seg_end,
            series=s_vals,
            years_tested=y_vals,
        )


class PettittTest(HomogenizationTest):
    """Pettitt test (Pettitt 1979).

    Non-parametric test based on the Mann-Whitney statistic.
    K = max_t |U_{t,n}|.  P-value approximated analytically:
    p ≈ 2 exp(−6K² / (n³ + n²)).  Supports any alpha > 0.
    """

    @property
    def name(self) -> str:
        """Return test name."""
        return "pettitt"

    def detect(self, q_series: pl.Series, years: pl.Series) -> TestResult:
        """Apply Pettitt test to q_series and return a TestResult."""
        rows = [(q, int(y)) for q, y in zip(q_series.to_list(), years.to_list(), strict=True) if q is not None]
        n = len(rows)
        seg_start = rows[0][1] if rows else (int(years[0]) if len(years) > 0 else 0)
        seg_end = rows[-1][1] if rows else (int(years[-1]) if len(years) > 0 else 0)
        # K_crit from p = alpha: K_crit = sqrt(−ln(α/2) · (n³+n²) / 6)
        k_crit = math.sqrt(-math.log(self.alpha / 2) * (n**3 + n**2) / 6) if n >= _MIN_YEARS else 0.0

        null = TestResult(
            test_name=self.name,
            is_significant=False,
            break_year=seg_start,
            test_statistic=0.0,
            critical_value=k_crit,
            n_years=n,
            segment_start=seg_start,
            segment_end=seg_end,
        )
        if n < _MIN_YEARS:
            return null

        q_vals = [r[0] for r in rows]
        y_vals = [r[1] for r in rows]

        # U_{t,n} = Σ_{i<t} Σ_{j≥t} sgn(x_j − x_i)  for t = 1, …, n−1
        u_vals: list[float] = []
        u = 0.0
        for t in range(1, n):
            # incremental update: U_t = U_{t-1} + Σ_{j=t}^{n-1} sgn(x_j − x_{t-1})
            x_new = q_vals[t - 1]
            for j in range(t, n):
                diff = q_vals[j] - x_new
                u += 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
            u_vals.append(u)

        k_stat = max(abs(u) for u in u_vals)
        t_star = max(range(len(u_vals)), key=lambda t: abs(u_vals[t]))
        break_year = y_vals[t_star + 1] if t_star + 1 < n else y_vals[-1]

        return TestResult(
            test_name=self.name,
            is_significant=k_stat > k_crit,
            break_year=break_year,
            test_statistic=k_stat,
            critical_value=k_crit,
            n_years=n,
            segment_start=seg_start,
            segment_end=seg_end,
            series=u_vals,
            years_tested=y_vals[1:],
        )


def _worsley_crit(n: int, alpha: float) -> float:
    """Approximate critical value for the Worsley test via Bonferroni correction.

    P(W_n > c) ≈ 2(n−1)(1−Φ(c)), so c = Φ⁻¹(1 − α / (2(n−1))).
    """
    if n <= 2:  # noqa: PLR2004
        return float("inf")
    p = 1.0 - alpha / (2.0 * (n - 1))
    return _NormalDist().inv_cdf(p)


class WorsleyTest(HomogenizationTest):
    """Worsley likelihood ratio test (Worsley 1979).

    Maximum standardized two-sample t-statistic over all possible change points:
    W = max_{1≤k≤n−1} | sqrt(k(n−k)/n) · (x̄_before − x̄_after) / s_pooled |

    Critical values are computed analytically via the Bonferroni approximation
    and support any alpha > 0.
    """

    @property
    def name(self) -> str:
        """Return test name."""
        return "worsley"

    def detect(self, q_series: pl.Series, years: pl.Series) -> TestResult:
        """Apply Worsley test to q_series and return a TestResult."""
        rows = [(q, int(y)) for q, y in zip(q_series.to_list(), years.to_list(), strict=True) if q is not None]
        n = len(rows)
        seg_start = rows[0][1] if rows else (int(years[0]) if len(years) > 0 else 0)
        seg_end = rows[-1][1] if rows else (int(years[-1]) if len(years) > 0 else 0)
        crit = _worsley_crit(n, self.alpha) if n >= _MIN_YEARS else 0.0

        null = TestResult(
            test_name=self.name,
            is_significant=False,
            break_year=seg_start,
            test_statistic=0.0,
            critical_value=crit,
            n_years=n,
            segment_start=seg_start,
            segment_end=seg_end,
        )
        if n < _MIN_YEARS:
            return null

        q_vals = [r[0] for r in rows]
        y_vals = [r[1] for r in rows]

        # O(n) running sums for means and sum-of-squares
        cum_x = [0.0] * (n + 1)
        cum_x2 = [0.0] * (n + 1)
        for i, x in enumerate(q_vals):
            cum_x[i + 1] = cum_x[i] + x
            cum_x2[i + 1] = cum_x2[i] + x * x

        total_x = cum_x[n]
        total_x2 = cum_x2[n]

        t_vals: list[float] = []
        for k in range(1, n):
            n1, n2 = k, n - k
            s1 = cum_x[k]
            s2 = total_x - s1
            ss1 = cum_x2[k] - s1 * s1 / n1
            ss2 = (total_x2 - cum_x2[k]) - s2 * s2 / n2
            pooled_var = (ss1 + ss2) / (n - 2)
            if pooled_var < _NEAR_ZERO:
                t_vals.append(0.0)
                continue
            t_k = math.sqrt(n1 * n2 / n) * (s1 / n1 - s2 / n2) / math.sqrt(pooled_var)
            t_vals.append(abs(t_k))

        w_stat = max(t_vals)
        k_star = t_vals.index(w_stat)
        break_year = y_vals[k_star + 1] if k_star + 1 < n else y_vals[-1]

        return TestResult(
            test_name=self.name,
            is_significant=w_stat > crit,
            break_year=break_year,
            test_statistic=w_stat,
            critical_value=crit,
            n_years=n,
            segment_start=seg_start,
            segment_end=seg_end,
            series=t_vals,
            years_tested=y_vals[1:],
        )


class EasterlingPetersonTest(HomogenizationTest):
    """Easterling–Peterson two-phase regression test (Easterling & Peterson 1995).

    Fits the model q_i = a + b·t_i + c·I(i > k) + ε for each candidate break k.
    The step change c is tested via its OLS t-statistic; the maximum |t_k| over
    all k is the test statistic. Unlike mean-shift tests, the linear trend term
    b prevents trend from being mistaken for a break — useful for temperature.

    Critical values use the same Bonferroni normal approximation as WorsleyTest.
    """

    @property
    def name(self) -> str:
        """Return test name."""
        return "easterling_peterson"

    def detect(self, q_series: pl.Series, years: pl.Series) -> TestResult:
        """Apply Easterling–Peterson test to q_series and return a TestResult."""
        rows = [(q, int(y)) for q, y in zip(q_series.to_list(), years.to_list(), strict=True) if q is not None]
        n = len(rows)
        seg_start = rows[0][1] if rows else (int(years[0]) if len(years) > 0 else 0)
        seg_end = rows[-1][1] if rows else (int(years[-1]) if len(years) > 0 else 0)
        crit = _worsley_crit(n, self.alpha) if n >= _MIN_YEARS else 0.0

        null = TestResult(
            test_name=self.name,
            is_significant=False,
            break_year=seg_start,
            test_statistic=0.0,
            critical_value=crit,
            n_years=n,
            segment_start=seg_start,
            segment_end=seg_end,
        )
        if n < _MIN_YEARS:
            return null

        q_vals = [r[0] for r in rows]
        y_vals = [r[1] for r in rows]

        # Center time axis so that X'X has zero off-diagonal in the (1,2) block,
        # which simplifies the 3×3 normal equations considerably.
        t_mean = sum(y_vals) / n
        t_c = [y - t_mean for y in y_vals]

        s_tt = sum(tc * tc for tc in t_c)
        if s_tt < _NEAR_ZERO:
            return null

        s_y = sum(q_vals)
        s_ty = sum(tc * q for tc, q in zip(t_c, q_vals, strict=True))
        s_yy = sum(q * q for q in q_vals)
        # RSS under H0 (single linear trend, no break)
        r0 = s_yy - s_y * s_y / n - s_ty * s_ty / s_tt

        # Initialise running sums for post-break segment {1, …, n-1} (k = 0)
        s_z: float = n - 1
        s_tz: float = sum(t_c[1:])
        s_zy: float = sum(q_vals[1:])

        t_stats: list[float] = []
        for k in range(n - 1):
            if k > 0:
                s_z -= 1
                s_tz -= t_c[k]
                s_zy -= q_vals[k]

            denom_c = s_z - s_z * s_z / n - s_tz * s_tz / s_tt
            if denom_c < _NEAR_ZERO:
                t_stats.append(0.0)
                continue

            numer_c = s_zy - s_z * s_y / n - s_tz * s_ty / s_tt
            rss = r0 - numer_c * numer_c / denom_c
            if rss < 0:
                rss = 0.0

            df = n - 3
            if df <= 0 or rss < _NEAR_ZERO:
                t_stats.append(0.0)
                continue

            t_k = numer_c / math.sqrt(denom_c * rss / df)
            t_stats.append(abs(t_k))

        if not t_stats:
            return null

        t_max = max(t_stats)
        k_opt = t_stats.index(t_max)
        break_year = y_vals[k_opt + 1] if k_opt + 1 < n else y_vals[-1]

        return TestResult(
            test_name=self.name,
            is_significant=t_max > crit,
            break_year=break_year,
            test_statistic=t_max,
            critical_value=crit,
            n_years=n,
            segment_start=seg_start,
            segment_end=seg_end,
            series=t_stats,
            years_tested=y_vals[1:],
        )


class StarsTest(HomogenizationTest):
    r"""STARS: Sequential T-test Analysis of Regime Shifts (Rodionov 2004).

    Scans the Q-series sequentially and declares a regime shift at year *j* when
    the Regime Shift Index (RSI) stays consistently positive (or negative) over
    the following *l* years.  Unlike the other tests that maximise a global
    statistic, STARS propagates a running regime mean, so it is sensitive to
    shifts that accumulate gradually across the series rather than appearing as a
    single dominant peak.

    Variance is estimated from the mean-square successive differences (MSSD),
    which is robust to the very breaks being detected:

    .. math::

        \\hat{\\sigma}^2 = \\frac{1}{2(n-1)} \\sum_{t=1}^{n-1}(x_{t+1} - x_t)^2

    The minimum detectable mean shift is :math:`\\delta = t_{\\text{crit}} \\cdot
    \\hat{\\sigma} \\sqrt{2/l}`, where :math:`t_{\\text{crit}}` is the two-tailed
    t-distribution critical value for :math:`\\nu = 2(l-1)` degrees of freedom.

    A shift at year *j* is confirmed when the RSI

    .. math::

        \\text{RSI}(k) = \\sum_{t=j}^{k}
            \\frac{x_t - (\\bar{x}_{\\text{prev}} \\pm \\delta)}{\\hat{\\sigma}\\, l}

    does not change sign for all :math:`k \\in [j,\\, j+l-1]`.  If multiple
    shifts are confirmed, the one with the largest t-statistic is returned as
    the primary break year.

    Parameters
    ----------
    l :
        Cut-off length in years — the minimum regime duration and the RSI
        confirmation window.  Sets :math:`\\nu = 2(l-1)` degrees of freedom
        for the t-test.  Typical values for annual climate data: 10–15.
    alpha :
        Significance level for the t-test.  Supported: ``{0.01, 0.05, 0.10}``.
    min_years_from_end :
        Inherited edge-effect guard (default: 5).
    min_relative_signal :
        Minimum ratio of the confirmed t-statistic to the critical value
        (default: 1.0).

    References
    ----------
    Rodionov, S. N. (2004): A sequential algorithm for testing climate regime
    shifts. *Geophys. Res. Lett.*, 31, L09204.
    https://doi.org/10.1029/2004GL019448

    """

    def __init__(
        self,
        l: int = 10,  # noqa: E741
        alpha: float = 0.05,
        min_years_from_end: int = 5,
        min_relative_signal: float = 1.0,
    ) -> None:
        """Initialise with cut-off length, significance level, and edge-guard settings."""
        super().__init__(alpha=alpha, min_years_from_end=min_years_from_end, min_relative_signal=min_relative_signal)
        self.l = l
        if alpha not in _T_CRIT:
            msg = f"StarsTest: alpha must be one of {sorted(_T_CRIT)}, got {alpha}"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        """Return test name."""
        return "stars"

    def __repr__(self) -> str:
        """Return short summary string."""
        return (
            f"{self.__class__.__name__}("
            f"l={self.l}, "
            f"alpha={self.alpha}, "
            f"min_years_from_end={self.min_years_from_end}, "
            f"min_relative_signal={self.min_relative_signal})"
        )

    def detect(self, q_series: pl.Series, years: pl.Series) -> TestResult:
        """Apply STARS to q_series and return a TestResult."""
        rows = [(q, int(y)) for q, y in zip(q_series.to_list(), years.to_list(), strict=True) if q is not None]
        n = len(rows)
        seg_start = rows[0][1] if rows else (int(years[0]) if len(years) > 0 else 0)
        seg_end = rows[-1][1] if rows else (int(years[-1]) if len(years) > 0 else 0)

        df = 2 * (self.l - 1)
        crit = _interpolate_crit(_T_CRIT[self.alpha], df) if df > 0 else float("inf")

        null = TestResult(
            test_name=self.name,
            is_significant=False,
            break_year=seg_start,
            test_statistic=0.0,
            critical_value=crit,
            n_years=n,
            segment_start=seg_start,
            segment_end=seg_end,
        )

        if n < max(_MIN_YEARS, 2 * self.l):
            return null

        q_vals = [r[0] for r in rows]
        y_vals = [r[1] for r in rows]

        # σ² via mean-square successive differences — robust to the breaks we detect
        mssd = sum((q_vals[i + 1] - q_vals[i]) ** 2 for i in range(n - 1)) / (2 * (n - 1))
        if mssd < _NEAR_ZERO:
            return null
        sigma = math.sqrt(mssd)

        # Minimum detectable mean difference at this significance level
        delta = crit * sigma * math.sqrt(2.0 / self.l)

        # Sequential detection ─────────────────────────────────────────────────
        # Track the best (highest t-statistic) confirmed break.
        # After a confirmed shift at index j, the next eligible check is j+l
        # (minimum regime-length constraint).
        best_break_idx: int | None = None
        best_t: float = 0.0

        mu = sum(q_vals[: self.l]) / self.l  # regime mean, initialised over first l values
        regime_start = 0
        next_check = self.l  # first index eligible for a shift test

        for j in range(self.l, n):
            if j >= next_check and j + self.l <= n and abs(q_vals[j] - mu) >= delta:
                direction = 1 if q_vals[j] > mu else -1
                boundary = mu + direction * delta

                # Confirm via RSI: must stay on the same side for l consecutive years
                rsi = 0.0
                confirmed = True
                for k in range(j, j + self.l):
                    rsi += (q_vals[k] - boundary) / (sigma * self.l)
                    if direction * rsi < 0:
                        confirmed = False
                        break

                if confirmed:
                    t_stat = abs(q_vals[j] - mu) / (sigma * math.sqrt(2.0 / self.l))
                    if t_stat > best_t:
                        best_t = t_stat
                        best_break_idx = j
                    # Start new regime: reset mean to single-point estimate and grow it
                    mu = q_vals[j]
                    regime_start = j
                    next_check = j + self.l
                    continue

            # Welford incremental update of the current regime mean
            count = j - regime_start + 1
            mu += (q_vals[j] - mu) / count

        if best_break_idx is None:
            return null

        return TestResult(
            test_name=self.name,
            is_significant=best_t > crit,
            break_year=y_vals[best_break_idx],
            test_statistic=best_t,
            critical_value=crit,
            n_years=n,
            segment_start=seg_start,
            segment_end=seg_end,
            years_tested=y_vals,
        )
