# Algorithm

rucola implements the six-step quality control and homogenization procedure
described in [González-Rouco et al. (2001)](https://doi.org/10.1175/1520-0442(2001)014<0964:QCAHOP>2.0.CO;2),
with six pluggable breakpoint tests (SNHT, Buishand, Pettitt, Worsley, Easterling–Peterson, STARS)
and an iteratively refined reference station pool.

## Problem statement

Long climate records from ground stations are regularly affected by
*non-climatic discontinuities* — abrupt shifts in the measured values caused by:

- station relocations
- instrument replacements or recalibrations
- changes in observation practice or time of observation
- changes in the immediate environment of the station (urbanisation, vegetation)

These shifts are indistinguishable from genuine climate signals in a single
station's record, but can be detected by comparing the station against a set of
nearby *reference stations* that share the same large-scale climate signal but
were not affected by the same change.

## The Q-series

The core input to every statistical test is the **Q-series** — a normalized
anomaly series that isolates the candidate station's behaviour relative to its
reference stations.

**Ratio mode** (multiplicative, for precipitation):

$$Q_t = \frac{P_t^{(i)}}{\bar{G}_t}$$

where $P_t^{(i)}$ is the candidate value and $\bar{G}_t$ is the
correlation-weighted mean of reference stations, each normalized by their
long-term mean:

$$G_t = \frac{\sum_j r_j^2 \cdot P_t^{(j)} / \bar{P}^{(j)}}{\sum_j r_j^2}$$

**Difference mode** (additive, for temperature):

$$Q_t = T_t^{(i)} - G_t, \quad G_t = \frac{\sum_j r_j^2 \cdot T_t^{(j)}}{\sum_j r_j^2}$$

In both cases weights are $r_j^2$ — the squared Pearson correlation between
the candidate and each reference station over the overlapping record.

## Reference station selection

For each candidate station, up to `max_neighbors` reference stations are
selected from the pool of available stations based on:

1. **Pearson correlation** — only stations with correlation ≥ `min_correlation`
   (default 0.5) over at least 10 overlapping years are considered
2. **Distance** (optional) — if `max_distance_km` is set, stations beyond that
   radius are excluded before correlation is computed
3. **Group membership** — in later steps, only stations already classified as
   homogeneous are eligible as references

## Statistical tests

Five breakpoint tests are available, all operating on the Q-series:

| Test | Reference | Type |
|---|---|---|
| **SNHT** | Alexandersson (1986) | Likelihood ratio, single break |
| **Buishand range** | Buishand (1982) | Cumulative deviation |
| **Pettitt** | Pettitt (1979) | Non-parametric rank |
| **Worsley likelihood ratio** | Worsley (1979) | Likelihood ratio |
| **Easterling–Peterson** | Easterling & Peterson (1995) | Two-phase regression |
| **STARS** | Rodionov (2004) | Sequential regime-shift (MSSD variance) |

STARS (`StarsTest`) is distinct from the other five in that it propagates a running regime mean rather than maximising a global statistic. This makes it sensitive to gradual shifts that accumulate across the series. Configurable parameters: `l` (minimum regime length / RSI confirmation window, default 10) and the shared `alpha`, `min_years_from_end`, `min_relative_signal`.

Each test returns a break year, a test statistic, and a significance flag
based on a configurable `alpha` level. When multiple tests are used, a
**consensus rule** (`any`, `majority`, `unanimous`, or `strongest_signal`)
determines whether a break is accepted.

## The six steps

The procedure iterates through six steps, progressively refining the reference
pool and correcting detected breaks. At each step, stations are classified into
groups — **H** (homogeneous) and **I** (inhomogeneous) — which determine their
eligibility as references in subsequent steps.

| Step | Candidates | Reference pool | Groups produced |
|---|---|---|---|
| **1** | ALL | ALL | H1, I1 |
| **2** | ALL | H1 + corrected I1 | H2, I2 |
| **3** | corrected I2 | H2 + corrected I2 | HC3, IC3 |
| **4** | H2 + corrected I2 | H2 + HC3 | H4, HC4, I4, IC4 |
| **5** | I4 | H4 + HC4 | HC5, IC5 |
| **6** | IC4 + IC5 | H4 + HC4 + HC5 | HCC6, ICC6 |

After each step, detected breaks are corrected *in-place* before the next step
begins. Step 6 applies a two-pass double-break correction: first the later
break is corrected on the post-first-break sub-series (6a), then the earlier
break is corrected on the full series (6b). A final diagnostic re-test (6c)
checks the doubly-corrected series against the same reference pool — if the
consensus still flags an inhomogeneity, the station is classified `ICC6`
("still inhomogeneous after two corrections", González-Rouco et al. 2001);
otherwise `HCC6`. The 6c result is stored as a `DetectionRecord` for
inspection but is never applied as a correction.

Stations that can never find a reference (isolated stations) are labelled
**UNTESTABLE**. Stations with fewer non-null years than `min_series_years`, or
with a consecutive null gap exceeding `max_gap_years`, are labelled
**INSUFFICIENT_DATA** and skipped entirely.

## Correction

Once breaks are detected, corrections are applied by `DetectionResult.normalize()`.
The `NormalizationConfig` controls:

- **min_years_from_end** — reject breaks within this many years of the series
  boundary; these are edge-effect artefacts (Hawkins 1977) that cannot be
  estimated reliably (default: 5, matching the detection-phase default)
- **min_relative_signal** — reject test results below this ratio of test
  statistic to critical value, even when statistically significant (default: 1.0)
- **consensus** — how many tests must agree post-hoc to accept a break
- **break_window_years** — tolerance (±years) for two tests to be considered
  in agreement on the same break year
- **tiebreak** — what to do when consensus is not reached
- **min_correction_magnitude** — ignore corrections below this threshold
- **max_corrections_per_station** — cap on the number of breaks applied

### Step size estimation

The correction factor $f$ is estimated from the Q-series itself, by comparing
its mean on either side of the detected break:

$$\bar{Q}_{\text{before}} = \frac{1}{n_b} \sum_{t < t_{\text{break}}} Q_t,
\qquad
\bar{Q}_{\text{after}} = \frac{1}{n_a} \sum_{t \ge t_{\text{break}}} Q_t$$

The form of $f$ depends on the mode (González-Rouco et al. 2001, Eq. 5):

$$f_{\text{ratio}} = \frac{\bar{Q}_{\text{after}}}{\bar{Q}_{\text{before}}},
\qquad
f_{\text{diff}} = \bar{Q}_{\text{after}} - \bar{Q}_{\text{before}}$$

Neutral values are $f_{\text{ratio}} = 1$ and $f_{\text{diff}} = 0$ (no
correction).  Because the Q-series already cancels the shared climate signal
against the reference pool, $f$ isolates the station-specific shift.

### Applying the correction

**Ratio correction** (precipitation): multiply all values *before* the break
by the inverse of the ratio factor:

$$P_t^{\text{corr}} = P_t \cdot f^{-1}, \quad t < t_{\text{break}}$$

**Difference correction** (temperature): subtract the additive factor from all
values before the break:

$$T_t^{\text{corr}} = T_t - f, \quad t < t_{\text{break}}$$

## References

- González-Rouco et al. (2001), *J. Climate* 14(5):964–978. [doi:10.1175/1520-0442(2001)014<0964:QCAHOP>2.0.CO;2](https://doi.org/10.1175/1520-0442(2001)014<0964:QCAHOP>2.0.CO;2)
- Hawkins (1977), *Biometrika* 64(2):279–288. [doi:10.1093/biomet/64.2.279](https://doi.org/10.1093/biomet/64.2.279) — edge-effect artefact in changepoint tests
- Alexandersson (1986), *Int. J. Climatol.* 6(6):661–675. [doi:10.1002/joc.3370060607](https://doi.org/10.1002/joc.3370060607)
- Alexandersson & Moberg (1997), *Int. J. Climatol.* 17(1):25–34. [doi:10.1002/(SICI)1097-0088(199701)17:1<25::AID-JOC103>3.0.CO;2-J](https://doi.org/10.1002/(SICI)1097-0088(199701)17:1<25::AID-JOC103>3.0.CO;2-J)
- Buishand (1982), *J. Hydrol.* 58(1–2):11–29. [doi:10.1016/0022-1694(82)90066-X](https://doi.org/10.1016/0022-1694(82)90066-X)
- Pettitt (1979), *Appl. Stat.* 28(2):126–135. [doi:10.2307/2346729](https://doi.org/10.2307/2346729)
- Worsley (1979), *J. Amer. Statist. Assoc.* 74(366):365–367. [doi:10.1080/01621459.1979.10482519](https://doi.org/10.1080/01621459.1979.10482519)
- Easterling & Peterson (1995), *Int. J. Climatol.* 15(4):369–377. [doi:10.1002/joc.3370150403](https://doi.org/10.1002/joc.3370150403)
- Rodionov (2004), *Geophys. Res. Lett.* 31, L09204. [doi:10.1029/2004GL019448](https://doi.org/10.1029/2004GL019448) — STARS sequential regime-shift test
