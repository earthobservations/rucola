# Normalization

`run()` performs the six-step detection procedure and returns a `DetectionResult` — the raw evidence of every breakpoint found, uncorrected. `normalize()` is a separate, configurable step that decides which of those detections to turn into actual corrections.

This separation means you can re-run normalization with different settings without re-running the (expensive) detection phase.

```python
detection = r.run(RunConfig(mode="ratio"))

# default normalization
result = detection.normalize()

# stricter normalization on the same detection
result = detection.normalize(NormalizationConfig(consensus="majority"))
```

---

## How normalization works

For each station and each procedure step where a break was detected, normalization runs a pipeline of filters. A break is applied only when it passes **all** of them in order:

1. **Edge-effect guard** — individual test results with a break year within `min_years_from_end` of either series boundary are excluded. Breaks near the endpoints cannot be estimated reliably (Hawkins 1977) and are statistical artefacts of the test window.
2. **Signal-strength filter** — test results below `min_relative_signal` (ratio of test statistic to critical value) are excluded at the same stage, even if technically significant. The default of 1.0 matches the binary significance threshold; raise it to demand stronger evidence.
3. **Consensus** — how many of the remaining (edge-safe, strong-enough) significant tests must agree that the break is real.
4. **Year agreement** — significant tests must agree on the break year within `break_window_years` (±years).
5. **Tiebreak** — when consensus or year agreement fails, optionally fall back to the test with the highest signal.
6. **Magnitude filter** — `min_correction_magnitude` discards trivially small corrections (`|factor − 1|` in ratio mode, `|factor|` in difference mode).
7. **Predicate** — an optional `BreakPredicate` expression that can inspect all break properties and return `True` (accept) or `False` (discard).

After filtering, corrections are sorted by break year and capped at `max_corrections_per_station`.

**Overrides bypass the entire pipeline.** Stations listed in `overrides` skip detection and all filters; their corrections are applied exactly as specified.

---

## Edge-effect guard

The `min_years_from_end` parameter (default: **5**) discards test results whose break year falls within that many years of the start or end of the test segment. Breakpoints detected near the boundaries of a series cannot be estimated reliably — the test window is too short on one side to distinguish a true break from random variation. This is the edge-effect artefact described by Hawkins (1977).

```python
NormalizationConfig(min_years_from_end=5)   # default — matches HomogenizationTest default
NormalizationConfig(min_years_from_end=10)  # stricter: wider exclusion zone
NormalizationConfig(min_years_from_end=0)   # disable the guard (not recommended)
```

The default of 5 matches the `min_years_from_end` used by all six bundled tests during the detection phase. If you configure a different `min_years_from_end` on your tests, set the same value here so detection and normalization use a consistent rule.

---

## Signal-strength filter

`min_relative_signal` (default: **1.0**) discards individual test results whose ratio of test statistic to critical value falls below the threshold, even when the binary significance flag is set. The default of 1.0 is equivalent to requiring significance at the configured `alpha` level. Raising it demands a more convincing signal.

```python
NormalizationConfig(min_relative_signal=1.0)   # default — accept any significant result
NormalizationConfig(min_relative_signal=1.5)   # require signal 1.5× the critical value
NormalizationConfig(min_relative_signal=2.0)   # very conservative
```

This filter is applied at the same stage as the edge-effect guard, before consensus is counted. It mirrors the `min_relative_signal` parameter on each `HomogenizationTest`; if you raise it during detection, raise it here too for consistent behaviour.

---

## Consensus modes

| `consensus` | Break is accepted when… |
|---|---|
| `"any"` | at least 1 test is significant |
| `"majority"` | more than half of the tests are significant (default) |
| `"unanimous"` | all tests are significant |
| `"strongest_signal"` | always use the most significant test, regardless of others |

```python
NormalizationConfig(consensus="unanimous")   # very conservative
NormalizationConfig(consensus="any")         # most sensitive
```

When consensus is not met, `tiebreak` controls what happens:

- `"strongest_signal"` (default) — apply the correction from the test with the highest signal
- `"skip"` — discard the break entirely

---

## Break predicates

A `BreakPredicate` is a callable object that receives a [`BreakInfo`][rucola.BreakInfo] and returns `True` to accept the break or `False` to discard it. Predicates are composable and fully serializable.

### Available predicates

| Predicate | Accepts a break when… |
|---|---|
| `YearBetween(min=, max=)` | break year is within `[min, max]` (both bounds optional, inclusive) |
| `StationIn(station_ids)` | station ID is in the whitelist |
| `StepIn(steps)` | break was detected at one of the given procedure steps |
| `MagnitudeAbove(threshold)` | correction magnitude exceeds the threshold |
| `TestSignificant(test_name)` | the named test flagged the series as significant |
| `SignalAbove(threshold)` | maximum relative signal across all tests exceeds the threshold |
| `NSignificantAbove(n)` | at least `n` tests are significant (absolute count) |
| `NeighborCountAbove(n)` | at least `n` reference stations were used to build the reference series |

```python
from rucola import (
    YearBetween, StationIn, StepIn,
    MagnitudeAbove, TestSignificant,
    SignalAbove, NSignificantAbove, NeighborCountAbove,
)

YearBetween(min=1960)              # no upper bound
YearBetween(max=2010)              # no lower bound
YearBetween(min=1960, max=2010)    # both bounds

StationIn({"S1", "S3", "S7"})     # frozenset or set

StepIn({3, 4, 5})                  # only trust later, more reliable steps

MagnitudeAbove(threshold=0.05)     # ratio: |factor − 1| > 0.05
                                   # difference: |factor| > 0.05

TestSignificant("snht")            # only if the SNHT was significant
TestSignificant("buishand")

SignalAbove(threshold=1.5)         # signal must be 1.5× the critical value
                                   # (stricter than the binary 95 % threshold)

NSignificantAbove(n=2)             # at least 2 tests must be significant
                                   # regardless of how many tests were run

NeighborCountAbove(n=4)            # require a solid reference pool
```

### Combining predicates

Use `&`, `|`, and `~` to compose predicates into arbitrary expressions:

```python
# AND — both conditions must pass
p = YearBetween(min=1960, max=2010) & MagnitudeAbove(threshold=0.05)

# OR — either condition suffices
p = YearBetween(max=1970) | YearBetween(min=2000)

# NOT — invert
p = ~StepIn({1, 2})

# freely nested
p = (MagnitudeAbove(threshold=0.05) & ~StepIn({1})) | StationIn({"S_priority"})
```

### Predicate and consensus together

The predicate runs **after** consensus. This means consensus first decides whether there is enough evidence for a break, and the predicate then decides whether that break should be applied given additional context.

```python
# Require majority agreement, but still restrict to a trusted year range
NormalizationConfig(
    consensus="majority",
    predicate=YearBetween(min=1960, max=2010),
)
```

### Predicate and min_correction_magnitude together

`min_correction_magnitude` and the predicate are independent filters — both must pass. Either can be used alone or together:

```python
# magnitude filter + predicate
NormalizationConfig(
    min_correction_magnitude=0.02,
    predicate=YearBetween(min=1960),
)
```

Use `MagnitudeAbove` as a predicate when you want to combine the magnitude condition with other predicate logic:

```python
# magnitude check as part of a composed predicate
NormalizationConfig(
    predicate=MagnitudeAbove(threshold=0.05) & ~StepIn({1, 2})
)
```

### Overrides and predicates

`overrides` bypass the predicate entirely. When a station is listed in `overrides`, its corrections are applied exactly as specified — no consensus check, no predicate, no magnitude filter.

```python
NormalizationConfig(
    predicate=YearBetween(min=1970),
    overrides={
        "S_special": [(1955, 1.08)],   # applied regardless of YearBetween
    },
)
```

---

## Serialization

All built-in predicates and compositions are serializable:

```python
from rucola import BreakPredicate, YearBetween, StationIn

p = YearBetween(min=1960) & ~StationIn({"S_bad"})

# round-trip
d = p.to_dict()
p2 = BreakPredicate.from_dict(d)
```

This means a full `NormalizationConfig` (including its predicate) can be stored alongside the serialized `DetectionResult` to reproduce results exactly.

---

## Manual overrides

Override the algorithmic detections for specific stations by providing exact `(break_year, factor)` pairs. This is useful when you have external metadata confirming a break that the algorithm missed or placed incorrectly.

```python
NormalizationConfig(
    overrides={
        "S1": [(1978, 1.12)],             # one manual correction
        "S2": [(1965, 0.95), (1990, 1.07)],  # two corrections
    }
)
```

Overrides replace all algorithmic detections for the named station. Other stations are unaffected.
