# Prediction Pipeline

> **Status:** implemented. This document is the design; the code that
> realises it lives under `src/nway/`. Where the two differ, the code is
> the source of truth and this document is a bug.

How a fixture becomes a set of calibrated probabilities, and the mechanisms
that stop future information getting in.

---

## 1. The path

```
fixture + as_of (prediction_timestamp)
        │
        ▼
 [1] Snapshot        record source watermarks -> snapshot_id
        ▼
 [2] Gate            data quality + completeness; may refuse to predict
        ▼
 [3] Features        FeatureContext(as_of) -> feature_value rows
        ▼
 [4] Goal model      dixon_coles -> λ_home, λ_away, τ
        ▼
 [5] Distribution    11x11 goal matrix
        ▼
 [6] Derivation      every goal/result market from the matrix
        ▼
 [7] Side models     corners_nb, cards_nb   [Recommended]
        ▼
 [8] Calibration     per (market, competition), temporally fitted
        ▼
 [9] Invariants      monotonicity, normalisation, coherence
        ▼
[10] Persist         prediction_run + prediction + prediction_explanation
```

Steps 1–10 are one transaction. A failure at step 9 rolls the run back rather
than storing a partially coherent prediction set.

## 2. The as-of contract

`prediction_timestamp` is the cutoff. Everything the run reads must satisfy
`knowledge_time <= prediction_timestamp`.

Four rules, enforced in four different places so that a single mistake does not
defeat them all:

1. **API.** Feature functions receive `FeatureContext(as_of)` and reach the
   database only through `AsOfRepository`, whose every method takes `as_of`.
   There is no unfiltered session to reach for from `features/`.
2. **Storage.** Each `feature_value` row records
   `source_max_knowledge_time` — the newest input that actually contributed.
   A check constraint and a nightly audit assert it never exceeds `as_of`.
3. **Model artefacts.** `model_version.train_window_end` must be ≤ the `as_of`
   of any prediction that uses it. A model trained through May 2026 may not
   produce a March 2026 backtest prediction. Asserted at load time, not review
   time.
4. **Tests.** The future-poisoning suite, §3.

### Sources of leakage this addresses

| Leak | Blocked by |
|---|---|
| Future match results in rolling windows | `knowledge_time <= as_of` filter |
| League table computed including later matches | Table is derived as-of, never stored as "current" |
| Post-match statistics of the fixture itself | Statistics carry the fixture's own `knowledge_time` = full-time |
| Injury known after the cutoff | Availability rows carry `knowledge_time` |
| Lineups announced after the cutoff | Same; and confirmed lineups are ~1h pre-kickoff, after the default cutoff |
| Closing odds | Odds never enter features at all |
| Calibrator fitted on the target period | `calibrator_version.fit_window_end` ≤ `as_of` |
| Weather *actuals* instead of forecast | Reanalysis endpoint banned; forecast archive used in backtest |
| Stats arriving late but treated as timely | `knowledge_time` = CSV fetch time, modelling the ~3-day lag |

That last one is the subtle one. The Wednesday match whose statistics appear in
Sunday's CSV was *played* on Wednesday but only became *knowable* on Sunday. A
naive implementation would use it for Saturday's prediction. The bitemporal
schema is what makes the distinction expressible at all.

## 3. Future poisoning — the leakage test harness

Enumerating leak paths catches the ones already thought of. This catches the
rest.

```python
def test_features_are_blind_to_the_future(fixture, as_of):
    before = compute_features(fixture, as_of)

    # Inject deliberately absurd data AFTER as_of: 9-0 scorelines, 40 corners,
    # red cards, a fabricated injury, a fabricated result for this very fixture.
    poison_database_after(as_of)

    after = compute_features(fixture, as_of)
    assert before == after      # any difference is a leak
```

If any feature value changes, some code path read past the cutoff. The test
does not need to know *how* — only that it happened. Run across a random sample
of fixtures and as-of times per season, per feature version, in CI.

Three companions:

- **Shuffled-future test.** Permute all post-`as_of` results and assert
  predictions are unchanged. Catches ordering-dependent leaks.
- **Monotone-information test.** Features at T-48 computed twice, once with the
  database as of T-48 and once with the full database filtered to T-48, must
  agree. Catches filters applied at the wrong layer.
- **Backtest sanity.** A held-out log loss meaningfully below the market's
  measured 0.956 is treated as a **leak alarm**, not a success.

## 4. Data-quality gate

Before predicting, the fixture is checked: both teams resolved and verified;
required features present at ≥ `min_data_completeness`; feature staleness
within bounds; fixture status schedulable; no unresolved `BLOCKING` issue.

Failure records a `prediction_skipped` event with the reason. The system
declines to predict rather than predicting from corrupt inputs, and the skipped
fixture is simply absent from the candidate pool.

## 5. Coherence invariants

Asserted after derivation, failing the run if violated:

```
|Σ matrix − 1|            < 1e-9
|P(H)+P(D)+P(A) − 1|      < 1e-9
P(Over 0.5) ≥ P(Over 1.5) ≥ P(Over 2.5) ≥ P(Over 3.5)
P(BTTS) ≤ P(Over 1.5)
E[total goals] ≈ λ_home + λ_away        (within 1e-6)
P(double chance HD) = P(H) + P(D)
every probability ∈ [0, 1]
```

These are the payoff of deriving markets from one distribution: they are
guaranteed by construction, and the assertions exist to catch implementation
bugs rather than modelling disagreements.

## 6. Explanations

Generated from stored quantities, never written freely:

1. Take the run's features and compare each to its reference (league-season
   mean, or the opponent's corresponding value).
2. Compute each feature's contribution by re-deriving the market probability
   with that feature at its reference value — a leave-one-out effect in
   probability units.
3. Keep the top *k* positive contributors as `SUPPORT` and the top *m* negative
   as `RISK`.
4. Render each through a fixed template keyed by `feature_key`.

Every rendered line traces to a `prediction_explanation` row carrying the
feature key, its value, the reference and the contribution. An explanation
that cannot be traced cannot be rendered — there is no fallback text.

## 7. Versioning and reproducibility

Every `prediction_run` stores `model_version`, `feature_version`,
`snapshot_id`, `config_hash` and `prediction_timestamp`. Together these answer
"why did the system make this prediction?" from stored rows alone: the feature
values as of that moment, the model artefact, the calibrator, the thresholds in
force, and the data watermark.

`nway explain --prediction-id N` prints exactly that, without re-running
anything.
