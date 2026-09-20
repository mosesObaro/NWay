# Testing Strategy

> **Status:** implemented. This document is the design; the code that
> realises it lives under `src/nway/`. Where the two differ, the code is
> the source of truth and this document is a bug.

Four suites, run in CI. No test makes a live network call: every provider test
replays recorded responses from `tests/fixtures/`.

---

## 1. Unit tests

**Feature calculations.** Rolling windows over hand-built fixture sequences with
known answers · window boundaries are half-open and exclude the fixture itself ·
`min_observations` yields NULL with `INSUFFICIENT_HISTORY`, never 0 ·
home-only and away-only splits select the right subset · opponent adjustment
reproduces a worked example · rest days and congestion across a known calendar.

**Probability and distributions.** Poisson pmf against scipy · the goal matrix
sums to 1 · Dixon–Coles τ alters only the four low-score cells · parameter
recovery from synthetic data with known attack/defence values · time-decay
weights are correct at known offsets.

**Market derivation.** Every market against a hand-computed matrix · all
coherence invariants ([PREDICTION_PIPELINE.md §5](PREDICTION_PIPELINE.md)) ·
`P(Over 0.5) ≥ P(Over 1.5) ≥ P(Over 2.5) ≥ P(Over 3.5)` · `P(BTTS) ≤ P(Over 1.5)` ·
double chance equals the sum of its components · matrix mean equals λ sum.

**Entity resolution.** Every curated alias resolves · `Manchester United` and
`Manchester City` never collide · unknown names queue rather than guess ·
accent and punctuation normalisation (`M'gladbach`, `Atlético`) · fuzzy matches
below the auto-accept threshold are not written.

**Recommendation ranking.** The Beta posterior lower bound on hand-computed
cases · a thin cell is shrunk toward the market prior · a miscalibrated market
ranks below a better-calibrated one at a higher raw probability — the brief's
82%-beats-85% case, asserted directly · family caps computed against actual
batch size · the relaxation step fires only when it would otherwise starve.

**Notification eligibility.** Each of the nine decision checks in isolation ·
the minimum and maximum bounds · lead-time arithmetic · cooldown boundaries.

**Timezones.** UTC → `Africa/Lagos` across a European DST transition · a naive
datetime anywhere raises · stored values are always UTC.

**Configuration.** Unknown keys rejected · `config_hash` stable under key
reordering · secrets never appear in a rendered config or a log line.

## 2. Integration tests

**Ingestion.** Recorded football-data.org responses parse into fixtures with
correct UTC kickoffs · football-data.co.uk CSVs parse with latin-1, BOM
stripping and both date formats · the `www` → apex 302 is followed · truncated
seasons (Ligue 1 2019/20 = 279 rows, Eredivisie 2019/20 = 232) parse without
tripping row-count assertions · re-parsing stored bytes is byte-deterministic ·
the rate limiter holds under concurrency.

**Feature pipeline.** End to end on a fixed dataset, with a checksum on the
resulting feature matrix.

**Prediction pipeline.** Fixture → features → model → calibration → stored
prediction, with all invariants asserted and full version metadata written.

**Recommendation and planner.** A seeded database produces an expected
decision payload.

**Scheduler.** `nway tick` twice in the same minute leaves identical state and
sends at most one email · a tick interrupted mid-way resumes cleanly.

**Email.** `FileProvider` output parses as valid multipart · the plain-text part
is generated from the data structure, not by stripping tags · a provider
failure is logged and the ledger is *not* written, so a retry is legitimate.

**Result validation.** Settlement truth table per market · extra-time matches
settle on 90 minutes · postponed fixtures void · a missing corner statistic
yields `UNSETTLEABLE`, never a loss.

## 3. Temporal leakage tests

The most important suite. Detail in
[PREDICTION_PIPELINE.md §3](PREDICTION_PIPELINE.md).

- **Future poisoning.** Compute features at `as_of`, inject absurd post-`as_of`
  data, recompute, assert identity. Random sample of fixtures and as-of times
  per season.
- **Shuffled future.** Permute all post-`as_of` results; predictions must not
  change.
- **Monotone information.** Features computed against a database truncated at
  `as_of` must equal those computed against the full database filtered to
  `as_of`.
- **Model artefact guard.** `model_version.train_window_end > as_of` raises at
  load time.
- **Calibrator guard.** `calibrator_version.fit_window_end > as_of` raises.
- **Database audit.** No `feature_value` row has
  `source_max_knowledge_time > as_of`.
- **Canary.** A deliberately leaky feature is added inside the test and the
  harness **must** catch it. This is what stops the suite silently degrading
  into one that passes because it has stopped checking.

## 4. Notification tests

Counts:

| Qualifying | Expected |
|---|---|
| 0 | no email · `NO_FIXTURES` / `NO_QUALIFYING_PREDICTIONS` |
| 3 | no email · `INSUFFICIENT_QUALIFYING` |
| 6 | no email · `INSUFFICIENT_QUALIFYING` |
| 7 | sends 7 |
| 12 | sends 12 |
| 20 | sends 20 |
| 25 | sends the best 20 |
| 100 | sends 20, all caps respected |

Behavioural:

- the same fixtures already notified → suppressed;
- probability moves 0.10 after 6h → update permitted, labelled as an update,
  previous probability shown;
- probability moves 0.03 → suppressed;
- the real 2024-11-11 → 11-19 international break → silence throughout, nothing
  logged at error level;
- a fixture postponed after notification → correction path;
- kickoff moved earlier below the minimum lead time → dropped, count re-checked,
  batch skipped if it falls under seven;
- 6 qualifying at T, a 7th enters at T+30min → sends at T+30min;
- the clock advanced past every kickoff → no send, no crash;
- 25 candidates of which 22 are double chance → family caps hold, and the batch
  is either ≥ 7 or skipped — never padded.

Property tests (Hypothesis): over randomly generated candidate pools, the
selected count is 0 or in [7, 20] · no fixture appears twice · no selection
lacks a stored explanation · the decision function is deterministic given
identical inputs · adding a *weak* candidate never changes which strong
candidates were selected (no threshold relaxation can sneak in).

## 5. Architectural tests

- `recommendation/` never imports `models/`.
- No module outside `clock.py` calls `datetime.now()` or `utcnow()`.
- No scheduling query references `matchday`.
- Every market in `markets.yaml` has a derivation or a model, and every
  `enabled_competitions` entry names a competition that exists.
- Every `feature_key` referenced by an explanation template exists in the
  feature registry.

## 6. What is deliberately not tested

Model *accuracy* is not a unit test. It is measured by the walk-forward
backtest and reported with confidence intervals. A test asserting "log loss <
1.02" would fail for legitimate reasons — a genuinely unusual season — and pass
for illegitimate ones, such as a leak. The backtest's red-flag checks
([BACKTESTING.md §8](BACKTESTING.md)) serve that purpose instead.
