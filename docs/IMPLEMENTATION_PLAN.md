# Implementation Plan

Fifteen phases. Each states its objective, the modules it touches, its tasks,
what it depends on, its tests, and the acceptance criteria that must hold
before the next phase starts.

Effort estimates assume one student working part-time. **Phases 0–8 are the
MVP**; a working, honest system exists at the end of Phase 8 even if nothing
after it is built.

---

## Phase 0 — Project foundation · [MVP] · ~2 days

**Objective.** A runnable skeleton with configuration, logging, the injectable
clock and CI.

**Modules.** `git init`, `pyproject.toml`, `requirements.txt`, `Makefile`,
`src/nway/{cli,clock,config}`, `tests/`, `.github/workflows/ci.yml`, `.env.example`.

**Tasks.** Initialise the repository and a 3.12 virtualenv · pin dependencies ·
typed YAML config loading with validation and a `config_hash` · structured JSON
logging with secret redaction · `clock.py` returning timezone-aware UTC, with a
`frozen_at` helper · `nway --help` · pytest and ruff in CI.

**Depends on.** Nothing.

**Tests.** Config rejects unknown keys and bad types; `config_hash` is stable
across reorderings; a lint test fails on any `datetime.now()` outside
`clock.py`; CI runs green.

**Acceptance.** `make test` and `make lint` pass. `nway --help` lists the
commands. No secret ever appears in a log line.

---

## Phase 1 — Data foundation · [MVP] · ~1 week

**Objective.** Ingest from football-data.org and football-data.co.uk into an
immutable raw store, with rate limiting and replayable parsing.

**Modules.** `ingestion/{http,raw_store}`, `ingestion/providers/*`,
`storage/{schema,migrations}`, `cli:ingest`.

**Tasks.** HTTP layer with token-bucket limiting, backoff with jitter,
`ETag`/`If-Modified-Since`, descriptive `User-Agent` with contact, on-disk cache
· `raw_response` persistence with sha256 content addressing · football-data.org
client for competitions, seasons, teams and fixtures, reading
`X-Requests-Available` · football-data.co.uk CSV client — **follow the 302 from
`www.`**, latin-1 decoding, BOM stripping, both date formats · Alembic
migration 001 for entity and raw tables · `knowledge_time` assignment.

**Depends on.** Phase 0.

**Tests.** Recorded-fixture tests for both providers (no live calls in CI) ·
BOM and latin-1 parsing · both date formats · 302 redirect handling · rate
limiter enforces the budget under concurrency · truncated seasons (Ligue 1
2019/20 = 279 rows, Eredivisie 2019/20 = 232) parse without tripping row-count
assertions · re-parsing stored bytes is deterministic.

**Acceptance.** `nway ingest --source football_data_couk --seasons 2017/18..2025/26`
loads ~21,500 matches for the seven leagues. Every raw response is on disk and
replayable. No live network call in the test suite.

---

## Phase 2 — Fixture and competition model · [MVP] · ~4 days

**Objective.** Canonical entities, entity resolution, and the fixture domain on
UTC timestamps.

**Modules.** `entities/`, `normalisation/`, `storage/repositories`,
`domain/fixtures`.

**Tasks.** Competition registry from `config/competitions.yaml`, including the
provider-code map and the **`CL` = Champions, `UCL` = Conference** trap ·
`provider_entity_map` and the curated alias table for the seven leagues ·
exact → normalised → scoped-fuzzy resolution with an auto-accept threshold and
a review queue · fixture upsert with revision tracking · the rolling-window
query · `matchday` stored, unindexed for scheduling.

**Depends on.** Phase 1.

**Tests.** Every alias in the table resolves · an unknown name queues rather
than guessing · `Manchester United` and `Manchester City` never collide ·
fixture upsert is idempotent · a kickoff change writes a revision · the window
query returns matches across competitions ordered by kickoff · a test asserts
no scheduling query references `matchday`.

**Acceptance.** All seven leagues' team names from both providers resolve to
the same canonical ids, with zero unresolved names for the current season. The
72-hour window query returns a multi-competition list.

---

## Phase 3 — Historical feature engineering · [MVP] · ~1.5 weeks

**Objective.** The as-of feature store, and the leakage harness that makes it
trustworthy. **This is the phase that determines whether any later number
means anything.**

**Modules.** `features/{context,registry,store,team_form,opponent_adjusted,match_context}`,
migration 002, `tests/leakage/`.

**Tasks.** `FeatureContext(as_of)` and `AsOfRepository` as the only data path
from feature code · declarative `FeatureSpec` registry hashed into
`feature_version` · rolling windows (3/5/10/STD × all/home/away) with
`min_observations` and `max_staleness_hours` · opponent-adjusted features and
strength of schedule · match context: rest days, congestion, season progress,
promoted flag, stage, leg · `is_null_reason` handling · **the future-poisoning
test harness**, plus shuffled-future and monotone-information tests.

**Depends on.** Phase 2.

**Tests.** Future poisoning across a random sample of fixtures and as-of times ·
a team's first-ever match yields NULL with `INSUFFICIENT_HISTORY`, not 0 ·
window boundaries are half-open and exclude the fixture itself · statistics
arriving after `as_of` are invisible · a deliberately leaky feature is added in
a test and the harness catches it.

**Acceptance.** The full leakage suite passes, **including the deliberately
leaky feature being caught**. Features compute for every fixture from 2018/19
onward. No `feature_value` row has `source_max_knowledge_time > as_of`.

---

## Phase 4 — Baseline models · [MVP] · ~3 days

**Objective.** Four baselines and the evaluation harness, so every later claim
has something to beat.

**Modules.** `models/{base,baselines}`, `evaluation/metrics`, migration 003.

**Tasks.** The `Model` protocol (fit/predict/version/artefact) · uniform,
base-rate, form and naive-Poisson baselines · log loss, Brier, ECE, reliability
diagrams, Wilson intervals · the walk-forward runner.

**Depends on.** Phase 3.

**Tests.** Metrics against hand-computed values · uniform scores exactly
ln 3 = 1.0986 · walk-forward never trains on test-fold data.

**Acceptance.** Reproduces the measured benchmarks: uniform 1.0986 / 0.6667 and
pooled base rate ≈ 1.071 / 0.648 on the seven leagues, 2017/18–2025/26.

---

## Phase 5 — Statistical goal model · [MVP] · ~1.5 weeks

**Objective.** Dixon–Coles, the goal matrix, and every derived market.

**Modules.** `models/goals/{dixon_coles,distribution,derive}`, migration 004.

**Tasks.** Time-weighted MLE for attack/defence/home-advantage with
per-competition intercepts · the low-score τ correction · L2 shrinkage · ξ and
the penalty fitted by held-out likelihood · `crowd_present` indicator for
2020/21 · the 11×11 matrix · derivation of 1X2, double chance, DNB,
over/under 0.5–3.5, BTTS, team goals, clean sheet, margin, ranges, expected
goals · coherence invariants.

**Depends on.** Phase 4.

**Tests.** Every coherence invariant · recovers known parameters from
synthetic data · monotonicity of over-lines · `P(BTTS) ≤ P(Over 1.5)` ·
matrix mean equals λ sum · τ affects only the four low-score cells.

**Acceptance.** Held-out 1X2 log loss **< 1.071** (beats the base rate), with
**≤ 1.00** as the target. All invariants hold on every fixture in a full
season. A log loss below 0.95 triggers the leakage suite before it is believed.

---

## Phase 6 — Calibration · [MVP] · ~4 days

**Objective.** Probabilities that mean what they say.

**Modules.** `calibration/`, migration 005.

**Tasks.** Isotonic, Platt and beta calibrators with the sample-size selection
rule · per (market, competition) with pooled fallback · temporal fitting with a
separate calibration fold · versioned artefacts recording ECE before and after ·
the market-eligibility sample floor.

**Depends on.** Phase 5.

**Tests.** A deliberately miscalibrated input is corrected · the fit window
never overlaps the target period · below the sample floor the identity
calibrator is used and the market is marked ineligible · ECE is computed
correctly against a hand-worked example.

**Acceptance.** Held-out ECE ≤ 0.03 for 1X2, Over 1.5, Over 2.5 and BTTS, with
reliability diagrams produced. Reference: de-vigged market ECE on Over 2.5 is
0.0132.

---

## Phase 7 — Backtesting · [MVP] · ~1 week

**Objective.** Walk-forward simulation of the whole product, with reconstructed
knowledge times.

**Modules.** `cli:backtest`, `evaluation/backtest`, `research/`.

**Tasks.** Simulated tick loop under the frozen clock · reconstructed
knowledge times with the measured 72-hour statistics lag and the estimated flag
· three expanding-window folds plus a rolling-window sensitivity run · output
artefacts including `decisions.parquet` · the six red-flag checks · determinism
test.

**Depends on.** Phase 6.

**Tests.** Identical inputs produce identical outputs · a deliberately leaky
feature makes the red-flag check fire · the 72h lag genuinely hides recent
statistics · no live network access.

**Acceptance.** A three-fold backtest over 2023/24–2025/26 produces per-market
and per-competition metrics, reliability diagrams and a decision log. The
report states which fields used estimated knowledge times.

---

## Phase 8 — Recommendation engine, planner, email · [MVP] · ~1.5 weeks

**Objective.** The product: select 7–20, decide when to send, send it.

**Modules.** `recommendation/`, `notifications/{planner,batch,render,delivery}`,
`scheduling/`, migrations 006–007.

**Tasks.** Eligibility gates · Beta-posterior reliability scoring · diversity
and correlation caps, with exact within-fixture correlation from the goal
matrix · selection with hard 7/20 bounds · `should_send_notification` as a pure
function with the nine checks and the structured decision · `PredictionBatch`
lifecycle · `notified_selection` ledger and the material-change rule · Jinja2
HTML plus independently generated plain text · `EmailProvider` protocol with
SMTP, console and file implementations · retries and `notification_log` ·
timezone rendering · the idempotent `nway tick`.

**Depends on.** Phase 6 (Phase 7 informs the thresholds).

**Tests.** The full count matrix 0/3/6/7/12/20/25/100 · every behavioural case
in [NOTIFICATION_ARCHITECTURE.md §12](NOTIFICATION_ARCHITECTURE.md) · the real
2024-11-11→19 international break produces silence · property tests on the
bounds and on selection monotonicity · an import-linting test that
`recommendation/` never imports `models/` · a DST-boundary timezone test ·
double-tick idempotency · plain text is generated from data, not from stripping
HTML.

**Acceptance.** `nway tick --dry-run` produces a decision with reasons on real
data. Two ticks in the same minute send at most one email. Never fewer than 7,
never more than 20. Nothing is sent during an international break, and nothing
is logged at error level. Every rendered explanation traces to a
`prediction_explanation` row.

> **MVP complete.** Everything below extends a working system.

---

## Phase 9 — Post-match validation and evaluation · [MVP-adjacent] · ~5 days

**Objective.** Close the loop: settle every prediction, measure everything.

**Modules.** `validation/`, `evaluation/aggregates`, `cli:report`, migration 008.

**Tasks.** Result polling with backoff · cross-source verification with
`BLOCKING` hold on disagreement · deterministic settlement per market,
excluding extra time · `VOID` and `UNSETTLEABLE` handling · per-prediction
metrics · aggregates at six grains with Wilson intervals · the
`min_samples_for_conclusion` rule · a static HTML report.

**Depends on.** Phase 8.

**Tests.** Settlement truth table per market · extra-time matches settle on 90
minutes · a postponed fixture voids · a missing corner statistic yields
`UNSETTLEABLE`, not a loss · aggregates match hand-computed values · a
corrected result creates a revision rather than a silent overwrite.

**Acceptance.** Every prediction older than kickoff + 6h is settled, voided or
explicitly unsettleable. `nway report --window 90d` renders hit rate,
calibration and sample size by market, competition and confidence bucket.

---

## Phase 10 — Corners and cards models · [Recommended] · ~1 week

**Objective.** Extend market coverage where the data supports it.

**Modules.** `models/corners/`, `models/cards/`, `config/markets.yaml`.

**Tasks.** Negative-binomial GLMs with fitted dispersion · per-competition card
intercepts (measured range 3.18–5.39) · a Premier League referee random effect,
with `referee_unknown` elsewhere · convolution to totals with a shared
match-level intensity · per-league eligibility gating on training sample.

**Depends on.** Phase 9.

**Tests.** Recovers dispersion from synthetic data · a Poisson fit is rejected
on the real corner distribution · Eredivisie and Primeira Liga are gated off
until their sample clears the floor · totals equal the convolution.

**Acceptance.** Held-out corners MAE beats the league-mean baseline; cards
likewise; ECE ≤ 0.05 on enabled corner and card over/under markets. Markets
enable per league only where they pass.

---

## Phase 11 — Half-time and segment markets · [Recommended] · ~4 days

Half-specific rates (measured 1.244 first half, 1.563 second), HT 1X2 and
first/second-half over-unders derived from separate matrices. Half-time data is
100% complete in all seven leagues, so this is cheap coverage. Acceptance:
coherence invariants hold jointly with full-time markets (HT goals ≤ FT goals
in every derived quantity), ECE ≤ 0.05.

## Phase 12 — Monitoring · [Recommended] · ~5 days

Data-quality checks (missing and duplicate fixtures, stale statistics, invalid
values, unresolved entities, source failures, schedule changes, rising
`UNSETTLEABLE` rate) and drift checks (rolling ECE, PSI on feature and
prediction distributions, per-competition degradation, season-over-season
comparison). A `BLOCKING` result stops prediction for affected fixtures. A
retraining recommendation is raised, never applied automatically.
Acceptance: an injected anomaly is detected; a simulated calibration drift
raises a breach; no alert fires during an international break.

## Phase 13 — Reporting · [Recommended] · ~4 days

Static HTML reports: reliability diagrams per market, performance by
competition and season, confidence-bucket tables, batch history with skip
reasons, and a model-comparison view. No dashboard server; a generated file is
enough and survives the project being put down for a month.

## Phase 14 — ML residual model · [Optional] · ~1 week

Gradient boosting on Dixon–Coles residuals using context features. Ships only
if it beats plain Dixon–Coles on held-out log likelihood by more than the
bootstrap standard error in walk-forward testing, with no competition
degrading beyond tolerance. Explicitly permitted to fail and be abandoned —
recording that it did not help is a legitimate result.

## Phase 15 — Player markets · [Future, blocked] · not scheduled

Blocked on data, not effort ([DATA_SOURCES.md §9](DATA_SOURCES.md)). Requires a
paid source for lineups, minutes and injuries. The pipeline shape is
player → expected minutes → expected opportunities → scoring probability, and
no player recommendation may be produced when availability is unknown, expected
minutes are too uncertain, or the market lacks validated calibration.

---

## Ordering rationale

Phase 3 precedes every model because the feature store's temporal contract
cannot be retrofitted. Phase 4 precedes Phase 5 so the goal model has a bar to
clear. Phase 6 precedes Phase 8 because the recommendation engine consumes
calibrated probabilities and reliability history, and without them the
selection logic is ranking noise. Phase 7 precedes threshold-setting in Phase 8
because the per-market floors are derived from a market proxy and must be
re-derived from the system's own behaviour.

Phases 10 and 11 come after Phase 9 deliberately: extending market coverage
before the validation loop exists means adding markets nobody can evaluate.

## Critical path

```
0 → 1 → 2 → 3 → 4 → 5 → 6 → 8 → 9      (MVP, ~7 weeks part-time)
                      ↘ 7 ↗            (informs Phase 8 thresholds)
                            → 10, 11, 12, 13, 14
```
