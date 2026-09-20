# System Design

> **Status:** implemented. This document is the design; the code that
> realises it lives under `src/nway/`. Where the two differ, the code is
> the source of truth and this document is a bug.

Labels: **[MVP]** · **[Recommended]** · **[Future]**

---

## 1. Repository assessment

`/Users/obaromoses/Documents/NWay` was **empty** when this design started — no
files, no git repository, no hidden directories. There is no existing
architecture, no existing capability, and therefore no migration constraint.
Everything below is greenfield.

**Host environment, as measured:**

| Capability | Present | Design consequence |
|---|---|---|
| Python | 3.12.4 (Homebrew) + Anaconda 3 | Target 3.12, `venv`, no conda dependency |
| Package managers | `pip` 24.0 only — no `uv`, `poetry`, `pipx` | Plain `pip` + `requirements.txt` + `venv` |
| Databases | `sqlite3` 3.51 only — no Postgres, no DuckDB | **SQLite**, schema written Postgres-portable |
| Containers | **no Docker** | No container orchestration; plain processes |
| Node | v26 | Not needed |
| Other | `git`, `gh`, `make`, `curl`, `jq` | `make` as the task runner |
| Already installed (Anaconda) | pandas 2.2.2, scikit-learn 1.5.1, statsmodels 0.14.2, xgboost 2.1.4, SQLAlchemy 2.0.34, pytest 7.4.4 | The statistical stack is available; LightGBM is not |

This rules out Airflow, Kafka, Spark and Kubernetes — all inappropriate for a
school project anyway. The scheduler is a single idempotent process invoked by
`launchd` or `cron`.

**The repository is not yet a git repository.** `git init` is the first task of
Phase 0.

## 2. Existing capabilities

None. Stated plainly so the roadmap is not mistaken for a refactor.

## 3. Major gaps

Everything is a gap; the useful work is ranking them.

| Gap | Severity | Note |
|---|---|---|
| Ingestion, raw-response persistence, entity resolution | **Critical** | Nothing works without it |
| Bitemporal storage with `knowledge_time` | **Critical** | Retrofitting as-of-time semantics later is a rewrite |
| As-of feature store + leakage test harness | **Critical** | The one thing that makes results meaningful |
| Fixture/competition domain model on UTC timestamps | **Critical** | Scheduling primitive |
| Goal model + market derivation | **Critical** | The product |
| Probability calibration | **Critical** | A probability without calibration is a number |
| Walk-forward backtest | **Critical** | The only honest evidence of quality |
| Recommendation engine (separate from models) | **High** | The 7–20 rule lives here |
| Notification planner + scheduler + dedup | **High** | The brief's distinguishing requirement |
| Email delivery with provider abstraction | **High** | |
| Post-match validation and evaluation store | **High** | Closes the loop |
| Monitoring: drift + data quality | **Medium** | |
| Corners / cards models | **Medium** | Data verified available; league-gated |
| Half / segment markets | **Medium** | HT data is 100% complete |
| Reporting dashboard | **Low** | Static HTML report is enough |
| Player markets | **Low (blocked)** | Blocked on data, not effort |
| Europa / Conference League | **Low (blocked)** | Blocked on budget |

## 4. Target architecture

```
                         config/*.yaml + .env
                                  │
┌─────────────────────────────────┼──────────────────────────────────┐
│                          INGESTION LAYER                           │
│  football-data.org   football-data.co.uk   openfootball  Open-Meteo│
│         │                    │                  │           │      │
│         └────────────┬───────┴──────────────────┴───────────┘      │
│                      ▼                                             │
│              raw_response  (immutable, hash-addressed, replayable) │
└──────────────────────┬─────────────────────────────────────────────┘
                       ▼
        Normalisation ──▶ Entity Resolution ──▶ Validation
                       │        (canonical ids)     (quality gates)
                       ▼
┌────────────────────────────────────────────────────────────────────┐
│   CORE STORE (bitemporal: every fact has event_time + knowledge_time)│
│   competitions · seasons · teams · venues · fixtures · results      │
│   team_match_stats · players · player_stats · availability          │
└──────────────────────┬─────────────────────────────────────────────┘
                       ▼
            ┌──────────────────────────┐
            │  AS-OF FEATURE STORE     │  every read requires as_of
            │  feature_value(as_of)    │  feature_set_version
            └──────────┬───────────────┘
                       ▼
     ┌─────────────────┴────────────────────────────┐
     │                MODELS                        │
     │  goals (Dixon-Coles) ─▶ goal matrix ─▶ all   │
     │                          goal/result markets │
     │  corners (neg-binomial)   cards (neg-binom)  │
     └─────────────────┬────────────────────────────┘
                       ▼
                 CALIBRATION  (isotonic / Platt, per market × league)
                       ▼
                 prediction  (immutable snapshot, versioned)
                       ▼
            RECOMMENDATION ENGINE   ← eligibility, reliability, diversity
                       ▼
            NOTIFICATION PLANNER    ← 72h window, 7–20, lead time, cooldown
                       ▼
                 prediction_batch ──▶ EMAIL (HTML + text)
                       ▼
                    kickoff
                       ▼
            RESULT COLLECTION ──▶ MARKET SETTLEMENT
                       ▼
            EVALUATION  (log loss, Brier, ECE, by market/league/season)
                       ▼
            MONITORING  (drift, data quality, alerting)
```

### Why this shape

- **Ingestion persists raw responses before parsing.** Backtests and bug fixes
  replay bytes rather than re-hitting rate-limited APIs, and a parser bug
  becomes recoverable instead of a lost week of data.
- **Entity resolution sits between raw and core**, so no downstream component
  ever sees a provider-specific team name.
- **One goal distribution feeds every goal-derived market.** Training separate
  models for Over 1.5, Over 2.5 and BTTS would let them contradict each other —
  e.g. P(Over 2.5) > P(Over 1.5). Deriving them from a single joint
  distribution over (home goals, away goals) makes that impossible by
  construction.
- **Calibration is its own stage**, fitted on temporally held-out data, because
  a model that discriminates well can still be badly calibrated, and this
  system's output *is* the probability.
- **The recommendation engine cannot see the models** — only stored, calibrated
  predictions plus historical reliability. Prediction quality and selection
  policy change for different reasons and at different rates.
- **The planner is independent of the email provider.** It emits a decision and
  a batch; delivery is a driver behind an interface.

## 5. Module layout

Adapted from the brief's sketch to this stack — flatter, with `pipelines/`
replaced by a single CLI because there is no orchestrator.

```
nway/
├── config/
│   ├── competitions.yaml        # the ten competitions + provider codes
│   ├── markets.yaml             # market registry, per-league eligibility
│   ├── models.yaml              # hyperparameters, training windows
│   ├── recommendations.yaml     # thresholds, caps, diversity rules
│   ├── notifications.yaml       # 7/20, 72h, 0.5h, cooldown, interval
│   └── sources.yaml             # endpoints, rate limits, retry policy
├── src/nway/
│   ├── cli.py                   # `nway tick|ingest|train|backtest|report`
│   ├── clock.py                 # the ONLY source of "now" (injectable)
│   ├── config/                  # typed config loading + validation
│   ├── ingestion/
│   │   ├── http.py              # rate limiting, retries, caching, UA
│   │   ├── raw_store.py         # immutable raw_response persistence
│   │   └── providers/           # footballdata_org.py, footballdata_couk.py,
│   │                            # openfootball.py, openmeteo.py
│   ├── normalisation/
│   ├── entities/                # canonical ids + provider mappings
│   ├── storage/                 # SQLAlchemy Core schema, migrations, repos
│   ├── features/
│   │   ├── context.py           # FeatureContext(as_of) — the leakage gate
│   │   ├── registry.py          # declarative FeatureSpec definitions
│   │   ├── team_form.py  opponent_adjusted.py  match_context.py
│   │   └── store.py             # as-of reads/writes, feature_set_version
│   ├── models/
│   │   ├── base.py              # fit/predict/version protocol
│   │   ├── baselines.py         # 4 required baselines
│   │   ├── goals/               # dixon_coles.py, distribution.py, derive.py
│   │   ├── corners/  cards/     # negative-binomial
│   │   └── players/             # [Future] scaffolding only
│   ├── calibration/
│   ├── prediction/              # orchestration, snapshotting, versioning
│   ├── recommendation/          # eligibility, scoring, diversity, selection
│   ├── notifications/
│   │   ├── planner.py           # should_send_notification()
│   │   ├── batch.py             # PredictionBatch lifecycle
│   │   ├── render.py            # Jinja2 HTML + plain text
│   │   └── delivery/            # EmailProvider protocol: smtp.py, console.py
│   ├── scheduling/              # the idempotent tick
│   ├── validation/              # market settlement from results
│   ├── evaluation/              # metrics, reliability diagrams, reports
│   └── monitoring/              # drift + data quality checks
├── tests/
│   ├── unit/  integration/  leakage/  notifications/  fixtures/
├── research/                    # one-off studies, not system code
├── docs/
└── data/                        # gitignored: nway.db, raw/, snapshots/
```

`clock.py` deserves its own line: **no module calls `datetime.now()` directly.**
A single injectable clock is what makes "simulate Saturday 13:30 UTC in March
2023" a one-line test rather than a mocking exercise, and it is the mechanism
that lets the backtester reuse the production code path unchanged.

## 6. Technology choices

| Concern | Choice | Reason |
|---|---|---|
| Language | Python 3.12 | Installed; the statistical ecosystem lives here |
| Storage | **SQLite** via SQLAlchemy Core | Present on the machine, zero setup, single-writer is fine at ~2,500 fixtures/season |
| Migrations | Alembic | Schema evolves through the phases |
| Portability | ISO-8601 UTC `TEXT` timestamps, no SQLite-only types | A later move to Postgres is a connection-string change |
| Numerics | NumPy, SciPy, pandas, statsmodels | Installed |
| ML | scikit-learn; XGBoost where a GBM earns its place | Installed; LightGBM optional |
| Templating | Jinja2 | HTML + text email from one data structure |
| Config | YAML + `.env`, validated into typed objects at load | Brief requires zero hard-coding |
| Scheduling | idempotent `nway tick` under `launchd`/`cron`; APScheduler for dev | No Docker, no orchestrator |
| Tests | pytest | Installed |
| Task runner | `make` | Installed |

**Why not Postgres:** it is not installed, and the workload is ~2,500
fixtures/season with a single writer. SQLite with WAL mode handles this with
room to spare. The schema avoids SQLite-specific constructs so the decision is
reversible.

## 7. Configuration principle

Nothing in the table below appears in code: competitions, market definitions
and per-league eligibility, probability thresholds, minimum/maximum
recommendations, prediction horizon, lead time, cooldown, scheduler interval,
model hyperparameters, training windows, data-source endpoints and rate
limits, email provider and credentials, user timezone.

Configuration is loaded once, validated into frozen typed objects, and the
**resolved configuration hash is stored on every prediction row**. A prediction
made under different thresholds is a different prediction, and the audit trail
must show which.

Secrets (API tokens, SMTP credentials) live only in `.env`, never in YAML,
never in the database, and are redacted from logs.

## 8. Critical design principles, and where each is enforced

| Principle | Enforced by |
|---|---|
| Fixtures, not gameweeks, drive scheduling | `fixture.kickoff_utc` is the only scheduling key; `matchday` is a nullable metadata column with no index used by the planner |
| Predictions are probabilistic | Every market row stores a probability; email templates carry an explicit uncertainty notice |
| Temporal leakage is unacceptable | `FeatureContext(as_of)` is the only DB access path for feature code; future-poisoning property tests |
| Every prediction has an as-of timestamp | `prediction.prediction_timestamp` NOT NULL; no feature read without `as_of` |
| Backtests reproduce historical availability | Replay from `raw_response` bytes with `knowledge_time <= as_of` |
| Calibration matters as much as accuracy | Calibration stage is mandatory; ECE is a release gate |
| Recommendation engine is separate | `recommendation/` imports from `storage/` only, never `models/` — enforced by an import-linting test |
| Never force weak recommendations | The planner may only *filter*; it has no code path that lowers a threshold |
| Never fewer than 7, never more than 20 | Hard bounds in the planner + property tests across 0–30 candidates |
| ~72h lookahead, ~0.5h lead time | Config-driven, both tested at boundaries |
| No emails on empty periods | `should_send` returns `NO_FIXTURES` — an ordinary outcome, not an error, never alerted |
| No repeat notifications | Unique constraint on (fixture, market, batch) + a material-change rule |
| Refresh as information changes | Refresh ladder at T-48/24/8/lead-time |
| Never overwrite snapshots | `prediction` is append-only; `superseded_by` links versions |
| UTC internally, local for display | DB stores UTC; timezone conversion happens only in the render layer |
| Do not lower standards when sparse | Thresholds are inputs, never outputs, of the planner |
| Explanations are traceable | Each explanation line carries the `feature_key` and value that produced it |
| Model probability ≠ market probability | Separate columns, separate tables, odds never enter training |
| Every prediction is validated | Settlement job walks every unsettled prediction after kickoff + buffer |

## 9. Acceptance criteria for the system as a whole

1. `nway tick` is idempotent: running it twice in the same minute produces
   identical database state and at most one email.
2. The leakage suite passes: no feature value changes when future data is
   injected after its `as_of`.
3. A walk-forward backtest over ≥ 5 seasons reports, per market and league:
   log loss, Brier, ECE, reliability diagram, and sample size.
4. The goal model beats the pooled base-rate baseline on 1X2 log loss
   (**< 1.071**, measured) on held-out seasons, and targets **≤ 1.00**.
5. Calibration: ECE ≤ 0.03 per enabled market on held-out data, with ≥ 500
   settled predictions before a market is allowed into recommendations.
6. Notification rules hold under property tests for 0–30 candidates.
7. Every prediction in the database can be explained from stored rows alone,
   without re-running the pipeline.
8. Running the system across an international break sends nothing and logs
   nothing at error level.
