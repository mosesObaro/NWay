# Data Model

> **Status:** implemented. This document is the design; the code that
> realises it lives under `src/nway/`. Where the two differ, the code is
> the source of truth and this document is a bug.

SQLite, written to stay Postgres-portable. All timestamps are **ISO-8601 UTC
strings with a `Z` suffix** (`2026-09-20T17:30:00Z`) — lexicographic order
equals chronological order, which makes `WHERE knowledge_time <= :as_of`
index-friendly in both engines.

---

## 1. The one idea that shapes everything: bitemporality

Every fact carries two timestamps:

| Column | Meaning |
|---|---|
| `event_time` | when the thing happened in the world (kickoff, goal, injury onset) |
| `knowledge_time` | when **this system could first have known it** |

A prediction made at `as_of` may read a row **only if
`knowledge_time <= as_of`**. This single rule is what makes historical
backtesting honest, and it cannot be retrofitted — it has to be in the schema
from the first migration.

`knowledge_time` is derived at ingestion, never guessed:

- API responses: the timestamp the response was fetched, clamped to be ≥ the
  provider's own `lastUpdated` when supplied.
- football-data.co.uk CSV rows: the fetch time of the CSV containing them.
  Because that file lags up to ~3 days, a match played Wednesday may not become
  *knowable* until Sunday. **The system must model that lag rather than pretend
  the statistics were available at kickoff + 2 hours.**
- Weather: the forecast issue time, not the valid time.

Where a source genuinely cannot tell us when a fact became known, ingestion
records a **conservative upper bound** (later than reality) and sets
`knowledge_time_is_estimated = 1`. Estimated knowledge times are pessimistic by
design — an over-late estimate loses a feature, an over-early one leaks.

---

## 2. Entity layer

```sql
CREATE TABLE competition (
    competition_id      INTEGER PRIMARY KEY,
    slug                TEXT    NOT NULL UNIQUE,   -- 'premier_league'
    name                TEXT    NOT NULL,
    country             TEXT,                       -- NULL for UEFA
    kind                TEXT    NOT NULL,           -- 'DOMESTIC_LEAGUE' | 'UEFA_CLUB'
    structure           TEXT    NOT NULL,           -- 'ROUND_ROBIN' | 'LEAGUE_PHASE' | 'KNOCKOUT'
    tier                INTEGER,
    enabled             INTEGER NOT NULL DEFAULT 1,
    created_at          TEXT    NOT NULL
);

CREATE TABLE season (
    season_id           INTEGER PRIMARY KEY,
    competition_id      INTEGER NOT NULL REFERENCES competition,
    label               TEXT    NOT NULL,           -- '2026/27'
    start_date          TEXT    NOT NULL,
    end_date            TEXT    NOT NULL,
    is_current          INTEGER NOT NULL DEFAULT 0,
    UNIQUE (competition_id, label)
);

CREATE TABLE team (
    team_id             INTEGER PRIMARY KEY,
    canonical_name      TEXT    NOT NULL,           -- 'Manchester United'
    short_name          TEXT,
    country             TEXT,
    venue_id            INTEGER REFERENCES venue,
    created_at          TEXT    NOT NULL
);

CREATE TABLE venue (
    venue_id            INTEGER PRIMARY KEY,
    name                TEXT    NOT NULL,
    city                TEXT,
    country             TEXT,
    latitude            REAL,                        -- for Open-Meteo
    longitude           REAL,
    timezone            TEXT                         -- IANA, e.g. 'Europe/London'
);

CREATE TABLE player (
    player_id           INTEGER PRIMARY KEY,
    canonical_name      TEXT    NOT NULL,
    date_of_birth       TEXT,
    position            TEXT,
    created_at          TEXT    NOT NULL
);
```

### Entity resolution

One table serves every entity kind, so adding a provider never adds a table:

```sql
CREATE TABLE provider_entity_map (
    map_id              INTEGER PRIMARY KEY,
    entity_kind         TEXT    NOT NULL,   -- 'TEAM'|'PLAYER'|'COMPETITION'|'FIXTURE'|'VENUE'
    canonical_id        INTEGER NOT NULL,
    provider            TEXT    NOT NULL,   -- 'football_data_org' | 'football_data_couk' | ...
    provider_entity_id  TEXT,               -- NULL when the provider has no stable id
    provider_name       TEXT    NOT NULL,   -- 'Man United'
    confidence          REAL    NOT NULL DEFAULT 1.0,
    resolved_by         TEXT    NOT NULL,   -- 'EXACT'|'ALIAS'|'FUZZY'|'MANUAL'
    verified            INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT    NOT NULL,
    UNIQUE (entity_kind, provider, provider_name)
);

CREATE TABLE entity_resolution_queue (
    queue_id            INTEGER PRIMARY KEY,
    entity_kind         TEXT    NOT NULL,
    provider            TEXT    NOT NULL,
    provider_name       TEXT    NOT NULL,
    best_guess_id       INTEGER,
    best_score          REAL,
    status              TEXT    NOT NULL,   -- 'PENDING'|'RESOLVED'|'REJECTED'
    created_at          TEXT    NOT NULL
);
```

**Resolution strategy**, in order: exact match on a curated alias table →
normalised match (casefold, strip accents/punctuation/`FC`/`CF`/`AFC`) →
token-set fuzzy match scoped to the same competition and season. A fuzzy match
above the auto-accept threshold is written with `resolved_by='FUZZY'` and
`verified=0`; below it, the name goes to the queue and **the fixture is not
ingested**. Silent guessing is how `Manchester United` and `Manchester City`
become the same team.

Real cases the alias table must cover, drawn from the two chosen providers:
`Man United`/`Manchester United FC`, `Man City`, `Nott'm Forest`/`Nottingham
Forest FC`, `Sheffield United`/`Sheffield Utd`, `Ath Bilbao`/`Athletic Club`,
`Ath Madrid`/`Club Atlético de Madrid`, `Sp Lisbon`/`Sporting CP`,
`Paris SG`/`Paris Saint-Germain FC`, `Ein Frankfurt`/`Eintracht Frankfurt`,
`M'gladbach`/`Borussia Mönchengladbach`, `AZ Alkmaar`/`AZ`.

A weekly job reports unverified mappings; a human confirms them. For a
school project that is a five-minute chore, and it removes an entire class of
silent corruption.

---

## 3. Fixtures — the scheduling primitive

```sql
CREATE TABLE fixture (
    fixture_id          INTEGER PRIMARY KEY,
    competition_id      INTEGER NOT NULL REFERENCES competition,
    season_id           INTEGER NOT NULL REFERENCES season,
    home_team_id        INTEGER NOT NULL REFERENCES team,
    away_team_id        INTEGER NOT NULL REFERENCES team,
    kickoff_utc         TEXT    NOT NULL,          -- THE scheduling key
    status              TEXT    NOT NULL,          -- SCHEDULED|TIMED|IN_PLAY|FINISHED|POSTPONED|CANCELLED|SUSPENDED|AWARDED
    venue_id            INTEGER REFERENCES venue,
    referee_id          INTEGER REFERENCES referee,

    -- metadata only: never used to schedule, filter windows, or batch
    matchday            INTEGER,
    stage               TEXT,                       -- 'LEAGUE_PHASE'|'ROUND_OF_16'|...
    leg                 INTEGER,                    -- 1 or 2, NULL otherwise
    tie_id              TEXT,                       -- links the two legs

    kickoff_is_confirmed INTEGER NOT NULL DEFAULT 1,
    knowledge_time      TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL,
    UNIQUE (competition_id, season_id, home_team_id, away_team_id, kickoff_utc)
);

CREATE INDEX ix_fixture_window  ON fixture (kickoff_utc, status);
CREATE INDEX ix_fixture_comp    ON fixture (competition_id, kickoff_utc);

-- Append-only audit of schedule changes; drives fixture-change detection.
CREATE TABLE fixture_revision (
    revision_id     INTEGER PRIMARY KEY,
    fixture_id      INTEGER NOT NULL REFERENCES fixture,
    field           TEXT    NOT NULL,       -- 'kickoff_utc' | 'status' | 'venue_id'
    old_value       TEXT,
    new_value       TEXT,
    observed_at     TEXT    NOT NULL,
    source          TEXT    NOT NULL
);
```

`ix_fixture_window` is the index the rolling-window query uses, and it is
deliberately keyed on `kickoff_utc` alone. **There is no index on `matchday`
used by any scheduling query** — a structural reminder that matchday is
reporting metadata. The brief's requirement that gameweeks must not drive
scheduling is therefore visible in the schema, not only in prose.

Two-legged ties get `tie_id` and `leg` so that "second leg, trailing by two
goals" is expressible as a *feature* without becoming a scheduling concept.

---

## 4. Results and statistics

```sql
CREATE TABLE match_result (
    fixture_id      INTEGER PRIMARY KEY REFERENCES fixture,
    home_goals      INTEGER NOT NULL,
    away_goals      INTEGER NOT NULL,
    ht_home_goals   INTEGER,
    ht_away_goals   INTEGER,
    outcome         TEXT    NOT NULL,       -- 'H'|'D'|'A'
    went_to_et      INTEGER NOT NULL DEFAULT 0,
    went_to_pens    INTEGER NOT NULL DEFAULT 0,
    -- 90-minute figures are authoritative for market settlement
    result_source   TEXT    NOT NULL,
    event_time      TEXT    NOT NULL,       -- full-time
    knowledge_time  TEXT    NOT NULL
);

CREATE TABLE team_match_stats (
    fixture_id      INTEGER NOT NULL REFERENCES fixture,
    team_id         INTEGER NOT NULL REFERENCES team,
    is_home         INTEGER NOT NULL,
    goals           INTEGER,
    goals_conceded  INTEGER,
    ht_goals        INTEGER,
    shots           INTEGER,
    shots_on_target INTEGER,
    corners         INTEGER,
    fouls           INTEGER,
    yellow_cards    INTEGER,
    red_cards       INTEGER,
    possession      REAL,        -- NULL: not in any free source
    xg              REAL,        -- NULL until a licensed provider is bought
    sxg_proxy       REAL,        -- shot-based proxy; see MODEL_DESIGN.md
    source          TEXT    NOT NULL,
    event_time      TEXT    NOT NULL,
    knowledge_time  TEXT    NOT NULL,
    PRIMARY KEY (fixture_id, team_id)
);
```

`xg` and `sxg_proxy` are separate columns on purpose. Collapsing them into one
`xg` column is how a proxy quietly becomes a claim.

Extra-time matters for UEFA knockouts: markets settle on **90 minutes**, so
`went_to_et` exists to make settlement explicit rather than accidental.

```sql
CREATE TABLE player_match_stats (      -- [Future] schema present, unpopulated
    fixture_id INTEGER NOT NULL, player_id INTEGER NOT NULL, team_id INTEGER NOT NULL,
    minutes_played INTEGER, started INTEGER, goals INTEGER, assists INTEGER,
    shots INTEGER, shots_on_target INTEGER, xg REAL,
    yellow_cards INTEGER, red_cards INTEGER,
    source TEXT NOT NULL, event_time TEXT NOT NULL, knowledge_time TEXT NOT NULL,
    PRIMARY KEY (fixture_id, player_id)
);

CREATE TABLE player_availability (     -- [Future]
    availability_id INTEGER PRIMARY KEY,
    player_id INTEGER NOT NULL REFERENCES player,
    team_id   INTEGER NOT NULL REFERENCES team,
    status    TEXT NOT NULL,           -- 'AVAILABLE'|'INJURED'|'SUSPENDED'|'DOUBTFUL'|'UNKNOWN'
    reason TEXT, expected_return_date TEXT,
    source TEXT NOT NULL, event_time TEXT NOT NULL, knowledge_time TEXT NOT NULL
);
```

---

## 5. Raw and snapshot layer

```sql
CREATE TABLE raw_response (
    raw_id          INTEGER PRIMARY KEY,
    source          TEXT    NOT NULL,
    endpoint        TEXT    NOT NULL,
    request_params  TEXT,                    -- JSON
    http_status     INTEGER,
    content_hash    TEXT    NOT NULL,        -- sha256; dedupes unchanged payloads
    body_path       TEXT    NOT NULL,        -- data/raw/<source>/<hash>.json|csv.gz
    fetched_at      TEXT    NOT NULL,        -- becomes knowledge_time downstream
    parsed_at       TEXT,
    parse_status    TEXT,                    -- 'OK'|'FAILED'|'PARTIAL'
    parse_error     TEXT
);
CREATE INDEX ix_raw_source_time ON raw_response (source, fetched_at);

CREATE TABLE data_snapshot (
    snapshot_id     INTEGER PRIMARY KEY,
    as_of           TEXT    NOT NULL,
    source_watermarks TEXT  NOT NULL,        -- JSON {source: max_knowledge_time}
    row_counts      TEXT,                    -- JSON, for drift detection
    created_at      TEXT    NOT NULL
);
```

Storing bodies on disk and hashes in the database keeps the database small
while making every parse replayable. A changed parser is re-run over stored
bytes; nothing is re-fetched.

---

## 6. Features

```sql
CREATE TABLE feature_set_version (
    feature_version   TEXT PRIMARY KEY,      -- 'fs_2026_09_20_a'
    description       TEXT NOT NULL,
    spec_hash         TEXT NOT NULL,         -- hash of the registry definitions
    created_at        TEXT NOT NULL
);

CREATE TABLE feature_value (
    fixture_id        INTEGER NOT NULL REFERENCES fixture,
    feature_version   TEXT    NOT NULL REFERENCES feature_set_version,
    as_of             TEXT    NOT NULL,      -- the point-in-time contract
    feature_key       TEXT    NOT NULL,      -- 'home.gf_per_match.last5.home_only'
    value             REAL,
    is_null_reason    TEXT,                  -- 'INSUFFICIENT_HISTORY'|'SOURCE_MISSING'|'STALE'
    source_max_knowledge_time TEXT NOT NULL, -- newest input actually used
    computed_at       TEXT    NOT NULL,
    PRIMARY KEY (fixture_id, feature_version, as_of, feature_key)
);
CREATE INDEX ix_feature_lookup ON feature_value (fixture_id, as_of);
```

`source_max_knowledge_time` is the audit column that makes leakage detectable
after the fact: **if it ever exceeds `as_of`, the feature leaked**, and a
database-level check catches it even if a code path slipped through review.

`is_null_reason` exists so that "no value" is never ambiguous. A model that
cannot distinguish *missing* from *zero* will learn nonsense from newly
promoted teams in August.

---

## 7. Predictions — append-only

```sql
CREATE TABLE model_version (
    model_version   TEXT PRIMARY KEY,        -- 'goals_dc_v3'
    model_family    TEXT NOT NULL,           -- 'DIXON_COLES'|'NEG_BINOMIAL'|'BASELINE'|'GBM'
    target          TEXT NOT NULL,           -- 'GOALS'|'CORNERS'|'CARDS'
    trained_at      TEXT NOT NULL,
    train_window_start TEXT NOT NULL,
    train_window_end   TEXT NOT NULL,        -- must be <= any prediction's as_of
    feature_version TEXT NOT NULL REFERENCES feature_set_version,
    hyperparameters TEXT NOT NULL,           -- JSON
    artifact_path   TEXT NOT NULL,
    metrics         TEXT,                    -- JSON: holdout metrics at training
    is_active       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE calibrator_version (
    calibrator_version TEXT PRIMARY KEY,
    model_version   TEXT NOT NULL REFERENCES model_version,
    market_key      TEXT NOT NULL,
    competition_id  INTEGER REFERENCES competition,   -- NULL = pooled
    method          TEXT NOT NULL,           -- 'ISOTONIC'|'PLATT'|'BETA'|'IDENTITY'
    fitted_at       TEXT NOT NULL,
    fit_window_start TEXT NOT NULL,
    fit_window_end  TEXT NOT NULL,
    n_samples       INTEGER NOT NULL,
    ece_before      REAL, ece_after REAL,
    artifact_path   TEXT NOT NULL
);

-- One row per fixture per prediction run. NEVER updated in place.
CREATE TABLE prediction_run (
    prediction_run_id     INTEGER PRIMARY KEY,
    fixture_id            INTEGER NOT NULL REFERENCES fixture,
    prediction_timestamp  TEXT    NOT NULL,   -- the as-of cutoff
    kickoff_utc           TEXT    NOT NULL,   -- denormalised for lead-time queries
    hours_to_kickoff      REAL    NOT NULL,
    refresh_stage         TEXT    NOT NULL,   -- 'T48'|'T24'|'T8'|'FINAL'|'ADHOC'
    model_version         TEXT    NOT NULL REFERENCES model_version,
    feature_version       TEXT    NOT NULL REFERENCES feature_set_version,
    snapshot_id           INTEGER NOT NULL REFERENCES data_snapshot,
    config_hash           TEXT    NOT NULL,
    lambda_home           REAL, lambda_away REAL,     -- the goal model's output
    data_completeness     REAL    NOT NULL,           -- 0..1
    feature_staleness_hours REAL  NOT NULL,
    superseded_by         INTEGER REFERENCES prediction_run,
    created_at            TEXT    NOT NULL,
    UNIQUE (fixture_id, prediction_timestamp, model_version)
);

-- One row per market per run.
CREATE TABLE prediction (
    prediction_id       INTEGER PRIMARY KEY,
    prediction_run_id   INTEGER NOT NULL REFERENCES prediction_run,
    fixture_id          INTEGER NOT NULL REFERENCES fixture,
    market_key          TEXT    NOT NULL,     -- 'OVER_1_5'|'HOME_WIN'|'BTTS'|...
    selection           TEXT    NOT NULL,     -- 'YES'|'NO'|'HOME'|'DRAW'|'AWAY'
    raw_probability     REAL    NOT NULL,
    calibrated_probability REAL NOT NULL,
    calibrator_version  TEXT    REFERENCES calibrator_version,
    expected_value      REAL,                 -- for numeric markets (xG, corners)
    uncertainty         REAL,                 -- posterior sd / bootstrap spread
    is_derived_from     TEXT,                 -- 'GOAL_MATRIX' etc.
    created_at          TEXT    NOT NULL,
    UNIQUE (prediction_run_id, market_key, selection)
);
CREATE INDEX ix_prediction_fixture ON prediction (fixture_id, market_key);

-- Traceable explanation. One row per contributing factor.
CREATE TABLE prediction_explanation (
    explanation_id  INTEGER PRIMARY KEY,
    prediction_id   INTEGER NOT NULL REFERENCES prediction,
    direction       TEXT    NOT NULL,        -- 'SUPPORT'|'RISK'
    feature_key     TEXT    NOT NULL,        -- must exist in feature_value
    feature_value   REAL,
    reference_value REAL,                    -- league/season baseline compared against
    contribution    REAL,                    -- signed effect on the market probability
    template_key    TEXT    NOT NULL,        -- fixed phrasing; no free text
    rank            INTEGER NOT NULL
);
```

Explanations reference a `feature_key` and carry the numbers that produced
them. There is no free-text column a model could fill with plausible-sounding
prose, which is the structural answer to "do not manufacture explanations".

---

## 8. Recommendations, batches, delivery

```sql
CREATE TABLE recommendation (
    recommendation_id   INTEGER PRIMARY KEY,
    prediction_id       INTEGER NOT NULL REFERENCES prediction,
    fixture_id          INTEGER NOT NULL REFERENCES fixture,
    evaluated_at        TEXT    NOT NULL,
    passed_eligibility  INTEGER NOT NULL,
    rejection_reasons   TEXT,                  -- JSON array; kept for ALL candidates
    reliability_score   REAL,                  -- Beta posterior lower bound
    ranking_score       REAL,
    confidence_band     TEXT,                  -- 'HIGH'|'MEDIUM'|'LOW'
    correlation_group   TEXT,                  -- fixture-level group key
    selected            INTEGER NOT NULL DEFAULT 0,
    batch_id            INTEGER REFERENCES prediction_batch
);
```

Rejected candidates are stored with their reasons. Without that, "why was there
no email on Saturday?" is unanswerable — and on a system that deliberately stays
silent most of the time, that question gets asked constantly.

```sql
CREATE TABLE prediction_batch (
    batch_id                INTEGER PRIMARY KEY,
    created_at              TEXT    NOT NULL,
    prediction_window_start TEXT    NOT NULL,
    prediction_window_end   TEXT    NOT NULL,
    scheduled_send_time     TEXT,
    actual_send_time        TEXT,
    match_count             INTEGER NOT NULL,
    recommendation_count    INTEGER NOT NULL,
    status                  TEXT    NOT NULL,  -- PENDING|READY|SCHEDULED|SENT|SKIPPED|CANCELLED|FAILED
    skip_reason             TEXT,
    decision_payload        TEXT    NOT NULL,  -- the full should_send() decision, JSON
    model_version           TEXT    NOT NULL,
    feature_version         TEXT    NOT NULL,
    config_hash             TEXT    NOT NULL
);

CREATE TABLE notification_log (
    notification_id INTEGER PRIMARY KEY,
    batch_id        INTEGER NOT NULL REFERENCES prediction_batch,
    channel         TEXT    NOT NULL,          -- 'EMAIL'
    provider        TEXT    NOT NULL,
    recipient       TEXT    NOT NULL,
    subject         TEXT    NOT NULL,
    attempt         INTEGER NOT NULL DEFAULT 1,
    status          TEXT    NOT NULL,          -- 'SENT'|'FAILED'|'RETRYING'
    provider_message_id TEXT,
    error           TEXT,
    sent_at         TEXT
);

-- The duplicate-prevention ledger.
CREATE TABLE notified_selection (
    fixture_id      INTEGER NOT NULL REFERENCES fixture,
    market_key      TEXT    NOT NULL,
    selection       TEXT    NOT NULL,
    batch_id        INTEGER NOT NULL REFERENCES prediction_batch,
    prediction_id   INTEGER NOT NULL REFERENCES prediction,
    probability_sent REAL   NOT NULL,
    sent_at         TEXT    NOT NULL,
    PRIMARY KEY (fixture_id, market_key, selection, batch_id)
);
CREATE INDEX ix_notified_recent ON notified_selection (fixture_id, sent_at);
```

## 9. Validation and evaluation

```sql
CREATE TABLE market_outcome (
    fixture_id      INTEGER NOT NULL REFERENCES fixture,
    market_key      TEXT    NOT NULL,
    selection       TEXT    NOT NULL,
    outcome         TEXT    NOT NULL,          -- 'WIN'|'LOSS'|'VOID'|'UNSETTLEABLE'
    outcome_value   REAL,                      -- realised numeric (goals, corners)
    settled_at      TEXT    NOT NULL,
    settlement_source TEXT  NOT NULL,
    PRIMARY KEY (fixture_id, market_key, selection)
);

CREATE TABLE prediction_evaluation (
    prediction_id   INTEGER PRIMARY KEY REFERENCES prediction,
    outcome         TEXT    NOT NULL,
    hit             INTEGER,                   -- NULL when VOID
    log_loss        REAL,
    brier           REAL,
    was_recommended INTEGER NOT NULL DEFAULT 0,
    was_notified    INTEGER NOT NULL DEFAULT 0,
    evaluated_at    TEXT    NOT NULL
);

CREATE TABLE evaluation_metric (
    metric_id       INTEGER PRIMARY KEY,
    computed_at     TEXT    NOT NULL,
    window_start    TEXT    NOT NULL,
    window_end      TEXT    NOT NULL,
    grain           TEXT    NOT NULL,          -- 'MARKET'|'MARKET_COMPETITION'|'CONFIDENCE_BUCKET'|...
    market_key      TEXT, competition_id INTEGER, season_id INTEGER,
    model_version   TEXT, confidence_bucket TEXT,
    n_samples       INTEGER NOT NULL,
    mean_predicted  REAL, actual_rate REAL,
    log_loss REAL, brier REAL, ece REAL, auc REAL,
    mae REAL, rmse REAL, poisson_deviance REAL,
    ci_low REAL, ci_high REAL                  -- Wilson interval on actual_rate
);
```

Every aggregate carries `n_samples` and a confidence interval. A market cannot
be reported on, or be allowed into recommendations, until it has enough settled
predictions — which is the schema's way of enforcing "do not use a small sample
to make strong conclusions".

## 10. Monitoring

```sql
CREATE TABLE data_quality_check (
    check_id INTEGER PRIMARY KEY, run_at TEXT NOT NULL,
    check_name TEXT NOT NULL,                 -- 'MISSING_FIXTURES'|'DUPLICATE_FIXTURE'|'STALE_STATS'|...
    severity TEXT NOT NULL,                   -- 'INFO'|'WARN'|'ERROR'|'BLOCKING'
    entity_kind TEXT, entity_id INTEGER,
    detail TEXT NOT NULL, resolved_at TEXT
);

CREATE TABLE drift_check (
    drift_id INTEGER PRIMARY KEY, run_at TEXT NOT NULL,
    scope TEXT NOT NULL,                      -- 'GLOBAL'|'COMPETITION'|'MARKET'
    competition_id INTEGER, market_key TEXT, model_version TEXT,
    metric TEXT NOT NULL,                     -- 'ECE'|'LOG_LOSS'|'PSI'|'MEAN_PRED'
    baseline_value REAL, current_value REAL,
    n_samples INTEGER NOT NULL,
    breach INTEGER NOT NULL DEFAULT 0, detail TEXT
);
```

A `BLOCKING` data-quality result stops the prediction stage for the affected
fixtures. The system declines to predict rather than predicting from corrupt
inputs.

## 11. Relationship summary

```
competition ─< season ─< fixture >─ team (home/away) ─ venue
                          │
                          ├─ match_result
                          ├─< team_match_stats
                          ├─< fixture_revision
                          ├─< feature_value            (fixture, as_of, key)
                          ├─< prediction_run ─< prediction ─< prediction_explanation
                          │                         │
                          │                         ├─< recommendation >─ prediction_batch
                          │                         │                          │
                          │                         │                          └─< notification_log
                          │                         └─< prediction_evaluation
                          └─< market_outcome
```

## 12. Retention

Raw bodies are gzipped and kept indefinitely — a season of JSON and CSV is
tens of megabytes, and they are what makes backtests reproducible. Feature
values for superseded `feature_version`s are pruned after a configurable
window, except those referenced by a `prediction_run`, which are kept forever
so that the audit question "why did the system make this prediction?" always
has an answer.
