"""Database schema.

SQLite, written to stay PostgreSQL-portable: timestamps are ISO-8601 UTC TEXT
with a Z suffix (lexicographic order == chronological order), no SQLite-only
types, no AUTOINCREMENT.

The defining property is bitemporality. Every fact carries ``event_time`` (when
it happened) and ``knowledge_time`` (when this system could first have known
it). A prediction made at ``as_of`` may read a row only when
``knowledge_time <= as_of``. That distinction is what makes backtests honest,
and it cannot be retrofitted, so it is here from the first migration.
"""

SCHEMA_VERSION = 1

DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL
);

-- ---------------------------------------------------------------- entities
CREATE TABLE IF NOT EXISTS competition (
    competition_id INTEGER PRIMARY KEY,
    slug           TEXT NOT NULL UNIQUE,
    name           TEXT NOT NULL,
    country        TEXT,
    kind           TEXT NOT NULL,
    structure      TEXT NOT NULL,
    enabled        INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS season (
    season_id      INTEGER PRIMARY KEY,
    competition_id INTEGER NOT NULL REFERENCES competition,
    label          TEXT NOT NULL,
    start_date     TEXT NOT NULL,
    end_date       TEXT NOT NULL,
    is_current     INTEGER NOT NULL DEFAULT 0,
    UNIQUE (competition_id, label)
);

CREATE TABLE IF NOT EXISTS venue (
    venue_id  INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    city      TEXT,
    country   TEXT,
    latitude  REAL,
    longitude REAL,
    timezone  TEXT
);

CREATE TABLE IF NOT EXISTS team (
    team_id        INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL UNIQUE,
    short_name     TEXT,
    country        TEXT,
    venue_id       INTEGER REFERENCES venue,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS referee (
    referee_id     INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL UNIQUE,
    country        TEXT
);

CREATE TABLE IF NOT EXISTS player (
    player_id      INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    date_of_birth  TEXT,
    position       TEXT,
    created_at     TEXT NOT NULL
);

-- One table for every entity kind, so a new provider never adds a table.
CREATE TABLE IF NOT EXISTS provider_entity_map (
    map_id             INTEGER PRIMARY KEY,
    entity_kind        TEXT NOT NULL,
    canonical_id       INTEGER NOT NULL,
    provider           TEXT NOT NULL,
    provider_entity_id TEXT,
    provider_name      TEXT NOT NULL,
    confidence         REAL NOT NULL DEFAULT 1.0,
    resolved_by        TEXT NOT NULL,
    verified           INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT NOT NULL,
    UNIQUE (entity_kind, provider, provider_name)
);

CREATE TABLE IF NOT EXISTS entity_resolution_queue (
    queue_id      INTEGER PRIMARY KEY,
    entity_kind   TEXT NOT NULL,
    provider      TEXT NOT NULL,
    provider_name TEXT NOT NULL,
    best_guess_id INTEGER,
    best_score    REAL,
    status        TEXT NOT NULL DEFAULT 'PENDING',
    created_at    TEXT NOT NULL,
    UNIQUE (entity_kind, provider, provider_name)
);

-- ---------------------------------------------------------------- fixtures
CREATE TABLE IF NOT EXISTS fixture (
    fixture_id     INTEGER PRIMARY KEY,
    competition_id INTEGER NOT NULL REFERENCES competition,
    season_id      INTEGER NOT NULL REFERENCES season,
    home_team_id   INTEGER NOT NULL REFERENCES team,
    away_team_id   INTEGER NOT NULL REFERENCES team,
    kickoff_utc    TEXT NOT NULL,
    status         TEXT NOT NULL,
    venue_id       INTEGER REFERENCES venue,
    referee_id     INTEGER REFERENCES referee,
    -- metadata only. Nothing in scheduling reads these.
    matchday       INTEGER,
    stage          TEXT,
    leg            INTEGER,
    tie_id         TEXT,
    kickoff_is_confirmed INTEGER NOT NULL DEFAULT 1,
    knowledge_time TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    UNIQUE (competition_id, season_id, home_team_id, away_team_id, kickoff_utc)
);

-- The index the rolling-window query uses. Deliberately keyed on kickoff
-- alone: matchday is reporting metadata, never a scheduling key.
CREATE INDEX IF NOT EXISTS ix_fixture_window ON fixture (kickoff_utc, status);
CREATE INDEX IF NOT EXISTS ix_fixture_comp   ON fixture (competition_id, kickoff_utc);
CREATE INDEX IF NOT EXISTS ix_fixture_teams  ON fixture (home_team_id, away_team_id);

CREATE TABLE IF NOT EXISTS fixture_revision (
    revision_id INTEGER PRIMARY KEY,
    fixture_id  INTEGER NOT NULL REFERENCES fixture,
    field       TEXT NOT NULL,
    old_value   TEXT,
    new_value   TEXT,
    observed_at TEXT NOT NULL,
    source      TEXT NOT NULL
);

-- ------------------------------------------------------- results and stats
CREATE TABLE IF NOT EXISTS match_result (
    fixture_id     INTEGER PRIMARY KEY REFERENCES fixture,
    home_goals     INTEGER NOT NULL,
    away_goals     INTEGER NOT NULL,
    ht_home_goals  INTEGER,
    ht_away_goals  INTEGER,
    outcome        TEXT NOT NULL,
    went_to_et     INTEGER NOT NULL DEFAULT 0,
    went_to_pens   INTEGER NOT NULL DEFAULT 0,
    result_source  TEXT NOT NULL,
    event_time     TEXT NOT NULL,
    knowledge_time TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS team_match_stats (
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
    possession      REAL,
    xg              REAL,          -- NULL until a licensed provider is bought
    sxg_proxy       REAL,          -- shot-based proxy; never called xg
    source          TEXT NOT NULL,
    event_time      TEXT NOT NULL,
    knowledge_time  TEXT NOT NULL,
    PRIMARY KEY (fixture_id, team_id)
);
CREATE INDEX IF NOT EXISTS ix_tms_team_know ON team_match_stats (team_id, knowledge_time);

-- ---------------------------------------------------------- raw + snapshot
CREATE TABLE IF NOT EXISTS raw_response (
    raw_id         INTEGER PRIMARY KEY,
    source         TEXT NOT NULL,
    endpoint       TEXT NOT NULL,
    request_params TEXT,
    http_status    INTEGER,
    content_hash   TEXT NOT NULL,
    body_path      TEXT NOT NULL,
    fetched_at     TEXT NOT NULL,
    parsed_at      TEXT,
    parse_status   TEXT,
    parse_error    TEXT
);
CREATE INDEX IF NOT EXISTS ix_raw_source_time ON raw_response (source, fetched_at);

CREATE TABLE IF NOT EXISTS data_snapshot (
    snapshot_id       INTEGER PRIMARY KEY,
    as_of             TEXT NOT NULL,
    source_watermarks TEXT NOT NULL,
    row_counts        TEXT,
    created_at        TEXT NOT NULL
);

-- ---------------------------------------------------------------- features
CREATE TABLE IF NOT EXISTS feature_set_version (
    feature_version TEXT PRIMARY KEY,
    description     TEXT NOT NULL,
    spec_hash       TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feature_value (
    fixture_id      INTEGER NOT NULL REFERENCES fixture,
    feature_version TEXT NOT NULL REFERENCES feature_set_version,
    as_of           TEXT NOT NULL,
    feature_key     TEXT NOT NULL,
    value           REAL,
    is_null_reason  TEXT,
    source_max_knowledge_time TEXT NOT NULL,
    computed_at     TEXT NOT NULL,
    PRIMARY KEY (fixture_id, feature_version, as_of, feature_key),
    -- the audit rule, enforced by the database rather than by review
    CHECK (source_max_knowledge_time <= as_of)
);
CREATE INDEX IF NOT EXISTS ix_feature_lookup ON feature_value (fixture_id, as_of);

-- ------------------------------------------------------------- predictions
CREATE TABLE IF NOT EXISTS model_version (
    model_version      TEXT PRIMARY KEY,
    model_family       TEXT NOT NULL,
    target             TEXT NOT NULL,
    trained_at         TEXT NOT NULL,
    train_window_start TEXT NOT NULL,
    train_window_end   TEXT NOT NULL,
    feature_version    TEXT,
    hyperparameters    TEXT NOT NULL,
    artifact_path      TEXT NOT NULL,
    metrics            TEXT,
    is_active          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS calibrator_version (
    calibrator_version TEXT PRIMARY KEY,
    model_version    TEXT NOT NULL REFERENCES model_version,
    market_key       TEXT NOT NULL,
    competition_id   INTEGER REFERENCES competition,
    method           TEXT NOT NULL,
    fitted_at        TEXT NOT NULL,
    fit_window_start TEXT NOT NULL,
    fit_window_end   TEXT NOT NULL,
    n_samples        INTEGER NOT NULL,
    ece_before       REAL,
    ece_after        REAL,
    artifact_path    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prediction_run (
    prediction_run_id    INTEGER PRIMARY KEY,
    fixture_id           INTEGER NOT NULL REFERENCES fixture,
    prediction_timestamp TEXT NOT NULL,
    kickoff_utc          TEXT NOT NULL,
    hours_to_kickoff     REAL NOT NULL,
    refresh_stage        TEXT NOT NULL,
    model_version        TEXT NOT NULL REFERENCES model_version,
    feature_version      TEXT NOT NULL,
    snapshot_id          INTEGER REFERENCES data_snapshot,
    config_hash          TEXT NOT NULL,
    lambda_home          REAL,
    lambda_away          REAL,
    data_completeness    REAL NOT NULL,
    feature_staleness_hours REAL NOT NULL,
    superseded_by        INTEGER REFERENCES prediction_run,
    created_at           TEXT NOT NULL,
    UNIQUE (fixture_id, prediction_timestamp, model_version)
);
CREATE INDEX IF NOT EXISTS ix_run_fixture ON prediction_run (fixture_id, prediction_timestamp);

CREATE TABLE IF NOT EXISTS prediction (
    prediction_id     INTEGER PRIMARY KEY,
    prediction_run_id INTEGER NOT NULL REFERENCES prediction_run,
    fixture_id        INTEGER NOT NULL REFERENCES fixture,
    market_key        TEXT NOT NULL,
    selection         TEXT NOT NULL,
    raw_probability   REAL NOT NULL,
    calibrated_probability REAL NOT NULL,
    calibrator_version TEXT,
    expected_value    REAL,
    uncertainty       REAL,
    is_derived_from   TEXT,
    created_at        TEXT NOT NULL,
    UNIQUE (prediction_run_id, market_key, selection)
);
CREATE INDEX IF NOT EXISTS ix_prediction_fixture ON prediction (fixture_id, market_key);

CREATE TABLE IF NOT EXISTS prediction_explanation (
    explanation_id  INTEGER PRIMARY KEY,
    prediction_id   INTEGER NOT NULL REFERENCES prediction,
    direction       TEXT NOT NULL,
    feature_key     TEXT NOT NULL,
    feature_value   REAL,
    reference_value REAL,
    contribution    REAL,
    template_key    TEXT NOT NULL,
    rank            INTEGER NOT NULL
);

-- ------------------------------------------- recommendations + batches
CREATE TABLE IF NOT EXISTS prediction_batch (
    batch_id                INTEGER PRIMARY KEY,
    created_at              TEXT NOT NULL,
    prediction_window_start TEXT NOT NULL,
    prediction_window_end   TEXT NOT NULL,
    scheduled_send_time     TEXT,
    actual_send_time        TEXT,
    match_count             INTEGER NOT NULL,
    recommendation_count    INTEGER NOT NULL,
    status                  TEXT NOT NULL,
    skip_reason             TEXT,
    decision_payload        TEXT NOT NULL,
    model_version           TEXT,
    feature_version         TEXT,
    config_hash             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_batch_status ON prediction_batch (status, created_at);

CREATE TABLE IF NOT EXISTS recommendation (
    recommendation_id  INTEGER PRIMARY KEY,
    prediction_id      INTEGER NOT NULL REFERENCES prediction,
    fixture_id         INTEGER NOT NULL REFERENCES fixture,
    evaluated_at       TEXT NOT NULL,
    passed_eligibility INTEGER NOT NULL,
    rejection_reasons  TEXT,
    reliability_score  REAL,
    ranking_score      REAL,
    confidence_band    TEXT,
    correlation_group  TEXT,
    selected           INTEGER NOT NULL DEFAULT 0,
    batch_id           INTEGER REFERENCES prediction_batch
);
CREATE INDEX IF NOT EXISTS ix_rec_eval ON recommendation (evaluated_at, selected);

CREATE TABLE IF NOT EXISTS notification_log (
    notification_id     INTEGER PRIMARY KEY,
    batch_id            INTEGER NOT NULL REFERENCES prediction_batch,
    channel             TEXT NOT NULL,
    provider            TEXT NOT NULL,
    recipient           TEXT NOT NULL,
    subject             TEXT NOT NULL,
    attempt             INTEGER NOT NULL DEFAULT 1,
    status              TEXT NOT NULL,
    provider_message_id TEXT,
    error               TEXT,
    sent_at             TEXT
);

-- The duplicate-prevention ledger, written inside the send transaction.
CREATE TABLE IF NOT EXISTS notified_selection (
    fixture_id       INTEGER NOT NULL REFERENCES fixture,
    market_key       TEXT NOT NULL,
    selection        TEXT NOT NULL,
    batch_id         INTEGER NOT NULL REFERENCES prediction_batch,
    prediction_id    INTEGER NOT NULL REFERENCES prediction,
    probability_sent REAL NOT NULL,
    sent_at          TEXT NOT NULL,
    PRIMARY KEY (fixture_id, market_key, selection, batch_id)
);
CREATE INDEX IF NOT EXISTS ix_notified_recent ON notified_selection (fixture_id, sent_at);

-- --------------------------------------------- validation and evaluation
CREATE TABLE IF NOT EXISTS market_outcome (
    fixture_id        INTEGER NOT NULL REFERENCES fixture,
    market_key        TEXT NOT NULL,
    selection         TEXT NOT NULL,
    outcome           TEXT NOT NULL,
    outcome_value     REAL,
    settled_at        TEXT NOT NULL,
    settlement_source TEXT NOT NULL,
    PRIMARY KEY (fixture_id, market_key, selection)
);

CREATE TABLE IF NOT EXISTS prediction_evaluation (
    prediction_id   INTEGER PRIMARY KEY REFERENCES prediction,
    outcome         TEXT NOT NULL,
    hit             INTEGER,
    log_loss        REAL,
    brier           REAL,
    was_recommended INTEGER NOT NULL DEFAULT 0,
    was_notified    INTEGER NOT NULL DEFAULT 0,
    evaluated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_metric (
    metric_id        INTEGER PRIMARY KEY,
    computed_at      TEXT NOT NULL,
    window_start     TEXT NOT NULL,
    window_end       TEXT NOT NULL,
    grain            TEXT NOT NULL,
    market_key       TEXT,
    competition_id   INTEGER,
    season_id        INTEGER,
    model_version    TEXT,
    confidence_bucket TEXT,
    n_samples        INTEGER NOT NULL,
    mean_predicted   REAL,
    actual_rate      REAL,
    log_loss         REAL,
    brier            REAL,
    ece              REAL,
    auc              REAL,
    mae              REAL,
    rmse             REAL,
    poisson_deviance REAL,
    ci_low           REAL,
    ci_high          REAL
);

-- ------------------------------------------------------------- monitoring
CREATE TABLE IF NOT EXISTS data_quality_check (
    check_id    INTEGER PRIMARY KEY,
    run_at      TEXT NOT NULL,
    check_name  TEXT NOT NULL,
    severity    TEXT NOT NULL,
    entity_kind TEXT,
    entity_id   INTEGER,
    detail      TEXT NOT NULL,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_dq_open ON data_quality_check (severity, resolved_at);

CREATE TABLE IF NOT EXISTS drift_check (
    drift_id       INTEGER PRIMARY KEY,
    run_at         TEXT NOT NULL,
    scope          TEXT NOT NULL,
    competition_id INTEGER,
    market_key     TEXT,
    model_version  TEXT,
    metric         TEXT NOT NULL,
    baseline_value REAL,
    current_value  REAL,
    n_samples      INTEGER NOT NULL,
    breach         INTEGER NOT NULL DEFAULT 0,
    detail         TEXT
);
"""
