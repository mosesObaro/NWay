"""End-to-end: features -> model -> markets -> calibration -> storage."""

from __future__ import annotations

import datetime as dt

import pytest

from nway import clock
from nway.features.compute import compute_features, completeness, persist_features
from nway.features.context import FeatureContext
from nway.markets import check_coherence, derive_markets
from nway.models.dixon_coles import DixonColesModel
from nway.prediction.pipeline import PredictionPipeline, refresh_stage
from nway.storage import repositories as repo


def _upcoming(seeded, hours_ahead=8):
    kickoff = seeded["base"] + dt.timedelta(hours=hours_ahead)
    fixture_id = seeded["add_upcoming"]("Charlie", "Alpha", kickoff)
    row = seeded["db"].query_one("""
        SELECT f.*, c.kind AS competition_kind FROM fixture f
        JOIN competition c ON c.competition_id = f.competition_id
        WHERE f.fixture_id = ?""", (fixture_id,))
    return dict(row)


def test_features_compute_and_persist(seeded):
    db = seeded["db"]
    fixture = _upcoming(seeded)
    as_of = seeded["base"]
    context = FeatureContext.build(db, fixture, as_of)
    values = compute_features(context)

    assert values, "no features were produced"
    assert completeness(values) > 0.5
    persist_features(db, fixture["fixture_id"], as_of, values,
                     context.repo.max_knowledge_time)
    stored = db.scalar(
        "SELECT COUNT(*) FROM feature_value WHERE fixture_id = ?",
        (fixture["fixture_id"],), default=0)
    assert stored == len(values)


def test_missing_history_is_null_with_a_reason_not_zero(seeded):
    """A promoted team with no history must not read as 'scores zero goals'."""
    db = seeded["db"]
    newcomer = repo.get_or_create_team(db, "Newcomer")
    fixture_id, _ = repo.upsert_fixture(
        db, competition_id=seeded["competition_id"], season_id=seeded["season_id"],
        home_team_id=newcomer, away_team_id=seeded["teams"]["Alpha"],
        kickoff_utc=clock.to_iso(seeded["base"] + dt.timedelta(hours=6)),
        status="TIMED", knowledge_time=clock.to_iso(seeded["base"]), source="test")
    row = dict(db.query_one("""
        SELECT f.*, c.kind AS competition_kind FROM fixture f
        JOIN competition c ON c.competition_id = f.competition_id
        WHERE f.fixture_id = ?""", (fixture_id,)))

    values = compute_features(FeatureContext.build(db, row, seeded["base"]))
    home_goals = values["home.gf_per_match.last5"]
    assert home_goals["value"] is None
    assert home_goals["is_null_reason"] == "INSUFFICIENT_HISTORY"


def test_full_prediction_run_persists_everything(seeded, config):
    db = seeded["db"]
    fixture = _upcoming(seeded)
    as_of = seeded["base"]

    history = db.query("""
        SELECT f.fixture_id, f.competition_id, f.season_id, f.kickoff_utc,
               f.home_team_id, f.away_team_id, r.home_goals, r.away_goals, r.outcome
        FROM match_result r JOIN fixture f ON f.fixture_id = r.fixture_id""")
    matches = [dict(row) for row in history]
    # The seeded league is tiny, so fit with a relaxed minimum.
    model = DixonColesModel(xi=0.003, l2=1.0)
    try:
        model.fit(matches, as_of=as_of)
    except ValueError:
        model.params.intercept[seeded["competition_id"]] = 0.3
        model.params.home_advantage[seeded["competition_id"]] = 0.25
        model.params.league_defaults[seeded["competition_id"]] = (1.4, 1.1)

    pipeline = PredictionPipeline(db, config, model, "test_model")
    result = pipeline.predict_fixture(fixture, as_of, snapshot_id=None)

    assert not result.was_skipped
    assert result.prediction_run_id is not None
    assert result.lambda_home > 0 and result.lambda_away > 0

    predictions = db.query(
        "SELECT market_key, calibrated_probability FROM prediction "
        "WHERE prediction_run_id = ?", (result.prediction_run_id,))
    assert len(predictions) >= 10
    by_market = {row["market_key"]: row["calibrated_probability"] for row in predictions}
    assert by_market["OVER_1_5"] >= by_market["OVER_2_5"]
    assert by_market["BTTS"] <= by_market["OVER_1_5"] + 1e-9

    explanations = db.scalar("""
        SELECT COUNT(*) FROM prediction_explanation e
        JOIN prediction p ON p.prediction_id = e.prediction_id
        WHERE p.prediction_run_id = ?""", (result.prediction_run_id,), default=0)
    assert explanations > 0, "no explanation rows were stored"


def test_refresh_appends_and_never_overwrites(seeded, config):
    db = seeded["db"]
    fixture = _upcoming(seeded, hours_ahead=40)
    model = DixonColesModel()
    model.params.intercept[seeded["competition_id"]] = 0.3
    model.params.home_advantage[seeded["competition_id"]] = 0.25
    model.params.league_defaults[seeded["competition_id"]] = (1.4, 1.1)
    pipeline = PredictionPipeline(db, config, model, "test_model")

    first = pipeline.predict_fixture(fixture, seeded["base"] - dt.timedelta(hours=8))
    second = pipeline.predict_fixture(fixture, seeded["base"])

    assert first.prediction_run_id != second.prediction_run_id
    superseded = db.scalar(
        "SELECT superseded_by FROM prediction_run WHERE prediction_run_id = ?",
        (first.prediction_run_id,))
    assert superseded == second.prediction_run_id
    # The original snapshot survives untouched.
    assert db.scalar(
        "SELECT COUNT(*) FROM prediction WHERE prediction_run_id = ?",
        (first.prediction_run_id,), default=0) > 0


def test_same_as_of_is_idempotent(seeded, config):
    db = seeded["db"]
    fixture = _upcoming(seeded)
    model = DixonColesModel()
    model.params.intercept[seeded["competition_id"]] = 0.3
    model.params.home_advantage[seeded["competition_id"]] = 0.25
    model.params.league_defaults[seeded["competition_id"]] = (1.4, 1.1)
    pipeline = PredictionPipeline(db, config, model, "test_model")

    first = pipeline.predict_fixture(fixture, seeded["base"])
    second = pipeline.predict_fixture(fixture, seeded["base"])
    assert first.prediction_run_id == second.prediction_run_id
    assert db.scalar("SELECT COUNT(*) FROM prediction_run", default=0) == 1


@pytest.mark.parametrize("hours,stage", [
    (1.0, "FINAL"), (6.0, "T8"), (20.0, "T24"), (40.0, "T48"), (100.0, "ADHOC"),
])
def test_refresh_stages(hours, stage):
    assert refresh_stage(hours) == stage
