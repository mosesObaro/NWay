"""Market settlement and evaluation."""

from __future__ import annotations

import datetime as dt

import pytest

from nway import clock
from nway.validation.settlement import settle_finished_fixtures


# The settlement truth table lives in tests/unit/test_new_markets.py, which
# exercises all 86 markets through MatchFacts. What is tested here is the
# database path: that a settled prediction records the right outcome, that a
# postponed fixture voids, and that missing statistics are UNSETTLEABLE rather
# than a loss.


def _predict(db, fixture_id, market, probability=0.8):
    db.insert("model_version", {
        "model_version": "m", "model_family": "DIXON_COLES", "target": "GOALS",
        "trained_at": "2025-12-01T00:00:00Z",
        "train_window_start": "2017-08-01T00:00:00Z",
        "train_window_end": "2025-12-01T00:00:00Z", "hyperparameters": "{}",
        "artifact_path": "x", "is_active": 1}, or_ignore=True)
    run_id = db.insert("prediction_run", {
        "fixture_id": fixture_id, "prediction_timestamp": "2026-01-01T10:00:00Z",
        "kickoff_utc": "2026-01-01T15:00:00Z", "hours_to_kickoff": 5.0,
        "refresh_stage": "T8", "model_version": "m", "feature_version": "f",
        "snapshot_id": None, "config_hash": "h", "lambda_home": 1.5,
        "lambda_away": 1.0, "data_completeness": 1.0,
        "feature_staleness_hours": 1.0, "created_at": "2026-01-01T10:00:00Z"})
    return db.insert("prediction", {
        "prediction_run_id": run_id, "fixture_id": fixture_id, "market_key": market,
        "selection": "YES", "raw_probability": probability,
        "calibrated_probability": probability, "created_at": "2026-01-01T10:00:00Z"})


def test_settles_and_records_metrics(seeded):
    db = seeded["db"]
    kickoff = seeded["base"] - dt.timedelta(hours=5)
    fixture_id = seeded["add_result"]("Charlie", "Echo", 2, 1, kickoff)
    prediction_id = _predict(db, fixture_id, "OVER_2_5", 0.75)

    summary = settle_finished_fixtures(db, now=seeded["base"])
    assert summary.predictions_settled == 1

    row = db.query_one(
        "SELECT outcome, hit, log_loss, brier FROM prediction_evaluation "
        "WHERE prediction_id = ?", (prediction_id,))
    assert row["outcome"] == "WIN"      # 3 goals is over 2.5
    assert row["hit"] == 1
    assert row["brier"] == pytest.approx((0.75 - 1) ** 2)


def test_postponed_fixture_voids_rather_than_losing(seeded):
    db = seeded["db"]
    kickoff = seeded["base"] - dt.timedelta(hours=5)
    fixture_id = seeded["add_upcoming"]("Delta", "Foxtrot", kickoff, status="POSTPONED")
    prediction_id = _predict(db, fixture_id, "OVER_2_5", 0.8)

    settle_finished_fixtures(db, now=seeded["base"])
    row = db.query_one(
        "SELECT outcome, hit FROM prediction_evaluation WHERE prediction_id = ?",
        (prediction_id,))
    assert row["outcome"] == "VOID"
    assert row["hit"] is None, "a voided prediction must not count as a loss"


def test_missing_result_is_unsettleable_not_a_loss(seeded):
    """A data failure must be visible as one, not scored as a wrong prediction."""
    db = seeded["db"]
    kickoff = seeded["base"] - dt.timedelta(hours=5)
    fixture_id = seeded["add_upcoming"]("Delta", "Charlie", kickoff, status="FINISHED")
    prediction_id = _predict(db, fixture_id, "OVER_2_5", 0.8)

    settle_finished_fixtures(db, now=seeded["base"])
    row = db.query_one(
        "SELECT outcome, hit FROM prediction_evaluation WHERE prediction_id = ?",
        (prediction_id,))
    assert row["outcome"] == "UNSETTLEABLE"
    assert row["hit"] is None


def test_settlement_is_not_repeated(seeded):
    db = seeded["db"]
    kickoff = seeded["base"] - dt.timedelta(hours=5)
    fixture_id = seeded["add_result"]("Echo", "Charlie", 1, 1, kickoff)
    _predict(db, fixture_id, "BTTS", 0.6)

    first = settle_finished_fixtures(db, now=seeded["base"])
    second = settle_finished_fixtures(db, now=seeded["base"])
    assert first.predictions_settled == 1
    assert second.predictions_settled == 0
