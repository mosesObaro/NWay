"""Temporal leakage suite — the most important tests in the project.

Enumerating leak paths only catches the ones already thought of. Poisoning the
future catches the rest: compute features at an as-of time, insert deliberately
absurd data *after* that time, recompute, and assert nothing changed. The test
does not need to know *how* a leak happened, only that it did.

The canary test is what stops this suite quietly decaying into one that passes
because it has stopped checking.
"""

from __future__ import annotations

import datetime as dt

import pytest

from nway import clock
from nway.features.compute import compute_features
from nway.features.context import AsOfRepository, AsOfViolation, FeatureContext
from nway.storage import repositories as repo


def _fixture_row(db, competition_id, season_id, teams, kickoff):
    fixture_id, _ = repo.upsert_fixture(
        db, competition_id=competition_id, season_id=season_id,
        home_team_id=teams["Alpha"], away_team_id=teams["Bravo"],
        kickoff_utc=clock.to_iso(kickoff), status="TIMED",
        knowledge_time=clock.to_iso(kickoff - dt.timedelta(days=20)), source="test")
    row = db.query_one("""
        SELECT f.*, c.kind AS competition_kind FROM fixture f
        JOIN competition c ON c.competition_id = f.competition_id
        WHERE f.fixture_id = ?""", (fixture_id,))
    return dict(row)


def test_future_poisoning_leaves_features_unchanged(seeded):
    """The general leak detector. Any change means something read past as_of."""
    db = seeded["db"]
    as_of = seeded["base"]
    kickoff = as_of + dt.timedelta(hours=6)
    fixture = _fixture_row(db, seeded["competition_id"], seeded["season_id"],
                           seeded["teams"], kickoff)

    before = compute_features(FeatureContext.build(db, fixture, as_of))

    # Absurd results, every one of them AFTER the cutoff, against opponents
    # reserved for this purpose so nothing collides with existing history.
    for offset in (1, 2, 3, 5, 8):
        seeded["add_result"]("Alpha", "Golf", 9, 0,
                             as_of + dt.timedelta(days=offset),
                             shots=(45, 1), corners=(30, 0))
        seeded["add_result"]("Bravo", "Hotel", 0, 9,
                             as_of + dt.timedelta(days=offset),
                             shots=(1, 45), corners=(0, 30))
        seeded["add_result"]("Golf", "Bravo", 8, 1,
                             as_of + dt.timedelta(days=offset),
                             shots=(40, 2), corners=(25, 1))

    after = compute_features(FeatureContext.build(db, fixture, as_of))
    assert before == after, "a feature changed when future data was injected"


def test_late_arriving_statistics_are_invisible(seeded):
    """A match PLAYED before as_of but only KNOWABLE afterwards must not count.

    This is the subtle one. football-data.co.uk lags up to ~3 days, so a
    Wednesday match can still be unknown on Saturday. Using event_time instead
    of knowledge_time would leak here and nowhere obvious.
    """
    db = seeded["db"]
    as_of = seeded["base"]
    kickoff = as_of + dt.timedelta(hours=6)
    fixture = _fixture_row(db, seeded["competition_id"], seeded["season_id"],
                           seeded["teams"], kickoff)

    before = compute_features(FeatureContext.build(db, fixture, as_of))

    seeded["add_result"](
        "Alpha", "Golf", 7, 0,
        kickoff=as_of - dt.timedelta(days=2),          # played BEFORE the cutoff
        knowledge=as_of + dt.timedelta(days=1),        # knowable AFTER it
        shots=(40, 2), corners=(20, 1))

    after = compute_features(FeatureContext.build(db, fixture, as_of))
    assert before == after, "a late-published result leaked into an earlier prediction"


def test_shuffled_future_does_not_change_features(seeded):
    """Catches ordering-dependent leaks that value-poisoning might miss."""
    db = seeded["db"]
    as_of = seeded["base"]
    kickoff = as_of + dt.timedelta(hours=6)
    fixture = _fixture_row(db, seeded["competition_id"], seeded["season_id"],
                           seeded["teams"], kickoff)
    before = compute_features(FeatureContext.build(db, fixture, as_of))

    db.execute(
        "UPDATE match_result SET home_goals = away_goals, away_goals = home_goals "
        "WHERE fixture_id IN (SELECT fixture_id FROM fixture WHERE kickoff_utc > ?)",
        (clock.to_iso(as_of),))

    after = compute_features(FeatureContext.build(db, fixture, as_of))
    assert before == after


def test_repository_refuses_rows_past_the_cutoff(seeded):
    """Belt and braces: the repository raises rather than silently filtering."""
    db = seeded["db"]
    as_of = seeded["base"]
    repository = AsOfRepository(db, as_of)
    repository.team_matches(seeded["teams"]["Alpha"])
    assert repository.max_knowledge_time <= clock.to_iso(as_of)

    with pytest.raises(AsOfViolation):
        repository._note(clock.to_iso(as_of + dt.timedelta(seconds=1)))


def test_source_max_knowledge_time_never_exceeds_as_of(seeded):
    """The database-level audit, independent of any code path."""
    db = seeded["db"]
    as_of = seeded["base"]
    kickoff = as_of + dt.timedelta(hours=6)
    fixture = _fixture_row(db, seeded["competition_id"], seeded["season_id"],
                           seeded["teams"], kickoff)

    context = FeatureContext.build(db, fixture, as_of)
    values = compute_features(context)
    from nway.features.compute import persist_features
    persist_features(db, fixture["fixture_id"], as_of, values,
                     context.repo.max_knowledge_time)

    violations = db.scalar(
        "SELECT COUNT(*) FROM feature_value WHERE source_max_knowledge_time > as_of",
        default=0)
    assert violations == 0


def test_check_constraint_rejects_a_leaking_row(db):
    """The schema itself refuses a leaked feature, even if code lets one through."""
    import sqlite3
    db.insert("feature_set_version", {
        "feature_version": "fs_test", "description": "test",
        "spec_hash": "test", "created_at": "2026-01-01T00:00:00Z"})
    competition_id = repo.upsert_competition(
        db, "x", "X", "DOMESTIC_LEAGUE", "ROUND_ROBIN", None, True)
    season_id = repo.upsert_season(db, competition_id, "2025/26",
                                   "2025-08-01", "2026-05-31", True)
    home = repo.get_or_create_team(db, "H")
    away = repo.get_or_create_team(db, "A")
    fixture_id, _ = repo.upsert_fixture(
        db, competition_id=competition_id, season_id=season_id,
        home_team_id=home, away_team_id=away,
        kickoff_utc="2026-02-01T15:00:00Z", status="TIMED",
        knowledge_time="2026-01-01T00:00:00Z", source="test")

    with pytest.raises(sqlite3.IntegrityError):
        db.insert("feature_value", {
            "fixture_id": fixture_id, "feature_version": "fs_test",
            "as_of": "2026-01-15T12:00:00Z", "feature_key": "leaky",
            "value": 1.0, "is_null_reason": None,
            "source_max_knowledge_time": "2026-01-20T12:00:00Z",  # after as_of
            "computed_at": "2026-01-15T12:00:00Z"})


def test_model_refuses_to_predict_before_its_training_window_ends():
    """A model trained through May cannot produce a March prediction.

    This is the easiest leak to introduce and the one that most flatters the
    results, so it is checked at load time rather than left to review.
    """
    from nway.models.base import LeakageError, ModelArtifact

    artifact = ModelArtifact(
        model_version="test", model_family="DIXON_COLES", target="GOALS",
        trained_at=dt.datetime(2026, 6, 1, tzinfo=clock.UTC),
        train_window_start=dt.datetime(2017, 8, 1, tzinfo=clock.UTC),
        train_window_end=dt.datetime(2026, 5, 31, tzinfo=clock.UTC),
        hyperparameters={}, parameters={})

    artifact.assert_usable_at(dt.datetime(2026, 6, 15, tzinfo=clock.UTC))
    with pytest.raises(LeakageError):
        artifact.assert_usable_at(dt.datetime(2026, 3, 14, tzinfo=clock.UTC))


def test_canary_a_deliberately_leaky_feature_is_caught(seeded):
    """The suite must catch a leak that is introduced on purpose.

    Without this, a refactor that quietly stopped the harness from checking
    anything would look like a green test run.
    """
    db = seeded["db"]
    as_of = seeded["base"]
    kickoff = as_of + dt.timedelta(hours=6)
    fixture = _fixture_row(db, seeded["competition_id"], seeded["season_id"],
                           seeded["teams"], kickoff)

    def leaky_feature() -> float:
        # Reads the raw table with no knowledge_time filter -- the classic bug.
        return db.scalar(
            "SELECT AVG(goals) FROM team_match_stats WHERE team_id = ?",
            (seeded["teams"]["Alpha"],), default=0.0) or 0.0

    before = leaky_feature()
    for offset in (1, 2, 3):
        seeded["add_result"]("Alpha", "Golf", 9, 0,
                             as_of + dt.timedelta(days=offset))
    after = leaky_feature()

    assert before != after, (
        "the canary did not leak; the poisoning fixture is no longer effective, "
        "which means the other tests in this file may be passing vacuously")
