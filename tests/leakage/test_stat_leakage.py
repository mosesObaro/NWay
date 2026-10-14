"""Leakage suite for the statistics the new markets depend on.

Corners, cards and half-time scores arrive through a different route from
goals: football-data.co.uk publishes them roughly twice a week, so they are
routinely knowable days after the match was played. Poisoning them separately
matters because a bug that leaked them would be invisible to a goals-only test.

The canary at the end asserts the poisoning fixture still bites, so this file
cannot pass vacuously after a refactor.
"""

from __future__ import annotations

import datetime as dt

import pytest

from nway import clock
from nway.features.compute import compute_features
from nway.features.context import AsOfRepository, FeatureContext
from nway.storage import repositories as repo

POISON_COLUMNS = ("corners", "yellow_cards", "red_cards", "shots",
                  "shots_on_target", "fouls")


def _fixture_row(db, seeded, kickoff):
    fixture_id, _ = repo.upsert_fixture(
        db, competition_id=seeded["competition_id"], season_id=seeded["season_id"],
        home_team_id=seeded["teams"]["Alpha"], away_team_id=seeded["teams"]["Bravo"],
        kickoff_utc=clock.to_iso(kickoff), status="TIMED",
        knowledge_time=clock.to_iso(kickoff - dt.timedelta(days=20)), source="test")
    row = db.query_one("""
        SELECT f.*, c.kind AS competition_kind, c.slug AS competition_slug
        FROM fixture f JOIN competition c ON c.competition_id = f.competition_id
        WHERE f.fixture_id = ?""", (fixture_id,))
    return dict(row)


def test_future_corner_and_card_columns_do_not_leak(seeded):
    """Poison HC, AC, HY, AY, HR, AR after as_of and assert zero drift."""
    db = seeded["db"]
    as_of = seeded["base"]
    fixture = _fixture_row(db, seeded, as_of + dt.timedelta(hours=6))

    before = compute_features(FeatureContext.build(db, fixture, as_of))

    for offset in (1, 2, 4, 7):
        seeded["add_result"]("Alpha", "Golf", 9, 0,
                             as_of + dt.timedelta(days=offset),
                             shots=(60, 1), corners=(40, 0))
        seeded["add_result"]("Golf", "Bravo", 0, 9,
                             as_of + dt.timedelta(days=offset),
                             shots=(1, 60), corners=(0, 40))
    # Absurd card counts on the same future fixtures.
    db.execute(
        "UPDATE team_match_stats SET yellow_cards = 11, red_cards = 4 "
        "WHERE fixture_id IN (SELECT fixture_id FROM fixture WHERE kickoff_utc > ?)",
        (clock.to_iso(as_of),))

    after = compute_features(FeatureContext.build(db, fixture, as_of))
    assert before == after, "a corner or card feature changed on future data"


def test_future_half_time_columns_do_not_leak(seeded):
    """HTHG and HTAG feed the half-share model and the HT features."""
    db = seeded["db"]
    as_of = seeded["base"]
    fixture = _fixture_row(db, seeded, as_of + dt.timedelta(hours=6))

    before = compute_features(FeatureContext.build(db, fixture, as_of))
    db.execute(
        "UPDATE match_result SET ht_home_goals = 7, ht_away_goals = 0 "
        "WHERE fixture_id IN (SELECT fixture_id FROM fixture WHERE kickoff_utc > ?)",
        (clock.to_iso(as_of),))
    after = compute_features(FeatureContext.build(db, fixture, as_of))
    assert before == after, "a half-time feature changed on future data"


def test_late_published_statistics_are_invisible(seeded):
    """Played before as_of, publishable only after it.

    The subtle case, and the realistic one: football-data.co.uk lags up to
    three days, so a Wednesday match's corner counts can still be unknown on
    Saturday. Filtering on kickoff instead of knowledge_time would leak here
    and nowhere obvious.
    """
    db = seeded["db"]
    as_of = seeded["base"]
    fixture = _fixture_row(db, seeded, as_of + dt.timedelta(hours=6))

    before = compute_features(FeatureContext.build(db, fixture, as_of))
    seeded["add_result"](
        "Alpha", "Hotel", 5, 0,
        kickoff=as_of - dt.timedelta(days=2),          # played BEFORE
        knowledge=as_of + dt.timedelta(days=2),        # knowable AFTER
        shots=(50, 1), corners=(35, 0))
    after = compute_features(FeatureContext.build(db, fixture, as_of))
    assert before == after, "late-published statistics leaked into an earlier as_of"


def test_training_rows_respect_the_statistics_cutoff(seeded):
    """The loader the corner and card models train on must filter stats too."""
    db = seeded["db"]
    as_of = seeded["base"]

    rows = AsOfRepository(db, as_of).training_rows()
    baseline_with_corners = sum(1 for r in rows if r["home_corners"] is not None)

    seeded["add_result"](
        "Charlie", "Golf", 3, 0,
        kickoff=as_of - dt.timedelta(days=1),
        knowledge=as_of + dt.timedelta(days=5),        # stats arrive later
        shots=(30, 2), corners=(20, 1))

    rows_after = AsOfRepository(db, as_of).training_rows()
    assert sum(1 for r in rows_after if r["home_corners"] is not None) \
        == baseline_with_corners


def test_training_rows_keep_goals_when_only_statistics_are_late(seeded):
    """A match with knowable goals but unpublished corners must still train the
    goal model -- dropping it would shrink the goal corpus for no reason."""
    db = seeded["db"]
    as_of = seeded["base"]
    kickoff = as_of - dt.timedelta(days=1)

    fixture_id, _ = repo.upsert_fixture(
        db, competition_id=seeded["competition_id"], season_id=seeded["season_id"],
        home_team_id=seeded["teams"]["Delta"], away_team_id=seeded["teams"]["Hotel"],
        kickoff_utc=clock.to_iso(kickoff), status="FINISHED",
        knowledge_time=clock.to_iso(kickoff), source="test")
    db.execute(
        "INSERT INTO match_result (fixture_id, home_goals, away_goals, outcome, "
        "went_to_et, went_to_pens, result_source, event_time, knowledge_time) "
        "VALUES (?,?,?,?,0,0,?,?,?)",
        (fixture_id, 2, 1, "H", "test",
         clock.to_iso(kickoff + dt.timedelta(hours=2)),
         clock.to_iso(kickoff + dt.timedelta(hours=2))))

    rows = AsOfRepository(db, as_of).training_rows()
    match = [r for r in rows if r["fixture_id"] == fixture_id]
    assert len(match) == 1, "a goal-only match was dropped from the goal corpus"
    assert match[0]["home_corners"] is None
    assert match[0]["total_cards"] is None


@pytest.mark.parametrize("column", POISON_COLUMNS)
def test_canary_each_poisoned_column_would_be_detectable(seeded, column):
    """Each column this suite poisons must actually be readable unfiltered.

    Without this, a column that stopped being used -- or a fixture that stopped
    writing it -- would make the corresponding leak test pass for the wrong
    reason.
    """
    db = seeded["db"]
    as_of = seeded["base"]

    def unfiltered() -> float:
        return db.scalar(
            f"SELECT AVG({column}) FROM team_match_stats WHERE team_id = ?",
            (seeded["teams"]["Alpha"],), default=0.0) or 0.0

    before = unfiltered()
    for offset in (1, 2, 3):
        seeded["add_result"]("Alpha", "Golf", 9, 0,
                             as_of + dt.timedelta(days=offset),
                             shots=(55, 1), corners=(33, 0))
    db.execute(
        "UPDATE team_match_stats SET yellow_cards = 10, red_cards = 3, fouls = 44 "
        "WHERE fixture_id IN (SELECT fixture_id FROM fixture WHERE kickoff_utc > ?)",
        (clock.to_iso(as_of),))

    assert unfiltered() != before, (
        f"poisoning did not move {column}; the leak tests for it may be "
        f"passing vacuously")
