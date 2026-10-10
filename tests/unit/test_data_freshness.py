"""Freshness is publication lag, not data age.

Those differ whenever football pauses. After an international break the newest
statistics are legitimately weeks old because nothing was played, and a gate on
absolute age refuses every first matchday back -- which is what happened on
2026-10-10: 705 candidates, none eligible, with the source fully caught up.
"""

from __future__ import annotations

import datetime as dt

import pytest

from nway import clock
from nway.features.context import AsOfRepository

NOW = dt.datetime(2026, 10, 10, 9, 0, tzinfo=clock.UTC)


def test_no_lag_when_the_source_has_everything_played(seeded):
    """Three weeks since the last match, but we hold its statistics."""
    db = seeded["db"]
    last_played = NOW - dt.timedelta(days=20)
    seeded["add_result"]("Alpha", "Golf", 2, 1, last_played,
                         knowledge=last_played + dt.timedelta(hours=2))

    lag = AsOfRepository(db, NOW).data_lag_hours(seeded["teams"]["Alpha"])
    assert lag == pytest.approx(0.0, abs=1.0), (
        "a quiet calendar was reported as stale data")


def test_lag_appears_when_the_source_falls_behind(seeded):
    """A match played and knowable, but with no statistics attached."""
    db = seeded["db"]
    from nway.storage import repositories as repo

    recent = NOW - dt.timedelta(days=2)
    fixture_id, _ = repo.upsert_fixture(
        db, competition_id=seeded["competition_id"], season_id=seeded["season_id"],
        home_team_id=seeded["teams"]["Alpha"], away_team_id=seeded["teams"]["Hotel"],
        kickoff_utc=clock.to_iso(recent), status="FINISHED",
        knowledge_time=clock.to_iso(recent), source="test")
    db.execute(
        "INSERT INTO match_result (fixture_id, home_goals, away_goals, outcome, "
        "went_to_et, went_to_pens, result_source, event_time, knowledge_time) "
        "VALUES (?,?,?,?,0,0,?,?,?)",
        (fixture_id, 1, 0, "H", "test",
         clock.to_iso(recent + dt.timedelta(hours=2)),
         clock.to_iso(recent + dt.timedelta(hours=2))))

    lag = AsOfRepository(db, NOW).data_lag_hours(seeded["teams"]["Alpha"])
    assert lag > 100, "the source is behind by weeks and the gate did not see it"


def test_a_team_with_no_statistics_reports_no_lag(seeded):
    """Completeness owns that case; double-counting it as staleness would hide
    the real reason a fixture was refused."""
    db = seeded["db"]
    from nway.storage import repositories as repo

    newcomer = repo.get_or_create_team(db, "Brand New")
    assert AsOfRepository(db, NOW).data_lag_hours(newcomer) == 0.0


def test_lag_never_goes_negative(seeded):
    db = seeded["db"]
    seeded["add_result"]("Bravo", "Golf", 1, 1, NOW - dt.timedelta(days=5))
    assert AsOfRepository(db, NOW).data_lag_hours(seeded["teams"]["Bravo"]) >= 0.0
