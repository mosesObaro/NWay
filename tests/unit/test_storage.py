"""Storage-layer behaviour that is easy to get subtly wrong."""

from __future__ import annotations

import datetime as dt

from nway import clock
from nway.storage import repositories as repo


def test_ignored_insert_returns_zero_not_a_stale_id(db):
    """sqlite keeps lastrowid across an ignored INSERT OR IGNORE.

    Returning it would make callers attach child rows to an unrelated parent --
    the failure mode that the idempotency test surfaced.
    """
    first = repo.get_or_create_team(db, "Alpha")
    assert first > 0
    db.insert("venue", {"name": "Somewhere"})          # bump lastrowid
    ignored = db.insert("team", {
        "canonical_name": "Alpha", "created_at": clock.to_iso(clock.now())},
        or_ignore=True)
    assert ignored == 0


def test_upsert_fixture_treats_a_reschedule_as_the_same_match(db):
    competition_id = repo.upsert_competition(
        db, "l", "L", "DOMESTIC_LEAGUE", "ROUND_ROBIN", None, True)
    season_id = repo.upsert_season(db, competition_id, "2025/26",
                                   "2025-08-01", "2026-05-31", True)
    home = repo.get_or_create_team(db, "H")
    away = repo.get_or_create_team(db, "A")
    original = dt.datetime(2026, 2, 1, 15, 0, tzinfo=clock.UTC)

    first, _ = repo.upsert_fixture(
        db, competition_id=competition_id, season_id=season_id,
        home_team_id=home, away_team_id=away, kickoff_utc=clock.to_iso(original),
        status="TIMED", knowledge_time=clock.to_iso(original), source="t")
    # Postponed by ten days: the same match, not a new one.
    second, changes = repo.upsert_fixture(
        db, competition_id=competition_id, season_id=season_id,
        home_team_id=home, away_team_id=away,
        kickoff_utc=clock.to_iso(original + dt.timedelta(days=10)),
        status="TIMED", knowledge_time=clock.to_iso(original), source="t")

    assert first == second
    assert "kickoff_utc" in changes
    assert db.scalar("SELECT COUNT(*) FROM fixture", default=0) == 1
    assert db.scalar(
        "SELECT COUNT(*) FROM fixture_revision WHERE field='kickoff_utc'",
        default=0) == 1


def test_upsert_fixture_keeps_distant_meetings_separate(db):
    """Two meetings months apart are different matches, not a reschedule."""
    competition_id = repo.upsert_competition(
        db, "l", "L", "DOMESTIC_LEAGUE", "ROUND_ROBIN", None, True)
    season_id = repo.upsert_season(db, competition_id, "2025/26",
                                   "2025-08-01", "2026-05-31", True)
    home = repo.get_or_create_team(db, "H")
    away = repo.get_or_create_team(db, "A")

    first, _ = repo.upsert_fixture(
        db, competition_id=competition_id, season_id=season_id,
        home_team_id=home, away_team_id=away,
        kickoff_utc="2025-09-01T15:00:00Z", status="FINISHED",
        knowledge_time="2025-09-01T17:00:00Z", source="t")
    second, _ = repo.upsert_fixture(
        db, competition_id=competition_id, season_id=season_id,
        home_team_id=home, away_team_id=away,
        kickoff_utc="2026-03-01T15:00:00Z", status="TIMED",
        knowledge_time="2026-02-01T00:00:00Z", source="t")

    assert first != second
    assert db.scalar("SELECT COUNT(*) FROM fixture", default=0) == 2


def test_rolling_window_query_spans_competitions(db):
    """One query, every enabled competition, ordered by kickoff."""
    now = dt.datetime(2026, 9, 20, 10, 0, tzinfo=clock.UTC)
    for index, slug in enumerate(("premier_league", "la_liga", "champions_league")):
        competition_id = repo.upsert_competition(
            db, slug, slug.title(), "DOMESTIC_LEAGUE", "ROUND_ROBIN", None, True)
        season_id = repo.upsert_season(db, competition_id, "2026/27",
                                       "2026-08-01", "2027-05-31", True)
        home = repo.get_or_create_team(db, f"H{index}")
        away = repo.get_or_create_team(db, f"A{index}")
        repo.upsert_fixture(
            db, competition_id=competition_id, season_id=season_id,
            home_team_id=home, away_team_id=away,
            kickoff_utc=clock.to_iso(now + dt.timedelta(hours=10 + index * 5)),
            status="TIMED", knowledge_time=clock.to_iso(now), source="t")

    rows = repo.fixtures_in_window(db, now, now + dt.timedelta(hours=72))
    assert len(rows) == 3
    slugs = [row["competition_slug"] for row in rows]
    assert len(set(slugs)) == 3
    kickoffs = [row["kickoff_utc"] for row in rows]
    assert kickoffs == sorted(kickoffs)


def test_rolling_window_excludes_postponed_fixtures(db):
    now = dt.datetime(2026, 9, 20, 10, 0, tzinfo=clock.UTC)
    competition_id = repo.upsert_competition(
        db, "pl", "PL", "DOMESTIC_LEAGUE", "ROUND_ROBIN", None, True)
    season_id = repo.upsert_season(db, competition_id, "2026/27",
                                   "2026-08-01", "2027-05-31", True)
    home = repo.get_or_create_team(db, "H")
    away = repo.get_or_create_team(db, "A")
    repo.upsert_fixture(
        db, competition_id=competition_id, season_id=season_id,
        home_team_id=home, away_team_id=away,
        kickoff_utc=clock.to_iso(now + dt.timedelta(hours=10)),
        status="POSTPONED", knowledge_time=clock.to_iso(now), source="t")
    assert repo.fixtures_in_window(db, now, now + dt.timedelta(hours=72)) == []
