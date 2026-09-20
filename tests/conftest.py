"""Shared fixtures.

Every test runs against an in-memory database built from the real schema, so
constraints (including the feature_value CHECK that enforces the as-of rule)
are exercised rather than mocked away.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nway import clock                                          # noqa: E402
from nway.config import load_config                             # noqa: E402
from nway.logging_setup import setup_logging                    # noqa: E402
from nway.storage import repositories as repo                   # noqa: E402
from nway.storage.db import Database                            # noqa: E402

setup_logging("ERROR")

BASE = dt.datetime(2026, 1, 1, 12, 0, tzinfo=clock.UTC)


@pytest.fixture
def db() -> Database:
    database = Database(":memory:")
    database.migrate()
    yield database
    database.close()


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def register_model(db):
    """prediction_run has a FOREIGN KEY to model_version, so a test that stores
    predictions must register the model first -- exactly as training does."""
    def _register(model_version: str = "test_model") -> str:
        db.insert("model_version", {
            "model_version": model_version, "model_family": "DIXON_COLES",
            "target": "GOALS", "trained_at": clock.to_iso(BASE - dt.timedelta(days=30)),
            "train_window_start": clock.to_iso(BASE - dt.timedelta(days=400)),
            "train_window_end": clock.to_iso(BASE - dt.timedelta(days=30)),
            "hyperparameters": "{}", "artifact_path": "test", "is_active": 1,
        }, or_ignore=True)
        return model_version
    return _register


@pytest.fixture
def seeded(db, register_model):
    register_model()
    """A small league: 6 teams, a round of completed matches, then upcoming ones."""
    competition_id = repo.upsert_competition(
        db, "test_league", "Test League", "DOMESTIC_LEAGUE", "ROUND_ROBIN", "XX", True)
    season_id = repo.upsert_season(
        db, competition_id, "2025/26", "2025-08-01", "2026-05-31", True)
    # Golf and Hotel are reserved for the leakage tests: they never appear in
    # the seeded pairings, so poisoning with them cannot collide with an
    # existing fixture and accidentally REMOVE history instead of adding it.
    teams = {name: repo.get_or_create_team(db, name)
             for name in ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot",
                          "Golf", "Hotel")}

    def add_result(home: str, away: str, home_goals: int, away_goals: int,
                   kickoff: dt.datetime, knowledge: dt.datetime | None = None,
                   shots: tuple[int, int] = (12, 9),
                   corners: tuple[int, int] = (6, 4)):
        knowledge = knowledge or (kickoff + dt.timedelta(hours=2))
        fixture_id, _ = repo.upsert_fixture(
            db, competition_id=competition_id, season_id=season_id,
            home_team_id=teams[home], away_team_id=teams[away],
            kickoff_utc=clock.to_iso(kickoff), status="FINISHED",
            knowledge_time=clock.to_iso(knowledge), source="test")
        outcome = ("H" if home_goals > away_goals
                   else "A" if away_goals > home_goals else "D")
        db.execute(
            "INSERT OR REPLACE INTO match_result (fixture_id, home_goals, away_goals,"
            " ht_home_goals, ht_away_goals, outcome, went_to_et, went_to_pens,"
            " result_source, event_time, knowledge_time) VALUES (?,?,?,?,?,?,0,0,?,?,?)",
            (fixture_id, home_goals, away_goals, home_goals // 2, away_goals // 2,
             outcome, "test", clock.to_iso(kickoff + dt.timedelta(hours=2)),
             clock.to_iso(knowledge)))
        for is_home, team, goals, conceded, shot, corner in (
                (1, home, home_goals, away_goals, shots[0], corners[0]),
                (0, away, away_goals, home_goals, shots[1], corners[1])):
            db.execute(
                "INSERT OR REPLACE INTO team_match_stats (fixture_id, team_id, is_home,"
                " goals, goals_conceded, ht_goals, shots, shots_on_target, corners,"
                " fouls, yellow_cards, red_cards, possession, xg, sxg_proxy, source,"
                " event_time, knowledge_time) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,"
                "NULL,NULL,?,?,?,?)",
                (fixture_id, teams[team], is_home, goals, conceded, goals // 2,
                 shot, shot // 3, corner, 11, 2, 0, round(0.3 * (shot // 3), 3),
                 "test", clock.to_iso(kickoff + dt.timedelta(hours=2)),
                 clock.to_iso(knowledge)))
        return fixture_id

    def add_upcoming(home: str, away: str, kickoff: dt.datetime,
                     status: str = "TIMED") -> int:
        fixture_id, _ = repo.upsert_fixture(
            db, competition_id=competition_id, season_id=season_id,
            home_team_id=teams[home], away_team_id=teams[away],
            kickoff_utc=clock.to_iso(kickoff), status=status,
            knowledge_time=clock.to_iso(BASE - dt.timedelta(days=10)), source="test")
        return fixture_id

    # Twelve completed matches over the preceding twelve weeks.
    pairs = [("Alpha", "Bravo"), ("Charlie", "Delta"), ("Echo", "Foxtrot"),
             ("Bravo", "Charlie"), ("Delta", "Echo"), ("Foxtrot", "Alpha"),
             ("Alpha", "Charlie"), ("Bravo", "Delta"), ("Echo", "Alpha"),
             ("Charlie", "Foxtrot"), ("Delta", "Alpha"), ("Bravo", "Echo")]
    for index, (home, away) in enumerate(pairs):
        add_result(home, away, (index % 4), (index % 3),
                   BASE - dt.timedelta(days=84 - index * 7))

    return {
        "db": db, "competition_id": competition_id, "season_id": season_id,
        "teams": teams, "add_result": add_result, "add_upcoming": add_upcoming,
        "base": BASE,
    }
