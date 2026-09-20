"""Repository helpers used across the pipeline.

These are the *unfiltered* accessors: they are used by ingestion, settlement
and reporting, which legitimately look at the present. Feature code must not
use them -- it goes through ``features.context.AsOfRepository``, which refuses
to run without an ``as_of``.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, Sequence

from nway import clock
from nway.storage.db import Database


# ------------------------------------------------------------- entities
def upsert_competition(db: Database, slug: str, name: str, kind: str,
                       structure: str, country: str | None, enabled: bool) -> int:
    row = db.query_one("SELECT competition_id FROM competition WHERE slug = ?", (slug,))
    if row:
        db.execute(
            "UPDATE competition SET name=?, kind=?, structure=?, country=?, enabled=? "
            "WHERE competition_id=?",
            (name, kind, structure, country, int(enabled), row["competition_id"]),
        )
        return row["competition_id"]
    return db.insert("competition", {
        "slug": slug, "name": name, "kind": kind, "structure": structure,
        "country": country, "enabled": int(enabled),
        "created_at": clock.to_iso(clock.now()),
    })


def upsert_season(db: Database, competition_id: int, label: str,
                  start_date: str, end_date: str, is_current: bool) -> int:
    row = db.query_one(
        "SELECT season_id FROM season WHERE competition_id=? AND label=?",
        (competition_id, label))
    if row:
        db.execute(
            "UPDATE season SET start_date=?, end_date=?, is_current=? WHERE season_id=?",
            (start_date, end_date, int(is_current), row["season_id"]))
        return row["season_id"]
    return db.insert("season", {
        "competition_id": competition_id, "label": label, "start_date": start_date,
        "end_date": end_date, "is_current": int(is_current),
    })


def get_or_create_team(db: Database, canonical_name: str,
                       country: str | None = None) -> int:
    row = db.query_one("SELECT team_id FROM team WHERE canonical_name = ?",
                       (canonical_name,))
    if row:
        return row["team_id"]
    return db.insert("team", {
        "canonical_name": canonical_name, "country": country,
        "created_at": clock.to_iso(clock.now()),
    })


def get_or_create_referee(db: Database, name: str) -> int:
    row = db.query_one("SELECT referee_id FROM referee WHERE canonical_name = ?", (name,))
    if row:
        return row["referee_id"]
    return db.insert("referee", {"canonical_name": name})


# ------------------------------------------------------------- fixtures
def upsert_fixture(db: Database, *, competition_id: int, season_id: int,
                   home_team_id: int, away_team_id: int, kickoff_utc: str,
                   status: str, knowledge_time: str, matchday: int | None = None,
                   stage: str | None = None, leg: int | None = None,
                   tie_id: str | None = None, referee_id: int | None = None,
                   kickoff_is_confirmed: bool = True,
                   source: str = "unknown",
                   rematch_window_days: int = 45) -> tuple[int, list[str]]:
    """Insert or update a fixture, recording every field change as a revision.

    Matching is by (competition, season, home, away) AND a kickoff within
    ``rematch_window_days``. Matching on the pairing alone would merge two
    genuinely distinct meetings -- competitions where sides meet twice at the
    same ground in a season, and two-legged ties replayed at one venue -- while
    matching on the exact kickoff would duplicate every rescheduled match.
    Postponements move a fixture by days; separate meetings are months apart,
    so the window separates them cleanly.
    """
    now = clock.to_iso(clock.now())
    window = dt.timedelta(days=rematch_window_days)
    target = clock.from_iso(kickoff_utc)
    existing = db.query_one(
        "SELECT * FROM fixture WHERE competition_id=? AND season_id=? "
        "AND home_team_id=? AND away_team_id=? "
        "AND kickoff_utc >= ? AND kickoff_utc <= ? "
        "ORDER BY ABS(julianday(kickoff_utc) - julianday(?)) LIMIT 1",
        (competition_id, season_id, home_team_id, away_team_id,
         clock.to_iso(target - window), clock.to_iso(target + window),
         kickoff_utc))

    if existing is None:
        fixture_id = db.insert("fixture", {
            "competition_id": competition_id, "season_id": season_id,
            "home_team_id": home_team_id, "away_team_id": away_team_id,
            "kickoff_utc": kickoff_utc, "status": status, "matchday": matchday,
            "stage": stage, "leg": leg, "tie_id": tie_id, "referee_id": referee_id,
            "kickoff_is_confirmed": int(kickoff_is_confirmed),
            "knowledge_time": knowledge_time, "updated_at": now,
        })
        return fixture_id, ["created"]

    fixture_id = existing["fixture_id"]
    changes: list[str] = []
    candidates = {
        "kickoff_utc": kickoff_utc, "status": status, "matchday": matchday,
        "stage": stage, "leg": leg, "referee_id": referee_id,
        "kickoff_is_confirmed": int(kickoff_is_confirmed),
    }
    updates: dict[str, Any] = {}
    for field, new_value in candidates.items():
        if new_value is None:
            continue
        if existing[field] != new_value:
            db.insert("fixture_revision", {
                "fixture_id": fixture_id, "field": field,
                "old_value": str(existing[field]), "new_value": str(new_value),
                "observed_at": now, "source": source,
            })
            updates[field] = new_value
            changes.append(field)
    if updates:
        assignments = ", ".join(f"{k}=?" for k in updates)
        db.execute(f"UPDATE fixture SET {assignments}, updated_at=? WHERE fixture_id=?",
                   (*updates.values(), now, fixture_id))
    return fixture_id, changes


def fixtures_in_window(db: Database, start: dt.datetime, end: dt.datetime,
                       competition_ids: Sequence[int] | None = None,
                       statuses: Sequence[str] = ("SCHEDULED", "TIMED")) -> list[Any]:
    """The rolling-window query.

    One query across every enabled competition, ordered by kickoff. There is no
    per-competition scheduling and no gameweek grouping: a batch spans whatever
    happens to fall inside the window.
    """
    params: list[Any] = [clock.to_iso(start), clock.to_iso(end)]
    sql = ["""
        SELECT f.*, c.slug AS competition_slug, c.name AS competition_name,
               c.kind AS competition_kind,
               h.canonical_name AS home_team, a.canonical_name AS away_team
        FROM fixture f
        JOIN competition c ON c.competition_id = f.competition_id
        JOIN team h ON h.team_id = f.home_team_id
        JOIN team a ON a.team_id = f.away_team_id
        WHERE f.kickoff_utc > ? AND f.kickoff_utc <= ?
    """]
    if statuses:
        sql.append(f"AND f.status IN ({','.join('?' * len(statuses))})")
        params.extend(statuses)
    if competition_ids:
        sql.append(f"AND f.competition_id IN ({','.join('?' * len(competition_ids))})")
        params.extend(competition_ids)
    sql.append("ORDER BY f.kickoff_utc, f.fixture_id")
    return db.query(" ".join(sql), params)


def record_quality_check(db: Database, name: str, severity: str, detail: str,
                         entity_kind: str | None = None,
                         entity_id: int | None = None) -> int:
    return db.insert("data_quality_check", {
        "run_at": clock.to_iso(clock.now()), "check_name": name,
        "severity": severity, "entity_kind": entity_kind, "entity_id": entity_id,
        "detail": detail,
    })


def create_snapshot(db: Database, as_of: dt.datetime) -> int:
    """Record the data watermark at prediction time.

    Storing the newest knowledge_time per source is what lets a stored
    prediction be explained later without re-running anything.
    """
    watermarks = {
        "fixture": db.scalar("SELECT MAX(knowledge_time) FROM fixture", default=""),
        "match_result": db.scalar("SELECT MAX(knowledge_time) FROM match_result", default=""),
        "team_match_stats": db.scalar(
            "SELECT MAX(knowledge_time) FROM team_match_stats", default=""),
    }
    counts = {
        "fixtures": db.scalar("SELECT COUNT(*) FROM fixture", default=0),
        "results": db.scalar("SELECT COUNT(*) FROM match_result", default=0),
        "team_stats": db.scalar("SELECT COUNT(*) FROM team_match_stats", default=0),
    }
    return db.insert("data_snapshot", {
        "as_of": clock.to_iso(as_of),
        "source_watermarks": json.dumps(watermarks),
        "row_counts": json.dumps(counts),
        "created_at": clock.to_iso(clock.now()),
    })
