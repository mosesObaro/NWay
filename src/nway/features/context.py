"""The as-of data access boundary.

Feature code never touches the database directly. It receives a
``FeatureContext`` and reads through ``AsOfRepository``, whose every method
requires an ``as_of`` and filters ``knowledge_time <= as_of``.

That is the whole trick: there is no unfiltered accessor within reach, so
leaking future information takes deliberate effort rather than inattention. The
``tests/leakage`` suite verifies the property directly by poisoning the
database with absurd post-``as_of`` rows and asserting that nothing changes.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Sequence

from nway import clock
from nway.storage.db import Database


class AsOfViolation(RuntimeError):
    """Raised when something tries to read past its own cutoff."""


@dataclass
class MatchRow:
    """One completed match, as known at some as-of time."""
    fixture_id: int
    competition_id: int
    season_id: int
    kickoff_utc: dt.datetime
    team_id: int
    opponent_id: int
    is_home: bool
    goals_for: int
    goals_against: int
    ht_goals_for: int | None
    ht_goals_against: int | None
    shots: int | None
    shots_on_target: int | None
    corners: int | None
    yellow_cards: int | None
    red_cards: int | None
    sxg_proxy: float | None
    knowledge_time: dt.datetime

    @property
    def points(self) -> int:
        if self.goals_for > self.goals_against:
            return 3
        return 1 if self.goals_for == self.goals_against else 0

    @property
    def clean_sheet(self) -> bool:
        return self.goals_against == 0

    @property
    def btts(self) -> bool:
        return self.goals_for > 0 and self.goals_against > 0


class AsOfRepository:
    """Point-in-time reads. Every method requires an explicit as_of."""

    def __init__(self, db: Database, as_of: dt.datetime) -> None:
        self.db = db
        self.as_of = clock.ensure_utc(as_of)
        self._as_of_iso = clock.to_iso(self.as_of)
        self._max_knowledge_seen: str = "0000-00-00T00:00:00Z"
        self._cache: dict[tuple, list[MatchRow]] = {}

    @property
    def max_knowledge_time(self) -> str:
        """Newest input actually used, stored alongside every feature value.

        If this ever exceeds as_of, the feature leaked -- and a CHECK
        constraint on feature_value rejects the row outright.
        """
        if self._max_knowledge_seen == "0000-00-00T00:00:00Z":
            return self._as_of_iso
        return self._max_knowledge_seen

    def _note(self, knowledge_time: str) -> None:
        if knowledge_time > self._as_of_iso:
            raise AsOfViolation(
                f"row with knowledge_time={knowledge_time} leaked past as_of={self._as_of_iso}")
        if knowledge_time > self._max_knowledge_seen:
            self._max_knowledge_seen = knowledge_time

    # -- match history ---------------------------------------------------
    def team_matches(self, team_id: int, *, competition_id: int | None = None,
                     season_id: int | None = None, venue: str = "ANY",
                     limit: int | None = None,
                     before: dt.datetime | None = None) -> list[MatchRow]:
        """Completed matches for a team that were knowable at as_of.

        Ordered most recent first. ``venue`` is ANY, HOME or AWAY.
        """
        cutoff = clock.to_iso(before) if before else self._as_of_iso
        key = (team_id, competition_id, season_id, venue, limit, cutoff)
        if key in self._cache:
            for row in self._cache[key]:
                self._note(clock.to_iso(row.knowledge_time))
            return self._cache[key]

        conditions = [
            "s.team_id = ?",
            # BOTH conditions matter. The knowledge filter is the leakage
            # guard; the kickoff filter stops a match that has been played but
            # not yet started relative to this fixture from counting.
            "s.knowledge_time <= ?",
            "r.knowledge_time <= ?",
            "f.kickoff_utc < ?",
            "f.status = 'FINISHED'",
        ]
        params: list[Any] = [team_id, self._as_of_iso, self._as_of_iso, cutoff]
        if competition_id is not None:
            conditions.append("f.competition_id = ?")
            params.append(competition_id)
        if season_id is not None:
            conditions.append("f.season_id = ?")
            params.append(season_id)
        if venue == "HOME":
            conditions.append("s.is_home = 1")
        elif venue == "AWAY":
            conditions.append("s.is_home = 0")

        sql = f"""
            SELECT f.fixture_id, f.competition_id, f.season_id, f.kickoff_utc,
                   s.team_id, s.is_home, s.goals, s.goals_conceded, s.ht_goals,
                   s.shots, s.shots_on_target, s.corners, s.yellow_cards,
                   s.red_cards, s.sxg_proxy, s.knowledge_time,
                   r.home_goals, r.away_goals, r.ht_home_goals, r.ht_away_goals,
                   f.home_team_id, f.away_team_id
            FROM team_match_stats s
            JOIN fixture f ON f.fixture_id = s.fixture_id
            JOIN match_result r ON r.fixture_id = s.fixture_id
            WHERE {' AND '.join(conditions)}
            ORDER BY f.kickoff_utc DESC
        """
        if limit:
            sql += f" LIMIT {int(limit)}"

        rows: list[MatchRow] = []
        for record in self.db.query(sql, params):
            self._note(record["knowledge_time"])
            is_home = bool(record["is_home"])
            ht_for = record["ht_home_goals"] if is_home else record["ht_away_goals"]
            ht_against = record["ht_away_goals"] if is_home else record["ht_home_goals"]
            rows.append(MatchRow(
                fixture_id=record["fixture_id"],
                competition_id=record["competition_id"],
                season_id=record["season_id"],
                kickoff_utc=clock.from_iso(record["kickoff_utc"]),
                team_id=record["team_id"],
                opponent_id=(record["away_team_id"] if is_home else record["home_team_id"]),
                is_home=is_home,
                goals_for=record["goals"] if record["goals"] is not None else (
                    record["home_goals"] if is_home else record["away_goals"]),
                goals_against=record["goals_conceded"] if record["goals_conceded"] is not None
                else (record["away_goals"] if is_home else record["home_goals"]),
                ht_goals_for=ht_for, ht_goals_against=ht_against,
                shots=record["shots"], shots_on_target=record["shots_on_target"],
                corners=record["corners"], yellow_cards=record["yellow_cards"],
                red_cards=record["red_cards"], sxg_proxy=record["sxg_proxy"],
                knowledge_time=clock.from_iso(record["knowledge_time"]),
            ))
        self._cache[key] = rows
        return rows

    def results_for_training(self, competition_ids: Sequence[int] | None = None,
                             since: dt.datetime | None = None) -> list[dict[str, Any]]:
        """Completed matches usable for fitting a model at this as_of."""
        conditions = ["r.knowledge_time <= ?", "f.kickoff_utc < ?", "f.status = 'FINISHED'"]
        params: list[Any] = [self._as_of_iso, self._as_of_iso]
        if since:
            conditions.append("f.kickoff_utc >= ?")
            params.append(clock.to_iso(since))
        if competition_ids:
            conditions.append(
                f"f.competition_id IN ({','.join('?' * len(competition_ids))})")
            params.extend(competition_ids)
        rows = self.db.query(f"""
            SELECT f.fixture_id, f.competition_id, f.season_id, f.kickoff_utc,
                   f.home_team_id, f.away_team_id, r.home_goals, r.away_goals,
                   r.ht_home_goals, r.ht_away_goals, r.outcome, r.knowledge_time
            FROM match_result r
            JOIN fixture f ON f.fixture_id = r.fixture_id
            WHERE {' AND '.join(conditions)}
            ORDER BY f.kickoff_utc
        """, params)
        for row in rows:
            self._note(row["knowledge_time"])
        return [dict(r) for r in rows]

    def last_match_before(self, team_id: int,
                          before: dt.datetime) -> MatchRow | None:
        matches = self.team_matches(team_id, limit=1, before=before)
        return matches[0] if matches else None

    def matches_in_period(self, team_id: int, start: dt.datetime,
                          end: dt.datetime) -> int:
        """Fixture congestion: how many matches in a window."""
        return self.db.scalar("""
            SELECT COUNT(*) FROM fixture f
            JOIN match_result r ON r.fixture_id = f.fixture_id
            WHERE (f.home_team_id = ? OR f.away_team_id = ?)
              AND f.kickoff_utc >= ? AND f.kickoff_utc < ?
              AND r.knowledge_time <= ?
        """, (team_id, team_id, clock.to_iso(start), clock.to_iso(end),
              self._as_of_iso), default=0)

    def competition_baseline(self, competition_id: int,
                             season_id: int | None = None) -> dict[str, float]:
        """League scoring baseline as known at as_of.

        Used as the reference value in explanations and as the shrinkage
        target for teams with thin history.
        """
        conditions = ["f.competition_id = ?", "r.knowledge_time <= ?",
                      "f.kickoff_utc < ?"]
        params: list[Any] = [competition_id, self._as_of_iso, self._as_of_iso]
        if season_id is not None:
            conditions.append("f.season_id = ?")
            params.append(season_id)
        row = self.db.query_one(f"""
            SELECT COUNT(*) AS n,
                   AVG(r.home_goals) AS home_goals,
                   AVG(r.away_goals) AS away_goals,
                   AVG(CASE WHEN r.outcome='H' THEN 1.0 ELSE 0.0 END) AS home_win_rate
            FROM match_result r JOIN fixture f ON f.fixture_id = r.fixture_id
            WHERE {' AND '.join(conditions)}
        """, params)
        if not row or not row["n"]:
            return {"n": 0.0, "home_goals": 1.5, "away_goals": 1.2, "home_win_rate": 0.44}
        return {
            "n": float(row["n"]),
            "home_goals": float(row["home_goals"] or 1.5),
            "away_goals": float(row["away_goals"] or 1.2),
            "home_win_rate": float(row["home_win_rate"] or 0.44),
        }


@dataclass
class FeatureContext:
    """Everything a feature function is allowed to see."""

    as_of: dt.datetime
    fixture: dict[str, Any]
    repo: AsOfRepository
    config: Any = None
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def home_team_id(self) -> int:
        return self.fixture["home_team_id"]

    @property
    def away_team_id(self) -> int:
        return self.fixture["away_team_id"]

    @property
    def kickoff(self) -> dt.datetime:
        return clock.from_iso(self.fixture["kickoff_utc"])

    @classmethod
    def build(cls, db: Database, fixture: dict[str, Any],
              as_of: dt.datetime, config: Any = None) -> "FeatureContext":
        return cls(as_of=clock.ensure_utc(as_of), fixture=fixture,
                   repo=AsOfRepository(db, as_of), config=config)
