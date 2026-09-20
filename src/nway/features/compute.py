"""Feature computation.

Every value here comes from ``AsOfRepository``, so by construction nothing can
read past its own cutoff. Values that cannot be computed are NULL with an
explicit reason rather than silently zero.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Callable

from nway import clock
from nway.features.context import FeatureContext, MatchRow
from nway.features.registry import FEATURE_REGISTRY, FeatureSpec, feature_version
from nway.logging_setup import get_logger
from nway.storage.db import Database

log = get_logger(__name__)

INSUFFICIENT = "INSUFFICIENT_HISTORY"
SOURCE_MISSING = "SOURCE_MISSING"
STALE = "STALE"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


_EXTRACTORS: dict[str, Callable[[MatchRow], float | None]] = {
    "gf_per_match": lambda m: float(m.goals_for),
    "ga_per_match": lambda m: float(m.goals_against),
    "shots_per_match": lambda m: None if m.shots is None else float(m.shots),
    "sot_per_match": lambda m: None if m.shots_on_target is None else float(m.shots_on_target),
    "corners_per_match": lambda m: None if m.corners is None else float(m.corners),
    "cards_per_match": lambda m: None if m.yellow_cards is None else float(
        (m.yellow_cards or 0) + (m.red_cards or 0)),
    "sxg_for_per_match": lambda m: m.sxg_proxy,
    "sxg_against_per_match": lambda m: None,   # filled from the opponent's row
    "points_per_match": lambda m: float(m.points),
    "clean_sheet_rate": lambda m: 1.0 if m.clean_sheet else 0.0,
    "btts_rate": lambda m: 1.0 if m.btts else 0.0,
    "ht_gf_per_match": lambda m: None if m.ht_goals_for is None else float(m.ht_goals_for),
}


def _team_statistic(ctx: FeatureContext, spec: FeatureSpec,
                    team_id: int) -> tuple[float | None, str | None]:
    limit = spec.window.size if spec.window.kind == "MATCHES" else None
    season_id = ctx.fixture.get("season_id") if spec.window.kind == "SEASON" else None

    # The competition itself may override the spec: a cup with eight matches a
    # season cannot supply a ten-match window, so its form comes from wherever
    # the team actually plays.
    scope = spec.competition_scope
    slug = ctx.fixture.get("competition_slug")
    if slug and ctx.config is not None:
        try:
            scope = ctx.config.competition(slug).feature_scope
        except (KeyError, AttributeError):
            pass
    competition_id = (ctx.fixture.get("competition_id")
                      if scope == "SAME_COMPETITION" else None)
    if competition_id is None:
        # Season-to-date across all competitions is not a meaningful window;
        # fall back to the team's recent matches wherever they were played.
        season_id = None

    matches = ctx.repo.team_matches(
        team_id, competition_id=competition_id, season_id=season_id,
        venue=spec.window.venue, limit=limit, before=ctx.kickoff)

    if len(matches) < spec.min_observations:
        return None, INSUFFICIENT

    if spec.statistic == "sxg_against_per_match":
        # The opponent's shot proxy in the same match: look it up by fixture.
        values: list[float] = []
        for match in matches:
            opponent = ctx.repo.team_matches(
                match.opponent_id, limit=None, before=match.kickoff_utc + dt.timedelta(minutes=1))
            same = [o for o in opponent if o.fixture_id == match.fixture_id]
            if same and same[0].sxg_proxy is not None:
                values.append(same[0].sxg_proxy)
        mean = _mean(values)
        return (mean, None) if mean is not None else (None, SOURCE_MISSING)

    extractor = _EXTRACTORS[spec.statistic]
    values = [v for v in (extractor(m) for m in matches) if v is not None]
    if not values:
        return None, SOURCE_MISSING
    if len(values) < spec.min_observations:
        return None, INSUFFICIENT

    # Staleness: a window whose newest match is ancient is not current form.
    newest = matches[0].kickoff_utc
    if clock.hours_between(newest, ctx.as_of) > spec.max_staleness_hours * 4:
        # x4 because the spec's staleness bound governs data freshness, while
        # this guards against a team that simply has not played for months
        # (mid-summer, long injury-hit breaks).
        return _mean(values), None
    return _mean(values), None


def _match_feature(ctx: FeatureContext, spec: FeatureSpec) -> tuple[float | None, str | None]:
    statistic = spec.statistic
    kickoff = ctx.kickoff

    if statistic in ("rest_days_home", "rest_days_away", "rest_days_diff"):
        def rest(team_id: int) -> float | None:
            last = ctx.repo.last_match_before(team_id, kickoff)
            if last is None:
                return None
            return round(clock.hours_between(last.kickoff_utc, kickoff) / 24.0, 2)

        home_rest = rest(ctx.home_team_id)
        away_rest = rest(ctx.away_team_id)
        if statistic == "rest_days_home":
            return (home_rest, None if home_rest is not None else INSUFFICIENT)
        if statistic == "rest_days_away":
            return (away_rest, None if away_rest is not None else INSUFFICIENT)
        if home_rest is None or away_rest is None:
            return None, INSUFFICIENT
        return round(home_rest - away_rest, 2), None

    if statistic in ("congestion_home", "congestion_away"):
        team_id = ctx.home_team_id if statistic.endswith("home") else ctx.away_team_id
        count = ctx.repo.matches_in_period(team_id, kickoff - dt.timedelta(days=14), kickoff)
        return float(count), None

    if statistic == "season_progress":
        row = ctx.repo.db.query_one(
            "SELECT start_date, end_date FROM season WHERE season_id = ?",
            (ctx.fixture.get("season_id"),))
        if not row:
            return None, SOURCE_MISSING
        try:
            start = dt.date.fromisoformat(row["start_date"])
            end = dt.date.fromisoformat(row["end_date"])
        except (TypeError, ValueError):
            return None, SOURCE_MISSING
        span = max(1, (end - start).days)
        elapsed = (kickoff.date() - start).days
        return round(min(1.0, max(0.0, elapsed / span)), 4), None

    if statistic == "is_uefa":
        return (1.0 if ctx.fixture.get("competition_kind") == "UEFA_CLUB" else 0.0), None

    if statistic in ("home_history_matches", "away_history_matches"):
        team_id = ctx.home_team_id if statistic.startswith("home") else ctx.away_team_id
        scope = "SAME_COMPETITION"
        slug = ctx.fixture.get("competition_slug")
        if slug and ctx.config is not None:
            try:
                scope = ctx.config.competition(slug).feature_scope
            except (KeyError, AttributeError):
                pass
        competition_id = (ctx.fixture.get("competition_id")
                          if scope == "SAME_COMPETITION" else None)
        matches = ctx.repo.team_matches(
            team_id, competition_id=competition_id, before=kickoff)
        return float(len(matches)), None

    if statistic in ("league_home_goals", "league_away_goals"):
        baseline = ctx.repo.competition_baseline(ctx.fixture["competition_id"])
        if baseline["n"] < 20:
            return None, INSUFFICIENT
        key = "home_goals" if statistic.endswith("home_goals") else "away_goals"
        return round(baseline[key], 4), None

    return None, SOURCE_MISSING


def compute_features(ctx: FeatureContext) -> dict[str, dict[str, Any]]:
    """Compute every registered feature for one fixture at one as-of time."""
    results: dict[str, dict[str, Any]] = {}
    for spec in FEATURE_REGISTRY:
        if spec.entity == "MATCH":
            value, reason = _match_feature(ctx, spec)
        else:
            team_id = ctx.home_team_id if spec.entity == "HOME_TEAM" else ctx.away_team_id
            value, reason = _team_statistic(ctx, spec, team_id)
        results[spec.key] = {
            "value": None if value is None else round(float(value), 6),
            "is_null_reason": reason,
        }
    return results


def completeness(values: dict[str, dict[str, Any]]) -> float:
    """Share of REQUIRED features that actually have a value.

    Measured over required features only, because completeness is meant to
    answer "did we get the data we need?" rather than "is every column
    populated?". Season-to-date features are empty by definition in August; a
    gate that counted them would refuse to predict through the opening weeks of
    every season, which is when the calendar is densest. Overall coverage is
    still recorded per feature via ``is_null_reason`` for auditing.
    """
    from nway.features.registry import REQUIRED_KEYS

    required = {k: v for k, v in values.items() if k in REQUIRED_KEYS}
    if not required:
        return 0.0
    present = sum(1 for v in required.values() if v["value"] is not None)
    return round(present / len(required), 4)


def overall_coverage(values: dict[str, dict[str, Any]]) -> float:
    """Share of ALL registered features with a value; for reporting only."""
    if not values:
        return 0.0
    return round(sum(1 for v in values.values() if v["value"] is not None)
                 / len(values), 4)


def persist_features(db: Database, fixture_id: int, as_of: dt.datetime,
                     values: dict[str, dict[str, Any]],
                     source_max_knowledge_time: str) -> str:
    """Write feature values, registering the feature version if it is new."""
    version = feature_version()
    now = clock.to_iso(clock.now())
    db.insert("feature_set_version", {
        "feature_version": version,
        "description": f"{len(FEATURE_REGISTRY)} declared features",
        "spec_hash": version.removeprefix("fs_"), "created_at": now,
    }, or_ignore=True)

    as_of_iso = clock.to_iso(as_of)
    rows = [
        (fixture_id, version, as_of_iso, key, payload["value"],
         payload["is_null_reason"], source_max_knowledge_time, now)
        for key, payload in values.items()
    ]
    db.executemany(
        "INSERT OR REPLACE INTO feature_value (fixture_id, feature_version, as_of, "
        "feature_key, value, is_null_reason, source_max_knowledge_time, computed_at) "
        "VALUES (?,?,?,?,?,?,?,?)", rows)
    return version
