"""Declarative feature registry.

Features are declared, not merely implemented. The registry is hashed into a
``feature_version``, so a window cannot be changed quietly: altering a spec
produces a new version, and existing predictions keep pointing at the
definition that produced them.

``min_observations`` and ``max_staleness_hours`` are part of the declaration
because "no value" must never be confused with "zero". A newly promoted team in
August genuinely has no season history, and a model that reads that as zero
goals per game will produce confident nonsense.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Window:
    kind: str = "MATCHES"        # MATCHES | SEASON
    size: int | None = None
    venue: str = "ANY"           # ANY | HOME | AWAY

    @property
    def suffix(self) -> str:
        base = f"last{self.size}" if self.kind == "MATCHES" else "std"
        return base if self.venue == "ANY" else f"{base}.{self.venue.lower()}_only"


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    entity: str                  # HOME_TEAM | AWAY_TEAM | MATCH
    statistic: str
    window: Window = field(default_factory=Window)
    min_observations: int = 3
    max_staleness_hours: float = 96.0
    competition_scope: str = "SAME_COMPETITION"
    requires_completed_fixtures: bool = True
    required: bool = True
    description: str = ""


# Statistics available from a MatchRow. Corner and card statistics are only
# present from 2017/18 in the Eredivisie and Primeira Liga, which is why market
# eligibility is gated per league rather than assumed globally.
_TEAM_STATISTICS = [
    ("gf_per_match", "goals for per match"),
    ("ga_per_match", "goals against per match"),
    ("shots_per_match", "shots per match"),
    ("sot_per_match", "shots on target per match"),
    ("corners_per_match", "corners per match"),
    ("cards_per_match", "cards per match"),
    ("sxg_for_per_match", "shot-based expected goals proxy for"),
    ("sxg_against_per_match", "shot-based expected goals proxy against"),
    ("points_per_match", "points per match"),
    ("clean_sheet_rate", "clean sheet rate"),
    ("btts_rate", "both teams to score rate"),
    ("ht_gf_per_match", "first-half goals for per match"),
]

_WINDOWS = [
    Window("MATCHES", 5, "ANY"),
    Window("MATCHES", 10, "ANY"),
    Window("MATCHES", 5, "HOME"),
    Window("MATCHES", 5, "AWAY"),
    Window("SEASON", None, "ANY"),
]


def _build_registry() -> tuple[FeatureSpec, ...]:
    specs: list[FeatureSpec] = []
    for side in ("home", "away"):
        entity = "HOME_TEAM" if side == "home" else "AWAY_TEAM"
        for statistic, description in _TEAM_STATISTICS:
            for window in _WINDOWS:
                # A home-only window for the away side is not useless, but it
                # halves an already thin sample; the away side's away-only form
                # is the informative split, and vice versa.
                if window.venue == "HOME" and side == "away":
                    continue
                if window.venue == "AWAY" and side == "home":
                    continue
                specs.append(FeatureSpec(
                    key=f"{side}.{statistic}.{window.suffix}",
                    entity=entity, statistic=statistic, window=window,
                    min_observations=3,
                    # Season-to-date is enrichment, not a requirement. It is
                    # legitimately empty every August, and gating on it would
                    # blind the system for the opening weeks of every season --
                    # exactly when the fixture calendar is densest.
                    required=(window.kind == "MATCHES"),
                    description=f"{side} team {description}, {window.suffix}",
                ))

    # Match context. Objective and measurable only -- nothing resembling
    # "motivation", which cannot be represented by data.
    context = [
        ("match.rest_days_home", "days since the home team last played"),
        ("match.rest_days_away", "days since the away team last played"),
        ("match.rest_days_diff", "rest advantage, home minus away"),
        ("match.congestion_home", "home team matches in the previous 14 days"),
        ("match.congestion_away", "away team matches in the previous 14 days"),
        ("match.season_progress", "fraction of the season elapsed"),
        ("match.is_uefa", "1 when this is a UEFA competition"),
        ("match.home_history_matches", "completed matches known for the home team"),
        ("match.away_history_matches", "completed matches known for the away team"),
        ("match.league_home_goals", "league baseline home goals per match"),
        ("match.league_away_goals", "league baseline away goals per match"),
    ]
    for key, description in context:
        specs.append(FeatureSpec(
            key=key, entity="MATCH", statistic=key.split(".", 1)[1],
            window=Window("SEASON", None, "ANY"), min_observations=0,
            # season_progress needs season bounds; the league baselines need a
            # populated season. Both are context, not prerequisites.
            required=key not in ("match.season_progress", "match.league_home_goals",
                                 "match.league_away_goals"),
            description=description,
        ))
    return tuple(specs)


FEATURE_REGISTRY: tuple[FeatureSpec, ...] = _build_registry()
FEATURE_KEYS: tuple[str, ...] = tuple(spec.key for spec in FEATURE_REGISTRY)
REQUIRED_KEYS: frozenset[str] = frozenset(
    spec.key for spec in FEATURE_REGISTRY if spec.required)
SPEC_BY_KEY: dict[str, FeatureSpec] = {spec.key: spec for spec in FEATURE_REGISTRY}


def spec_hash() -> str:
    payload = json.dumps([asdict(s) for s in FEATURE_REGISTRY], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def feature_version() -> str:
    """Stable identifier for the current set of feature definitions."""
    return f"fs_{spec_hash()}"
