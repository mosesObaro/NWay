"""Market settlement rules: one predicate per market, from stored match facts.

These are the single definition of what each market means. The same resolvers
compute the base rates in ``config/markets.yaml`` and settle live predictions,
so a market cannot be scored against one rule and floored against another.

All markets settle on ninety minutes. Extra time and penalties are excluded,
which matters for UEFA knockouts, so ``MatchFacts.went_to_et`` exists to make
the exclusion explicit rather than accidental.

Every predicate is strictly Bernoulli: half-line handicaps cannot push, and
whole-number handicaps are expressed as three mutually exclusive outcomes. The
Beta posterior ranker needs 0/1 outcomes, and a pushed bet would break it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class MatchFacts:
    """Everything settlement is allowed to look at."""
    home_goals: int
    away_goals: int
    ht_home_goals: int | None = None
    ht_away_goals: int | None = None
    home_corners: int | None = None
    away_corners: int | None = None
    total_cards: int | None = None
    went_to_et: bool = False

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals

    @property
    def margin(self) -> int:
        return self.home_goals - self.away_goals

    @property
    def result(self) -> str:
        return "H" if self.margin > 0 else ("D" if self.margin == 0 else "A")

    @property
    def ht_result(self) -> str | None:
        if self.ht_home_goals is None or self.ht_away_goals is None:
            return None
        diff = self.ht_home_goals - self.ht_away_goals
        return "H" if diff > 0 else ("D" if diff == 0 else "A")

    @property
    def ht_total(self) -> int | None:
        if self.ht_home_goals is None or self.ht_away_goals is None:
            return None
        return self.ht_home_goals + self.ht_away_goals

    @property
    def total_corners(self) -> int | None:
        if self.home_corners is None or self.away_corners is None:
            return None
        return self.home_corners + self.away_corners

    @property
    def corner_margin(self) -> int | None:
        if self.home_corners is None or self.away_corners is None:
            return None
        return self.home_corners - self.away_corners


Resolver = Callable[[MatchFacts], bool]


def _goal_resolvers() -> dict[str, Resolver]:
    return {
        "HOME_WIN": lambda f: f.margin > 0,
        "DRAW": lambda f: f.margin == 0,
        "AWAY_WIN": lambda f: f.margin < 0,
        "DOUBLE_CHANCE_1X": lambda f: f.margin >= 0,
        "DOUBLE_CHANCE_X2": lambda f: f.margin <= 0,
        "DOUBLE_CHANCE_12": lambda f: f.margin != 0,
        "OVER_0_5": lambda f: f.total_goals > 0,
        "OVER_1_5": lambda f: f.total_goals > 1,
        "OVER_2_5": lambda f: f.total_goals > 2,
        "OVER_3_5": lambda f: f.total_goals > 3,
        "UNDER_1_5": lambda f: f.total_goals <= 1,
        "UNDER_2_5": lambda f: f.total_goals <= 2,
        "UNDER_3_5": lambda f: f.total_goals <= 3,
        "BTTS": lambda f: f.home_goals > 0 and f.away_goals > 0,
        "BTTS_NO": lambda f: f.home_goals == 0 or f.away_goals == 0,
        "HOME_TO_SCORE": lambda f: f.home_goals > 0,
        "AWAY_TO_SCORE": lambda f: f.away_goals > 0,
        "HOME_OVER_1_5": lambda f: f.home_goals > 1,
        "HOME_OVER_2_5": lambda f: f.home_goals > 2,
        "AWAY_OVER_1_5": lambda f: f.away_goals > 1,
        "AWAY_OVER_2_5": lambda f: f.away_goals > 2,
        "HOME_CLEAN_SHEET": lambda f: f.away_goals == 0,
        "AWAY_CLEAN_SHEET": lambda f: f.home_goals == 0,
        "HOME_WIN_TO_NIL": lambda f: f.margin > 0 and f.away_goals == 0,
        "AWAY_WIN_TO_NIL": lambda f: f.margin < 0 and f.home_goals == 0,
        # Half lines: no push.
        "AH_HOME_MINUS_1_5": lambda f: f.margin >= 2,
        "AH_AWAY_PLUS_1_5": lambda f: f.margin <= 1,
        "AH_AWAY_MINUS_1_5": lambda f: f.margin <= -2,
        "AH_HOME_PLUS_1_5": lambda f: f.margin >= -1,
        # Whole-number handicaps, split three ways so none can push.
        "EH_MINUS_1_HOME": lambda f: f.margin >= 2,
        "EH_MINUS_1_DRAW": lambda f: f.margin == 1,
        "EH_MINUS_1_AWAY": lambda f: f.margin <= 0,
        "EH_PLUS_1_HOME": lambda f: f.margin >= 0,
        "EH_PLUS_1_DRAW": lambda f: f.margin == -1,
        "EH_PLUS_1_AWAY": lambda f: f.margin <= -2,
        "HOME_AND_BTTS": lambda f: f.margin > 0 and f.away_goals > 0,
        "DRAW_AND_BTTS": lambda f: f.margin == 0 and f.home_goals > 0,
        "AWAY_AND_BTTS": lambda f: f.margin < 0 and f.home_goals > 0,
        "HOME_AND_OVER_2_5": lambda f: f.margin > 0 and f.total_goals > 2,
        "DRAW_AND_OVER_2_5": lambda f: f.margin == 0 and f.total_goals > 2,
        "AWAY_AND_OVER_2_5": lambda f: f.margin < 0 and f.total_goals > 2,
    }


def _half_resolvers() -> dict[str, Resolver]:
    resolvers: dict[str, Resolver] = {
        "HT_OVER_0_5": lambda f: f.ht_total > 0,
        "HT_OVER_1_5": lambda f: f.ht_total > 1,
        "HT_UNDER_0_5": lambda f: f.ht_total < 1,
        "HT_UNDER_1_5": lambda f: f.ht_total < 2,
        "HT_HOME_WIN": lambda f: f.ht_result == "H",
        "HT_DRAW": lambda f: f.ht_result == "D",
        "HT_AWAY_WIN": lambda f: f.ht_result == "A",
        "HT_DOUBLE_CHANCE_1X": lambda f: f.ht_result in ("H", "D"),
        "HT_DOUBLE_CHANCE_X2": lambda f: f.ht_result in ("A", "D"),
    }
    for first in ("H", "D", "A"):
        for second in ("H", "D", "A"):
            resolvers[f"HTFT_{first}_{second}"] = (
                lambda f, a=first, b=second: f.ht_result == a and f.result == b)
    return resolvers


def _corner_resolvers() -> dict[str, Resolver]:
    resolvers: dict[str, Resolver] = {}
    for line in (8.5, 9.5, 10.5, 11.5):
        label = str(line).replace(".", "_")
        resolvers[f"CORNERS_OVER_{label}"] = lambda f, l=line: f.total_corners > l
        resolvers[f"CORNERS_UNDER_{label}"] = lambda f, l=line: f.total_corners < l
    for line in (3.5, 4.5, 5.5):
        label = str(line).replace(".", "_")
        resolvers[f"HOME_CORNERS_OVER_{label}"] = lambda f, l=line: f.home_corners > l
        resolvers[f"AWAY_CORNERS_OVER_{label}"] = lambda f, l=line: f.away_corners > l
    resolvers["CORNERS_1X2_HOME"] = lambda f: f.corner_margin > 0
    resolvers["CORNERS_1X2_DRAW"] = lambda f: f.corner_margin == 0
    resolvers["CORNERS_1X2_AWAY"] = lambda f: f.corner_margin < 0
    resolvers["CORNERS_AH_HOME_MINUS_1_5"] = lambda f: f.corner_margin >= 2
    resolvers["CORNERS_AH_AWAY_PLUS_1_5"] = lambda f: f.corner_margin <= 1
    resolvers["CORNERS_AH_AWAY_MINUS_1_5"] = lambda f: f.corner_margin <= -2
    resolvers["CORNERS_AH_HOME_PLUS_1_5"] = lambda f: f.corner_margin >= -1
    return resolvers


def _card_resolvers() -> dict[str, Resolver]:
    resolvers: dict[str, Resolver] = {}
    for line in (3.5, 4.5, 5.5):
        label = str(line).replace(".", "_")
        resolvers[f"CARDS_OVER_{label}"] = lambda f, l=line: f.total_cards > l
        resolvers[f"CARDS_UNDER_{label}"] = lambda f, l=line: f.total_cards < l
    return resolvers


RESOLVERS: dict[str, Resolver] = {
    **_goal_resolvers(), **_half_resolvers(),
    **_corner_resolvers(), **_card_resolvers(),
}

# Which stored facts each market needs. A market whose facts are missing
# settles UNSETTLEABLE rather than as a loss -- a data gap is not a wrong
# prediction, and conflating them would quietly flatter the model.
REQUIRED_FACTS: dict[str, tuple[str, ...]] = {
    **{key: () for key in _goal_resolvers()},
    **{key: ("ht_home_goals", "ht_away_goals") for key in _half_resolvers()},
    **{key: ("home_corners", "away_corners") for key in _corner_resolvers()},
    **{key: ("total_cards",) for key in _card_resolvers()},
}


def can_settle(market_key: str, facts: MatchFacts) -> bool:
    required = REQUIRED_FACTS.get(market_key)
    if required is None:
        return False
    return all(getattr(facts, name) is not None for name in required)


def settle(market_key: str, facts: MatchFacts) -> bool | None:
    """True/False, or None when the required facts are not available."""
    resolver = RESOLVERS.get(market_key)
    if resolver is None or not can_settle(market_key, facts):
        return None
    return bool(resolver(facts))
