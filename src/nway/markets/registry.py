"""Which markets each grid produces, and how a key splits into a selection.

The pipeline stores a fixed list and the configuration must declare every one
of it; an architecture test asserts the two agree. Six undeclared markets once
cost 282 of 705 candidates on a matchday, discarded as UNKNOWN_MARKET with no
error anywhere.
"""

from __future__ import annotations

from nway.markets.cards import CARD_MARKETS
from nway.markets.corners import CORNER_MARKETS
from nway.markets.goals import goal_masks
from nway.markets.halves import HALF_MARKETS

GOAL_MARKETS: tuple[str, ...] = tuple(goal_masks().keys())

# Grid of origin, which is also the unit the diversity caps work on: one bet
# per grid per fixture, because every market from a grid is a view of the same
# estimate.
MARKET_GRID: dict[str, str] = {
    **{key: "GOAL_MATRIX" for key in GOAL_MARKETS},
    **{key: "HALF_MATRIX" for key in HALF_MARKETS},
    **{key: "CORNER_MATRIX" for key in CORNER_MARKETS},
    **{key: "CARD_DISTRIBUTION" for key in CARD_MARKETS},
}

ALL_DERIVED_MARKETS: tuple[str, ...] = (
    GOAL_MARKETS + HALF_MARKETS + CORNER_MARKETS + CARD_MARKETS)

# Markets whose "selection" is not a plain yes/no.
_THREE_WAY = {
    "HOME_WIN": "HOME", "DRAW": "DRAW", "AWAY_WIN": "AWAY",
    "HT_HOME_WIN": "HOME", "HT_DRAW": "DRAW", "HT_AWAY_WIN": "AWAY",
    "CORNERS_1X2_HOME": "HOME", "CORNERS_1X2_DRAW": "DRAW",
    "CORNERS_1X2_AWAY": "AWAY",
    "EH_MINUS_1_HOME": "HOME", "EH_MINUS_1_DRAW": "DRAW",
    "EH_MINUS_1_AWAY": "AWAY",
    "EH_PLUS_1_HOME": "HOME", "EH_PLUS_1_DRAW": "DRAW",
    "EH_PLUS_1_AWAY": "AWAY",
}


def market_selection(market_key: str) -> tuple[str, str]:
    """Split a derived key into (stored market key, selection label)."""
    return market_key, _THREE_WAY.get(market_key, "YES")


def grid_of(market_key: str) -> str:
    return MARKET_GRID.get(market_key, "GOAL_MATRIX")
