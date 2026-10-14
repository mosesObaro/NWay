"""Market derivation: goal, corner, card and half-time grids.

Deliberately a top-level package rather than part of ``nway.models``. Market
derivation is a shared domain concept: the models produce rates, the prediction
pipeline derives markets from them, and the recommendation engine needs the
same grids to compute exact within-fixture correlation. Putting it under
``models`` would force the recommendation engine to import the model package,
which the architecture test forbids for good reason -- selection policy and
prediction quality must be able to change independently.

Each event type has exactly ONE joint grid, and every sub-market is a sum over
its cells. That is what makes impossible orderings (P(Over 2.5) > P(Over 1.5),
or a team-total exceeding a match-total) unreachable rather than merely
unlikely.
"""

from nway.markets.cards import (
    CARD_MARKETS, card_distribution, derive_card_markets,
)
from nway.markets.coherence import CoherenceError, check_coherence
from nway.markets.corners import (
    CORNER_GRID, CORNER_MARKETS, corner_matrix, corner_masks,
    derive_corner_markets,
)
from nway.markets.goals import (
    GRID, TOLERANCE, MarketProbabilities, derive_markets, dixon_coles_tau,
    goal_masks, goal_matrix,
)
from nway.markets.halves import (
    HALF_MARKETS, derive_half_markets, half_time_matrix,
)
from nway.markets.registry import ALL_DERIVED_MARKETS, market_selection

__all__ = [
    "GRID", "TOLERANCE", "MarketProbabilities", "derive_markets",
    "dixon_coles_tau", "goal_masks", "goal_matrix",
    "CORNER_GRID", "CORNER_MARKETS", "corner_matrix", "corner_masks",
    "derive_corner_markets",
    "CARD_MARKETS", "card_distribution", "derive_card_markets",
    "HALF_MARKETS", "derive_half_markets", "half_time_matrix",
    "CoherenceError", "check_coherence",
    "ALL_DERIVED_MARKETS", "market_selection",
]
