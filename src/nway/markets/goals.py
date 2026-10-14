"""Goal distribution and every market derived from it.

One joint distribution over (home goals, away goals) produces every
goal-derived market by summation over cells. Training separate models per
market would let them contradict each other -- P(Over 2.5) > P(Over 1.5) is a
real failure mode when markets are modelled independently -- and deriving them
from a single matrix makes that impossible by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import poisson

GRID = 11          # 0..10 goals per side; covers >99.99% of realistic mass
TOLERANCE = 1e-9


def dixon_coles_tau(home_goals: int, away_goals: int, lam_home: float,
                    lam_away: float, rho: float) -> float:
    """Low-score dependence correction.

    Independent Poisson is known to misfit the four lowest scorelines: it
    understates 0-0 and 1-1 and overstates 1-0 and 0-1. Dixon and Coles (1997)
    correct exactly those cells and leave the rest untouched.
    """
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lam_home * lam_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lam_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lam_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def goal_matrix(lam_home: float, lam_away: float, rho: float = 0.0,
                grid: int = GRID) -> np.ndarray:
    """Joint P(home=h, away=a), renormalised over the grid."""
    lam_home = max(1e-6, float(lam_home))
    lam_away = max(1e-6, float(lam_away))
    matrix = np.outer(poisson.pmf(np.arange(grid), lam_home),
                      poisson.pmf(np.arange(grid), lam_away))
    if rho:
        for h in range(min(2, grid)):
            for a in range(min(2, grid)):
                matrix[h, a] *= dixon_coles_tau(h, a, lam_home, lam_away, rho)
        matrix = np.clip(matrix, 0.0, None)
    total = matrix.sum()
    return matrix / total if total > 0 else matrix


def goal_masks(grid: int = GRID) -> dict[str, np.ndarray]:
    """Boolean masks over the (home, away) goal grid.

    Full (grid, grid) index arrays via np.indices: broadcast row/column vectors
    silently yield (1, grid) masks for conditions touching only one side (e.g.
    ``away == 0``), which then index the matrix wrongly.
    """
    home, away = np.indices((grid, grid))
    total = home + away
    margin = home - away
    return {
        # --- match result -------------------------------------------------
        "HOME_WIN": home > away,
        "DRAW": home == away,
        "AWAY_WIN": home < away,
        "DOUBLE_CHANCE_1X": home >= away,
        "DOUBLE_CHANCE_X2": home <= away,
        "DOUBLE_CHANCE_12": home != away,
        # --- total goals --------------------------------------------------
        "OVER_0_5": total > 0,
        "OVER_1_5": total > 1,
        "OVER_2_5": total > 2,
        "OVER_3_5": total > 3,
        "UNDER_1_5": total <= 1,
        "UNDER_2_5": total <= 2,
        "UNDER_3_5": total <= 3,
        "BTTS": (home > 0) & (away > 0),
        "BTTS_NO": (home == 0) | (away == 0),
        # --- team totals --------------------------------------------------
        # HOME_OVER_0_5 is HOME_TO_SCORE by definition; only the latter is
        # registered, so the same event cannot enter a batch twice under two
        # names.
        "HOME_TO_SCORE": home > 0,
        "AWAY_TO_SCORE": away > 0,
        "HOME_OVER_1_5": home > 1,
        "HOME_OVER_2_5": home > 2,
        "AWAY_OVER_1_5": away > 1,
        "AWAY_OVER_2_5": away > 2,
        # --- clean sheets and win to nil ----------------------------------
        "HOME_CLEAN_SHEET": away == 0,
        "AWAY_CLEAN_SHEET": home == 0,
        "HOME_WIN_TO_NIL": (home > away) & (away == 0),
        "AWAY_WIN_TO_NIL": (away > home) & (home == 0),
        # --- Asian handicap, half lines only ------------------------------
        # Half lines cannot push, so settlement stays strictly Bernoulli --
        # which the Beta posterior ranker requires. The +/-0.5 lines are
        # omitted deliberately: home +0.5 IS double chance 1X, and away +0.5
        # IS X2, both already registered.
        "AH_HOME_MINUS_1_5": margin >= 2,
        "AH_AWAY_PLUS_1_5": margin <= 1,
        "AH_AWAY_MINUS_1_5": margin <= -2,
        "AH_HOME_PLUS_1_5": margin >= -1,
        # --- European 3-way handicap --------------------------------------
        # Three mutually exclusive binary outcomes per line, so each settles
        # 0 or 1 and the trio sums to one. A whole-number handicap on a
        # two-way market would push; split three ways it cannot.
        "EH_MINUS_1_HOME": margin >= 2,
        "EH_MINUS_1_DRAW": margin == 1,
        "EH_MINUS_1_AWAY": margin <= 0,
        "EH_PLUS_1_HOME": margin >= 0,
        "EH_PLUS_1_DRAW": margin == -1,
        "EH_PLUS_1_AWAY": margin <= -2,
        # --- result combinations ------------------------------------------
        "HOME_AND_BTTS": (home > away) & (home > 0) & (away > 0),
        "DRAW_AND_BTTS": (home == away) & (home > 0),
        "AWAY_AND_BTTS": (away > home) & (home > 0),
        "HOME_AND_OVER_2_5": (home > away) & (total > 2),
        "DRAW_AND_OVER_2_5": (home == away) & (total > 2),
        "AWAY_AND_OVER_2_5": (away > home) & (total > 2),
    }


@dataclass
class MarketProbabilities:
    """Every market derived from one joint grid."""
    probabilities: dict[str, float]
    lambda_home: float
    lambda_away: float
    matrix: np.ndarray
    masks: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def get(self, market_key: str) -> float | None:
        return self.probabilities.get(market_key)

    def joint(self, first: str, second: str) -> float | None:
        """Exact joint probability of two markets from the same grid.

        Because both are derived from one distribution, correlation between
        them is computable rather than assumed -- which is what lets the
        recommendation engine avoid presenting one estimate as several
        independent opportunities.
        """
        masks = self.masks or goal_masks(self.matrix.shape[0])
        if first not in masks or second not in masks:
            return None
        return float(self.matrix[masks[first] & masks[second]].sum())

    def correlation(self, first: str, second: str) -> float | None:
        """Phi coefficient for two binary market outcomes."""
        p_first = self.probabilities.get(first)
        p_second = self.probabilities.get(second)
        joint = self.joint(first, second)
        if p_first is None or p_second is None or joint is None:
            return None
        denominator = p_first * (1 - p_first) * p_second * (1 - p_second)
        if denominator <= 0:
            return 1.0
        return float((joint - p_first * p_second) / np.sqrt(denominator))


def derive_markets(lam_home: float, lam_away: float,
                   rho: float = 0.0, grid: int = GRID) -> MarketProbabilities:
    matrix = goal_matrix(lam_home, lam_away, rho, grid)
    masks = goal_masks(grid)
    probabilities = {key: float(matrix[mask].sum()) for key, mask in masks.items()}

    # Numeric quantities: reported for context, never selected.
    axis = np.arange(grid)
    expected_home = float((matrix.sum(axis=1) * axis).sum())
    expected_away = float((matrix.sum(axis=0) * axis).sum())
    probabilities["EXPECTED_HOME_GOALS"] = expected_home
    probabilities["EXPECTED_AWAY_GOALS"] = expected_away
    probabilities["EXPECTED_GOALS"] = expected_home + expected_away

    margins = axis[:, None] - axis[None, :]
    probabilities["GOAL_MARGIN"] = float((matrix * margins).sum())

    return MarketProbabilities(probabilities, lam_home, lam_away, matrix, masks)
