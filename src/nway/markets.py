"""Goal distribution and market derivation.

Deliberately a top-level module rather than part of ``nway.models``. Market
derivation is a shared domain concept: the models produce goal rates, the
prediction pipeline derives markets from them, and the recommendation engine
needs the same matrix to compute exact within-fixture correlation. Putting it
under ``models`` would force the recommendation engine to import from the model
package, which the architecture test forbids for good reason -- selection
policy and prediction quality must be able to change independently.


One joint distribution over (home goals, away goals) produces every
goal-derived market by summation. Training separate models for Over 1.5, Over
2.5 and BTTS would let them contradict each other -- P(Over 2.5) > P(Over 1.5)
is a real failure mode when markets are modelled independently. Deriving them
from a single matrix makes that impossible by construction, and the invariants
in ``check_coherence`` verify the implementation rather than the modelling.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    home_pmf = poisson.pmf(np.arange(grid), lam_home)
    away_pmf = poisson.pmf(np.arange(grid), lam_away)
    matrix = np.outer(home_pmf, away_pmf)
    if rho:
        for h in range(min(2, grid)):
            for a in range(min(2, grid)):
                matrix[h, a] *= dixon_coles_tau(h, a, lam_home, lam_away, rho)
        matrix = np.clip(matrix, 0.0, None)
    total = matrix.sum()
    return matrix / total if total > 0 else matrix


@dataclass
class MarketProbabilities:
    """Every market derived from one goal matrix."""
    probabilities: dict[str, float]
    lambda_home: float
    lambda_away: float
    matrix: np.ndarray

    def get(self, market_key: str) -> float | None:
        return self.probabilities.get(market_key)

    def joint(self, first: str, second: str) -> float | None:
        """Exact joint probability of two markets from the same matrix.

        Because both are derived from the same distribution, correlation
        between them is computable rather than assumed -- which is what the
        recommendation engine uses to avoid presenting one estimate as several
        independent opportunities.
        """
        masks = _market_masks(self.matrix.shape[0])
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


def _market_masks(grid: int) -> dict[str, np.ndarray]:
    # Full (grid, grid) index arrays. Using broadcast row/column vectors here
    # silently yields (1, grid) masks for conditions that touch only one side
    # (e.g. ``away == 0``), which then index the matrix wrongly.
    home, away = np.indices((grid, grid))
    total = home + away
    return {
        "HOME_WIN": home > away,
        "DRAW": home == away,
        "AWAY_WIN": home < away,
        "DOUBLE_CHANCE_1X": home >= away,
        "DOUBLE_CHANCE_X2": home <= away,
        "DOUBLE_CHANCE_12": home != away,
        "OVER_0_5": total > 0,
        "OVER_1_5": total > 1,
        "OVER_2_5": total > 2,
        "OVER_3_5": total > 3,
        "UNDER_1_5": total <= 1,
        "UNDER_2_5": total <= 2,
        "UNDER_3_5": total <= 3,
        "BTTS": (home > 0) & (away > 0),
        "BTTS_NO": (home == 0) | (away == 0),
        "HOME_CLEAN_SHEET": away == 0,
        "AWAY_CLEAN_SHEET": home == 0,
        "HOME_TO_SCORE": home > 0,
        "AWAY_TO_SCORE": away > 0,
        "HOME_OVER_1_5": home > 1,
        "AWAY_OVER_1_5": away > 1,
    }


def derive_markets(lam_home: float, lam_away: float,
                   rho: float = 0.0, grid: int = GRID) -> MarketProbabilities:
    matrix = goal_matrix(lam_home, lam_away, rho, grid)
    probabilities = {
        key: float(matrix[mask].sum()) for key, mask in _market_masks(grid).items()
    }

    # Numeric markets: reported for context, never selected as recommendations.
    home_axis = np.arange(grid)
    expected_home = float((matrix.sum(axis=1) * home_axis).sum())
    expected_away = float((matrix.sum(axis=0) * home_axis).sum())
    probabilities["EXPECTED_HOME_GOALS"] = expected_home
    probabilities["EXPECTED_AWAY_GOALS"] = expected_away
    probabilities["EXPECTED_GOALS"] = expected_home + expected_away

    margins = home_axis[:, None] - home_axis[None, :]
    probabilities["GOAL_MARGIN"] = float((matrix * margins).sum())
    for margin in (1, 2, 3):
        probabilities[f"HOME_BY_{margin}_PLUS"] = float(matrix[margins >= margin].sum())
        probabilities[f"AWAY_BY_{margin}_PLUS"] = float(matrix[margins <= -margin].sum())

    return MarketProbabilities(probabilities, lam_home, lam_away, matrix)


class CoherenceError(AssertionError):
    """A derived market set violated an invariant that must hold by construction."""


def check_coherence(markets: MarketProbabilities) -> None:
    """Assert the invariants that deriving from one matrix guarantees.

    These catch implementation bugs, not modelling disagreements: every one of
    them is true of any valid joint distribution, so a failure means the code
    is wrong.
    """
    probabilities = markets.probabilities
    total = float(markets.matrix.sum())
    if abs(total - 1.0) > 1e-6:
        raise CoherenceError(f"matrix does not sum to 1: {total}")

    one_x_two = (probabilities["HOME_WIN"] + probabilities["DRAW"]
                 + probabilities["AWAY_WIN"])
    if abs(one_x_two - 1.0) > 1e-6:
        raise CoherenceError(f"1X2 does not sum to 1: {one_x_two}")

    ladder = ["OVER_0_5", "OVER_1_5", "OVER_2_5", "OVER_3_5"]
    for higher, lower in zip(ladder, ladder[1:]):
        if probabilities[lower] > probabilities[higher] + TOLERANCE:
            raise CoherenceError(f"{lower} exceeds {higher}")

    if probabilities["BTTS"] > probabilities["OVER_1_5"] + TOLERANCE:
        raise CoherenceError("BTTS exceeds Over 1.5, which is impossible")

    expected = probabilities["EXPECTED_GOALS"]
    if abs(expected - (markets.lambda_home + markets.lambda_away)) > 0.05:
        raise CoherenceError(
            f"expected goals {expected:.4f} differs from lambda sum "
            f"{markets.lambda_home + markets.lambda_away:.4f}")

    if abs(probabilities["DOUBLE_CHANCE_1X"]
           - (probabilities["HOME_WIN"] + probabilities["DRAW"])) > 1e-6:
        raise CoherenceError("double chance 1X is not P(H)+P(D)")

    for key, value in probabilities.items():
        if key.startswith("EXPECTED") or key == "GOAL_MARGIN":
            continue
        if not (-TOLERANCE <= value <= 1.0 + TOLERANCE):
            raise CoherenceError(f"{key} outside [0,1]: {value}")


def market_selection(market_key: str) -> tuple[str, str]:
    """Split a derived key into (stored market key, selection label)."""
    if market_key in ("HOME_WIN", "DRAW", "AWAY_WIN"):
        return market_key, {"HOME_WIN": "HOME", "DRAW": "DRAW",
                            "AWAY_WIN": "AWAY"}[market_key]
    return market_key, "YES"
