"""Corner markets from a single joint 26x26 grid.

Corners are NOT modelled as two independent counts. Measured over 21,919
matches, home and away corners correlate at **-0.276**: one side having more
means the other has fewer, which is what possession implies. An independent
product would give var(home) + var(away) = 16.09 against a measured total
variance of 11.68 -- overstating the spread of total corners by 38%, precisely
at the 8.5 to 11.5 lines where the mass sits.

So the grid is built the other way round:

    total corners  T  ~ NegBinomial(mu, r)
    home share     H | T ~ BetaBinomial(T, alpha, beta)
    P(c_h, c_a)      = P(T = c_h + c_a) * P(H = c_h | T)

Two things fall out for free. The total marginal is exactly the fitted negative
binomial, so the total over/under lines -- the primary markets -- are calibrated
by construction rather than by luck. And because the split is conditional on
the total, the negative covariance is induced by the mechanism itself:
var(H) + var(A) + 2cov = var(T) holds identically.

Measured dispersion: total corners var/mean 1.184, home 1.653, away 1.603.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import betaln, gammaln

CORNER_GRID = 26        # 0..25 per side; P(either side > 25) measured at 9.1e-5


def negative_binomial_pmf(k: np.ndarray, mean: float, dispersion: float) -> np.ndarray:
    """NB2 parameterisation: var = mean + mean^2 / dispersion.

    dispersion -> infinity recovers the Poisson, so a fitted value says how far
    the data sits from Poisson rather than assuming an answer.
    """
    mean = max(1e-6, float(mean))
    dispersion = max(1e-6, float(dispersion))
    log_pmf = (gammaln(k + dispersion) - gammaln(dispersion) - gammaln(k + 1.0)
               + dispersion * np.log(dispersion / (dispersion + mean))
               + k * np.log(mean / (dispersion + mean)))
    return np.exp(log_pmf)


def beta_binomial_pmf(k: np.ndarray, n: int, alpha: float, beta: float) -> np.ndarray:
    """P(H = k | T = n) for a beta-binomial share.

    A plain binomial split would leave per-side variance too low; the beta
    mixing adds exactly the extra spread the data shows (home var/mean 1.65).
    """
    if n <= 0:
        return np.where(k == 0, 1.0, 0.0)
    alpha = max(1e-6, float(alpha))
    beta = max(1e-6, float(beta))
    log_pmf = (gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
               + betaln(k + alpha, n - k + beta) - betaln(alpha, beta))
    return np.exp(log_pmf)


def share_parameters(share: float, concentration: float) -> tuple[float, float]:
    """(alpha, beta) for a mean share and a concentration alpha+beta."""
    share = min(max(float(share), 1e-3), 1 - 1e-3)
    concentration = max(0.2, float(concentration))
    return share * concentration, (1.0 - share) * concentration


def corner_matrix(mean_total: float, dispersion: float, home_share: float,
                  concentration: float, grid: int = CORNER_GRID) -> np.ndarray:
    """Joint P(home corners = c_h, away corners = c_a)."""
    alpha, beta = share_parameters(home_share, concentration)
    totals = np.arange(2 * grid - 1)
    total_pmf = negative_binomial_pmf(totals, mean_total, dispersion)

    matrix = np.zeros((grid, grid))
    for total in totals:
        weight = total_pmf[total]
        if weight <= 0.0:
            continue
        # Only splits that fit inside the grid contribute; the truncated tail
        # is removed by the renormalisation below.
        low = max(0, total - grid + 1)
        high = min(grid - 1, total)
        if low > high:
            continue
        homes = np.arange(low, high + 1)
        split = beta_binomial_pmf(homes, int(total), alpha, beta)
        matrix[homes, total - homes] += weight * split

    mass = matrix.sum()
    return matrix / mass if mass > 0 else matrix


def corner_masks(grid: int = CORNER_GRID) -> dict[str, np.ndarray]:
    home, away = np.indices((grid, grid))
    total = home + away
    margin = home - away
    masks: dict[str, np.ndarray] = {}
    for line in (8.5, 9.5, 10.5, 11.5):
        label = str(line).replace(".", "_")
        masks[f"CORNERS_OVER_{label}"] = total > line
        masks[f"CORNERS_UNDER_{label}"] = total < line
    for line in (3.5, 4.5, 5.5):
        label = str(line).replace(".", "_")
        masks[f"HOME_CORNERS_OVER_{label}"] = home > line
        masks[f"AWAY_CORNERS_OVER_{label}"] = away > line
    masks["CORNERS_1X2_HOME"] = home > away
    masks["CORNERS_1X2_DRAW"] = home == away
    masks["CORNERS_1X2_AWAY"] = home < away
    # Half lines: no push, so settlement stays Bernoulli.
    masks["CORNERS_AH_HOME_MINUS_1_5"] = margin >= 2
    masks["CORNERS_AH_AWAY_PLUS_1_5"] = margin <= 1
    masks["CORNERS_AH_AWAY_MINUS_1_5"] = margin <= -2
    masks["CORNERS_AH_HOME_PLUS_1_5"] = margin >= -1
    return masks


CORNER_MARKETS: tuple[str, ...] = tuple(corner_masks().keys())


@dataclass
class CornerProbabilities:
    probabilities: dict[str, float]
    mean_total: float
    mean_home: float
    mean_away: float
    matrix: np.ndarray
    masks: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def joint(self, first: str, second: str) -> float | None:
        masks = self.masks or corner_masks(self.matrix.shape[0])
        if first not in masks or second not in masks:
            return None
        return float(self.matrix[masks[first] & masks[second]].sum())

    def correlation(self, first: str, second: str) -> float | None:
        p_a = self.probabilities.get(first)
        p_b = self.probabilities.get(second)
        joint = self.joint(first, second)
        if p_a is None or p_b is None or joint is None:
            return None
        denominator = p_a * (1 - p_a) * p_b * (1 - p_b)
        if denominator <= 0:
            return 1.0
        return float((joint - p_a * p_b) / np.sqrt(denominator))


def derive_corner_markets(mean_total: float, dispersion: float, home_share: float,
                          concentration: float,
                          grid: int = CORNER_GRID) -> CornerProbabilities:
    matrix = corner_matrix(mean_total, dispersion, home_share, concentration, grid)
    masks = corner_masks(grid)
    probabilities = {key: float(matrix[mask].sum()) for key, mask in masks.items()}

    axis = np.arange(grid)
    expected_home = float((matrix.sum(axis=1) * axis).sum())
    expected_away = float((matrix.sum(axis=0) * axis).sum())
    probabilities["EXPECTED_HOME_CORNERS"] = expected_home
    probabilities["EXPECTED_AWAY_CORNERS"] = expected_away
    probabilities["EXPECTED_CORNERS"] = expected_home + expected_away

    return CornerProbabilities(probabilities, expected_home + expected_away,
                               expected_home, expected_away, matrix, masks)
