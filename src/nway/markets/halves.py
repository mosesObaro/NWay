"""First-half and half-time/full-time markets.

Derived from the authoritative full-time goal grid rather than from a second,
independently fitted model. Two fitted grids for the same match would disagree:
the full-time marginal implied by a half model would not match the full-time
matrix the result markets come from, and P(1H Over 1.5) could exceed
P(FT Over 1.5).

Instead, condition on the full-time score. If goals arrive as a Poisson process
over the ninety minutes, then given a side's final total the number scored
before the interval is Binomial(total, share). So:

    P(HT = (h1, a1), FT = (h, a))
        = P_FT(h, a) * Binom(h1; h, share) * Binom(a1; a, share)

The full-time marginal is then *exactly* the input grid -- coherence holds by
construction, not by tuning -- and the only parameter is the share of goals
scored in the first half.

Measured across 21,919 matches: 0.4429 overall, ranging from 0.4359 (Serie A)
to 0.4512 (Primeira Liga). Markedly below half, so splitting a full-time rate
evenly would overstate first-half scoring by about 13%.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import binom

DEFAULT_FIRST_HALF_SHARE = 0.4429       # measured, all competitions pooled

RESULT_LABELS = ("H", "D", "A")


def half_time_matrix(full_time: np.ndarray,
                     share: float = DEFAULT_FIRST_HALF_SHARE) -> np.ndarray:
    """Marginal P(HT home = h1, HT away = a1) over the first half."""
    grid = full_time.shape[0]
    share = min(max(float(share), 1e-3), 1 - 1e-3)
    # splits[total, scored] = P(scored before the interval | total at full time)
    splits = np.zeros((grid, grid))
    for total in range(grid):
        splits[total, : total + 1] = binom.pmf(np.arange(total + 1), total, share)

    half = np.zeros((grid, grid))
    for home in range(grid):
        for away in range(grid):
            weight = full_time[home, away]
            if weight <= 0.0:
                continue
            half += weight * np.outer(splits[home], splits[away])
    mass = half.sum()
    return half / mass if mass > 0 else half


def _result(home: int, away: int) -> str:
    return "H" if home > away else ("D" if home == away else "A")


def half_masks(grid: int) -> dict[str, np.ndarray]:
    home, away = np.indices((grid, grid))
    total = home + away
    return {
        "HT_OVER_0_5": total > 0,
        "HT_OVER_1_5": total > 1,
        "HT_UNDER_0_5": total < 1,
        "HT_UNDER_1_5": total < 2,
        "HT_HOME_WIN": home > away,
        "HT_DRAW": home == away,
        "HT_AWAY_WIN": home < away,
        "HT_DOUBLE_CHANCE_1X": home >= away,
        "HT_DOUBLE_CHANCE_X2": home <= away,
    }


HALF_MARKETS: tuple[str, ...] = tuple(half_masks(2).keys()) + tuple(
    f"HTFT_{first}_{second}" for first in RESULT_LABELS for second in RESULT_LABELS)


@dataclass
class HalfProbabilities:
    probabilities: dict[str, float]
    half_matrix: np.ndarray
    share: float
    masks: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def joint(self, first: str, second: str) -> float | None:
        masks = self.masks or half_masks(self.half_matrix.shape[0])
        if first not in masks or second not in masks:
            return None
        return float(self.half_matrix[masks[first] & masks[second]].sum())

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


def derive_half_markets(full_time: np.ndarray,
                        share: float = DEFAULT_FIRST_HALF_SHARE) -> HalfProbabilities:
    grid = full_time.shape[0]
    half = half_time_matrix(full_time, share)
    masks = half_masks(grid)
    probabilities = {key: float(half[mask].sum()) for key, mask in masks.items()}

    axis = np.arange(grid)
    probabilities["EXPECTED_HT_GOALS"] = float(
        (half.sum(axis=1) * axis).sum() + (half.sum(axis=0) * axis).sum())

    # HT/FT needs the joint, not the two marginals: a side leading at the
    # interval is far likelier to win, so multiplying marginals would be wrong.
    share_clamped = min(max(float(share), 1e-3), 1 - 1e-3)
    splits = np.zeros((grid, grid))
    for total in range(grid):
        splits[total, : total + 1] = binom.pmf(np.arange(total + 1), total,
                                               share_clamped)

    joint = {f"HTFT_{a}_{b}": 0.0 for a in RESULT_LABELS for b in RESULT_LABELS}
    for home in range(grid):
        for away in range(grid):
            weight = full_time[home, away]
            if weight <= 0.0:
                continue
            final = _result(home, away)
            block = np.outer(splits[home][: home + 1], splits[away][: away + 1])
            for h1 in range(home + 1):
                for a1 in range(away + 1):
                    joint[f"HTFT_{_result(h1, a1)}_{final}"] += weight * block[h1, a1]
    total_mass = sum(joint.values())
    if total_mass > 0:
        joint = {k: v / total_mass for k, v in joint.items()}
    probabilities.update(joint)

    return HalfProbabilities(probabilities, half, share, masks)
