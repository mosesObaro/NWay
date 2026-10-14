"""Card markets from a single total-cards distribution.

Measured over 21,919 matches: mean 4.26 cards, variance 5.69, var/mean 1.336.
Clearly overdispersed, so negative binomial rather than Poisson -- a Poisson
fit would understate the spread exactly at the 3.5 to 5.5 lines.

Only total-cards markets are derived. A home/away split is deliberately not
offered: unlike corners, where the per-side counts are a direct consequence of
territory, card counts per side depend heavily on game state and referee
tendency, and football-data.co.uk carries referee identity for the Premier
League alone (15.8% of matches). Splitting on that evidence would be guessing
with a decimal point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nway.markets.corners import negative_binomial_pmf

CARD_GRID = 21          # 0..20 total cards; red cards count as one each


def card_masks(grid: int = CARD_GRID) -> dict[str, np.ndarray]:
    totals = np.arange(grid)
    masks: dict[str, np.ndarray] = {}
    for line in (3.5, 4.5, 5.5):
        label = str(line).replace(".", "_")
        masks[f"CARDS_OVER_{label}"] = totals > line
        masks[f"CARDS_UNDER_{label}"] = totals < line
    return masks


CARD_MARKETS: tuple[str, ...] = tuple(card_masks().keys())


def card_distribution(mean_total: float, dispersion: float,
                      grid: int = CARD_GRID) -> np.ndarray:
    totals = np.arange(grid)
    pmf = negative_binomial_pmf(totals, mean_total, dispersion)
    mass = pmf.sum()
    return pmf / mass if mass > 0 else pmf


@dataclass
class CardProbabilities:
    probabilities: dict[str, float]
    mean_total: float
    distribution: np.ndarray
    masks: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def joint(self, first: str, second: str) -> float | None:
        masks = self.masks or card_masks(len(self.distribution))
        if first not in masks or second not in masks:
            return None
        return float(self.distribution[masks[first] & masks[second]].sum())

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


def derive_card_markets(mean_total: float, dispersion: float,
                        grid: int = CARD_GRID) -> CardProbabilities:
    distribution = card_distribution(mean_total, dispersion, grid)
    masks = card_masks(grid)
    probabilities = {key: float(distribution[mask].sum())
                     for key, mask in masks.items()}
    expected = float((distribution * np.arange(grid)).sum())
    probabilities["EXPECTED_CARDS"] = expected
    return CardProbabilities(probabilities, expected, distribution, masks)
