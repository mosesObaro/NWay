"""Evaluation metrics.

Raw accuracy is deliberately absent from every headline. For a market with a
77% base rate, always answering "yes" scores 77% accuracy and is worthless.
Log loss, Brier and calibration are what distinguish a probability estimate
from a guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

EPSILON = 1e-12


def log_loss_binary(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    if not len(probabilities):
        return float("nan")
    p = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)
    y = np.asarray(outcomes, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def log_loss_multiclass(probabilities: Sequence[Sequence[float]],
                        outcome_indices: Sequence[int]) -> float:
    if not len(probabilities):
        return float("nan")
    p = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1.0)
    rows = np.arange(len(outcome_indices))
    return float(-np.mean(np.log(p[rows, np.asarray(outcome_indices)])))


def brier_binary(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    if not len(probabilities):
        return float("nan")
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def brier_multiclass(probabilities: Sequence[Sequence[float]],
                     outcome_indices: Sequence[int]) -> float:
    if not len(probabilities):
        return float("nan")
    p = np.asarray(probabilities, dtype=float)
    actual = np.zeros_like(p)
    actual[np.arange(len(outcome_indices)), np.asarray(outcome_indices)] = 1.0
    return float(np.mean(np.sum((p - actual) ** 2, axis=1)))


@dataclass
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_predicted: float
    actual_rate: float

    @property
    def gap(self) -> float:
        return self.actual_rate - self.mean_predicted


def calibration_bins(probabilities: Sequence[float], outcomes: Sequence[int],
                     n_bins: int = 10, equal_count: bool = False) -> list[CalibrationBin]:
    """Reliability diagram data.

    ``equal_count`` uses quantile bins, which is the honest choice when
    predictions cluster (they do: most goal-market probabilities sit between
    0.4 and 0.8, leaving fixed-width extreme bins nearly empty).
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.size == 0:
        return []
    if equal_count:
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    for lower, upper in zip(edges, edges[1:]):
        mask = (p >= lower) & (p < upper)
        if upper == edges[-1]:
            mask = (p >= lower) & (p <= upper)
        if not mask.any():
            continue
        bins.append(CalibrationBin(
            lower=float(lower), upper=float(upper), count=int(mask.sum()),
            mean_predicted=float(p[mask].mean()), actual_rate=float(y[mask].mean())))
    return bins


def expected_calibration_error(probabilities: Sequence[float], outcomes: Sequence[int],
                               n_bins: int = 10, equal_count: bool = False) -> float:
    bins = calibration_bins(probabilities, outcomes, n_bins, equal_count)
    total = sum(b.count for b in bins)
    if not total:
        return float("nan")
    return float(sum(b.count / total * abs(b.gap) for b in bins))


def maximum_calibration_error(probabilities: Sequence[float], outcomes: Sequence[int],
                              n_bins: int = 10) -> float:
    bins = calibration_bins(probabilities, outcomes, n_bins)
    return max((abs(b.gap) for b in bins), default=float("nan"))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval.

    Used everywhere a hit rate is reported. A 500-prediction market at an 80%
    hit rate has an interval about +/-3.5 points wide, which is most of the
    reason small differences between seasons mean nothing.
    """
    if total <= 0:
        return (0.0, 1.0)
    phat = successes / total
    denominator = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    spread = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))
    return (max(0.0, (centre - spread) / denominator),
            min(1.0, (centre + spread) / denominator))


def roc_auc(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """AUC via the rank-sum identity; ties get average ranks."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    positives, negatives = int(y.sum()), int((1 - y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1, dtype=float)
    sorted_p = p[order]
    start = 0
    for i in range(1, len(sorted_p) + 1):
        if i == len(sorted_p) or sorted_p[i] != sorted_p[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def mae(predicted: Sequence[float], actual: Sequence[float]) -> float:
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual))))


def rmse(predicted: Sequence[float], actual: Sequence[float]) -> float:
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(actual)) ** 2)))


def poisson_deviance(predicted: Sequence[float], actual: Sequence[float]) -> float:
    mu = np.clip(np.asarray(predicted, dtype=float), EPSILON, None)
    y = np.asarray(actual, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * np.mean(term - (y - mu)))


@dataclass
class MarketReport:
    market_key: str
    n: int
    mean_predicted: float
    actual_rate: float
    log_loss: float
    brier: float
    ece: float
    auc: float
    ci_low: float
    ci_high: float

    @property
    def calibrated(self) -> bool:
        """Is the observed rate inside the interval for the predicted rate?"""
        return self.ci_low <= self.mean_predicted <= self.ci_high

    def summary(self) -> str:
        verdict = "within interval" if self.calibrated else "OUTSIDE interval"
        return (f"{self.market_key:<20} n={self.n:<6} pred={self.mean_predicted:.3f} "
                f"actual={self.actual_rate:.3f} [{self.ci_low:.3f},{self.ci_high:.3f}] "
                f"ll={self.log_loss:.4f} brier={self.brier:.4f} ece={self.ece:.4f} "
                f"-> {verdict}")


def market_report(market_key: str, probabilities: Sequence[float],
                  outcomes: Sequence[int]) -> MarketReport:
    n = len(probabilities)
    successes = int(np.sum(outcomes))
    low, high = wilson_interval(successes, n)
    return MarketReport(
        market_key=market_key, n=n,
        mean_predicted=float(np.mean(probabilities)) if n else float("nan"),
        actual_rate=successes / n if n else float("nan"),
        log_loss=log_loss_binary(probabilities, outcomes),
        brier=brier_binary(probabilities, outcomes),
        ece=expected_calibration_error(probabilities, outcomes, equal_count=True),
        auc=roc_auc(probabilities, outcomes),
        ci_low=low, ci_high=high,
    )
