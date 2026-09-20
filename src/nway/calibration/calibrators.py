"""Probability calibration.

A model can rank matches well and still be badly calibrated, and this system's
output *is* the probability, so calibration is a first-class stage rather than
a finishing touch.

Method is chosen by sample size, not by preference: isotonic needs a lot of
data to avoid fitting steps into noise, Platt is a two-parameter fit that
survives smaller samples, and below the floor the identity calibrator is used
and the market is marked ineligible for recommendation rather than pretending
to a precision it has not earned.

Every calibrator is fitted on a window that ends before the predictions it
adjusts. Fitting on data that includes the target period is a subtle and very
common leak.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from nway.evaluation.metrics import expected_calibration_error, log_loss_binary

ISOTONIC_MIN_SAMPLES = 1000
PLATT_MIN_SAMPLES = 300


class Calibrator(Protocol):
    method: str

    def transform(self, probabilities: Sequence[float]) -> np.ndarray: ...


@dataclass
class IdentityCalibrator:
    """Pass-through, used below the sample floor.

    Its presence is a signal, not a default: a market calibrated by identity
    has not earned the right to be recommended.
    """
    method: str = "IDENTITY"
    n_samples: int = 0

    def transform(self, probabilities: Sequence[float]) -> np.ndarray:
        return np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)


@dataclass
class PlattCalibrator:
    """Logistic recalibration: p' = sigmoid(a * logit(p) + b)."""
    a: float = 1.0
    b: float = 0.0
    method: str = "PLATT"
    n_samples: int = 0

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, probabilities: Sequence[float],
            outcomes: Sequence[int]) -> "PlattCalibrator":
        from scipy.optimize import minimize

        x = self._logit(np.asarray(probabilities, dtype=float))
        y = np.asarray(outcomes, dtype=float)

        def negative_log_likelihood(theta):
            z = np.clip(theta[0] * x + theta[1], -30, 30)
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

        result = minimize(negative_log_likelihood, np.array([1.0, 0.0]),
                          method="L-BFGS-B")
        self.a, self.b = float(result.x[0]), float(result.x[1])
        self.n_samples = len(y)
        return self

    def transform(self, probabilities: Sequence[float]) -> np.ndarray:
        z = np.clip(self.a * self._logit(np.asarray(probabilities, dtype=float)) + self.b,
                    -30, 30)
        return np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)


@dataclass
class IsotonicCalibrator:
    """Isotonic regression by pool-adjacent-violators.

    Implemented here rather than pulled from scikit-learn to keep the runtime
    dependency list to numpy and scipy; PAVA is short and the behaviour at the
    boundaries matters enough to want it explicit.
    """
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    method: str = "ISOTONIC"
    n_samples: int = 0

    @staticmethod
    def _pava(values: list[float], weights: list[float]) -> tuple[list[float], list[float]]:
        """Pool adjacent violators. Returns block values and block weights."""
        values, weights = list(values), list(weights)
        i = 0
        while i < len(values) - 1:
            if values[i] <= values[i + 1]:
                i += 1
                continue
            total = weights[i] + weights[i + 1]
            values[i] = (values[i] * weights[i]
                         + values[i + 1] * weights[i + 1]) / total
            weights[i] = total
            del values[i + 1], weights[i + 1]
            if i > 0:
                i -= 1
        return values, weights

    def fit(self, probabilities: Sequence[float],
            outcomes: Sequence[int]) -> "IsotonicCalibrator":
        p = np.asarray(probabilities, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        order = np.argsort(p, kind="mergesort")
        xs, ys = p[order], y[order]

        values, weights = self._pava(list(ys), [1.0] * len(ys))

        # Laplace-smooth each block: (hits + 1) / (block size + 2).
        #
        # Raw PAVA returns each block's empirical rate, so a top block whose
        # every sample happened to be a hit yields exactly 1.0 -- the
        # calibrator asserting certainty about a football match from a handful
        # of tail observations. The correction scales with block size:
        # negligible for the large blocks mid-range, decisive for the small
        # ones in the tails, which is exactly where saturation happens.
        smoothed = [(value * weight + 1.0) / (weight + 2.0)
                    for value, weight in zip(values, weights)]

        # Smoothing shrinks small blocks toward 0.5, which can pull a small
        # high block BELOW a large one beneath it and break monotonicity -- and
        # a calibrator that reorders predictions destroys the ranking the
        # recommendation engine depends on. A second pooling pass restores the
        # ordering while keeping the shrinkage.
        values, weights = self._pava(smoothed, weights)

        fitted = np.empty(len(ys))
        cursor = 0
        for value, weight in zip(values, weights):
            span = int(round(weight))
            fitted[cursor:cursor + span] = value
            cursor += span

        self.x, self.y = xs, fitted
        self.n_samples = len(ys)
        return self

    def transform(self, probabilities: Sequence[float]) -> np.ndarray:
        if self.x.size == 0:
            return np.asarray(probabilities, dtype=float)
        return np.clip(
            np.interp(np.asarray(probabilities, dtype=float), self.x, self.y),
            1e-6, 1 - 1e-6)


@dataclass
class BetaCalibrator:
    """Beta calibration (Kull et al.).

    Kept because isotonic's step functions behave poorly near 0 and 1, which is
    exactly where Over 1.5 lives -- its base rate is already 77%, so the
    interesting region is the top of the range.
    """
    a: float = 1.0
    b: float = 1.0
    c: float = 0.0
    method: str = "BETA"
    n_samples: int = 0

    def fit(self, probabilities: Sequence[float],
            outcomes: Sequence[int]) -> "BetaCalibrator":
        from scipy.optimize import minimize

        p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(outcomes, dtype=float)
        log_p, log_1mp = np.log(p), np.log(1 - p)

        def negative_log_likelihood(theta):
            z = np.clip(theta[0] * log_p - theta[1] * log_1mp + theta[2], -30, 30)
            q = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-9, 1 - 1e-9)
            return float(-np.sum(y * np.log(q) + (1 - y) * np.log(1 - q)))

        result = minimize(negative_log_likelihood, np.array([1.0, 1.0, 0.0]),
                          method="L-BFGS-B")
        self.a, self.b, self.c = (float(result.x[0]), float(result.x[1]),
                                  float(result.x[2]))
        self.n_samples = len(y)
        return self

    def transform(self, probabilities: Sequence[float]) -> np.ndarray:
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        z = np.clip(self.a * np.log(p) - self.b * np.log(1 - p) + self.c, -30, 30)
        return np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)


@dataclass
class CalibrationResult:
    calibrator: Calibrator
    method: str
    n_samples: int
    ece_before: float
    ece_after: float
    recommendable: bool
    fit_window_start: dt.datetime | None = None
    fit_window_end: dt.datetime | None = None

    @property
    def improved(self) -> bool:
        if math.isnan(self.ece_before) or math.isnan(self.ece_after):
            return False
        return self.ece_after <= self.ece_before


def _candidates(n: int, isotonic_min: int, platt_min: int) -> list[str]:
    """Which methods are worth trying at this sample size."""
    methods = ["IDENTITY"]
    if n >= platt_min:
        methods += ["PLATT", "BETA"]
    if n >= isotonic_min:
        methods.append("ISOTONIC")
    return methods


def _build(method: str, probabilities, outcomes) -> Calibrator:
    if method == "ISOTONIC":
        return IsotonicCalibrator().fit(probabilities, outcomes)
    if method == "PLATT":
        return PlattCalibrator().fit(probabilities, outcomes)
    if method == "BETA":
        return BetaCalibrator().fit(probabilities, outcomes)
    return IdentityCalibrator(n_samples=len(probabilities))


def fit_calibrator(probabilities: Sequence[float], outcomes: Sequence[int],
                   fit_window_start: dt.datetime | None = None,
                   fit_window_end: dt.datetime | None = None,
                   isotonic_min: int = ISOTONIC_MIN_SAMPLES,
                   platt_min: int = PLATT_MIN_SAMPLES,
                   holdout_fraction: float = 0.30) -> CalibrationResult:
    """Fit every applicable method and keep the one that scores best out of sample.

    Two decisions here matter more than the mechanics.

    *The score is measured on a held-out tail, never on the fitting data.*
    Scoring a calibrator on its own training set returns an ECE near zero by
    construction -- it is fitting those exact points -- which looks like
    perfect calibration and tells you nothing. The split is temporal, matching
    how the calibrator is actually used: fitted on the past, applied to the
    future.

    *Selection is on log loss, not ECE.* ECE alone cannot distinguish a useful
    calibrator from one that has flattened every prediction to the base rate --
    a constant output can score a fine ECE while carrying no information.
    Isotonic does exactly that at the top of the range here, where the sample
    thins out. Log loss is a proper scoring rule, so it penalises both
    miscalibration and lost discrimination, which is what the recommendation
    engine needs preserved.
    """
    n = len(probabilities)
    ece_before = expected_calibration_error(probabilities, outcomes, equal_count=True)

    if n < platt_min:
        calibrator = IdentityCalibrator(n_samples=n)
        return CalibrationResult(calibrator, calibrator.method, n, ece_before,
                                 ece_before, recommendable=False,
                                 fit_window_start=fit_window_start,
                                 fit_window_end=fit_window_end)

    split = int(n * (1.0 - holdout_fraction))
    train_p, train_y = probabilities[:split], outcomes[:split]
    test_p, test_y = probabilities[split:], outcomes[split:]

    best_method, best_loss, best_ece = "IDENTITY", float("inf"), ece_before
    if split >= platt_min and len(test_p) >= 50:
        for method in _candidates(split, isotonic_min, platt_min):
            try:
                trial = _build(method, train_p, train_y)
                adjusted = trial.transform(test_p)
            except Exception:  # noqa: BLE001 - a failed fit is simply not a candidate
                continue
            loss = log_loss_binary(adjusted, test_y)
            if math.isnan(loss):
                continue
            if loss < best_loss:
                best_method = method
                best_loss = loss
                best_ece = expected_calibration_error(adjusted, test_y,
                                                      equal_count=True)

    # The shipped calibrator is refitted on the whole window; only its METHOD
    # and its SCORE came from the holdout.
    calibrator = _build(best_method, probabilities, outcomes)
    return CalibrationResult(
        calibrator, calibrator.method, n, ece_before, best_ece,
        recommendable=True, fit_window_start=fit_window_start,
        fit_window_end=fit_window_end)
