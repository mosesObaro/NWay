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

from nway.evaluation.metrics import expected_calibration_error

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

    def fit(self, probabilities: Sequence[float],
            outcomes: Sequence[int]) -> "IsotonicCalibrator":
        p = np.asarray(probabilities, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        order = np.argsort(p, kind="mergesort")
        xs, ys = p[order], y[order]

        values = list(ys)
        weights = [1.0] * len(ys)
        positions = list(range(len(ys)))
        i = 0
        while i < len(values) - 1:
            if values[i] <= values[i + 1]:
                i += 1
                continue
            total_weight = weights[i] + weights[i + 1]
            pooled = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / total_weight
            values[i] = pooled
            weights[i] = total_weight
            del values[i + 1], weights[i + 1], positions[i + 1]
            if i > 0:
                i -= 1

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


def fit_calibrator(probabilities: Sequence[float], outcomes: Sequence[int],
                   fit_window_start: dt.datetime | None = None,
                   fit_window_end: dt.datetime | None = None,
                   isotonic_min: int = ISOTONIC_MIN_SAMPLES,
                   platt_min: int = PLATT_MIN_SAMPLES) -> CalibrationResult:
    """Choose and fit a calibrator by sample size."""
    n = len(probabilities)
    ece_before = expected_calibration_error(probabilities, outcomes, equal_count=True)

    if n >= isotonic_min:
        calibrator: Calibrator = IsotonicCalibrator().fit(probabilities, outcomes)
    elif n >= platt_min:
        calibrator = PlattCalibrator().fit(probabilities, outcomes)
    else:
        calibrator = IdentityCalibrator(n_samples=n)
        return CalibrationResult(calibrator, calibrator.method, n, ece_before,
                                 ece_before, recommendable=False,
                                 fit_window_start=fit_window_start,
                                 fit_window_end=fit_window_end)

    ece_after = expected_calibration_error(
        calibrator.transform(probabilities), outcomes, equal_count=True)

    # If recalibration made things worse on its own fitting data, something is
    # wrong; fall back rather than ship a calibrator that hurts.
    if not math.isnan(ece_after) and not math.isnan(ece_before) and ece_after > ece_before * 1.5:
        identity = IdentityCalibrator(n_samples=n)
        return CalibrationResult(identity, identity.method, n, ece_before,
                                 ece_before, recommendable=True,
                                 fit_window_start=fit_window_start,
                                 fit_window_end=fit_window_end)

    return CalibrationResult(calibrator, calibrator.method, n, ece_before, ece_after,
                             recommendable=True, fit_window_start=fit_window_start,
                             fit_window_end=fit_window_end)
