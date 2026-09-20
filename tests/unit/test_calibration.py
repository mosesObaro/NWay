"""Calibration behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from nway.calibration.calibrators import (
    IdentityCalibrator, IsotonicCalibrator, PlattCalibrator, fit_calibrator,
)
from nway.evaluation.metrics import expected_calibration_error

RNG = np.random.default_rng(20260920)


def miscalibrated(n=4000, shift=0.15):
    """Predictions that are systematically over-confident by `shift`."""
    truth = RNG.uniform(0.15, 0.85, n)
    outcomes = (RNG.uniform(0, 1, n) < truth).astype(int)
    predicted = np.clip(truth + shift, 0.01, 0.99)
    return predicted, outcomes


def test_isotonic_corrects_a_systematic_shift():
    predicted, outcomes = miscalibrated()
    before = expected_calibration_error(predicted, outcomes, equal_count=True)
    calibrator = IsotonicCalibrator().fit(predicted, outcomes)
    after = expected_calibration_error(
        calibrator.transform(predicted), outcomes, equal_count=True)
    assert before > 0.10
    assert after < before / 2


def test_platt_corrects_a_systematic_shift():
    predicted, outcomes = miscalibrated(n=800)
    before = expected_calibration_error(predicted, outcomes, equal_count=True)
    calibrator = PlattCalibrator().fit(predicted, outcomes)
    after = expected_calibration_error(
        calibrator.transform(predicted), outcomes, equal_count=True)
    assert after < before


def test_below_the_sample_floor_nothing_is_fitted():
    """Too little data to calibrate means the market is not recommendable."""
    predicted, outcomes = miscalibrated(n=100)
    result = fit_calibrator(predicted, outcomes)
    assert result.method == "IDENTITY"
    assert result.recommendable is False


def test_method_is_chosen_by_held_out_performance():
    """Selection competes every applicable method rather than picking by size.

    A fixed size rule shipped isotonic for the goal markets, where it collapsed
    everything above raw 0.85 to a single value.
    """
    predicted, outcomes = miscalibrated(n=5000)
    result = fit_calibrator(predicted, outcomes)
    assert result.method in ("ISOTONIC", "PLATT", "BETA", "IDENTITY")
    # Whatever wins must actually help on data it did not see.
    assert result.ece_after <= result.ece_before + 0.02


def test_calibration_never_claims_certainty():
    """A displayed "100%" would contradict the uncertainty notice the email
    carries, and no football outcome is certain."""
    # An extreme case: every high prediction in the fitting data came in.
    truth = np.concatenate([RNG.uniform(0.2, 0.6, 3000),
                            RNG.uniform(0.93, 0.99, 600)])
    outcomes = np.concatenate([
        (RNG.uniform(0, 1, 3000) < truth[:3000]).astype(int),
        np.ones(600, dtype=int),          # every tail sample a hit
    ])
    result = fit_calibrator(truth, outcomes)
    transformed = result.calibrator.transform(
        [0.90, 0.95, 0.97, 0.99, 0.995, 0.999])
    assert transformed.max() < 1.0, "the calibrator asserted certainty"
    assert transformed.min() > 0.0


def test_calibration_preserves_discrimination():
    """A calibrator that flattens the top of the range is useless to ranking.

    Isotonic did exactly this: raw 0.85 through 0.99 all mapped to 0.832, so a
    genuinely 99% match and an 85% one became indistinguishable.
    """
    predicted, outcomes = miscalibrated(n=5000)
    result = fit_calibrator(predicted, outcomes)
    grid = np.array([0.70, 0.78, 0.85, 0.90, 0.95, 0.99])
    out = result.calibrator.transform(grid)
    distinct = len(set(np.round(out, 4)))
    assert distinct >= 4, (
        f"only {distinct} distinct outputs from 6 distinct inputs; "
        f"the calibrator has flattened the range")


def test_reported_ece_is_measured_out_of_sample():
    """Scoring a calibrator on its own fitting data returns ~0 by construction.

    Pure noise cannot be calibrated, so an honest score must NOT be near zero.
    """
    noise = RNG.uniform(0.05, 0.95, 3000)
    coin = (RNG.uniform(0, 1, 3000) < 0.5).astype(int)
    result = fit_calibrator(noise, coin)
    assert result.ece_after > 0.005, (
        "an ECE this low on unpredictable data means the calibrator was "
        "scored on the points it was fitted to")


def test_identity_calibrator_is_a_true_passthrough():
    values = [0.1, 0.5, 0.9]
    assert IdentityCalibrator().transform(values) == pytest.approx(values, abs=1e-6)


def test_isotonic_is_monotone():
    """Calibration may not reorder predictions; that would destroy ranking."""
    predicted, outcomes = miscalibrated()
    calibrator = IsotonicCalibrator().fit(predicted, outcomes)
    grid = np.linspace(0.02, 0.98, 60)
    transformed = calibrator.transform(grid)
    assert np.all(np.diff(transformed) >= -1e-9)


def test_output_stays_inside_the_unit_interval():
    predicted, outcomes = miscalibrated()
    for calibrator in (IsotonicCalibrator().fit(predicted, outcomes),
                       PlattCalibrator().fit(predicted, outcomes)):
        values = calibrator.transform([0.0, 0.001, 0.5, 0.999, 1.0])
        assert np.all((values > 0) & (values < 1))


def test_already_calibrated_input_is_left_roughly_alone():
    truth = RNG.uniform(0.2, 0.8, 4000)
    outcomes = (RNG.uniform(0, 1, 4000) < truth).astype(int)
    result = fit_calibrator(truth, outcomes)
    transformed = result.calibrator.transform(truth)
    assert np.mean(np.abs(transformed - truth)) < 0.06


def test_calibrator_records_its_fit_window():
    """A calibrator fitted on the target period is a leak; the window is stored
    so that it can be checked."""
    import datetime as dt

    from nway import clock

    predicted, outcomes = miscalibrated(n=2000)
    start = dt.datetime(2025, 1, 1, tzinfo=clock.UTC)
    end = dt.datetime(2025, 12, 31, tzinfo=clock.UTC)
    result = fit_calibrator(predicted, outcomes, start, end)
    assert result.fit_window_start == start
    assert result.fit_window_end == end
