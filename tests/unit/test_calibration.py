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


def test_method_is_chosen_by_sample_size():
    """Isotonic needs data; below the floor the market is not recommendable."""
    predicted, outcomes = miscalibrated(n=5000)
    assert fit_calibrator(predicted, outcomes).method == "ISOTONIC"

    predicted, outcomes = miscalibrated(n=500)
    assert fit_calibrator(predicted, outcomes).method == "PLATT"

    predicted, outcomes = miscalibrated(n=100)
    result = fit_calibrator(predicted, outcomes)
    assert result.method == "IDENTITY"
    assert result.recommendable is False, \
        "a market below the sample floor must not be recommendable"


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
