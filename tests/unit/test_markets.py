import math

import pytest

from nway.markets import (
    CoherenceError, check_coherence, derive_markets, dixon_coles_tau, goal_matrix,
)


def test_matrix_normalises():
    matrix = goal_matrix(1.8, 1.1)
    assert matrix.sum() == pytest.approx(1.0, abs=1e-9)


def test_tau_touches_only_low_scores():
    # Dixon-Coles corrects 0-0, 1-0, 0-1 and 1-1 and nothing else.
    for home, away in ((0, 0), (0, 1), (1, 0), (1, 1)):
        assert dixon_coles_tau(home, away, 1.5, 1.2, -0.05) != 1.0
    for home, away in ((2, 0), (0, 2), (2, 2), (3, 1)):
        assert dixon_coles_tau(home, away, 1.5, 1.2, -0.05) == 1.0


def test_independent_poisson_matches_hand_computation():
    markets = derive_markets(1.5, 1.0, rho=0.0)
    # P(0-0) under independence is exp(-(1.5+1.0))
    assert markets.matrix[0, 0] == pytest.approx(math.exp(-2.5), rel=1e-6)
    assert markets.probabilities["EXPECTED_GOALS"] == pytest.approx(2.5, abs=0.01)


@pytest.mark.parametrize("lam_home,lam_away,rho", [
    (0.5, 0.4, 0.0), (1.5, 1.2, -0.05), (3.2, 0.3, -0.10),
    (2.1, 2.4, 0.05), (0.2, 4.0, -0.02),
])
def test_coherence_holds_everywhere(lam_home, lam_away, rho):
    check_coherence(derive_markets(lam_home, lam_away, rho))


def test_over_lines_are_monotone():
    markets = derive_markets(2.0, 1.4, -0.04).probabilities
    assert (markets["OVER_0_5"] >= markets["OVER_1_5"]
            >= markets["OVER_2_5"] >= markets["OVER_3_5"])


def test_btts_cannot_exceed_over_1_5():
    # Both teams scoring implies at least two goals, so this is structural.
    for lam_home, lam_away in ((1.0, 1.0), (2.5, 2.5), (0.4, 3.0)):
        markets = derive_markets(lam_home, lam_away).probabilities
        assert markets["BTTS"] <= markets["OVER_1_5"] + 1e-12


def test_double_chance_equals_component_sum():
    markets = derive_markets(1.7, 1.3, -0.03).probabilities
    assert markets["DOUBLE_CHANCE_1X"] == pytest.approx(
        markets["HOME_WIN"] + markets["DRAW"], abs=1e-9)
    assert markets["DOUBLE_CHANCE_X2"] == pytest.approx(
        markets["AWAY_WIN"] + markets["DRAW"], abs=1e-9)


def test_one_x_two_sums_to_one():
    markets = derive_markets(1.9, 0.8, -0.06).probabilities
    assert (markets["HOME_WIN"] + markets["DRAW"]
            + markets["AWAY_WIN"]) == pytest.approx(1.0, abs=1e-9)


def test_correlation_is_exact_not_assumed():
    markets = derive_markets(1.8, 1.2, -0.04)
    # Over 1.5 and Over 2.5 come from the same matrix, so they are strongly
    # dependent -- this is what stops them being presented as two opportunities.
    assert markets.correlation("OVER_1_5", "OVER_2_5") > 0.4
    # A perfectly nested pair: joint == the narrower probability.
    joint = markets.joint("OVER_1_5", "OVER_2_5")
    assert joint == pytest.approx(markets.probabilities["OVER_2_5"], abs=1e-9)


def test_coherence_error_is_raised_on_tampering():
    markets = derive_markets(1.5, 1.2)
    markets.probabilities["OVER_2_5"] = markets.probabilities["OVER_1_5"] + 0.1
    with pytest.raises(CoherenceError):
        check_coherence(markets)


def test_clean_sheet_marginals():
    markets = derive_markets(1.5, 1.0)
    # Away clean sheet means the home side failed to score.
    assert markets.probabilities["HOME_CLEAN_SHEET"] == pytest.approx(
        math.exp(-1.0), rel=1e-6)
    assert markets.probabilities["AWAY_CLEAN_SHEET"] == pytest.approx(
        math.exp(-1.5), rel=1e-6)
