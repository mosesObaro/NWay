"""The expanded market set: coherence, settlement and grid construction."""

from __future__ import annotations

import numpy as np
import pytest

from nway.markets import (
    ALL_DERIVED_MARKETS, CoherenceError, check_coherence, corner_matrix,
    derive_card_markets, derive_corner_markets, derive_half_markets,
    derive_markets,
)
from nway.markets.corners import beta_binomial_pmf, negative_binomial_pmf
from nway.validation.resolvers import MatchFacts, RESOLVERS, settle


def bundle(lam_home=1.8, lam_away=1.1, rho=-0.04, share=0.4429,
           corner_total=9.8, corner_disp=91.0, corner_share=0.548,
           corner_conc=15.3, cards=4.0, card_disp=36.0):
    goals = derive_markets(lam_home, lam_away, rho)
    merged = dict(goals.probabilities)
    merged.update(derive_half_markets(goals.matrix, share).probabilities)
    merged.update(derive_corner_markets(corner_total, corner_disp,
                                        corner_share, corner_conc).probabilities)
    merged.update(derive_card_markets(cards, card_disp).probabilities)

    class _Merged:
        probabilities = merged
        matrix = goals.matrix
        lambda_home = lam_home
        lambda_away = lam_away
    return _Merged()


# ----------------------------------------------------------- coherence
@pytest.mark.parametrize("lam_home,lam_away", [
    (0.4, 0.3), (1.2, 1.2), (3.4, 0.4), (0.3, 3.1), (2.2, 2.4),
])
def test_every_grid_coheres_across_the_range(lam_home, lam_away):
    check_coherence(bundle(lam_home, lam_away))


def test_half_markets_cannot_exceed_full_time():
    """A first half cannot hold more goals than the match."""
    merged = bundle().probabilities
    assert merged["HT_OVER_0_5"] <= merged["OVER_0_5"] + 1e-9
    assert merged["HT_OVER_1_5"] <= merged["OVER_1_5"] + 1e-9


def test_team_totals_cannot_exceed_match_totals():
    merged = bundle().probabilities
    assert merged["HOME_OVER_1_5"] <= merged["OVER_1_5"] + 1e-9
    assert merged["AWAY_OVER_2_5"] <= merged["OVER_2_5"] + 1e-9


def test_three_way_handicaps_partition():
    merged = bundle().probabilities
    for prefix in ("EH_MINUS_1", "EH_PLUS_1"):
        total = sum(merged[f"{prefix}_{side}"] for side in ("HOME", "DRAW", "AWAY"))
        assert total == pytest.approx(1.0, abs=1e-9)


def test_asian_handicap_pairs_are_complements():
    """Half lines cannot push, so each pair must sum to exactly one."""
    merged = bundle().probabilities
    assert merged["AH_HOME_MINUS_1_5"] + merged["AH_AWAY_PLUS_1_5"] == \
        pytest.approx(1.0, abs=1e-9)
    assert merged["CORNERS_AH_HOME_MINUS_1_5"] + \
        merged["CORNERS_AH_AWAY_PLUS_1_5"] == pytest.approx(1.0, abs=1e-9)


def test_combinations_cannot_exceed_their_components():
    merged = bundle().probabilities
    assert merged["HOME_AND_BTTS"] <= min(merged["HOME_WIN"], merged["BTTS"]) + 1e-9
    assert merged["HOME_WIN_TO_NIL"] <= min(
        merged["HOME_WIN"], merged["HOME_CLEAN_SHEET"]) + 1e-9


def test_ht_ft_partitions_the_match():
    merged = bundle().probabilities
    total = sum(v for k, v in merged.items() if k.startswith("HTFT_"))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_corner_ladders_are_monotone():
    merged = bundle().probabilities
    overs = [merged[f"CORNERS_OVER_{l}"] for l in ("8_5", "9_5", "10_5", "11_5")]
    assert overs == sorted(overs, reverse=True)
    home = [merged[f"HOME_CORNERS_OVER_{l}"] for l in ("3_5", "4_5", "5_5")]
    assert home == sorted(home, reverse=True)


def test_card_ladder_is_monotone():
    merged = bundle().probabilities
    overs = [merged[f"CARDS_OVER_{l}"] for l in ("3_5", "4_5", "5_5")]
    assert overs == sorted(overs, reverse=True)


def test_tampering_is_caught():
    broken = bundle()
    broken.probabilities["CORNERS_OVER_11_5"] = \
        broken.probabilities["CORNERS_OVER_8_5"] + 0.1
    with pytest.raises(CoherenceError):
        check_coherence(broken)


# ------------------------------------------------------- grid mechanics
def test_half_split_preserves_the_full_time_marginal_exactly():
    """The reason halves are a split of the goal grid, not a second model.

    Two fitted grids would disagree, and P(1H Over 1.5) could exceed
    P(FT Over 1.5).
    """
    goals = derive_markets(2.1, 1.3, -0.05)
    half = derive_half_markets(goals.matrix, 0.4429)
    assert half.half_matrix.sum() == pytest.approx(1.0, abs=1e-12)
    # Expected half-time goals must be the share of expected full-time goals.
    expected_ft = goals.probabilities["EXPECTED_GOALS"]
    assert half.probabilities["EXPECTED_HT_GOALS"] == \
        pytest.approx(0.4429 * expected_ft, rel=1e-6)


def test_corner_grid_induces_negative_correlation():
    """Measured home/away corner correlation is -0.276.

    An independent product cannot produce that: it forces zero covariance and
    overstates total variance by 38% (16.09 against a measured 11.68), exactly
    at the 8.5 to 11.5 lines. Conditioning the split on the total induces the
    negative covariance by construction.
    """
    matrix = corner_matrix(9.8, 91.0, 0.548, 15.3)
    axis = np.arange(matrix.shape[0])
    mean_home = float((matrix.sum(1) * axis).sum())
    mean_away = float((matrix.sum(0) * axis).sum())
    var_home = float((matrix.sum(1) * axis ** 2).sum() - mean_home ** 2)
    var_away = float((matrix.sum(0) * axis ** 2).sum() - mean_away ** 2)
    totals = np.add.outer(axis, axis)
    mean_total = float((matrix * totals).sum())
    var_total = float((matrix * totals ** 2).sum() - mean_total ** 2)
    covariance = 0.5 * (var_total - var_home - var_away)
    assert covariance < 0, "the corner grid did not induce negative covariance"
    correlation = covariance / (var_home * var_away) ** 0.5
    assert -0.45 < correlation < -0.10


def test_corner_total_marginal_matches_the_fitted_negative_binomial():
    """Total over/under are the primary corner markets, so their marginal is
    calibrated by construction rather than by tuning."""
    mean, dispersion = 9.8, 91.0
    matrix = corner_matrix(mean, dispersion, 0.548, 15.3)
    axis = np.arange(matrix.shape[0])
    totals = np.add.outer(axis, axis)
    grid_total = float((matrix * totals).sum())
    assert grid_total == pytest.approx(mean, rel=0.01)


def test_negative_binomial_reduces_to_poisson_at_high_dispersion():
    from scipy.stats import poisson

    k = np.arange(20)
    nb = negative_binomial_pmf(k, 4.0, 1e6)
    assert np.allclose(nb, poisson.pmf(k, 4.0), atol=1e-4)


def test_beta_binomial_reduces_to_binomial_at_high_concentration():
    from scipy.stats import binom

    k = np.arange(11)
    bb = beta_binomial_pmf(k, 10, 0.5 * 1e6, 0.5 * 1e6)
    assert np.allclose(bb, binom.pmf(k, 10, 0.5), atol=1e-4)


def test_corner_grid_normalises_despite_truncation():
    """The 26x26 grid drops P(either side > 25), measured at 9.1e-5."""
    matrix = corner_matrix(14.0, 50.0, 0.55, 12.0)
    assert matrix.sum() == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------- settlement
def test_every_derived_market_has_a_resolver():
    missing = sorted(set(ALL_DERIVED_MARKETS) - set(RESOLVERS))
    assert not missing, f"derivable but unsettleable: {missing}"


def test_no_resolver_lacks_a_derivation():
    extra = sorted(set(RESOLVERS) - set(ALL_DERIVED_MARKETS))
    assert not extra, f"settleable but never derived: {extra}"


FACTS = MatchFacts(home_goals=3, away_goals=1, ht_home_goals=1, ht_away_goals=1,
                   home_corners=7, away_corners=4, total_cards=5)


@pytest.mark.parametrize("market,expected", [
    ("HOME_WIN", True), ("AH_HOME_MINUS_1_5", True), ("AH_AWAY_PLUS_1_5", False),
    ("EH_MINUS_1_HOME", True), ("EH_MINUS_1_DRAW", False), ("EH_MINUS_1_AWAY", False),
    ("EH_PLUS_1_HOME", True), ("HOME_OVER_2_5", True), ("HOME_OVER_1_5", True),
    ("AWAY_OVER_1_5", False), ("HOME_WIN_TO_NIL", False),
    ("HOME_AND_BTTS", True), ("HOME_AND_OVER_2_5", True), ("DRAW_AND_BTTS", False),
    ("HT_OVER_0_5", True), ("HT_OVER_1_5", True), ("HT_DRAW", True),
    ("HT_HOME_WIN", False), ("HTFT_D_H", True), ("HTFT_H_H", False),
    ("CORNERS_OVER_8_5", True), ("CORNERS_OVER_11_5", False),
    ("HOME_CORNERS_OVER_5_5", True), ("AWAY_CORNERS_OVER_4_5", False),
    ("CORNERS_1X2_HOME", True), ("CORNERS_AH_HOME_MINUS_1_5", True),
    ("CARDS_OVER_3_5", True), ("CARDS_OVER_5_5", False), ("CARDS_UNDER_5_5", True),
])
def test_settlement_truth_table(market, expected):
    assert settle(market, FACTS) is expected


def test_boundary_lines_settle_on_the_right_side():
    """Exactly on the line is the case that gets written wrong."""
    exact = MatchFacts(home_goals=1, away_goals=1, ht_home_goals=0,
                       ht_away_goals=0, home_corners=5, away_corners=4,
                       total_cards=4)
    assert settle("CORNERS_OVER_8_5", exact) is True      # 9 corners
    assert settle("CORNERS_OVER_9_5", exact) is False
    assert settle("CARDS_OVER_3_5", exact) is True        # 4 cards
    assert settle("CARDS_OVER_4_5", exact) is False
    assert settle("CARDS_UNDER_4_5", exact) is True
    assert settle("HT_OVER_0_5", exact) is False


def test_missing_facts_are_unsettleable_not_a_loss():
    """A data gap is not a wrong prediction; conflating them flatters the model."""
    goals_only = MatchFacts(home_goals=2, away_goals=0)
    assert settle("HOME_WIN", goals_only) is True
    assert settle("CORNERS_OVER_8_5", goals_only) is None
    assert settle("CARDS_OVER_3_5", goals_only) is None
    assert settle("HT_OVER_0_5", goals_only) is None


def test_whole_number_handicaps_never_push():
    """Each 3-way trio has exactly one winner for any scoreline, so every
    outcome is strictly Bernoulli -- which the Beta posterior ranker needs."""
    for home in range(6):
        for away in range(6):
            facts = MatchFacts(home_goals=home, away_goals=away)
            for prefix in ("EH_MINUS_1", "EH_PLUS_1"):
                wins = [settle(f"{prefix}_{s}", facts)
                        for s in ("HOME", "DRAW", "AWAY")]
                assert sum(bool(w) for w in wins) == 1, \
                    f"{prefix} at {home}-{away} did not resolve to exactly one"
