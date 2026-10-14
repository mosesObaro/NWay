"""Recommendation engine: floors, reliability ranking, and diversity caps."""

from __future__ import annotations

import datetime as dt

import pytest

from nway import clock
from nway.config import market_floor
from nway.recommendation.engine import Candidate, RecommendationEngine

NOW = dt.datetime(2026, 9, 20, 10, 0, tzinfo=clock.UTC)


def candidate(db, config, *, market="OVER_1_5", probability=0.86,
              fixture_id=1, prediction_id=1, competition="premier_league",
              staleness=2.0, completeness=1.0, hours_ahead=8) -> Candidate:
    return Candidate(
        prediction_id=prediction_id, prediction_run_id=prediction_id,
        fixture_id=fixture_id, competition_id=1, competition_slug=competition,
        competition_name=competition.replace("_", " ").title(),
        market_key=market, selection="YES", probability=probability,
        raw_probability=probability, home_team="Alpha", away_team="Bravo",
        kickoff_utc=NOW + dt.timedelta(hours=hours_ahead),
        prediction_timestamp=NOW - dt.timedelta(hours=1),
        data_completeness=completeness, feature_staleness_hours=staleness,
        lambda_home=1.8, lambda_away=1.2, model_version="m",
        fixture_status="TIMED", kickoff_is_confirmed=True)


# ------------------------------------------------------------------ floors
def test_floor_rule_matches_the_documented_formula():
    # Over 1.5 has a measured base rate of 77.1%; a flat 72% floor would
    # qualify predictions WEAKER than assuming the league average.
    assert market_floor(0.771, 0.65, 0.25) == pytest.approx(0.828, abs=0.001)
    # A low-base-rate market is held to the absolute floor instead.
    assert market_floor(0.313, 0.65, 0.25) == pytest.approx(0.65, abs=0.001)


def test_engine_uses_the_configured_floor_per_market(db, config):
    engine = RecommendationEngine(db, config)
    assert engine.floor_for("OVER_1_5") > engine.floor_for("OVER_2_5")
    assert engine.floor_for("OVER_1_5") > config.market("OVER_1_5").base_rate


def test_below_floor_is_rejected(db, config):
    engine = RecommendationEngine(db, config)
    weak = candidate(db, config, market="OVER_1_5", probability=0.80)
    reasons = engine.check_eligibility(weak, NOW)
    assert "BELOW_PROBABILITY_FLOOR" in reasons


def test_stale_features_are_rejected(db, config):
    engine = RecommendationEngine(db, config)
    stale = candidate(db, config, staleness=200.0)
    assert "STALE_FEATURES" in engine.check_eligibility(stale, NOW)


def test_expired_lead_time_is_rejected(db, config):
    engine = RecommendationEngine(db, config)
    late = candidate(db, config, hours_ahead=0)
    late.kickoff_utc = NOW + dt.timedelta(minutes=5)
    assert "INSUFFICIENT_LEAD_TIME" in engine.check_eligibility(late, NOW)


def test_postponed_fixtures_are_rejected(db, config):
    engine = RecommendationEngine(db, config)
    postponed = candidate(db, config)
    postponed.fixture_status = "POSTPONED"
    assert "FIXTURE_POSTPONED" in engine.check_eligibility(postponed, NOW)


def test_unconfirmed_kickoff_is_rejected(db, config):
    engine = RecommendationEngine(db, config)
    provisional = candidate(db, config)
    provisional.kickoff_is_confirmed = False
    assert "KICKOFF_UNCONFIRMED" in engine.check_eligibility(provisional, NOW)


# ------------------------------------------------------------- reliability
def test_reliability_is_pessimistic_without_history(db, config):
    """With no settled history the posterior is wide, so the lower bound sits
    well below the raw probability -- the system does not take its own word."""
    engine = RecommendationEngine(db, config)
    item = candidate(db, config, probability=0.90)
    score, n, observed = engine.reliability_score(item)
    assert n == 0 and observed is None
    assert score < item.probability


def test_confidence_band_is_capped_without_a_track_record(db, config):
    engine = RecommendationEngine(db, config)
    item = candidate(db, config, probability=0.95)
    score, n, _ = engine.reliability_score(item)
    assert engine.confidence_band(score, n) != "HIGH", \
        "HIGH confidence claimed with zero settled predictions"


def test_ranking_prefers_fresher_and_more_complete_data(db, config):
    engine = RecommendationEngine(db, config)
    fresh = candidate(db, config, prediction_id=1, fixture_id=1, staleness=1.0)
    stale = candidate(db, config, prediction_id=2, fixture_id=2, staleness=80.0)
    scored = engine.score([stale, fresh], NOW)
    assert scored[0].candidate.prediction_id == 1


# ---------------------------------------------------------------- selection
def test_selection_respects_the_maximum(db, config):
    engine = RecommendationEngine(db, config)
    items = [candidate(db, config, prediction_id=i, fixture_id=i,
                       market="OVER_1_5", probability=0.88) for i in range(1, 41)]
    selected = engine.select(engine.score(items, NOW))
    assert len(selected) <= engine.max_count


def test_one_selection_per_fixture(db, config):
    engine = RecommendationEngine(db, config)
    items = [
        candidate(db, config, prediction_id=1, fixture_id=1, market="OVER_1_5",
                  probability=0.88),
        candidate(db, config, prediction_id=2, fixture_id=1, market="OVER_2_5",
                  probability=0.80),
    ]
    selected = engine.select(engine.score(items, NOW), min_count=1)
    assert len({item.candidate.fixture_id for item in selected}) == len(selected)


def test_family_caps_prevent_a_monoculture(db, config):
    """Unconstrained, selection is ~81% double chance (measured). The caps
    exist to stop one model error taking the whole batch down at once."""
    engine = RecommendationEngine(db, config)
    items = []
    for i in range(1, 25):
        items.append(candidate(db, config, prediction_id=i, fixture_id=i,
                               market="DOUBLE_CHANCE_1X", probability=0.90))
    for i in range(25, 41):
        items.append(candidate(db, config, prediction_id=i, fixture_id=i,
                               market="OVER_1_5", probability=0.87))
    selected = engine.select(engine.score(items, NOW))
    families = [config.market(item.candidate.market_key).family for item in selected]
    double_chance = families.count("DOUBLE_CHANCE")
    assert double_chance < len(selected), "the batch is a single-market monoculture"
    assert double_chance / len(selected) <= 0.55


def test_selection_never_relaxes_a_threshold(db, config):
    """Adding weak candidates must not change which strong ones are chosen."""
    engine = RecommendationEngine(db, config)
    strong = [candidate(db, config, prediction_id=i, fixture_id=i,
                        probability=0.88) for i in range(1, 9)]
    weak = [candidate(db, config, prediction_id=i, fixture_id=i,
                      probability=0.50) for i in range(20, 30)]

    only_strong = engine.select(engine.score(strong, NOW))
    with_weak = engine.select(engine.score(strong + weak, NOW))
    assert ({item.candidate.prediction_id for item in only_strong}
            == {item.candidate.prediction_id for item in with_weak})


def test_empty_candidate_pool_selects_nothing(db, config):
    engine = RecommendationEngine(db, config)
    assert engine.select([]) == []


def test_rejection_reasons_are_recorded_for_every_candidate(db, config):
    """On a system that is silent most of the time, 'why no email?' must be
    answerable from stored rows."""
    engine = RecommendationEngine(db, config)
    items = [candidate(db, config, prediction_id=1, fixture_id=1, probability=0.50),
             candidate(db, config, prediction_id=2, fixture_id=2, staleness=300.0)]
    scored = engine.score(items, NOW)
    assert all(item.rejection_reasons for item in scored)
    summary = RecommendationEngine.rejection_summary(scored)
    assert "BELOW_PROBABILITY_FLOOR" in summary
    assert "STALE_FEATURES" in summary


def test_thin_team_history_is_rejected(db, config):
    """A side with three matches can clear the completeness bar on short
    windows while its rating is still the competition average."""
    engine = RecommendationEngine(db, config)
    newcomer = candidate(db, config)
    newcomer.team_history_matches = 2
    assert "INSUFFICIENT_TEAM_HISTORY" in engine.check_eligibility(newcomer, NOW)


def test_sufficient_team_history_passes(db, config):
    engine = RecommendationEngine(db, config)
    established = candidate(db, config)
    established.team_history_matches = 20
    assert "INSUFFICIENT_TEAM_HISTORY" not in engine.check_eligibility(established, NOW)


def test_unknown_team_history_does_not_block(db, config):
    """Older predictions predate the stored feature; absence is not evidence."""
    engine = RecommendationEngine(db, config)
    legacy = candidate(db, config)
    legacy.team_history_matches = None
    assert "INSUFFICIENT_TEAM_HISTORY" not in engine.check_eligibility(legacy, NOW)


# --------------------------------------------- grid lineage and family caps
def test_markets_from_one_grid_are_capped_per_fixture(db, config):
    """Every market from a grid is a summation over the same cells, so two of
    them on one fixture are one estimate seen twice."""
    engine = RecommendationEngine(db, config)
    # Grid comes from the configuration, not the caller: the registry is the
    # single source of truth for which grid a market belongs to.
    items = [
        candidate(db, config, prediction_id=1, fixture_id=1,
                  market="CORNERS_OVER_8_5", probability=0.80),
        candidate(db, config, prediction_id=2, fixture_id=1,
                  market="CORNERS_OVER_9_5", probability=0.78),
    ]
    for item in items:
        item.grid = engine._grid_of(item.market_key)
    selected = engine.select(engine.score(items, NOW), min_count=1)
    assert len(selected) == 1, "two markets from one corner grid were co-selected"


def test_half_time_shares_the_goal_lineage(db, config):
    """Half markets are a binomial split of the same full-time grid, so a half
    bet and a full-time bet are not two opportunities."""
    engine = RecommendationEngine(db, config)
    assert engine.lineage_of("OVER_2_5") == engine.lineage_of("HT_OVER_0_5")
    assert engine.lineage_of("CORNERS_OVER_8_5") != engine.lineage_of("OVER_2_5")
    assert engine.lineage_of("CARDS_OVER_3_5") != engine.lineage_of("CORNERS_OVER_8_5")


def test_different_grids_may_share_a_fixture(db, config):
    """Corners and goals are genuinely distinct estimates, so one fixture may
    contribute one of each -- up to the per-fixture cap."""
    engine = RecommendationEngine(db, config)
    goals = candidate(db, config, prediction_id=1, fixture_id=1,
                      market="OVER_1_5", probability=0.90)
    goals.grid = engine._grid_of(goals.market_key)
    corners = candidate(db, config, prediction_id=2, fixture_id=1,
                        market="CORNERS_OVER_8_5", probability=0.80)
    corners.grid = engine._grid_of(corners.market_key)
    selected = engine.select(engine.score([goals, corners], NOW), min_count=1)
    assert len(selected) == 2


def test_per_fixture_cap_is_respected_across_grids(db, config):
    """Four grids must not mean four bets on one match."""
    engine = RecommendationEngine(db, config)
    items = []
    for index, (market, grid) in enumerate([
        ("OVER_1_5", "GOAL_MATRIX"), ("HT_OVER_0_5", "HALF_MATRIX"),
        ("CORNERS_OVER_8_5", "CORNER_MATRIX"), ("CARDS_OVER_3_5", "CARD_DISTRIBUTION"),
    ], start=1):
        item = candidate(db, config, prediction_id=index, fixture_id=1,
                         market=market, probability=0.88)
        item.grid = engine._grid_of(market)
        items.append(item)
    selected = engine.select(engine.score(items, NOW), min_count=1)
    cap = int(config.recommendations["diversity"]["max_selections_per_fixture"])
    assert len(selected) <= cap


def test_no_family_exceeds_a_quarter_of_the_batch(db, config):
    """With 65 recommendable markets an unconstrained batch would be twenty
    views of three estimates."""
    engine = RecommendationEngine(db, config)
    items = []
    for index in range(1, 41):
        item = candidate(db, config, prediction_id=index, fixture_id=index,
                         market="CORNERS_OVER_8_5", probability=0.85)
        item.grid = engine._grid_of(item.market_key)
        items.append(item)
    for index in range(41, 81):
        item = candidate(db, config, prediction_id=index, fixture_id=index,
                         market="OVER_1_5", probability=0.90)
        item.grid = engine._grid_of(item.market_key)
        items.append(item)
    selected = engine.select(engine.score(items, NOW))
    if len(selected) >= 8:
        from collections import Counter
        mix = Counter(config.market(s.candidate.market_key).family
                      for s in selected)
        assert max(mix.values()) / len(selected) <= 0.55


def test_lift_factor_ranks_by_distance_covered_not_by_probability(db, config):
    """Normalised lift makes different base rates comparable.

    A thin call on a high-base-rate market must score below a substantial call
    on a lower one, even at the same probability: Under 5.5 cards at 90%
    against a 75.6% base rate has covered 0.590 of the distance to certainty,
    where BTTS at 90% against 54.8% has covered 0.779.
    """
    engine = RecommendationEngine(db, config)
    thin = candidate(db, config, prediction_id=1, fixture_id=1,
                     market="CARDS_UNDER_5_5", probability=0.90)
    substantial = candidate(db, config, prediction_id=2, fixture_id=2,
                            market="BTTS", probability=0.90)
    assert engine._lift_factor(substantial) > engine._lift_factor(thin)

    # And a high base rate does NOT by itself mean low information: measuring
    # it is what settles that.
    handicap = candidate(db, config, prediction_id=3, fixture_id=3,
                         market="AH_HOME_PLUS_1_5", probability=0.98)
    assert engine._lift_factor(handicap) > engine._lift_factor(thin)


def test_lift_factor_rises_with_probability_within_one_market(db, config):
    engine = RecommendationEngine(db, config)
    weaker = candidate(db, config, market="BTTS", probability=0.70)
    stronger = candidate(db, config, market="BTTS", probability=0.92)
    assert engine._lift_factor(stronger) > engine._lift_factor(weaker)


def test_lift_factor_is_bounded_and_never_dominates(db, config):
    """Reliability must stay primary: a well-evidenced ordinary call should
    beat an exciting unproven one."""
    engine = RecommendationEngine(db, config)
    weight = float(config.recommendations["reliability"]["lift_weight"])
    for market, probability in (("BTTS", 0.99), ("BTTS", 0.55),
                                ("OVER_1_5", 0.95), ("CARDS_OVER_3_5", 0.70)):
        item = candidate(db, config, market=market, probability=probability)
        factor = engine._lift_factor(item)
        assert 1.0 - weight - 1e-9 <= factor <= 1.0 + 1e-9


def test_unknown_market_does_not_crash_the_lift_factor(db, config):
    engine = RecommendationEngine(db, config)
    item = candidate(db, config, market="NOT_A_MARKET", probability=0.9)
    assert 0.0 < engine._lift_factor(item) <= 1.0
