"""Explanations must argue for the selection they accompany.

A support line that actually favours the opposite outcome is worse than no line
at all: it reads as evidence while being the reverse of it.
"""

from __future__ import annotations

import pytest

from nway.prediction.explain import TEMPLATES, build_explanations

BASELINE = {"home_goals": 1.5, "away_goals": 1.2}


def features(**overrides):
    base = {
        "home.gf_per_match.last5": 1.5, "away.gf_per_match.last5": 1.2,
        "home.ga_per_match.last5": 1.2, "away.ga_per_match.last5": 1.5,
    }
    base.update(overrides)
    return {key: {"value": value, "is_null_reason": None}
            for key, value in base.items()}


def build(market, feature_values, probability=0.8):
    return build_explanations(
        market_key=market, probability=probability, features=feature_values,
        lambdas=(1.8, 1.0), base_rate=0.5, league_baseline=BASELINE)


def test_home_market_treats_away_scoring_as_a_risk():
    """The bug this test exists for: the away side's form listed as SUPPORT
    for 'Home Win or Draw'."""
    lines = build("DOUBLE_CHANCE_1X", features(**{"away.gf_per_match.last5": 2.4}))
    away_lines = [line for line in lines if line.feature_key == "away.gf_per_match.last5"]
    assert away_lines, "the away side's strong scoring form was not mentioned at all"
    assert all(line.direction == "RISK" for line in away_lines)


def test_home_market_treats_home_scoring_as_support():
    lines = build("HOME_WIN", features(**{"home.gf_per_match.last5": 2.6}))
    home_lines = [line for line in lines if line.feature_key == "home.gf_per_match.last5"]
    assert home_lines and all(line.direction == "SUPPORT" for line in home_lines)


def test_away_market_mirrors_the_home_market():
    lines = build("DOUBLE_CHANCE_X2", features(**{"away.gf_per_match.last5": 2.4}))
    away_lines = [line for line in lines if line.feature_key == "away.gf_per_match.last5"]
    assert away_lines and all(line.direction == "SUPPORT" for line in away_lines)


def test_totals_treat_both_sides_scoring_as_support():
    lines = build("OVER_2_5", features(**{"home.gf_per_match.last5": 2.5,
                                          "away.gf_per_match.last5": 2.2}))
    scoring = [line for line in lines
               if line.feature_key in ("home.gf_per_match.last5",
                                       "away.gf_per_match.last5")]
    assert scoring and all(line.direction == "SUPPORT" for line in scoring)


def test_under_market_inverts_the_totals_orientation():
    lines = build("UNDER_2_5", features(**{"home.gf_per_match.last5": 2.5}))
    scoring = [line for line in lines if line.feature_key == "home.gf_per_match.last5"]
    assert scoring and all(line.direction == "RISK" for line in scoring)


def test_tight_defence_is_a_risk_for_an_over_market():
    lines = build("OVER_2_5", features(**{"away.ga_per_match.last5": 0.4}))
    defence = [line for line in lines if line.feature_key == "away.ga_per_match.last5"]
    assert defence and all(line.direction == "RISK" for line in defence)


def test_draw_gets_no_directional_lines():
    """No orientation is honest for a draw; inventing one would not be."""
    lines = build("DRAW", features(**{"home.gf_per_match.last5": 2.6}))
    directional = [line for line in lines if line.feature_key.startswith(("home.", "away."))]
    assert directional == []


def test_templates_are_directionally_neutral():
    """The +/- marker carries the sign; wording must not contradict it.

    Directional wording was wrong half the time once orientation became
    table-driven: 1.4 goals a game is encouraging for a totals market and
    discouraging for a home-win market, and one sentence cannot be both.
    """
    loaded = [t.lower() for t in TEMPLATES.values()]
    for text in loaded:
        for word in (" only ", " strong ", " weak ", " poor ", " excellent "):
            if "only {value:.0f} completed matches" in text:
                continue          # a genuine statement of thin data
            assert word not in text, f"directional wording in template: {text!r}"


def test_every_rendered_line_has_a_template_and_a_feature():
    lines = build("OVER_1_5", features(**{"home.gf_per_match.last5": 2.4,
                                          "away.gf_per_match.last5": 2.1}))
    assert lines
    for line in lines:
        assert line.text, "an explanation line rendered as empty text"
        assert line.template_key
        # Model and market lines are the only non-feature templates permitted.
        if line.feature_key.startswith(("home.", "away.", "match.")):
            assert line.feature_key in TEMPLATES


def test_near_average_features_are_not_mentioned():
    """Reporting a league-average value as a 'factor' is noise dressed as insight."""
    lines = build("OVER_2_5", features())
    feature_lines = [line for line in lines if line.feature_key.startswith(("home.", "away."))]
    assert feature_lines == []
