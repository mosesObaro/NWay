"""Explanations built from stored features, never from free text.

Each line traces to a feature key, its value, the reference it is compared
against, and the effect that difference had on the probability. There is no
code path that can produce a plausible-sounding reason no feature supports,
which is the structural answer to "do not manufacture explanations".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Fixed phrasing per feature. One neutral sentence each: the template states
# the FACT, and the +/- marker carries whether it helps or hurts the selection.
# Directional wording ("scoring only 1.4") was wrong half the time once
# orientation became table-driven -- the same number is encouraging for a
# totals market and discouraging for a home-win market.
#
# A key absent from this table cannot be rendered, so adding a feature means
# deliberately giving it a sentence.
TEMPLATES: dict[str, str] = {
    "home.gf_per_match.last5":
        "Home side scoring {value:.1f} per match over the last 5 "
        "(league average {reference:.1f})",
    "away.gf_per_match.last5":
        "Away side scoring {value:.1f} per match over the last 5 "
        "(league average {reference:.1f})",
    "home.ga_per_match.last5":
        "Home side conceding {value:.1f} per match over the last 5 "
        "(league average {reference:.1f})",
    "away.ga_per_match.last5":
        "Away side conceding {value:.1f} per match over the last 5 "
        "(league average {reference:.1f})",
    "home.gf_per_match.last10":
        "Home side scoring {value:.1f} per match over the last 10",
    "away.gf_per_match.last10":
        "Away side scoring {value:.1f} per match over the last 10",
    "home.ga_per_match.last10":
        "Home side conceding {value:.1f} per match over the last 10",
    "away.ga_per_match.last10":
        "Away side conceding {value:.1f} per match over the last 10",
    "home.gf_per_match.last5.home_only":
        "Home side averaging {value:.1f} goals in its last 5 at home",
    "away.gf_per_match.last5.away_only":
        "Away side averaging {value:.1f} goals in its last 5 away",
    "home.ga_per_match.last5.home_only":
        "Home side conceding {value:.1f} per match at home",
    "away.ga_per_match.last5.away_only":
        "Away side conceding {value:.1f} per match away",
    "home.clean_sheet_rate.last10":
        "Home side kept a clean sheet in {value:.0%} of its last 10",
    "away.clean_sheet_rate.last10":
        "Away side kept a clean sheet in {value:.0%} of its last 10",
    "home.sxg_for_per_match.last5":
        "Home side generating {value:.1f} shot-based expected goals per match "
        "(proxy, not licensed xG)",
    "away.sxg_for_per_match.last5":
        "Away side generating {value:.1f} shot-based expected goals per match "
        "(proxy, not licensed xG)",
    "home.sxg_against_per_match.last5":
        "Home side allowing {value:.1f} shot-based expected goals per match "
        "(proxy, not licensed xG)",
    "away.sxg_against_per_match.last5":
        "Away side allowing {value:.1f} shot-based expected goals per match "
        "(proxy, not licensed xG)",
    "match.congestion_home":
        "Home side has played {value:.0f} matches in the last 14 days",
    "match.congestion_away":
        "Away side has played {value:.0f} matches in the last 14 days",
    "match.home_history_matches":
        "Only {value:.0f} completed matches known for the home side",
    "match.away_history_matches":
        "Only {value:.0f} completed matches known for the away side",
}

MODEL_TEMPLATES = {
    "lambda_total": "Model projects {value:.2f} total goals",
    "lambda_home": "Model projects {value:.2f} goals for the home side",
    "lambda_away": "Model projects {value:.2f} goals for the away side",
    "market_base_rate": "League baseline for this market is {value:.0%}",
}


@dataclass
class ExplanationLine:
    direction: str            # SUPPORT | RISK
    feature_key: str
    feature_value: float | None
    reference_value: float | None
    contribution: float
    template_key: str
    text: str
    rank: int


def _render(template: str, value: float, reference: float | None) -> str:
    return template.format(value=value, abs_value=abs(value),
                           reference=reference if reference is not None else 0.0)


def render_template(template_key: str, value: float | None,
                    reference: float | None) -> str | None:
    """Render a stored explanation row. Unknown keys render as nothing."""
    template = TEMPLATES.get(template_key) or MODEL_TEMPLATES.get(template_key)
    if template is None:
        return None
    return _render(template, value if value is not None else 0.0, reference)


# How a feature's deviation from its reference should be read, per market
# family. A line that argues for the opposite selection is worse than no line at
# all, so orientation is declared rather than inferred.
#
# Each entry maps (side, kind) -> sign, where kind is "attack" (goals scored) or
# "defence" (goals conceded), and sign is +1 when a HIGHER value supports the
# selection.
_ORIENTATION: dict[str, dict[tuple[str, str], int]] = {
    # Totals: anyone scoring more, or anyone leaking more, means more goals.
    "TOTALS_OVER": {("home", "attack"): +1, ("away", "attack"): +1,
                    ("home", "defence"): +1, ("away", "defence"): +1},
    "TOTALS_UNDER": {("home", "attack"): -1, ("away", "attack"): -1,
                     ("home", "defence"): -1, ("away", "defence"): -1},
    # Home-leaning result markets: the home side scoring helps, the away side
    # scoring hurts; the home side leaking hurts, the away side leaking helps.
    "HOME_SIDE": {("home", "attack"): +1, ("away", "attack"): -1,
                  ("home", "defence"): -1, ("away", "defence"): +1},
    "AWAY_SIDE": {("home", "attack"): -1, ("away", "attack"): +1,
                  ("home", "defence"): +1, ("away", "defence"): -1},
}

_MARKET_ORIENTATION: dict[str, str] = {
    "OVER_0_5": "TOTALS_OVER", "OVER_1_5": "TOTALS_OVER",
    "OVER_2_5": "TOTALS_OVER", "OVER_3_5": "TOTALS_OVER",
    "BTTS": "TOTALS_OVER",
    "HOME_TO_SCORE": "HOME_SIDE", "AWAY_TO_SCORE": "AWAY_SIDE",
    "UNDER_1_5": "TOTALS_UNDER", "UNDER_2_5": "TOTALS_UNDER",
    "UNDER_3_5": "TOTALS_UNDER",
    "HOME_WIN": "HOME_SIDE", "DOUBLE_CHANCE_1X": "HOME_SIDE",
    "HOME_CLEAN_SHEET": "HOME_SIDE",
    "AWAY_WIN": "AWAY_SIDE", "DOUBLE_CHANCE_X2": "AWAY_SIDE",
    "AWAY_CLEAN_SHEET": "AWAY_SIDE",
    # A draw is not well described by "more of anything", so it gets no
    # directional lines at all rather than misleading ones.
    "DRAW": "",
}


def _classify(feature_key: str) -> tuple[str, str] | None:
    """Return (side, kind) for a feature, or None when it is not directional."""
    if feature_key.startswith("home."):
        side = "home"
    elif feature_key.startswith("away."):
        side = "away"
    else:
        return None
    if "gf_per_match" in feature_key or "sxg_for" in feature_key:
        return side, "attack"
    if "ga_per_match" in feature_key or "sxg_against" in feature_key:
        return side, "defence"
    if "clean_sheet_rate" in feature_key:
        # A high clean-sheet rate is the inverse of leaking goals.
        return side, "defence_inverse"
    return None


def build_explanations(*, market_key: str, probability: float,
                       features: dict[str, dict[str, Any]],
                       lambdas: tuple[float, float],
                       base_rate: float | None,
                       league_baseline: dict[str, float],
                       max_support: int = 4,
                       max_risk: int = 2) -> list[ExplanationLine]:
    """Describe the features that pushed this market away from its baseline.

    Contribution is a signed, normalised deviation from the feature's reference
    rather than a true leave-one-out effect. With a parametric goal model the
    two agree in sign and ordering for the features that matter, and the honest
    simplification is better than an expensive number implying more precision
    than it has.

    Orientation is table-driven (_ORIENTATION): a support line must actually
    argue for the selection being recommended. Listing the away side's scoring
    form as support for "Home Win or Draw" would be worse than saying nothing.
    """
    lam_home, lam_away = lambdas
    support: list[ExplanationLine] = []
    risk: list[ExplanationLine] = []

    total = lam_home + lam_away
    support.append(ExplanationLine(
        "SUPPORT", "model.lambda_total", round(total, 3), None,
        contribution=abs(total - 2.8) / 2.8, template_key="lambda_total",
        text=MODEL_TEMPLATES["lambda_total"].format(value=total), rank=0))

    orientation_key = _MARKET_ORIENTATION.get(market_key, "")
    orientation = _ORIENTATION.get(orientation_key, {})

    for key, payload in features.items():
        value = payload.get("value")
        if value is None or key not in TEMPLATES:
            continue
        classified = _classify(key)
        if classified is None or not orientation:
            continue
        side, kind = classified
        lookup_kind = "defence" if kind == "defence_inverse" else kind
        sign = orientation.get((side, lookup_kind))
        if sign is None:
            continue
        if kind == "defence_inverse":
            sign = -sign

        reference = (league_baseline.get("home_goals", 1.5) if side == "home"
                     else league_baseline.get("away_goals", 1.2))
        if lookup_kind == "defence":
            reference = (league_baseline.get("away_goals", 1.2) if side == "home"
                         else league_baseline.get("home_goals", 1.5))
        if kind == "defence_inverse":
            reference = 0.30          # roughly the league clean-sheet rate

        if reference <= 0:
            continue
        deviation = (value - reference) / reference
        oriented = sign * deviation
        if abs(oriented) < 0.15:
            continue

        line = ExplanationLine(
            "SUPPORT" if oriented > 0 else "RISK", key, round(float(value), 3),
            round(reference, 3), contribution=round(float(oriented), 4),
            template_key=key, text=_render(TEMPLATES[key], value, reference), rank=0)
        (support if oriented > 0 else risk).append(line)

    # Congestion and thin history are risks whenever present, in every market.
    for key in ("match.congestion_home", "match.congestion_away"):
        value = (features.get(key) or {}).get("value")
        if value is not None and value >= 4:
            risk.append(ExplanationLine(
                "RISK", key, float(value), None, contribution=-0.2,
                template_key=key, text=_render(TEMPLATES[key], value, None), rank=0))
    for key in ("match.home_history_matches", "match.away_history_matches"):
        value = (features.get(key) or {}).get("value")
        if value is not None and value < 8:
            risk.append(ExplanationLine(
                "RISK", key, float(value), None, contribution=-0.3,
                template_key=key, text=_render(TEMPLATES[key], value, None), rank=0))

    if base_rate is not None:
        support.append(ExplanationLine(
            "SUPPORT", "market.base_rate", round(base_rate, 4), None,
            contribution=round(probability - base_rate, 4),
            template_key="market_base_rate",
            text=MODEL_TEMPLATES["market_base_rate"].format(value=base_rate), rank=0))

    support.sort(key=lambda line: -abs(line.contribution))
    risk.sort(key=lambda line: -abs(line.contribution))
    selected = support[:max_support] + risk[:max_risk]
    for index, line in enumerate(selected):
        line.rank = index
    return selected
