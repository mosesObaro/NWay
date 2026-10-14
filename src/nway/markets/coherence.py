"""Invariants that deriving every market from one grid guarantees.

These catch implementation bugs, not modelling disagreements: each is true of
any valid joint distribution, so a failure means the code is wrong. They run
after every prediction and fail the run rather than storing a quietly
inconsistent set of markets.
"""

from __future__ import annotations

from typing import Any

TOLERANCE = 1e-6


class CoherenceError(AssertionError):
    """A derived market set violated an invariant that must hold by construction."""


def _ladder(probabilities: dict[str, float], keys: list[str], label: str) -> None:
    present = [k for k in keys if k in probabilities]
    for higher, lower in zip(present, present[1:]):
        if probabilities[lower] > probabilities[higher] + TOLERANCE:
            raise CoherenceError(
                f"{label}: {lower} ({probabilities[lower]:.6f}) exceeds "
                f"{higher} ({probabilities[higher]:.6f})")


def _complements(probabilities: dict[str, float],
                 pairs: list[tuple[str, str]], label: str) -> None:
    for first, second in pairs:
        if first in probabilities and second in probabilities:
            total = probabilities[first] + probabilities[second]
            if abs(total - 1.0) > TOLERANCE:
                raise CoherenceError(
                    f"{label}: {first} + {second} = {total:.9f}, expected 1")


def _trio(probabilities: dict[str, float], keys: list[str], label: str) -> None:
    if all(k in probabilities for k in keys):
        total = sum(probabilities[k] for k in keys)
        if abs(total - 1.0) > TOLERANCE:
            raise CoherenceError(
                f"{label}: {' + '.join(keys)} = {total:.9f}, expected 1")


def check_coherence(markets: Any) -> None:
    """Goal-grid invariants."""
    probabilities = markets.probabilities
    total = float(markets.matrix.sum())
    if abs(total - 1.0) > TOLERANCE:
        raise CoherenceError(f"matrix does not sum to 1: {total}")

    _trio(probabilities, ["HOME_WIN", "DRAW", "AWAY_WIN"], "1X2")
    _trio(probabilities, ["EH_MINUS_1_HOME", "EH_MINUS_1_DRAW",
                          "EH_MINUS_1_AWAY"], "European handicap -1")
    _trio(probabilities, ["EH_PLUS_1_HOME", "EH_PLUS_1_DRAW",
                          "EH_PLUS_1_AWAY"], "European handicap +1")
    _trio(probabilities, ["CORNERS_1X2_HOME", "CORNERS_1X2_DRAW",
                          "CORNERS_1X2_AWAY"], "corner 1X2")
    _trio(probabilities, ["HT_HOME_WIN", "HT_DRAW", "HT_AWAY_WIN"], "half-time 1X2")

    _ladder(probabilities, ["OVER_0_5", "OVER_1_5", "OVER_2_5", "OVER_3_5"],
            "total goals")
    _ladder(probabilities, ["HOME_TO_SCORE", "HOME_OVER_1_5", "HOME_OVER_2_5"],
            "home team goals")
    _ladder(probabilities, ["AWAY_TO_SCORE", "AWAY_OVER_1_5", "AWAY_OVER_2_5"],
            "away team goals")
    _ladder(probabilities, ["HT_OVER_0_5", "HT_OVER_1_5"], "half-time goals")
    _ladder(probabilities, [f"CORNERS_OVER_{l}" for l in
                            ("8_5", "9_5", "10_5", "11_5")], "total corners")
    _ladder(probabilities, [f"HOME_CORNERS_OVER_{l}" for l in
                            ("3_5", "4_5", "5_5")], "home corners")
    _ladder(probabilities, [f"AWAY_CORNERS_OVER_{l}" for l in
                            ("3_5", "4_5", "5_5")], "away corners")
    _ladder(probabilities, [f"CARDS_OVER_{l}" for l in ("3_5", "4_5", "5_5")],
            "total cards")

    _complements(probabilities, [
        ("AH_HOME_MINUS_1_5", "AH_AWAY_PLUS_1_5"),
        ("AH_AWAY_MINUS_1_5", "AH_HOME_PLUS_1_5"),
        ("CORNERS_AH_HOME_MINUS_1_5", "CORNERS_AH_AWAY_PLUS_1_5"),
        ("CORNERS_AH_AWAY_MINUS_1_5", "CORNERS_AH_HOME_PLUS_1_5"),
        ("BTTS", "BTTS_NO"),
    ], "complement pair")

    # A team cannot score more than the match total.
    for team, match in (("HOME_OVER_1_5", "OVER_1_5"),
                        ("AWAY_OVER_1_5", "OVER_1_5"),
                        ("HOME_OVER_2_5", "OVER_2_5"),
                        ("AWAY_OVER_2_5", "OVER_2_5")):
        if team in probabilities and match in probabilities:
            if probabilities[team] > probabilities[match] + TOLERANCE:
                raise CoherenceError(f"{team} exceeds {match}")

    if "BTTS" in probabilities and "OVER_1_5" in probabilities:
        if probabilities["BTTS"] > probabilities["OVER_1_5"] + TOLERANCE:
            raise CoherenceError("BTTS exceeds Over 1.5, which is impossible")

    # A first half cannot contain more goals than the match.
    for half, full in (("HT_OVER_0_5", "OVER_0_5"), ("HT_OVER_1_5", "OVER_1_5")):
        if half in probabilities and full in probabilities:
            if probabilities[half] > probabilities[full] + TOLERANCE:
                raise CoherenceError(f"{half} exceeds {full}")

    # A combination cannot be likelier than either of its parts.
    for combo, parts in (
        ("HOME_AND_BTTS", ("HOME_WIN", "BTTS")),
        ("DRAW_AND_BTTS", ("DRAW", "BTTS")),
        ("AWAY_AND_BTTS", ("AWAY_WIN", "BTTS")),
        ("HOME_AND_OVER_2_5", ("HOME_WIN", "OVER_2_5")),
        ("DRAW_AND_OVER_2_5", ("DRAW", "OVER_2_5")),
        ("AWAY_AND_OVER_2_5", ("AWAY_WIN", "OVER_2_5")),
        ("HOME_WIN_TO_NIL", ("HOME_WIN", "HOME_CLEAN_SHEET")),
        ("AWAY_WIN_TO_NIL", ("AWAY_WIN", "AWAY_CLEAN_SHEET")),
    ):
        if combo not in probabilities:
            continue
        for part in parts:
            if part in probabilities and \
                    probabilities[combo] > probabilities[part] + TOLERANCE:
                raise CoherenceError(f"{combo} exceeds its component {part}")

    # HT/FT is a partition of the match.
    htft = [k for k in probabilities if k.startswith("HTFT_")]
    if htft:
        total = sum(probabilities[k] for k in htft)
        if abs(total - 1.0) > TOLERANCE:
            raise CoherenceError(f"HT/FT does not partition: {total:.9f}")

    expected = probabilities.get("EXPECTED_GOALS")
    if expected is not None:
        implied = markets.lambda_home + markets.lambda_away
        if abs(expected - implied) > 0.05:
            raise CoherenceError(
                f"expected goals {expected:.4f} differs from lambda sum "
                f"{implied:.4f}")

    for key, value in probabilities.items():
        if key.startswith("EXPECTED") or key == "GOAL_MARGIN":
            continue
        if not (-TOLERANCE <= value <= 1.0 + TOLERANCE):
            raise CoherenceError(f"{key} outside [0,1]: {value}")
