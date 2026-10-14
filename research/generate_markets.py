"""Regenerate config/markets.yaml from measured base rates.

Run after changing the resolver set, adding a market, or when a new season
shifts the base rates enough to matter.

    python3 research/generate_markets.py            # print a summary
    python3 research/generate_markets.py --write     # rewrite the config

Every base rate is produced by settling the market through
``nway.validation.resolvers`` -- the same code that settles live predictions --
so a market cannot be floored against one definition and scored against
another. `min_probability` is computed, never typed:

    floor = max(absolute_floor, base_rate + lift_k * (1 - base_rate))
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nway.logging_setup import setup_logging                      # noqa: E402
from nway.markets.registry import MARKET_GRID                     # noqa: E402
from nway.storage.db import connect                               # noqa: E402
from nway.validation.resolvers import MatchFacts, RESOLVERS, settle  # noqa: E402

ABSOLUTE_FLOOR = 0.65
LIFT_K = 0.25
# The current regime, not the nine-season average: total cards have fallen from
# 4.54 a match in 2019/20 to 3.68 so far in 2026/27, and averaging over all of
# it would set the card floors too high.
BASE_RATE_SEASONS = ("2024/25", "2025/26", "2026/27")
DOMESTIC = ("premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1",
            "eredivisie", "primeira_liga")

GRID_COMMENT = {
    "GOAL_MATRIX": "Derived from the Dixon-Coles full-time goal grid.",
    "HALF_MATRIX": ("Derived from the full-time grid by a binomial split, so the "
                    "full-time marginal is preserved exactly and a half market "
                    "cannot exceed its full-time counterpart."),
    "CORNER_MATRIX": ("Derived from the 26x26 corner grid: negative-binomial "
                      "total with a beta-binomial home share."),
    "CARD_DISTRIBUTION": "Derived from the negative-binomial total-card distribution.",
}


def floor_for(base_rate: float) -> float:
    return round(max(ABSOLUTE_FLOOR, base_rate + LIFT_K * (1 - base_rate)), 3)


def family_of(key: str) -> str:
    if key.startswith("HTFT_"):
        return "HTFT"
    if key.startswith("HT_"):
        return "HALVES"
    if key.startswith("CARDS_"):
        return "CARDS"
    if key.startswith(("CORNERS_OVER", "CORNERS_UNDER")):
        return "CORNERS_TOTAL"
    if "CORNERS_OVER" in key:
        return "CORNERS_TEAM"
    if key.startswith("CORNERS_"):
        return "CORNERS_RESULT"
    if key.startswith(("AH_", "EH_")):
        return "HANDICAP"
    if "_AND_" in key:
        return "COMBO"
    if key in ("HOME_CLEAN_SHEET", "AWAY_CLEAN_SHEET",
               "HOME_WIN_TO_NIL", "AWAY_WIN_TO_NIL"):
        return "CLEAN_SHEET"
    if key in ("HOME_TO_SCORE", "AWAY_TO_SCORE", "HOME_OVER_1_5",
               "HOME_OVER_2_5", "AWAY_OVER_1_5", "AWAY_OVER_2_5"):
        return "TEAM_GOALS"
    if key in ("BTTS", "BTTS_NO"):
        return "BTTS"
    if key.startswith("DOUBLE_CHANCE"):
        return "DOUBLE_CHANCE"
    if key.startswith(("OVER_", "UNDER_")):
        return "TOTALS"
    return "RESULT"


def recommendable(key: str, base_rate: float) -> tuple[bool, str | None]:
    """Structural reachability, decided from the measured base rate.

    A floor of 0.65 on a market that comes in 20% of the time demands a lift of
    0.45, which no derived market reaches; above 0.92 the lift the floor permits
    is smaller than the noise.
    """
    if key.startswith("HTFT_"):
        return False, ("nine mutually exclusive outcomes, none with a base rate "
                       "near the floor; derived and evaluated, never recommended")
    if key == "DRAW":
        return False, ("draw probability rarely exceeds ~33% for any model, so "
                       "this can never clear its floor")
    if base_rate > 0.92:
        return False, (f"base rate {base_rate:.3f} leaves almost no information: "
                       f"the floor of {floor_for(base_rate):.3f} is a lift of "
                       f"{floor_for(base_rate) - base_rate:.3f}, inside the noise")
    if base_rate < 0.20:
        return False, (f"base rate {base_rate:.3f} against an absolute floor of "
                       f"0.65 demands a lift of {0.65 - base_rate:.2f}, which no "
                       f"derived market reaches")
    return True, None


def load_facts(db) -> list[MatchFacts]:
    placeholders = ",".join("?" * len(BASE_RATE_SEASONS))
    rows = db.query(f"""
        SELECT r.home_goals, r.away_goals, r.ht_home_goals, r.ht_away_goals,
               h.corners AS hc, a.corners AS ac,
               h.yellow_cards AS hy, a.yellow_cards AS ay,
               h.red_cards AS hr, a.red_cards AS ar, r.went_to_et
        FROM match_result r
        JOIN fixture f ON f.fixture_id = r.fixture_id
        JOIN season s ON s.season_id = f.season_id
        LEFT JOIN team_match_stats h
          ON h.fixture_id = f.fixture_id AND h.is_home = 1
        LEFT JOIN team_match_stats a
          ON a.fixture_id = f.fixture_id AND a.is_home = 0
        WHERE s.label IN ({placeholders})
    """, list(BASE_RATE_SEASONS))
    facts = []
    for row in rows:
        cards = None
        if row["hy"] is not None and row["ay"] is not None:
            cards = row["hy"] + row["ay"] + (row["hr"] or 0) + (row["ar"] or 0)
        facts.append(MatchFacts(
            row["home_goals"], row["away_goals"], row["ht_home_goals"],
            row["ht_away_goals"], row["hc"], row["ac"], cards,
            bool(row["went_to_et"])))
    return facts


def measure(facts: list[MatchFacts]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key in sorted(RESOLVERS):
        outcomes = [settle(key, f) for f in facts]
        outcomes = [o for o in outcomes if o is not None]
        if not outcomes:
            continue
        out[key] = {"base_rate": round(sum(outcomes) / len(outcomes), 4),
                    "n": len(outcomes),
                    "grid": MARKET_GRID.get(key, "GOAL_MATRIX")}
    return out


def render(rates: dict[str, dict], n_facts: int) -> str:
    grids = ["GOAL_MATRIX", "HALF_MATRIX", "CORNER_MATRIX", "CARD_DISTRIBUTION"]
    corner_n = max((v["n"] for k, v in rates.items()
                    if v["grid"] in ("CORNER_MATRIX", "CARD_DISTRIBUTION")), default=0)
    lines = [
        "# Market registry. GENERATED -- see research/generate_markets.py.",
        "#",
        "# `base_rate` is MEASURED by settling each market through",
        f"# nway.validation.resolvers over the last three seasons ({n_facts:,} matches;",
        f"# {corner_n:,} for corners and cards, which need statistics). Base rates and",
        "# settlement therefore cannot disagree: they share one definition.",
        "#",
        "# The last three seasons rather than all nine because the regime moves: total",
        "# cards have fallen from 4.54 a match in 2019/20 to 3.68 so far in 2026/27, and",
        "# a nine-season average would set the card floors too high.",
        "#",
        "# `min_probability` is not hand-picked:",
        "#     floor = max(absolute_floor, base_rate + lift_k * (1 - base_rate))",
        "# The relative term makes the floor mean the same thing in every market -- the",
        "# model must be a quarter of the way from the market's own base rate to",
        "# certainty. The absolute term stops a low-base-rate market counting as high",
        "# confidence merely by beating its base rate.",
        "",
        "thresholds:",
        f"  absolute_floor: {ABSOLUTE_FLOOR}",
        f"  lift_k: {LIFT_K}",
        "",
        "markets:",
    ]
    order = sorted(rates, key=lambda k: (grids.index(rates[k]["grid"]),
                                         family_of(k), -rates[k]["base_rate"]))
    last_grid = None
    for key in order:
        info = rates[key]
        base = info["base_rate"]
        grid = info["grid"]
        if grid != last_grid:
            lines += ["", f"  # ---- {grid} " + "-" * max(4, 62 - len(grid)),
                      f"  # {GRID_COMMENT[grid]}"]
            last_grid = grid
        rec, reason = recommendable(key, base)
        lines += ["", f"  - key: {key}", f"    family: {family_of(key)}",
                  f"    grid: {grid}", "    mvp: true",
                  f"    base_rate: {base}", f"    base_rate_n: {info['n']}",
                  f"    min_probability: {floor_for(base)}"]
        if rec and grid in ("CORNER_MATRIX", "CARD_DISTRIBUTION"):
            # Corners and cards need per-match statistics, which
            # football-data.org does not supply for the Champions League.
            lines.append(f"    enabled_competitions: [{', '.join(DOMESTIC)}]")
            lines.append("    requires_stats: " + ("[corners]"
                         if grid == "CORNER_MATRIX" else "[yellow_cards, red_cards]"))
        elif rec:
            lines.append("    enabled_competitions: [ALL]")
        else:
            lines.append("    recommendable: false")
            lines.append("    disabled_reason: >-")
            for start in range(0, len(reason), 66):
                lines.append(f"      {reason[start:start + 66]}")

    lines += [
        "", "  # ---- numeric, reported as context and never selected -------------------",
        "", "  - key: EXPECTED_GOALS", "    family: GOALS_NUMERIC",
        "    grid: GOAL_MATRIX", "    mvp: true", "    recommendable: false",
        "", "  - key: GOAL_MARGIN", "    family: GOALS_NUMERIC",
        "    grid: GOAL_MATRIX", "    mvp: true", "    recommendable: false",
        "", "  # ---- blocked on data, not effort ---------------------------------------",
        "", "  - key: ANYTIME_SCORER", "    family: PLAYER", "    mvp: false",
        "    enabled_competitions: []", "    blocked_reason: >-",
        "      No free, terms-compliant source for lineups, expected minutes or",
        "      injuries. See docs/DATA_SOURCES.md section 9.",
        "",
        "# A market is enabled for a competition only when its training sample clears",
        "# this floor. Eredivisie and Primeira Liga carry corner and card statistics",
        "# only from 2017/18 (~2,750 matches each) against ~9,900 for England.",
        "eligibility:", "  min_training_matches:",
        "    GOAL_MATRIX: 1500", "    HALF_MATRIX: 1500",
        "    CORNER_MATRIX: 2000", "    CARD_DISTRIBUTION: 2000",
        "  min_settled_predictions: 500", "  max_holdout_ece: 0.05", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--database", default=None)
    args = parser.parse_args()

    setup_logging("ERROR")
    db = connect(args.database)
    facts = load_facts(db)
    if not facts:
        print("no matches in the base-rate window; ingest history first")
        return 1
    rates = measure(facts)

    print(f"{len(facts):,} matches, {len(rates)} markets")
    counts = Counter(family_of(k) for k in rates)
    recommended = sum(1 for k in rates
                      if recommendable(k, rates[k]["base_rate"])[0])
    print(f"  {recommended} recommendable, {len(rates) - recommended} reported-only")
    print(f"  families: {dict(counts)}")

    if args.write:
        target = Path(__file__).resolve().parents[1] / "config" / "markets.yaml"
        target.write_text(render(rates, len(facts)))
        print(f"  wrote {target}")
    else:
        print("  (pass --write to rewrite config/markets.yaml)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
