"""
Threshold and selection study, run 2026-09-20, BEFORE implementation.

Answers the question the notification rules depend on: what probability floor
and what diversity caps produce batches of 7-20, often enough to be a useful
service, without a monoculture of one market?

Method
------
Bookmaker closing odds (B365, proportionally de-vigged) are inverted into an
independent-Poisson goal model per match: the implied total-goal rate is
recovered from the Over 2.5 line, and the home share from the implied
P(home) - P(away). Every market is then derived from that goal matrix, exactly
as the real system derives markets (docs/MODEL_DESIGN.md section 3.3).

This is an UPPER BOUND on a home-grown model's sharpness. Closing odds price
team news and money this system will not have, so real supply at a given floor
will be lower. Treat every number here as a ceiling and re-derive from the
system's own backtest before launch (docs/BACKTESTING.md).

Usage:
    python3 research/threshold_study.py          # needs research/evidence/raw
"""

import bisect
import collections
import csv
import datetime as dt
import math
import os
import statistics

DATA_DIR = os.path.join(os.path.dirname(__file__), "evidence", "raw")
LEAGUES = ["E0", "SP1", "D1", "I1", "F1", "N1", "P1"]
SEASONS = ["2223", "2324", "2425", "2526"]

# Measured over 21,545 matches, 2017/18-2025/26 (see feasibility_study.py).
BASE_RATE = {
    "HOME_WIN": 0.438, "AWAY_WIN": 0.313,
    "DOUBLE_CHANCE_1X": 0.687, "DOUBLE_CHANCE_X2": 0.562,
    "OVER_1_5": 0.771, "OVER_2_5": 0.533, "UNDER_2_5": 0.467, "BTTS": 0.535,
}
FAMILY = {
    "HOME_WIN": "RESULT", "AWAY_WIN": "RESULT",
    "DOUBLE_CHANCE_1X": "DOUBLE_CHANCE", "DOUBLE_CHANCE_X2": "DOUBLE_CHANCE",
    "OVER_1_5": "TOTALS", "OVER_2_5": "TOTALS", "UNDER_2_5": "TOTALS",
    "BTTS": "BTTS",
}
FAMILY_SHARE = {"DOUBLE_CHANCE": 0.30, "TOTALS": 0.40, "RESULT": 0.40, "BTTS": 0.30}
ABSOLUTE_FLOOR = 0.65
LIFT_K = 0.25
MIN_RECOMMENDATIONS = 7
MAX_RECOMMENDATIONS = 20
HORIZON_HOURS = 72


def market_floor(market, absolute=ABSOLUTE_FLOOR, k=LIFT_K):
    """floor = max(absolute, base + k * (1 - base)).

    The relative term makes the floor mean the same thing across markets: the
    model must be k of the way from the market's own base rate to certainty.
    The absolute term stops a low-base-rate market (an away win at 49%) from
    qualifying as "high confidence" merely because it beat its base rate.
    """
    base = BASE_RATE[market]
    return max(absolute, base + k * (1 - base))


def _poisson(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def implied_lambdas(p_home, p_draw, p_away, p_over25):
    total = _bisect_solve(
        lambda t: 1 - math.exp(-t) * (1 + t + t * t / 2), p_over25, 1.0, 6.0)

    def margin(share):
        lh, la = total * share, total * (1 - share)
        return sum(_poisson(h, lh) * _poisson(a, la) * (1 if h > a else -1 if a > h else 0)
                   for h in range(9) for a in range(9))

    share = _bisect_solve(margin, p_home - p_away, 0.05, 0.95)
    return total * share, total * (1 - share)


def _bisect_solve(fn, target, lo, hi, iterations=40):
    for _ in range(iterations):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fn(mid) < target else (lo, mid)
    return (lo + hi) / 2


def derive_markets(lam_home, lam_away, grid=11):
    matrix = [[_poisson(h, lam_home) * _poisson(a, lam_away) for a in range(grid)]
              for h in range(grid)]
    total = sum(map(sum, matrix))
    matrix = [[cell / total for cell in row] for row in matrix]
    home = sum(matrix[h][a] for h in range(grid) for a in range(grid) if h > a)
    draw = sum(matrix[h][h] for h in range(grid))
    over = lambda n: sum(matrix[h][a] for h in range(grid) for a in range(grid) if h + a > n)  # noqa: E731
    btts = (1 - sum(matrix[0]) - sum(matrix[h][0] for h in range(grid)) + matrix[0][0])
    return {
        "HOME_WIN": home, "AWAY_WIN": 1 - home - draw,
        "DOUBLE_CHANCE_1X": home + draw, "DOUBLE_CHANCE_X2": 1 - home,
        "OVER_1_5": over(1.5), "OVER_2_5": over(2.5), "UNDER_2_5": 1 - over(2.5),
        "BTTS": btts,
    }


def load():
    def num(v):
        v = (v or "").strip()
        try:
            return float(v)
        except ValueError:
            return None

    out = []
    for code in LEAGUES:
        for season in SEASONS:
            path = os.path.join(DATA_DIR, f"{code}-{season}.csv")
            if not os.path.exists(path):
                continue
            with open(path, encoding="latin-1") as fh:
                for row in csv.DictReader(fh):
                    if not row.get("HomeTeam") or not (row.get("FTHG") or "").strip():
                        continue
                    h = num(row.get("B365CH")) or num(row.get("B365H"))
                    d = num(row.get("B365CD")) or num(row.get("B365D"))
                    a = num(row.get("B365CA")) or num(row.get("B365A"))
                    o = num(row.get("B365C>2.5")) or num(row.get("B365>2.5"))
                    u = num(row.get("B365C<2.5")) or num(row.get("B365<2.5"))
                    if not all([h, d, a, o, u]):
                        continue
                    kickoff = None
                    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
                        try:
                            kickoff = dt.datetime.strptime((row.get("Date") or "").strip(), fmt)
                            break
                        except ValueError:
                            continue
                    if kickoff is None:
                        continue
                    clock = (row.get("Time") or "").strip()
                    if clock:
                        try:
                            hh, mm = clock.split(":")
                            kickoff = kickoff.replace(hour=int(hh), minute=int(mm))
                        except ValueError:
                            pass
                    vig, vig2 = 1 / h + 1 / d + 1 / a, 1 / o + 1 / u
                    lam_h, lam_a = implied_lambdas(
                        (1 / h) / vig, (1 / d) / vig, (1 / a) / vig, (1 / o) / vig2)
                    out.append((season, kickoff, derive_markets(lam_h, lam_a)))
    return out


def greedy(window, floors, family_cap, max_r=MAX_RECOMMENDATIONS):
    """One selection per fixture, highest probability first, respecting caps."""
    pool = sorted((-p, i, m)
                  for i, (_, markets) in enumerate(window)
                  for m, p in markets.items() if p >= floors[m])
    selected, used, per_family = [], set(), collections.Counter()
    for neg_p, i, market in pool:
        fam = FAMILY[market]
        if i in used or len(selected) >= max_r:
            continue
        if per_family[fam] >= family_cap.get(fam, max_r):
            continue
        selected.append((-neg_p, i, market))
        used.add(i)
        per_family[fam] += 1
    return selected


def select(window, floors, min_r=MIN_RECOMMENDATIONS):
    """Two passes: size the batch, then apply caps relative to that size."""
    unconstrained = greedy(window, floors, {})
    if len(unconstrained) < min_r:
        return unconstrained
    n = len(unconstrained)
    caps = {f: max(2, math.ceil(share * n)) for f, share in FAMILY_SHARE.items()}
    chosen = greedy(window, floors, caps)
    if len(chosen) < min_r:                       # caps starved a viable batch
        caps = {f: v + 1 for f, v in caps.items()}
        chosen = greedy(window, floors, caps)
    return chosen


def sweep(rows):
    print("\n=== Floor sweep: share of 72h windows yielding >= 7 selections ===")
    print("    (one selection per fixture, no diversity caps)\n")
    print("    " + "abs".rjust(6) + "k".rjust(7) + "qualifying fixtures".rjust(22)
          + "median/window".rjust(16) + ">=7 windows".rjust(14))
    for absolute in (0.60, 0.65, 0.70):
        for k in (0.15, 0.20, 0.25, 0.30):
            floors = {m: market_floor(m, absolute, k) for m in BASE_RATE}
            qualifying = [(s, t) for s, t, mk in rows
                          if any(p >= floors[m] for m, p in mk.items())]
            medians, shares = [], []
            for season in SEASONS:
                times = sorted(t for s, t in qualifying if s == season)
                allt = sorted(t for s, t, _ in rows if s == season)
                if not times:
                    continue
                counts, cursor = [], allt[0].replace(hour=0, minute=0)
                while cursor < allt[-1]:
                    lo = bisect.bisect_left(times, cursor)
                    hi = bisect.bisect_left(times, cursor + dt.timedelta(hours=HORIZON_HOURS))
                    counts.append(hi - lo)
                    cursor += dt.timedelta(hours=6)
                medians.append(statistics.median(counts))
                shares.append(100 * sum(1 for c in counts if c >= MIN_RECOMMENDATIONS) / len(counts))
            print("    " + f"{absolute:.2f}".rjust(6) + f"{k:.2f}".rjust(7)
                  + f"{100 * len(qualifying) / len(rows):.1f}%".rjust(22)
                  + f"{statistics.mean(medians):.1f}".rjust(16)
                  + f"{statistics.mean(shares):.1f}%".rjust(14))


def simulate(rows):
    floors = {m: market_floor(m) for m in BASE_RATE}
    print(f"\n=== Chosen rule: floor = max({ABSOLUTE_FLOOR}, base + {LIFT_K}(1-base)) ===\n")
    print("    " + "market".ljust(20) + "base".rjust(8) + "floor".rjust(8)
          + "lift".rjust(8) + "fixtures clearing".rjust(20))
    for market in BASE_RATE:
        clearing = sum(1 for _, _, mk in rows if mk[market] >= floors[market])
        print("    " + market.ljust(20) + f"{BASE_RATE[market]:>8.3f}{floors[market]:>8.3f}"
              + f"{floors[market] - BASE_RATE[market]:>+8.3f}"
              + f"{100 * clearing / len(rows):>19.1f}%")

    for label, caps_on in (("no diversity caps", False), ("with family caps", True)):
        mix, sizes, starved, windows = collections.Counter(), [], 0, 0
        print(f"\n    --- {label} ---")
        print("    " + "season".ljust(8) + "windows>=7".rjust(12)
              + "median".rjust(8) + "empty".rjust(8))
        for season in SEASONS:
            pairs = sorted((t, i) for i, (s, t, _) in enumerate(rows) if s == season)
            times = [t for t, _ in pairs]
            cursor, ok, total, sizes_s, empty = times[0].replace(hour=0, minute=0), 0, 0, [], 0
            while cursor < times[-1]:
                lo = bisect.bisect_left(times, cursor)
                hi = bisect.bisect_left(times, cursor + dt.timedelta(hours=HORIZON_HOURS))
                total += 1
                windows += 1
                window = [(times[j], rows[pairs[j][1]][2]) for j in range(lo, hi)]
                if not window:
                    empty += 1
                    cursor += dt.timedelta(hours=6)
                    continue
                raw = greedy(window, floors, {})
                chosen = select(window, floors) if caps_on else raw
                if len(raw) >= MIN_RECOMMENDATIONS and len(chosen) < MIN_RECOMMENDATIONS:
                    starved += 1
                if len(chosen) >= MIN_RECOMMENDATIONS:
                    ok += 1
                    sizes_s.append(len(chosen))
                    for _, _, m in chosen:
                        mix[FAMILY[m]] += 1
                cursor += dt.timedelta(hours=6)
            sizes += sizes_s
            print("    " + season.ljust(8) + f"{100 * ok / total:.1f}%".rjust(12)
                  + f"{statistics.median(sizes_s) if sizes_s else 0:.0f}".rjust(8)
                  + f"{100 * empty / total:.1f}%".rjust(8))
        total_mix = sum(mix.values())
        print("    family mix: " + str({k: f"{100 * v / total_mix:.0f}%" for k, v in mix.most_common()}))
        print(f"    batch size: median {statistics.median(sizes):.0f}  mean {statistics.mean(sizes):.1f}"
              f"  min {min(sizes)}  max {max(sizes)}")
        if caps_on:
            print(f"    caps starved an otherwise-viable batch in {starved} windows "
                  f"({100 * starved / windows:.2f}%) -- the measured cost of diversity")


def main():
    rows = load()
    print(f"Loaded {len(rows)} matches with complete odds "
          f"({SEASONS[0]}-{SEASONS[-1]}, 7 leagues)")
    sweep(rows)
    simulate(rows)
    print("\nReminder: market odds are sharper than this system's model will be.")
    print("Every figure above is a CEILING. Re-derive from the backtest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
