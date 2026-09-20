"""
Feasibility study run 2026-09-20, BEFORE any system implementation.

This is a one-off research script, not system code. It exists so that every
empirical claim in docs/DATA_SOURCES.md, docs/MODEL_DESIGN.md and
docs/NOTIFICATION_ARCHITECTURE.md can be reproduced and challenged.

It answers five questions that drive the design:

  Q1  Which of the 7 target leagues actually have shots / corners / cards
      history, and how far back?
  Q2  What are the real base rates per league (and therefore what does a
      "high probability" prediction mean per market)?
  Q3  Is "at least 7 qualifying matches inside a rolling 72h window"
      actually achievable, and how often is the window empty?
  Q4  What is the realistic *supply* of high-probability opportunities at
      each probability threshold?
  Q5  Which model family is justified by the data (Poisson vs negative
      binomial) for goals, corners and cards?

Usage:
    python3 research/feasibility_study.py --download   # fetch CSVs first
    python3 research/feasibility_study.py

Only stdlib is required. Data: https://www.football-data.co.uk (robots.txt
permits crawling; see docs/DATA_SOURCES.md for the licensing assessment).
"""

import argparse
import bisect
import collections
import csv
import datetime as dt
import math
import os
import statistics
import sys
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "evidence", "raw")
BASE_URL = "https://football-data.co.uk/mmz4281"

# football-data.co.uk division codes -> our canonical competition slugs
LEAGUES = {
    "E0": "premier_league",
    "SP1": "la_liga",
    "D1": "bundesliga",
    "I1": "serie_a",
    "F1": "ligue_1",
    "N1": "eredivisie",
    "P1": "primeira_liga",
}
SHORT = {
    "E0": "Premier League", "SP1": "La Liga", "D1": "Bundesliga",
    "I1": "Serie A", "F1": "Ligue 1", "N1": "Eredivisie",
    "P1": "Primeira Liga",
}
# Seasons used for the coverage probe (Q1) and the main analysis window.
PROBE_SEASONS = ["9394", "9596", "0001", "0506", "1011", "1516", "1617", "1718"]
MAIN_SEASONS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def download(seasons):
    os.makedirs(DATA_DIR, exist_ok=True)
    for season in seasons:
        for code in LEAGUES:
            path = os.path.join(DATA_DIR, f"{code}-{season}.csv")
            if os.path.exists(path) and os.path.getsize(path) > 200:
                continue
            url = f"{BASE_URL}/{season}/{code}.csv"
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    body = r.read()
                if len(body) > 200:
                    with open(path, "wb") as fh:
                        fh.write(body)
                    print(f"  fetched {code}-{season} ({len(body)} bytes)")
            except Exception as exc:  # noqa: BLE001 - research script
                print(f"  skip {code}-{season}: {exc}")


def _num(value):
    value = (value or "").strip()
    try:
        return float(value)
    except ValueError:
        return None


def _parse_kickoff(date_str, time_str):
    """football-data.co.uk dates are dd/mm/yy or dd/mm/yyyy, times are LOCAL.

    The production system must NOT use these timestamps for scheduling; it
    uses the UTC timestamps from football-data.org. They are good enough for
    a fixture-density study.
    """
    date_str = (date_str or "").strip()
    parsed = None
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            parsed = dt.datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        return None
    time_str = (time_str or "").strip()
    if time_str:
        try:
            hh, mm = time_str.split(":")
            parsed = parsed.replace(hour=int(hh), minute=int(mm))
        except ValueError:
            pass
    return parsed


def load(seasons):
    matches = []
    for code in LEAGUES:
        for season in seasons:
            path = os.path.join(DATA_DIR, f"{code}-{season}.csv")
            if not os.path.exists(path) or os.path.getsize(path) < 200:
                continue
            # latin-1: the files are not UTF-8 and the first header cell
            # carries a UTF-8 BOM ("﻿Div"). Both are ingestion traps.
            with open(path, encoding="latin-1") as fh:
                for row in csv.DictReader(fh):
                    if not row.get("HomeTeam") or not (row.get("FTHG") or "").strip():
                        continue
                    kickoff = _parse_kickoff(row.get("Date"), row.get("Time"))
                    if kickoff is None:
                        continue
                    try:
                        hg, ag = int(row["FTHG"]), int(row["FTAG"])
                    except (ValueError, KeyError):
                        continue
                    matches.append({
                        "league": code, "season": season, "kickoff": kickoff,
                        "hg": hg, "ag": ag, "result": row.get("FTR"),
                        "hs": _num(row.get("HS")), "as_": _num(row.get("AS")),
                        "hst": _num(row.get("HST")), "ast": _num(row.get("AST")),
                        "hc": _num(row.get("HC")), "ac": _num(row.get("AC")),
                        "hy": _num(row.get("HY")), "ay": _num(row.get("AY")),
                        "hr": _num(row.get("HR")), "ar": _num(row.get("AR")),
                        "hthg": _num(row.get("HTHG")), "htag": _num(row.get("HTAG")),
                        "referee": (row.get("Referee") or "").strip() or None,
                        "odds": _devig(row),
                    })
    return matches


def _devig(row):
    """Normalise B365 closing (fallback opening) odds to probabilities.

    Proportional de-vigging is crude but adequate as an upper-bound proxy for
    model sharpness. See docs/MODEL_DESIGN.md for why we use it as a ceiling
    benchmark only, never as a training target.
    """
    home = _num(row.get("B365CH")) or _num(row.get("B365H"))
    draw = _num(row.get("B365CD")) or _num(row.get("B365D"))
    away = _num(row.get("B365CA")) or _num(row.get("B365A"))
    out = {}
    if home and draw and away:
        total = 1 / home + 1 / draw + 1 / away
        out["1x2"] = ((1 / home) / total, (1 / draw) / total, (1 / away) / total)
    over = _num(row.get("B365C>2.5")) or _num(row.get("B365>2.5"))
    under = _num(row.get("B365C<2.5")) or _num(row.get("B365<2.5"))
    if over and under:
        out["o25"] = (1 / over) / (1 / over + 1 / under)
    return out or None


# --------------------------------------------------------------------------
# Q1 - stats coverage by league and season
# --------------------------------------------------------------------------

def q1_coverage():
    print("\n=== Q1  Detailed-stat coverage by league and season ===")
    print("    Y = column present and populated; . = absent/empty\n")
    seasons = PROBE_SEASONS + MAIN_SEASONS
    seasons = sorted(set(seasons), key=lambda s: (int(s[:2]) + 100 if int(s[:2]) < 90 else int(s[:2])))
    header = "    " + "league".ljust(16) + "".join(s.rjust(6) for s in seasons)
    print(header)
    for code in LEAGUES:
        cells = []
        for season in seasons:
            path = os.path.join(DATA_DIR, f"{code}-{season}.csv")
            if not os.path.exists(path) or os.path.getsize(path) < 200:
                cells.append("-")
                continue
            with open(path, encoding="latin-1") as fh:
                rows = [r for r in csv.DictReader(fh) if r.get("HomeTeam")]
            if not rows:
                cells.append("-")
                continue
            has_shots = any((r.get("HS") or "").strip() for r in rows)
            has_corners = any((r.get("HC") or "").strip() for r in rows)
            cells.append("Y" if (has_shots and has_corners) else ".")
        print("    " + SHORT[code].ljust(16) + "".join(c.rjust(6) for c in cells))
    print("\n    -> corners/cards models are only trainable from the first 'Y'.")


# --------------------------------------------------------------------------
# Q2 - base rates
# --------------------------------------------------------------------------

def q2_base_rates(matches):
    print("\n=== Q2  Base rates per league (these ARE the naive baseline) ===\n")
    cols = ("N", "H%", "D%", "A%", "AvgG", "O1.5", "O2.5", "O3.5", "BTTS", "Corners", "Cards")
    print("    " + "league".ljust(16) + "".join(c.rjust(9) for c in cols))
    for code in list(LEAGUES) + ["ALL"]:
        sel = matches if code == "ALL" else [m for m in matches if m["league"] == code]
        if not sel:
            continue
        n = len(sel)
        pct = lambda fn: 100 * sum(1 for m in sel if fn(m)) / n  # noqa: E731
        goals = [m["hg"] + m["ag"] for m in sel]
        corners = [m["hc"] + m["ac"] for m in sel if m["hc"] is not None and m["ac"] is not None]
        cards = [m["hy"] + m["ay"] + m["hr"] + m["ar"] for m in sel
                 if m["hy"] is not None and m["ar"] is not None]
        vals = [
            n,
            pct(lambda m: m["result"] == "H"), pct(lambda m: m["result"] == "D"),
            pct(lambda m: m["result"] == "A"), statistics.mean(goals),
            pct(lambda m: m["hg"] + m["ag"] > 1.5), pct(lambda m: m["hg"] + m["ag"] > 2.5),
            pct(lambda m: m["hg"] + m["ag"] > 3.5),
            pct(lambda m: m["hg"] > 0 and m["ag"] > 0),
            statistics.mean(corners) if corners else float("nan"),
            statistics.mean(cards) if cards else float("nan"),
        ]
        name = "ALL" if code == "ALL" else SHORT[code]
        print("    " + name.ljust(16) + f"{vals[0]:>9}" +
              "".join(f"{v:>9.2f}" for v in vals[1:]))


# --------------------------------------------------------------------------
# Q3 - rolling 72h fixture density
# --------------------------------------------------------------------------

def q3_fixture_density(matches, season="2425", horizon_hours=72, step_hours=6):
    print(f"\n=== Q3  Rolling {horizon_hours}h fixture density ({season}) ===\n")
    sel = sorted([m for m in matches if m["season"] == season], key=lambda m: m["kickoff"])
    if not sel:
        print("    no data for season")
        return
    times = [m["kickoff"] for m in sel]
    cursor = times[0].replace(hour=0, minute=0)
    windows = []
    while cursor < times[-1]:
        lo = bisect.bisect_left(times, cursor)
        hi = bisect.bisect_left(times, cursor + dt.timedelta(hours=horizon_hours))
        windows.append((cursor, hi - lo))
        cursor += dt.timedelta(hours=step_hours)
    counts = [c for _, c in windows]
    print(f"    {len(sel)} matches, {times[0].date()} -> {times[-1].date()}, "
          f"{len(windows)} windows")
    print(f"    median={statistics.median(counts):.0f}  mean={statistics.mean(counts):.1f}  "
          f"max={max(counts)}  min={min(counts)}\n")
    for threshold in (1, 3, 5, 7, 10, 15, 20, 30):
        share = 100 * sum(1 for c in counts if c >= threshold) / len(counts)
        print(f"    windows with >= {threshold:2} matches: {share:5.1f}%")
    empty = [t for t, c in windows if c == 0]
    print(f"\n    windows with ZERO matches: {len(empty)} "
          f"({100 * len(empty) / len(windows):.1f}%)  <- no-match suppression must handle these")
    if empty:
        runs, current = [], [empty[0]]
        for prev, nxt in zip(empty, empty[1:]):
            if nxt - prev <= dt.timedelta(hours=step_hours):
                current.append(nxt)
            else:
                runs.append(current)
                current = [nxt]
        runs.append(current)
        runs.sort(key=len, reverse=True)
        print("    longest empty stretches:")
        for run in runs[:5]:
            print(f"      {run[0].date()} -> {run[-1].date()}  ({(run[-1] - run[0]).days} days)")
    dow = collections.Counter(m["kickoff"].strftime("%a") for m in sel)
    print("    matches by weekday:", dict(sorted(dow.items(), key=lambda kv: -kv[1])))


# --------------------------------------------------------------------------
# Q4 - supply of high-probability opportunities
# --------------------------------------------------------------------------

def _lambda_from_over25(p_over):
    """Invert P(total > 2.5) under Poisson to recover the implied goal rate."""
    lo, hi = 0.2, 7.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p = 1 - math.exp(-mid) * (1 + mid + mid * mid / 2)
        lo, hi = (mid, hi) if p < p_over else (lo, mid)
    return (lo + hi) / 2


def q4_opportunity_supply(matches, median_window=21):
    print("\n=== Q4  Supply of high-probability opportunities ===")
    print("    (market-implied, de-vigged; an UPPER BOUND on a home-grown model)\n")
    withodds = [m for m in matches if m["odds"] and "1x2" in m["odds"]]
    witho25 = [m for m in withodds if "o25" in m["odds"]]
    if not witho25:
        print("    no odds data")
        return
    over15 = []
    for m in witho25:
        lam = _lambda_from_over25(m["odds"]["o25"])
        over15.append(1 - math.exp(-lam) * (1 + lam))
    cols = ("1X2 any", "O2.5", "U2.5", "O1.5*", ">=1 mkt", f"exp in {median_window}")
    print("    " + "thresh".ljust(9) + "".join(c.rjust(12) for c in cols))
    for threshold in (0.60, 0.65, 0.70, 0.72, 0.75, 0.80, 0.85, 0.90):
        p1x2 = sum(1 for m in withodds if max(m["odds"]["1x2"]) >= threshold) / len(withodds)
        pov = sum(1 for m in witho25 if m["odds"]["o25"] >= threshold) / len(witho25)
        pun = sum(1 for m in witho25 if 1 - m["odds"]["o25"] >= threshold) / len(witho25)
        p15 = sum(1 for p in over15 if p >= threshold) / len(over15)
        pany = sum(
            1 for i, m in enumerate(witho25)
            if max(max(m["odds"]["1x2"]), m["odds"]["o25"], 1 - m["odds"]["o25"], over15[i]) >= threshold
        ) / len(witho25)
        print("    " + f"{threshold:.0%}".ljust(9) +
              f"{p1x2:>11.1%} {pov:>11.1%} {pun:>11.1%} {p15:>11.1%} {pany:>11.1%}"
              f" {median_window * pany:>11.1f}")
    print("\n    * O1.5 derived by inverting the O2.5 line to a Poisson goal rate.")
    print("    -> a flat 80% floor yields ~6 matches in a median window: BELOW the")
    print("       minimum of 7. A ~72-75% floor plus per-market caps is the")
    print("       defensible choice. See docs/NOTIFICATION_ARCHITECTURE.md.")


# --------------------------------------------------------------------------
# Q4b - benchmark scores the models must beat
# --------------------------------------------------------------------------

def q4b_benchmarks(matches):
    print("\n=== Q4b  Benchmark scores for 1X2 (targets for MODEL acceptance) ===\n")
    sel = [m for m in matches if m["odds"] and "1x2" in m["odds"]]
    if not sel:
        return
    index = {"H": 0, "D": 1, "A": 2}
    n = len(sel)
    base = (
        sum(1 for m in sel if m["result"] == "H") / n,
        sum(1 for m in sel if m["result"] == "D") / n,
        sum(1 for m in sel if m["result"] == "A") / n,
    )

    def logloss(probs):
        return -sum(math.log(max(p[index[m["result"]]], 1e-12))
                    for p, m in zip(probs, sel)) / n

    def brier(probs):
        total = 0.0
        for p, m in zip(probs, sel):
            actual = [0, 0, 0]
            actual[index[m["result"]]] = 1
            total += sum((a - b) ** 2 for a, b in zip(p, actual))
        return total / n

    candidates = [
        ("uniform (1/3 each)", [(1 / 3, 1 / 3, 1 / 3)] * n),
        ("pooled base rate", [base] * n),
        ("market (de-vigged)", [m["odds"]["1x2"] for m in sel]),
    ]
    print("    " + "model".ljust(24) + "LogLoss".rjust(10) + "Brier(multi)".rjust(14))
    for name, probs in candidates:
        print("    " + name.ljust(24) + f"{logloss(probs):>10.4f}{brier(probs):>14.4f}")
    print(f"\n    n = {n} matches. The model must beat 'pooled base rate';")
    print("    'market' is the practical ceiling, not a realistic target.")

    print("\n    Calibration of the de-vigged market on Over 2.5 (method demo):")
    witho25 = [m for m in sel if "o25" in m["odds"]]
    buckets = collections.defaultdict(list)
    for m in witho25:
        buckets[min(int(m["odds"]["o25"] * 10), 9)].append(m)
    ece = 0.0
    print("    " + "bucket".ljust(12) + "n".rjust(7) + "predicted".rjust(11) + "actual".rjust(9) + "gap".rjust(9))
    for b in sorted(buckets):
        grp = buckets[b]
        pred = sum(m["odds"]["o25"] for m in grp) / len(grp)
        act = sum(1 for m in grp if m["hg"] + m["ag"] > 2.5) / len(grp)
        ece += len(grp) / len(witho25) * abs(pred - act)
        print("    " + f"{b / 10:.1f}-{b / 10 + 0.1:.1f}".ljust(12) +
              f"{len(grp):>7}{pred:>11.3f}{act:>9.3f}{act - pred:>+9.3f}")
    print(f"    Expected Calibration Error (10 bins): {ece:.4f}")


# --------------------------------------------------------------------------
# Q5 - model family justification
# --------------------------------------------------------------------------

def q5_dispersion(matches):
    print("\n=== Q5  Which distribution? (var/mean = 1.00 => Poisson-consistent) ===\n")

    def report(values, label):
        if not values:
            return
        mean = statistics.mean(values)
        var = statistics.variance(values)
        verdict = "Poisson OK" if var / mean < 1.15 else "overdispersed -> negative binomial"
        print(f"    {label:<24} n={len(values):>6}  mean={mean:>6.2f}  "
              f"var={var:>6.2f}  var/mean={var / mean:>5.2f}   {verdict}")

    report([m["hg"] + m["ag"] for m in matches], "total goals")
    report([m["hg"] for m in matches], "home goals")
    report([m["ag"] for m in matches], "away goals")
    report([m["hc"] + m["ac"] for m in matches if m["hc"] is not None], "total corners")
    report([m["hc"] for m in matches if m["hc"] is not None], "home corners")
    report([m["hy"] + m["ay"] + m["hr"] + m["ar"] for m in matches
            if m["hy"] is not None and m["ar"] is not None], "total cards")

    ht = [m for m in matches if m["hthg"] is not None]
    if ht:
        first = statistics.mean([m["hthg"] + m["htag"] for m in ht])
        second = statistics.mean([(m["hg"] + m["ag"]) - (m["hthg"] + m["htag"]) for m in ht])
        print(f"\n    half-time coverage: {len(ht)}/{len(matches)} "
              f"({100 * len(ht) / len(matches):.1f}%)")
        print(f"    mean 1st-half goals {first:.3f} | 2nd-half {second:.3f} "
              f"(2H/1H = {second / first:.3f})")
        print("    -> half markets need half-specific rates, NOT lambda/2.")

    print("\n    League card rates (why a single card intercept cannot work):")
    for code in LEAGUES:
        sel = [m for m in matches if m["league"] == code and m["hy"] is not None]
        if not sel:
            continue
        cards = [m["hy"] + m["ay"] + m["hr"] + m["ar"] for m in sel]
        refs = sum(1 for m in sel if m["referee"])
        print(f"      {SHORT[code]:<16} mean cards {statistics.mean(cards):.2f}   "
              f"referee known for {100 * refs / len(sel):5.1f}% of matches")

    print("\n    Home advantage by season (drift evidence):")
    for season in MAIN_SEASONS:
        sel = [m for m in matches if m["season"] == season]
        if not sel:
            continue
        hw = 100 * sum(1 for m in sel if m["result"] == "H") / len(sel)
        gd = statistics.mean([m["hg"] - m["ag"] for m in sel])
        flag = "  <- COVID, empty stadiums" if season == "2021" else ""
        print(f"      {season}: n={len(sel):>5}  home-win {hw:5.1f}%  mean(HG-AG) {gd:+.3f}{flag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="fetch CSVs before analysing")
    args = parser.parse_args()

    if args.download:
        print("Downloading football-data.co.uk CSVs ...")
        download(sorted(set(PROBE_SEASONS + MAIN_SEASONS)))

    if not os.path.isdir(DATA_DIR):
        print(f"No data in {DATA_DIR}. Run with --download first.", file=sys.stderr)
        return 1

    q1_coverage()
    matches = load(MAIN_SEASONS)
    print(f"\nLoaded {len(matches)} completed matches "
          f"({MAIN_SEASONS[0]}-{MAIN_SEASONS[-1]}, 7 leagues)")
    q2_base_rates(matches)
    q3_fixture_density(matches)
    q4_opportunity_supply(matches)
    q4b_benchmarks(matches)
    q5_dispersion(matches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
