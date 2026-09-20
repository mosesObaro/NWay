"""Walk-forward backtest.

A backtest that uses today's knowledge of history is not a backtest. Every
simulated prediction at time T sees only rows with ``knowledge_time <= T`` --
not ``event_time <= T``. A match played before T whose statistics only reached
us afterwards is invisible, exactly as it was at the time.

The backtester drives the *production* code path with a frozen clock rather
than reimplementing it, so a divergence between backtest and live behaviour is
a bug in one shared implementation instead of a discrepancy between two.

Red flags are checked automatically and reported. A held-out log loss below
~0.95 beats de-vigged bookmaker closing odds, which is implausible for this
feature set and is treated as a leakage alarm rather than a result.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nway import clock
from nway.config import PROJECT_ROOT, Config
from nway.evaluation.metrics import (
    brier_multiclass, expected_calibration_error, log_loss_multiclass, market_report,
)
from nway.features.context import AsOfRepository
from nway.logging_setup import get_logger
from nway.models.dixon_coles import DixonColesModel
from nway.markets import derive_markets
from nway.models.training import MARKET_RESOLVERS, OUTCOME_INDEX
from nway.notifications.planner import NotificationConfig
from nway.storage.db import Database

log = get_logger(__name__)

BACKTEST_DIR = PROJECT_ROOT / "data" / "backtests"
MARKET_CEILING = 0.9563      # de-vigged closing odds, measured on 21,544 matches
BASELINE = 1.0711            # pooled base rate, measured on the same sample


@dataclass
class BacktestResult:
    run_id: str
    start: dt.datetime
    end: dt.datetime
    n_predictions: int
    n_matches: int
    retrains: int
    log_loss_1x2: float
    brier_1x2: float
    baseline_log_loss: float
    market_reports: dict[str, dict[str, float]] = field(default_factory=dict)
    decisions: Counter = field(default_factory=Counter)
    batches: int = 0
    red_flags: list[str] = field(default_factory=list)
    output_dir: Path | None = None

    def summary(self) -> str:
        lines = [
            f"Backtest {self.run_id}",
            "=" * 74,
            f"period            {self.start.date()} .. {self.end.date()}",
            f"matches predicted {self.n_matches}",
            f"model refits      {self.retrains}",
            "",
            "1X2 (uncalibrated model against the measured benchmarks)",
            f"  model           {self.log_loss_1x2:.4f}   brier {self.brier_1x2:.4f}",
            f"  base rate       {self.baseline_log_loss:.4f}  (must beat)",
            f"  market ceiling  {MARKET_CEILING:.4f}  (not a realistic target)",
        ]
        verdict = ("BEATS the base rate" if self.log_loss_1x2 < self.baseline_log_loss
                   else "DOES NOT beat the base rate")
        lines.append(f"  verdict         {verdict}")

        if self.market_reports:
            lines += ["", "BY MARKET", "-" * 74,
                      f"{'market':<20}{'n':>7}{'pred':>8}{'actual':>8}"
                      f"{'brier':>9}{'ece':>8}{'auc':>8}"]
            for market_key, report in sorted(self.market_reports.items()):
                lines.append(
                    f"{market_key:<20}{report['n']:>7}{report['mean_predicted']:>8.3f}"
                    f"{report['actual_rate']:>8.3f}{report['brier']:>9.4f}"
                    f"{report['ece']:>8.4f}{report['auc']:>8.3f}")

        if self.decisions:
            lines += ["", "NOTIFICATION DECISIONS", "-" * 74]
            for code, count in self.decisions.most_common():
                lines.append(f"  {code:<34} {count}")
            lines.append(f"  batches that would have been sent: {self.batches}")

        lines += ["", "RED FLAGS", "-" * 74]
        if self.red_flags:
            lines += [f"  ! {flag}" for flag in self.red_flags]
        else:
            lines.append("  none")
        if self.output_dir:
            lines += ["", f"artefacts: {self.output_dir}"]
        return "\n".join(lines)


def run_backtest(db: Database, config: Config, *, start: dt.datetime,
                 end: dt.datetime, step_hours: float = 24.0,
                 retrain_days: int = 30, lookback_years: int = 9,
                 simulate_notifications: bool = True) -> BacktestResult:
    start = clock.ensure_utc(start)
    end = clock.ensure_utc(end)
    run_id = f"bt_{clock.now().strftime('%Y%m%d_%H%M%S')}"
    notifications = NotificationConfig.from_dict(config.notifications or {})

    competition_ids = [row["competition_id"] for row in db.query(
        "SELECT competition_id FROM competition WHERE enabled = 1")]

    model: DixonColesModel | None = None
    last_trained: dt.datetime | None = None
    retrains = 0

    probabilities_1x2: list[list[float]] = []
    outcomes_1x2: list[int] = []
    baseline_1x2: list[list[float]] = []
    per_market: dict[str, tuple[list[float], list[int]]] = {}
    decisions: Counter = Counter()
    batches = 0
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()

    cursor = start
    while cursor <= end:
        # Refit on a cadence, using only what was knowable at the time.
        if model is None or last_trained is None or \
                (cursor - last_trained).days >= retrain_days:
            repo = AsOfRepository(db, cursor)
            history = repo.results_for_training(
                competition_ids, since=cursor - dt.timedelta(days=365 * lookback_years))
            history = [m for m in history if m["home_goals"] is not None]
            if len(history) >= 500:
                model = DixonColesModel(xi=0.0030, l2=0.5).fit(history, as_of=cursor)
                last_trained = cursor
                retrains += 1
                base_rates = _base_rates(history)
            else:
                cursor += dt.timedelta(hours=step_hours)
                continue

        # Fixtures inside the rolling window at this moment.
        window_end = cursor + dt.timedelta(hours=notifications.prediction_horizon_hours)
        fixtures = db.query("""
            SELECT f.fixture_id, f.competition_id, f.home_team_id, f.away_team_id,
                   f.kickoff_utc, r.home_goals, r.away_goals, r.outcome
            FROM fixture f JOIN match_result r ON r.fixture_id = f.fixture_id
            WHERE f.kickoff_utc > ? AND f.kickoff_utc <= ?
              AND f.competition_id IN ({})
            ORDER BY f.kickoff_utc
        """.format(",".join("?" * len(competition_ids))),
            [clock.to_iso(cursor), clock.to_iso(window_end), *competition_ids])

        qualifying = 0
        for fixture in fixtures:
            markets = derive_markets(
                *model.predict_lambdas(fixture["home_team_id"],
                                       fixture["away_team_id"],
                                       fixture["competition_id"]), model.params.rho)
            # Each fixture scores once, at the first window that contains it.
            if fixture["fixture_id"] not in seen:
                seen.add(fixture["fixture_id"])
                probabilities_1x2.append([markets.probabilities["HOME_WIN"],
                                          markets.probabilities["DRAW"],
                                          markets.probabilities["AWAY_WIN"]])
                outcomes_1x2.append(OUTCOME_INDEX[fixture["outcome"]])
                rates = base_rates.get(fixture["competition_id"],
                                       (0.438, 0.248, 0.313))
                baseline_1x2.append(list(rates))
                for market_key, resolver in MARKET_RESOLVERS.items():
                    probability = markets.probabilities.get(market_key)
                    if probability is None:
                        continue
                    bucket = per_market.setdefault(market_key, ([], []))
                    bucket[0].append(probability)
                    bucket[1].append(resolver(fixture["home_goals"],
                                              fixture["away_goals"]))
                rows.append({
                    "as_of": clock.to_iso(cursor),
                    "fixture_id": fixture["fixture_id"],
                    "kickoff_utc": fixture["kickoff_utc"],
                    "competition_id": fixture["competition_id"],
                    "home_win": markets.probabilities["HOME_WIN"],
                    "draw": markets.probabilities["DRAW"],
                    "away_win": markets.probabilities["AWAY_WIN"],
                    "over_2_5": markets.probabilities["OVER_2_5"],
                    "btts": markets.probabilities["BTTS"],
                    "outcome": fixture["outcome"],
                    "total_goals": fixture["home_goals"] + fixture["away_goals"],
                })

            if simulate_notifications:
                # Would this fixture have produced a qualifying selection?
                for market_key in ("HOME_WIN", "AWAY_WIN", "DOUBLE_CHANCE_1X",
                                   "DOUBLE_CHANCE_X2", "OVER_1_5", "OVER_2_5",
                                   "UNDER_2_5", "BTTS"):
                    probability = markets.probabilities.get(market_key)
                    if probability is None:
                        continue
                    try:
                        floor = config.market(market_key).min_probability
                    except KeyError:
                        continue
                    if floor is not None and probability >= floor:
                        qualifying += 1
                        break

        if simulate_notifications:
            if not fixtures:
                decisions["NO_FIXTURES"] += 1
            elif qualifying < notifications.min_recommendations:
                decisions["INSUFFICIENT_QUALIFYING"] += 1
            else:
                decisions["SEND"] += 1
                batches += 1
        cursor += dt.timedelta(hours=step_hours)

    if not probabilities_1x2:
        raise RuntimeError("backtest produced no predictions; widen the period")

    log_loss = log_loss_multiclass(probabilities_1x2, outcomes_1x2)
    brier = brier_multiclass(probabilities_1x2, outcomes_1x2)
    baseline_log_loss = log_loss_multiclass(baseline_1x2, outcomes_1x2)

    reports: dict[str, dict[str, float]] = {}
    for market_key, (probabilities, outcomes) in sorted(per_market.items()):
        report = market_report(market_key, probabilities, outcomes)
        reports[market_key] = {
            "n": report.n, "mean_predicted": report.mean_predicted,
            "actual_rate": report.actual_rate, "log_loss": report.log_loss,
            "brier": report.brier, "ece": report.ece, "auc": report.auc,
            "ci_low": report.ci_low, "ci_high": report.ci_high,
        }

    result = BacktestResult(
        run_id=run_id, start=start, end=end, n_predictions=len(rows),
        n_matches=len(seen), retrains=retrains, log_loss_1x2=log_loss,
        brier_1x2=brier, baseline_log_loss=baseline_log_loss,
        market_reports=reports, decisions=decisions, batches=batches)
    result.red_flags = _red_flags(result)
    result.output_dir = _persist(run_id, config, result, rows)
    return result


def _base_rates(history: list[dict[str, Any]]) -> dict[int, tuple[float, float, float]]:
    tally: dict[int, Counter] = {}
    for match in history:
        tally.setdefault(match["competition_id"], Counter())[match["outcome"]] += 1
    rates: dict[int, tuple[float, float, float]] = {}
    for competition_id, counts in tally.items():
        total = sum(counts.values()) or 1
        rates[competition_id] = (counts["H"] / total, counts["D"] / total,
                                 counts["A"] / total)
    return rates


def _red_flags(result: BacktestResult) -> list[str]:
    """Signals treated as bugs until disproven."""
    flags: list[str] = []
    if result.log_loss_1x2 < 0.95:
        flags.append(
            f"log loss {result.log_loss_1x2:.4f} beats de-vigged closing odds "
            f"({MARKET_CEILING}). Implausible for this feature set — run the "
            f"leakage suite before believing it.")
    if result.log_loss_1x2 >= result.baseline_log_loss:
        flags.append(
            f"log loss {result.log_loss_1x2:.4f} does not beat the base rate "
            f"({result.baseline_log_loss:.4f}); the model has learned nothing.")
    for market_key, report in result.market_reports.items():
        if report["n"] < 200:
            continue
        gap = report["actual_rate"] - report["mean_predicted"]
        if abs(gap) > 0.08:
            flags.append(
                f"{market_key}: predicted {report['mean_predicted']:.3f} but "
                f"actual {report['actual_rate']:.3f} (gap {gap:+.3f}) — check "
                f"settlement and the window.")
    if result.decisions and result.decisions.get("SEND", 0) == len(
            [d for d in result.decisions.elements()]):
        flags.append("every window would have sent; the planner is not being exercised.")
    return flags


def _persist(run_id: str, config: Config, result: BacktestResult,
             rows: list[dict[str, Any]]) -> Path:
    directory = BACKTEST_DIR / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({
        "config_hash": config.config_hash,
        "start": clock.to_iso(result.start), "end": clock.to_iso(result.end),
    }, indent=2))
    (directory / "metrics.json").write_text(json.dumps({
        "log_loss_1x2": result.log_loss_1x2, "brier_1x2": result.brier_1x2,
        "baseline_log_loss": result.baseline_log_loss,
        "n_matches": result.n_matches, "retrains": result.retrains,
        "markets": result.market_reports,
    }, indent=2, default=str))
    # The decision log matters as much as the predictions: the distribution of
    # skip reasons over a season is the evidence for whether the thresholds are
    # set sensibly.
    (directory / "decisions.json").write_text(
        json.dumps(dict(result.decisions), indent=2))
    with (directory / "predictions.csv").open("w") as handle:
        if rows:
            handle.write(",".join(rows[0]) + "\n")
            for row in rows:
                handle.write(",".join(str(row[k]) for k in rows[0]) + "\n")
    (directory / "report.txt").write_text(result.summary())
    return directory
