"""Command-line interface.

    nway check                       # is everything configured?
    nway init-db
    nway ingest --history --seasons 2017/18..2025/26
    nway ingest                      # live fixtures and results
    nway train --through 2026-06-30
    nway backtest --from 2023-08-01 --to 2025-05-31
    nway tick [--dry-run] [--as-of ...]
    nway report [--window 90]
    nway explain --prediction-id 8812
    nway demo                        # replay a historical matchday, no keys needed
    nway status
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from nway import clock
from nway.config import load_config
from nway.logging_setup import get_logger, setup_logging
from nway.storage.db import connect

log = get_logger(__name__)


def _parse_moment(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    text = value.strip()
    if len(text) == 10:
        text += "T00:00:00Z"
    return clock.from_iso(text)


def _season_range(spec: str) -> list[str]:
    """'2017/18..2025/26' or a comma-separated list."""
    if ".." in spec:
        start, end = spec.split("..")
        first = int(start.split("/")[0])
        last = int(end.split("/")[0])
        return [f"{y}/{str(y + 1)[-2:]}" for y in range(first, last + 1)]
    return [s.strip() for s in spec.split(",") if s.strip()]


# --------------------------------------------------------------- commands
def cmd_init_db(args, config) -> int:
    db = connect(args.database)
    print(f"schema ready at {db.path}")
    return 0


def cmd_ingest(args, config) -> int:
    from nway.ingestion.pipeline import ingest_history, ingest_live

    db = connect(args.database)
    if args.history:
        seasons = _season_range(args.seasons)
        competitions = args.competitions.split(",") if args.competitions else None
        summary = ingest_history(db, config, seasons, competitions)
    else:
        summary = ingest_live(db, config, discovery_days=args.days)
    print(json.dumps(summary.as_dict(), indent=2))
    return 0


def cmd_train(args, config) -> int:
    from nway.models.training import train_goal_model

    db = connect(args.database)
    report = train_goal_model(
        db, config, train_end=_parse_moment(args.through),
        calibration_months=args.calibration_months,
        lookback_years=args.lookback_years)
    print(report.summary())

    benchmark = (config.benchmarks or {}).get("one_x_two_log_loss", {}) or {}
    achieved = report.metrics.get("log_loss_1x2")
    baseline = benchmark.get("pooled_base_rate")
    if achieved and baseline:
        verdict = "BEATS baseline" if achieved < baseline else "DOES NOT beat baseline"
        print(f"\n{verdict}: {achieved:.4f} vs {baseline:.4f}")
        alarm = benchmark.get("leak_alarm_below")
        if alarm and achieved < alarm:
            print(f"WARNING: {achieved:.4f} is below the leak alarm threshold "
                  f"{alarm}. Run the leakage suite before believing this.")
    return 0


def cmd_tick(args, config) -> int:
    from nway.scheduling.tick import tick

    db = connect(args.database)
    moment = _parse_moment(args.as_of)
    if moment:
        with clock.frozen_at(moment):
            report = tick(config, db, dry_run=args.dry_run,
                          skip_ingest=args.skip_ingest,
                          provider_override=args.provider)
    else:
        report = tick(config, db, dry_run=args.dry_run,
                      skip_ingest=args.skip_ingest, provider_override=args.provider)
    if report is None:
        print("another tick is running")
        return 0
    print(report.summary())
    return 0


def cmd_backtest(args, config) -> int:
    from nway.evaluation.backtest import run_backtest

    db = connect(args.database)
    result = run_backtest(
        db, config, start=_parse_moment(args.start), end=_parse_moment(args.end),
        step_hours=args.step_hours, retrain_days=args.retrain_days)
    print(result.summary())
    return 0


def cmd_report(args, config) -> int:
    from nway.evaluation.report import build_report

    db = connect(args.database)
    print(build_report(db, config, window_days=args.window))
    return 0


def cmd_explain(args, config) -> int:
    from nway.evaluation.report import explain_prediction

    db = connect(args.database)
    print(explain_prediction(db, args.prediction_id))
    return 0


def cmd_check(args, config) -> int:
    from nway.diagnostics import render, run_checks

    checks = run_checks(config)
    print(render(checks))
    return 1 if any(c.status == "fail" for c in checks) else 0


def cmd_demo(args, config) -> int:
    from nway.scheduling.demo import run_demo

    db = connect(args.database)
    report = run_demo(config, db, as_of=_parse_moment(args.as_of),
                      provider=args.provider)
    if report is None:
        return 1
    print(report.summary())
    return 0


def cmd_status(args, config) -> int:
    db = connect(args.database)
    rows = [
        ("competitions enabled", len(config.enabled_competitions())),
        ("fixtures", db.scalar("SELECT COUNT(*) FROM fixture", default=0)),
        ("results", db.scalar("SELECT COUNT(*) FROM match_result", default=0)),
        ("team stats", db.scalar("SELECT COUNT(*) FROM team_match_stats", default=0)),
        ("teams", db.scalar("SELECT COUNT(*) FROM team", default=0)),
        ("predictions", db.scalar("SELECT COUNT(*) FROM prediction", default=0)),
        ("settled", db.scalar("SELECT COUNT(*) FROM prediction_evaluation", default=0)),
        ("batches sent", db.scalar(
            "SELECT COUNT(*) FROM prediction_batch WHERE status='SENT'", default=0)),
        ("batches skipped", db.scalar(
            "SELECT COUNT(*) FROM prediction_batch WHERE status='SKIPPED'", default=0)),
        ("unresolved names", db.scalar(
            "SELECT COUNT(*) FROM entity_resolution_queue WHERE status='PENDING'",
            default=0)),
        ("config hash", config.config_hash),
    ]
    width = max(len(str(name)) for name, _ in rows)
    for name, value in rows:
        print(f"  {name:<{width}}  {value}")

    active = db.query_one(
        "SELECT model_version, trained_at, metrics FROM model_version "
        "WHERE is_active=1 ORDER BY trained_at DESC LIMIT 1")
    if active:
        metrics = json.loads(active["metrics"] or "{}")
        print(f"\n  active model  {active['model_version']}")
        print(f"  trained       {active['trained_at']}")
        if "log_loss_1x2" in metrics:
            print(f"  log loss      {metrics['log_loss_1x2']:.4f} "
                  f"(baseline {metrics.get('baseline_log_loss_1x2', float('nan')):.4f})")
    else:
        print("\n  no active model — run `nway train`")

    recent = db.query(
        "SELECT status, skip_reason, created_at, recommendation_count "
        "FROM prediction_batch ORDER BY batch_id DESC LIMIT 5")
    if recent:
        print("\n  recent decisions")
        for row in recent:
            label = row["skip_reason"] or row["status"]
            print(f"    {row['created_at']}  {label:<28} "
                  f"n={row['recommendation_count']}")
    return 0


# ------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nway", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", help="database URL (default from NWAY_DATABASE_URL)")
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--json-logs", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create the schema").set_defaults(func=cmd_init_db)

    ingest = sub.add_parser("ingest", help="ingest fixtures, results and statistics")
    ingest.add_argument("--history", action="store_true",
                        help="load historical statistics instead of live fixtures")
    ingest.add_argument("--seasons", default="2017/18..2025/26")
    ingest.add_argument("--competitions", default=None, help="comma-separated slugs")
    ingest.add_argument("--days", type=int, default=None, help="live discovery horizon")
    ingest.set_defaults(func=cmd_ingest)

    train = sub.add_parser("train", help="fit the goal model and calibrators")
    train.add_argument("--through", default=None, help="train window end (YYYY-MM-DD)")
    train.add_argument("--calibration-months", type=int, default=12)
    train.add_argument("--lookback-years", type=int, default=9)
    train.set_defaults(func=cmd_train)

    tick_parser = sub.add_parser("tick", help="one scheduler cycle")
    tick_parser.add_argument("--dry-run", action="store_true",
                             help="decide and report, send nothing")
    tick_parser.add_argument("--as-of", default=None, help="replay a moment in history")
    tick_parser.add_argument("--skip-ingest", action="store_true")
    tick_parser.add_argument("--provider", default=None,
                             help="override the email provider (console/file)")
    tick_parser.set_defaults(func=cmd_tick)

    backtest = sub.add_parser("backtest", help="walk-forward historical simulation")
    backtest.add_argument("--from", dest="start", required=True)
    backtest.add_argument("--to", dest="end", required=True)
    backtest.add_argument("--step-hours", type=float, default=24.0)
    backtest.add_argument("--retrain-days", type=int, default=30)
    backtest.set_defaults(func=cmd_backtest)

    report = sub.add_parser("report", help="evaluation report")
    report.add_argument("--window", type=int, default=90, help="days")
    report.set_defaults(func=cmd_report)

    explain = sub.add_parser("explain", help="why a prediction was made")
    explain.add_argument("--prediction-id", type=int, required=True)
    explain.set_defaults(func=cmd_explain)

    sub.add_parser(
        "check", help="verify credentials and data are set up correctly"
    ).set_defaults(func=cmd_check)

    demo = sub.add_parser(
        "demo", help="replay a historical matchday end to end (no credentials needed)")
    demo.add_argument("--as-of", default=None, help="historical moment to replay")
    demo.add_argument("--provider", default="console",
                      help="console (default) or file")
    demo.set_defaults(func=cmd_demo)

    sub.add_parser("status", help="system state").set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level, json_output=args.json_logs)
    config = load_config()
    try:
        return args.func(args, config)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        log.error("command failed", context={"command": args.command, "error": str(exc)})
        if (args.log_level or "").upper() == "DEBUG":
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
