"""Evaluation reporting and prediction auditing."""

from __future__ import annotations

import datetime as dt
import json

from nway import clock
from nway.evaluation.metrics import (
    brier_binary, expected_calibration_error, log_loss_binary, wilson_interval,
)
from nway.storage.db import Database

MIN_SAMPLES_FOR_CONCLUSION = 200


def build_report(db: Database, config, window_days: int = 90) -> str:
    """Performance by market, competition and confidence band.

    Every row carries n and a Wilson interval, and rows below the sample floor
    are shown as "insufficient sample" rather than omitted -- an omitted row
    looks like an absent problem.
    """
    now = clock.now()
    since = clock.to_iso(now - dt.timedelta(days=window_days))
    lines: list[str] = [
        f"NWay evaluation report — last {window_days} days",
        f"generated {clock.to_iso(now)}",
        "=" * 78, "",
    ]

    total = db.scalar(
        "SELECT COUNT(*) FROM prediction_evaluation WHERE evaluated_at >= ?",
        (since,), default=0)
    if not total:
        lines.append("No settled predictions in this window.")
        lines.append("")
        lines.append("This is normal before the first matches have been played and")
        lines.append("settled. Run `nway tick` through a matchday, then retry.")
        return "\n".join(lines)

    lines.append("BY MARKET")
    lines.append("-" * 78)
    lines.append(f"{'market':<20}{'n':>6}{'pred':>8}{'actual':>8}"
                 f"{'95% CI':>18}{'brier':>9}{'ece':>8}")
    for row in db.query("""
        SELECT p.market_key AS market, COUNT(*) AS n,
               AVG(p.calibrated_probability) AS pred, AVG(CAST(e.hit AS REAL)) AS actual,
               SUM(e.hit) AS hits
        FROM prediction_evaluation e
        JOIN prediction p ON p.prediction_id = e.prediction_id
        WHERE e.hit IS NOT NULL AND e.evaluated_at >= ?
        GROUP BY p.market_key ORDER BY n DESC
    """, (since,)):
        probabilities, outcomes = _market_samples(db, row["market"], since)
        low, high = wilson_interval(int(row["hits"] or 0), row["n"])
        ece = expected_calibration_error(probabilities, outcomes, equal_count=True)
        flag = "" if row["n"] >= MIN_SAMPLES_FOR_CONCLUSION else "  (insufficient sample)"
        lines.append(
            f"{row['market']:<20}{row['n']:>6}{row['pred']:>8.3f}{row['actual']:>8.3f}"
            f"{f'[{low:.3f},{high:.3f}]':>18}"
            f"{brier_binary(probabilities, outcomes):>9.4f}{ece:>8.4f}{flag}")

    lines += ["", "BY COMPETITION", "-" * 78,
              f"{'competition':<22}{'n':>6}{'pred':>8}{'actual':>8}{'log loss':>11}"]
    for row in db.query("""
        SELECT c.name AS competition, COUNT(*) AS n,
               AVG(p.calibrated_probability) AS pred,
               AVG(CAST(e.hit AS REAL)) AS actual, AVG(e.log_loss) AS ll
        FROM prediction_evaluation e
        JOIN prediction p ON p.prediction_id = e.prediction_id
        JOIN fixture f ON f.fixture_id = p.fixture_id
        JOIN competition c ON c.competition_id = f.competition_id
        WHERE e.hit IS NOT NULL AND e.evaluated_at >= ?
        GROUP BY c.name ORDER BY n DESC
    """, (since,)):
        flag = "" if row["n"] >= MIN_SAMPLES_FOR_CONCLUSION else "  (insufficient sample)"
        lines.append(f"{row['competition']:<22}{row['n']:>6}{row['pred']:>8.3f}"
                     f"{row['actual']:>8.3f}{row['ll'] or 0:>11.4f}{flag}")

    lines += ["", "RECOMMENDED vs ALL", "-" * 78]
    for label, clause in (("all predictions", ""),
                          ("recommended only", "AND e.was_recommended = 1"),
                          ("notified only", "AND e.was_notified = 1")):
        row = db.query_one(f"""
            SELECT COUNT(*) AS n, AVG(p.calibrated_probability) AS pred,
                   AVG(CAST(e.hit AS REAL)) AS actual, SUM(e.hit) AS hits
            FROM prediction_evaluation e
            JOIN prediction p ON p.prediction_id = e.prediction_id
            WHERE e.hit IS NOT NULL AND e.evaluated_at >= ? {clause}
        """, (since,))
        if not row or not row["n"]:
            lines.append(f"{label:<22}   none")
            continue
        low, high = wilson_interval(int(row["hits"] or 0), row["n"])
        lines.append(f"{label:<22}n={row['n']:<6} predicted {row['pred']:.3f}  "
                     f"actual {row['actual']:.3f}  [{low:.3f},{high:.3f}]")
    lines.append("")
    lines.append("Recommended predictions are selected for high probability and good")
    lines.append("historical reliability, so their hit rate is NOT an unbiased estimate")
    lines.append("of model quality. Both are shown for that reason.")

    lines += ["", "NOTIFICATION DECISIONS", "-" * 78]
    for row in db.query("""
        SELECT COALESCE(skip_reason, status) AS outcome, COUNT(*) AS n
        FROM prediction_batch WHERE created_at >= ?
        GROUP BY outcome ORDER BY n DESC
    """, (since,)):
        lines.append(f"  {row['outcome']:<32} {row['n']}")
    return "\n".join(lines)


def _market_samples(db: Database, market_key: str, since: str):
    rows = db.query("""
        SELECT p.calibrated_probability AS p, e.hit AS y
        FROM prediction_evaluation e
        JOIN prediction p ON p.prediction_id = e.prediction_id
        WHERE p.market_key = ? AND e.hit IS NOT NULL AND e.evaluated_at >= ?
    """, (market_key, since))
    return [r["p"] for r in rows], [r["y"] for r in rows]


def explain_prediction(db: Database, prediction_id: int) -> str:
    """Answer 'why did the system make this prediction?' from stored rows alone.

    Nothing is recomputed: the feature values as of that moment, the model and
    calibrator versions, the data watermark and the configuration hash are all
    persisted precisely so this question never requires re-running a pipeline.
    """
    row = db.query_one("""
        SELECT p.*, r.prediction_timestamp, r.kickoff_utc, r.hours_to_kickoff,
               r.refresh_stage, r.model_version, r.feature_version, r.config_hash,
               r.lambda_home, r.lambda_away, r.data_completeness,
               r.feature_staleness_hours, r.snapshot_id,
               c.name AS competition, h.canonical_name AS home,
               a.canonical_name AS away
        FROM prediction p
        JOIN prediction_run r ON r.prediction_run_id = p.prediction_run_id
        JOIN fixture f ON f.fixture_id = p.fixture_id
        JOIN competition c ON c.competition_id = f.competition_id
        JOIN team h ON h.team_id = f.home_team_id
        JOIN team a ON a.team_id = f.away_team_id
        WHERE p.prediction_id = ?
    """, (prediction_id,))
    if not row:
        return f"no prediction with id {prediction_id}"

    lines = [
        f"Prediction {prediction_id}",
        "=" * 70,
        f"Match          {row['home']} vs {row['away']}  ({row['competition']})",
        f"Kickoff        {row['kickoff_utc']}",
        f"Market         {row['market_key']} / {row['selection']}",
        f"Probability    {row['calibrated_probability']:.4f} "
        f"(raw {row['raw_probability']:.4f})",
        f"Derived from   {row['is_derived_from']}",
        "",
        "PROVENANCE",
        f"  as-of cutoff      {row['prediction_timestamp']}",
        f"  hours to kickoff  {row['hours_to_kickoff']:.2f}  (stage {row['refresh_stage']})",
        f"  model version     {row['model_version']}",
        f"  feature version   {row['feature_version']}",
        f"  calibrator        {row['calibrator_version'] or 'identity (uncalibrated)'}",
        f"  config hash       {row['config_hash']}",
        f"  data snapshot     {row['snapshot_id']}",
        f"  completeness      {row['data_completeness']:.2%}",
        f"  feature staleness {row['feature_staleness_hours']:.1f}h",
        "",
        "GOAL MODEL",
        f"  lambda home {row['lambda_home']:.3f}   lambda away {row['lambda_away']:.3f}"
        f"   total {row['lambda_home'] + row['lambda_away']:.3f}",
        "",
        "EXPLANATION (each line traces to a stored feature)",
    ]
    for line in db.query(
            "SELECT direction, feature_key, feature_value, reference_value, "
            "contribution FROM prediction_explanation WHERE prediction_id = ? "
            "ORDER BY rank", (prediction_id,)):
        marker = "+" if line["direction"] == "SUPPORT" else "-"
        reference = ("" if line["reference_value"] is None
                     else f"  vs reference {line['reference_value']:.3f}")
        lines.append(f"  {marker} {line['feature_key']:<42} "
                     f"value {line['feature_value']}{reference}  "
                     f"(contribution {line['contribution']:+.3f})")

    snapshot = db.query_one(
        "SELECT source_watermarks FROM data_snapshot WHERE snapshot_id = ?",
        (row["snapshot_id"],))
    if snapshot:
        lines += ["", "DATA WATERMARKS AT PREDICTION TIME"]
        for source, watermark in json.loads(snapshot["source_watermarks"]).items():
            lines.append(f"  {source:<20} {watermark}")

    evaluation = db.query_one(
        "SELECT outcome, hit, log_loss, brier, was_recommended, was_notified "
        "FROM prediction_evaluation WHERE prediction_id = ?", (prediction_id,))
    lines += ["", "OUTCOME"]
    if evaluation:
        lines.append(f"  {evaluation['outcome']}  hit={evaluation['hit']}  "
                     f"log_loss={evaluation['log_loss']}  brier={evaluation['brier']}")
        lines.append(f"  recommended={bool(evaluation['was_recommended'])}  "
                     f"notified={bool(evaluation['was_notified'])}")
    else:
        lines.append("  not yet settled")
    return "\n".join(lines)
