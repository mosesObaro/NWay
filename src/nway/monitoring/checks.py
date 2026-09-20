"""Data-quality and drift monitoring.

A BLOCKING result stops prediction for the affected fixtures. The system
declines to predict rather than predicting from corrupt input -- silence is
recoverable, a confidently wrong prediction built on bad data is not.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from nway import clock
from nway.logging_setup import get_logger
from nway.storage import repositories as repo
from nway.storage.db import Database

log = get_logger(__name__)


@dataclass
class QualityReport:
    checks_run: int = 0
    warnings: int = 0
    errors: int = 0
    blocking: int = 0
    details: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"checks": self.checks_run, "warnings": self.warnings,
                "errors": self.errors, "blocking": self.blocking}


def run_quality_checks(db: Database, now: dt.datetime | None = None) -> QualityReport:
    now = clock.ensure_utc(now or clock.now())
    report = QualityReport()

    def record(name: str, severity: str, detail: str,
               entity_kind: str | None = None, entity_id: int | None = None) -> None:
        repo.record_quality_check(db, name, severity, detail, entity_kind, entity_id)
        report.checks_run += 1
        report.details.append(f"{severity}: {name} — {detail}")
        if severity == "WARN":
            report.warnings += 1
        elif severity == "ERROR":
            report.errors += 1
        elif severity == "BLOCKING":
            report.blocking += 1

    # Duplicate fixtures: the same pairing twice in one season.
    for row in db.query("""
        SELECT competition_id, season_id, home_team_id, away_team_id, COUNT(*) AS n
        FROM fixture GROUP BY 1,2,3,4 HAVING n > 1
    """):
        record("DUPLICATE_FIXTURE", "ERROR",
               f"{row['n']} rows for the same pairing in season {row['season_id']}")

    # Finished fixtures with no result, past the point where one should exist.
    stale_cutoff = clock.to_iso(now - dt.timedelta(hours=6))
    for row in db.query("""
        SELECT f.fixture_id, f.kickoff_utc FROM fixture f
        LEFT JOIN match_result r ON r.fixture_id = f.fixture_id
        WHERE f.status = 'FINISHED' AND r.fixture_id IS NULL AND f.kickoff_utc <= ?
        LIMIT 25
    """, (stale_cutoff,)):
        record("MISSING_RESULT", "ERROR",
               f"finished fixture has no result (kickoff {row['kickoff_utc']})",
               "FIXTURE", row["fixture_id"])

    # Impossible values.
    for row in db.query(
            "SELECT fixture_id FROM match_result WHERE home_goals < 0 OR away_goals < 0 "
            "OR home_goals > 20 OR away_goals > 20 LIMIT 10"):
        record("INVALID_SCORE", "BLOCKING", "implausible scoreline",
               "FIXTURE", row["fixture_id"])

    for row in db.query(
            "SELECT fixture_id, team_id FROM team_match_stats "
            "WHERE corners < 0 OR corners > 40 OR shots < 0 OR shots > 60 LIMIT 10"):
        record("INVALID_STATISTIC", "ERROR", "implausible team statistic",
               "FIXTURE", row["fixture_id"])

    # Entity names awaiting review: ingestion refused to guess them.
    pending = db.scalar(
        "SELECT COUNT(*) FROM entity_resolution_queue WHERE status='PENDING'", default=0)
    if pending:
        record("UNRESOLVED_ENTITIES", "WARN",
               f"{pending} provider names await manual review")

    # Statistics that have stopped arriving. football-data.co.uk lags up to ~3
    # days by design; well beyond that means the source has changed or broken.
    newest = db.scalar("SELECT MAX(knowledge_time) FROM team_match_stats", default=None)
    if newest:
        age = clock.hours_between(clock.from_iso(newest), now)
        if age > 24 * 14:
            record("STALE_STATISTICS", "WARN",
                   f"newest team statistics are {age / 24:.1f} days old")

    # Rising unsettleable rate is a source failure wearing a modelling disguise.
    recent = db.query_one("""
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN outcome='UNSETTLEABLE' THEN 1 ELSE 0 END) AS bad
        FROM prediction_evaluation WHERE evaluated_at >= ?
    """, (clock.to_iso(now - dt.timedelta(days=30)),))
    if recent and recent["n"] and recent["n"] >= 50:
        share = (recent["bad"] or 0) / recent["n"]
        if share > 0.05:
            record("UNSETTLEABLE_RATE", "ERROR",
                   f"{share:.1%} of recent predictions could not be settled")

    log.info("quality checks complete", context=report.as_dict())
    return report


def run_drift_checks(db: Database, now: dt.datetime | None = None,
                     window_days: int = 90) -> list[dict]:
    """Compare recent calibration against the longer-run baseline."""
    now = clock.ensure_utc(now or clock.now())
    recent_start = clock.to_iso(now - dt.timedelta(days=window_days))
    breaches: list[dict] = []

    for row in db.query("""
        SELECT p.market_key,
               COUNT(*) AS n,
               AVG(p.calibrated_probability) AS mean_pred,
               AVG(CAST(e.hit AS REAL)) AS actual
        FROM prediction_evaluation e
        JOIN prediction p ON p.prediction_id = e.prediction_id
        WHERE e.hit IS NOT NULL AND e.evaluated_at >= ?
        GROUP BY p.market_key HAVING n >= 100
    """, (recent_start,)):
        gap = abs((row["mean_pred"] or 0) - (row["actual"] or 0))
        breach = gap > 0.05
        db.insert("drift_check", {
            "run_at": clock.to_iso(now), "scope": "MARKET",
            "market_key": row["market_key"], "metric": "CALIBRATION_GAP",
            "baseline_value": row["mean_pred"], "current_value": row["actual"],
            "n_samples": row["n"], "breach": int(breach),
            "detail": f"predicted {row['mean_pred']:.3f} vs actual {row['actual']:.3f}",
        })
        if breach:
            breaches.append({"market": row["market_key"], "gap": round(gap, 4),
                             "n": row["n"]})
            log.warning("calibration drift", context={
                "market": row["market_key"], "gap": round(gap, 4), "n": row["n"]})
    return breaches
