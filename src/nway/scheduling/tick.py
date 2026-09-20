"""The scheduler tick.

One idempotent command evaluates the whole world. Per-fixture cron entries
would mean thousands of jobs a season, each needing rescheduling whenever a
kickoff moves, and no way to reason about the system's state.

Idempotency is the core property: two ticks in the same minute must leave
identical state and send at most one email. It comes from a file lock, natural
keys on every write, prediction timestamps quantised to the tick boundary, and
the notified-selection ledger.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nway import clock
from nway.config import PROJECT_ROOT, Config
from nway.logging_setup import get_logger
from nway.monitoring.checks import run_quality_checks
from nway.notifications import batch as batch_mod
from nway.notifications.delivery.base import build_provider
from nway.notifications.planner import (
    NotificationConfig, should_send_notification,
)
from nway.notifications.render import render
from nway.prediction.pipeline import load_pipeline, refresh_stage, snapshot_for
from nway.recommendation.engine import RecommendationEngine
from nway.storage import repositories as repo
from nway.storage.db import Database
from nway.validation.settlement import settle_finished_fixtures

log = get_logger(__name__)

LOCK_PATH = PROJECT_ROOT / "data" / "nway.lock"

# Predictions are refreshed on this ladder. Each stage appends a NEW run; the
# previous one is marked superseded and never modified.
REFRESH_TARGETS = (60.0, 30.0, 12.0, 2.0)


class TickLock:
    """Exclusive file lock. A second tick exits quietly rather than interleaving."""

    def __init__(self, path: Path = LOCK_PATH) -> None:
        self.path = path
        self.handle: int | None = None

    def __enter__(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.handle = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.handle, str(os.getpid()).encode())
            return True
        except FileExistsError:
            # A stale lock from a killed process must not wedge the scheduler
            # forever, so anything older than an hour is reclaimed.
            try:
                age = clock.now().timestamp() - self.path.stat().st_mtime
                if age > 3600:
                    self.path.unlink()
                    return self.__enter__()
            except FileNotFoundError:
                return self.__enter__()
            return False

    def __exit__(self, *_exc: Any) -> None:
        if self.handle is not None:
            os.close(self.handle)
            self.path.unlink(missing_ok=True)


@dataclass
class TickReport:
    started_at: dt.datetime
    ingested: dict[str, int] = field(default_factory=dict)
    settled: dict[str, int] = field(default_factory=dict)
    quality: dict[str, int] = field(default_factory=dict)
    fixtures_in_window: int = 0
    predictions_made: int = 0
    candidates: int = 0
    eligible: int = 0
    selected: int = 0
    decision_code: str = ""
    decision_reason: str = ""
    batch_id: int | None = None
    email_sent: bool = False
    skipped: bool = False

    def summary(self) -> str:
        lines = [
            f"tick at {clock.to_iso(self.started_at)}",
            f"  fixtures in window   {self.fixtures_in_window}",
            f"  predictions made     {self.predictions_made}",
            f"  candidates / eligible {self.candidates} / {self.eligible}",
            f"  decision             {self.decision_code}",
            f"  reason               {self.decision_reason}",
        ]
        if self.selected:
            lines.append(f"  selected             {self.selected}")
        if self.batch_id:
            lines.append(f"  batch                {self.batch_id}"
                         f"{' (sent)' if self.email_sent else ''}")
        return "\n".join(lines)


def _quantise(moment: dt.datetime, minutes: int) -> dt.datetime:
    """Round down to the tick boundary.

    Two runs inside the same interval then share a prediction_timestamp and
    reuse the same prediction_run instead of creating a twin.
    """
    minute = (moment.minute // max(1, minutes)) * max(1, minutes)
    return moment.replace(minute=minute, second=0, microsecond=0)


def _needs_refresh(db: Database, fixture_id: int, hours_to_kickoff: float,
                   as_of: dt.datetime) -> bool:
    """Has this fixture crossed into a new refresh stage?"""
    latest = db.query_one(
        "SELECT prediction_timestamp, refresh_stage FROM prediction_run "
        "WHERE fixture_id=? AND superseded_by IS NULL "
        "ORDER BY prediction_timestamp DESC LIMIT 1", (fixture_id,))
    if latest is None:
        return True
    stage_now = refresh_stage(hours_to_kickoff)
    if stage_now != latest["refresh_stage"]:
        return True
    # Same stage: refresh only if the existing prediction has aged out.
    age = clock.hours_between(clock.from_iso(latest["prediction_timestamp"]), as_of)
    return age >= 12.0


def run_tick(db: Database, config: Config, *, now: dt.datetime | None = None,
             dry_run: bool = False, skip_ingest: bool = False,
             provider_override: str | None = None) -> TickReport:
    now = clock.ensure_utc(now or clock.now())
    notifications = NotificationConfig.from_dict(config.notifications or {})
    as_of = _quantise(now, int(notifications.scheduler_interval_minutes))
    report = TickReport(started_at=now)

    # 1-3. refresh fixtures, kickoff times and results
    if not skip_ingest:
        from nway.ingestion.pipeline import ingest_live
        try:
            report.ingested = ingest_live(db, config).as_dict()
        except Exception as exc:  # noqa: BLE001 — ingestion must not abort the tick
            log.error("ingestion failed during tick", context={"error": str(exc)})
            report.ingested = {"errors": 1}

    # 4-5. settle anything that has finished
    report.settled = settle_finished_fixtures(db, now).as_dict()

    # 6. data-quality gate
    report.quality = run_quality_checks(db, now).as_dict()

    # 7. the rolling window: one query across every enabled competition
    window_end = now + dt.timedelta(hours=notifications.prediction_horizon_hours)
    competition_ids = [
        row["competition_id"] for row in db.query(
            "SELECT competition_id FROM competition WHERE enabled = 1")]
    fixtures = repo.fixtures_in_window(db, now, window_end, competition_ids)
    report.fixtures_in_window = len(fixtures)

    # No supported fixtures is an ordinary outcome, not an error. Measured:
    # 15.7% of 72h windows in 2024/25 were completely empty, in runs of a week
    # or more during international breaks.
    if not fixtures:
        decision, _ = should_send_notification(
            now=now, upcoming_fixtures=[], scored=[], previous_batches=[],
            notified=[], config=notifications)
        report.decision_code = decision.decision_code
        report.decision_reason = decision.reason
        report.skipped = True
        log.info("no supported fixtures in window", context={
            "window_end": clock.to_iso(window_end)})
        return report

    # 8-9. predict and refresh where due
    try:
        pipeline = load_pipeline(db, config)
    except RuntimeError as exc:
        log.error("no model available", context={"error": str(exc)})
        report.decision_code = "NO_MODEL"
        report.decision_reason = str(exc)
        return report

    snapshot_id = snapshot_for(db, as_of)
    for fixture in fixtures:
        fixture_dict = dict(fixture)
        hours = clock.hours_between(now, clock.from_iso(fixture_dict["kickoff_utc"]))
        if hours > max(REFRESH_TARGETS):
            continue
        if not _needs_refresh(db, fixture_dict["fixture_id"], hours, as_of):
            continue
        try:
            result = pipeline.predict_fixture(fixture_dict, as_of, snapshot_id)
            if not result.was_skipped:
                report.predictions_made += 1
        except Exception as exc:  # noqa: BLE001 — one bad fixture must not stop the tick
            log.error("prediction failed", context={
                "fixture_id": fixture_dict["fixture_id"], "error": str(exc)})

    # 10. score candidates
    engine = RecommendationEngine(db, config)
    candidates = engine.load_candidates(now, window_end, as_of)
    scored = engine.score(candidates, now)
    report.candidates = len(scored)
    report.eligible = sum(1 for item in scored if item.passed)

    # 11. the decision
    previous = batch_mod.load_previous_batches(db)
    notified = batch_mod.load_notified(db, now - dt.timedelta(days=7))
    decision, qualifying = should_send_notification(
        now=now, upcoming_fixtures=fixtures, scored=scored,
        previous_batches=previous, notified=notified, config=notifications)
    report.decision_code = decision.decision_code
    report.decision_reason = decision.reason

    if not decision.should_send:
        engine.persist(scored, now)
        batch_mod.record_skip(db, decision, config.config_hash)
        report.skipped = True
        log.info("no notification", context={
            "decision": decision.decision_code, "reason": decision.reason})
        return report

    # 12. select, render, send
    selections = engine.select(qualifying)
    if len(selections) < notifications.min_recommendations:
        # The diversity caps starved an otherwise-viable batch. Measured cost:
        # 8.45% of windows. Skip rather than relax quality to reach seven.
        decision.should_send = False
        decision.decision_code = "INSUFFICIENT_QUALIFYING"
        decision.reason = (f"{len(selections)} selections after diversity caps, "
                           f"minimum is {notifications.min_recommendations}")
        engine.persist(scored, now)
        batch_mod.record_skip(db, decision, config.config_hash)
        report.decision_code = decision.decision_code
        report.decision_reason = decision.reason
        report.skipped = True
        return report

    decision.prediction_count = len(selections)
    decision.match_count = len({item.candidate.fixture_id for item in selections})
    decision.prediction_ids = [item.candidate.prediction_id for item in selections]
    decision.market_mix = {}
    for item in selections:
        key = item.candidate.market_key
        decision.market_mix[key] = decision.market_mix.get(key, 0) + 1

    engine.persist(scored, now)
    report.selected = len(selections)

    rendered = render(
        db, selections=selections, decision=decision, now=now,
        timezone=notifications.user_timezone,
        model_version=pipeline.model_version,
        feature_version=(selections[0].candidate.prediction_run_id and
                         db.scalar("SELECT feature_version FROM prediction_run "
                                   "WHERE prediction_run_id=?",
                                   (selections[0].candidate.prediction_run_id,),
                                   default="") or ""),
        subject_template=(config.notifications or {}).get("subject_template"))

    provider = build_provider(config, force=provider_override)
    result = batch_mod.create_and_send(
        db, decision=decision, selections=selections, provider=provider,
        rendered=rendered, config=config, dry_run=dry_run)
    report.batch_id = result.batch_id
    report.email_sent = result.status == batch_mod.STATUS_SENT
    return report


def tick(config: Config, db: Database, **kwargs: Any) -> TickReport | None:
    """Entry point with the lock. Returns None when another tick holds it."""
    lock = TickLock()
    with lock as acquired:
        if not acquired:
            log.info("another tick is running; exiting quietly")
            return None
        return run_tick(db, config, **kwargs)
