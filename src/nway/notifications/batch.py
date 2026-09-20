"""PredictionBatch lifecycle.

The batch is persisted before anything is sent, so a crash between persisting
and sending is recoverable rather than ambiguous. The duplicate-prevention
ledger is written only on confirmed delivery -- a failed send leaves the next
tick free to retry the same content instead of silently losing it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from nway import clock
from nway.logging_setup import get_logger
from nway.notifications.delivery.base import DeliveryResult, EmailMessage, EmailProvider
from nway.notifications.planner import (
    NotificationDecision, NotifiedSelection, PreviousBatch,
)
from nway.recommendation.engine import ScoredRecommendation
from nway.storage.db import Database

log = get_logger(__name__)

STATUS_PENDING = "PENDING"
STATUS_READY = "READY"
STATUS_SENT = "SENT"
STATUS_SKIPPED = "SKIPPED"
STATUS_FAILED = "FAILED"
STATUS_CANCELLED = "CANCELLED"


def load_previous_batches(db: Database, limit: int = 20) -> list[PreviousBatch]:
    rows = db.query(
        "SELECT batch_id, status, actual_send_time FROM prediction_batch "
        "ORDER BY batch_id DESC LIMIT ?", (limit,))
    batches: list[PreviousBatch] = []
    for row in rows:
        fixtures = {
            record["fixture_id"] for record in db.query(
                "SELECT DISTINCT fixture_id FROM notified_selection WHERE batch_id = ?",
                (row["batch_id"],))
        }
        batches.append(PreviousBatch(
            batch_id=row["batch_id"], status=row["status"],
            actual_send_time=(clock.from_iso(row["actual_send_time"])
                              if row["actual_send_time"] else None),
            fixture_ids=frozenset(fixtures)))
    return batches


def load_notified(db: Database, since: dt.datetime) -> list[NotifiedSelection]:
    rows = db.query(
        "SELECT fixture_id, market_key, selection, probability_sent, sent_at "
        "FROM notified_selection WHERE sent_at >= ?", (clock.to_iso(since),))
    return [NotifiedSelection(
        fixture_id=row["fixture_id"], market_key=row["market_key"],
        selection=row["selection"], probability_sent=row["probability_sent"],
        sent_at=clock.from_iso(row["sent_at"])) for row in rows]


def record_skip(db: Database, decision: NotificationDecision,
                config_hash: str) -> int:
    """Persist a decision not to send.

    Every silence gets a row. On a system that stays quiet most of the time,
    this is what makes "why was there no email on Saturday?" answerable.
    """
    return db.insert("prediction_batch", {
        "created_at": clock.to_iso(clock.now()),
        "prediction_window_start": decision.window_start or "",
        "prediction_window_end": decision.window_end or "",
        "scheduled_send_time": decision.recommended_send_time,
        "actual_send_time": None,
        "match_count": decision.match_count,
        "recommendation_count": decision.prediction_count,
        "status": STATUS_SKIPPED, "skip_reason": decision.decision_code,
        "decision_payload": json.dumps(decision.to_dict()),
        "model_version": None, "feature_version": None,
        "config_hash": config_hash,
    })


@dataclass
class BatchResult:
    batch_id: int
    status: str
    delivery: DeliveryResult | None = None


def create_and_send(db: Database, *, decision: NotificationDecision,
                    selections: Sequence[ScoredRecommendation],
                    provider: EmailProvider, rendered, config,
                    dry_run: bool = False) -> BatchResult:
    notifications = config.notifications or {}
    email_config = notifications.get("email", {}) or {}
    model_version = selections[0].candidate.model_version if selections else None

    batch_id = db.insert("prediction_batch", {
        "created_at": clock.to_iso(clock.now()),
        "prediction_window_start": decision.window_start or "",
        "prediction_window_end": decision.window_end or "",
        "scheduled_send_time": decision.recommended_send_time,
        "actual_send_time": None,
        "match_count": decision.match_count,
        "recommendation_count": decision.prediction_count,
        "status": STATUS_READY, "skip_reason": None,
        "decision_payload": json.dumps(decision.to_dict()),
        "model_version": model_version,
        "feature_version": None, "config_hash": config.config_hash,
    })

    if dry_run:
        db.execute("UPDATE prediction_batch SET status=? WHERE batch_id=?",
                   (STATUS_CANCELLED, batch_id))
        return BatchResult(batch_id, STATUS_CANCELLED)

    recipient = email_config.get("to_address")
    sender = email_config.get("from_address")
    if not recipient or not sender:
        db.execute(
            "UPDATE prediction_batch SET status=?, skip_reason=? WHERE batch_id=?",
            (STATUS_FAILED, "EMAIL_NOT_CONFIGURED", batch_id))
        log.error("email addresses not configured; set NWAY_EMAIL_FROM/TO in .env")
        return BatchResult(batch_id, STATUS_FAILED)

    # A stable key over the exact content: an ambiguous timeout cannot cause
    # the provider to deliver the same batch twice.
    fingerprint = hashlib.sha256(
        (str(batch_id) + "|" + "|".join(
            f"{item.candidate.fixture_id}:{item.candidate.market_key}"
            f":{item.candidate.probability:.4f}" for item in selections)
         ).encode()).hexdigest()[:32]

    message = EmailMessage(
        subject=rendered.subject, html=rendered.html, text=rendered.text,
        to=recipient, sender=sender,
        reply_to=email_config.get("reply_to") or None,
        idempotency_key=fingerprint)

    result = provider.send(message)

    db.insert("notification_log", {
        "batch_id": batch_id, "channel": "EMAIL", "provider": result.provider,
        "recipient": recipient, "subject": rendered.subject,
        "attempt": result.attempt, "status": "SENT" if result.success else "FAILED",
        "provider_message_id": result.message_id, "error": result.error,
        "sent_at": clock.to_iso(clock.now()) if result.success else None,
    })

    if not result.success:
        db.execute("UPDATE prediction_batch SET status=? WHERE batch_id=?",
                   (STATUS_FAILED, batch_id))
        log.error("batch delivery failed", context={
            "batch_id": batch_id, "error": result.error})
        return BatchResult(batch_id, STATUS_FAILED, result)

    sent_at = clock.to_iso(clock.now())
    # Ledger and status in one transaction: either the batch counts as sent and
    # its selections are suppressed later, or neither happens.
    with db.transaction():
        db.execute(
            "UPDATE prediction_batch SET status=?, actual_send_time=? WHERE batch_id=?",
            (STATUS_SENT, sent_at, batch_id))
        db.executemany(
            "INSERT OR REPLACE INTO notified_selection (fixture_id, market_key, "
            "selection, batch_id, prediction_id, probability_sent, sent_at) "
            "VALUES (?,?,?,?,?,?,?)",
            [(item.candidate.fixture_id, item.candidate.market_key,
              item.candidate.selection, batch_id, item.candidate.prediction_id,
              item.candidate.probability, sent_at) for item in selections])
        db.executemany(
            "UPDATE recommendation SET batch_id=? WHERE prediction_id=? AND selected=1",
            [(batch_id, item.candidate.prediction_id) for item in selections])

    log.info("batch sent", context={
        "batch_id": batch_id, "selections": len(selections),
        "message_id": result.message_id})
    return BatchResult(batch_id, STATUS_SENT, result)
