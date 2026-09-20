"""Self-scheduling: when should the next run happen?

GitHub Actions has no delayed dispatch -- a job cannot ask to be re-run in four
hours. The only durable channel between runs is state the next run reads, so
each run computes its own next wake time, writes it down, and rewrites the
workflow's cron to match.

The decision is driven by real fixtures, not a fixed interval:

  * fixtures already inside the prediction horizon -> wake just before the
    send window opens;
  * fixtures further out -> wake when the earliest one enters the horizon;
  * no fixtures at all (a 19-day gap around international windows is normal)
    -> wake tomorrow.

The wake time is always clamped to at most 24 hours ahead. Sleeping longer
would risk the GitHub Actions cache expiring (7 days idle), the scheduled
workflow being disabled (60 days of repository inactivity), and any bug in this
function becoming unrecoverable without manual intervention.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nway import clock
from nway.config import PROJECT_ROOT, Config
from nway.logging_setup import get_logger
from nway.storage import repositories as repo
from nway.storage.db import Database

log = get_logger(__name__)

STATE_PATH = PROJECT_ROOT / "state" / "nway-state.json"
STATE_VERSION = 1

MIN_GAP_MINUTES = 10          # never re-run immediately; GitHub schedules drift
MAX_SLEEP_HOURS = 24.0        # always wake at least daily, whatever happens
# GitHub's scheduled runs are routinely late under load, so aim early enough
# that a delayed run still lands before the target send window.
SCHEDULE_SLACK_MINUTES = 20


@dataclass
class ScheduleDecision:
    next_run_at: str
    reason: str
    code: str
    fixtures_in_window: int
    next_kickoff: str | None
    sleep_hours: float
    cron: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_next_run(db: Database, config: Config,
                    now: dt.datetime | None = None) -> ScheduleDecision:
    now = clock.ensure_utc(now or clock.now())
    notifications = config.notifications or {}
    horizon = float(notifications.get("prediction_horizon_hours", 72))
    interval = float(notifications.get("scheduler_interval_minutes", 30))
    lead_time = float(notifications.get("target_lead_time_hours", 0.5))
    minimum = int(notifications.get("min_recommendations", 7))

    competition_ids = [row["competition_id"] for row in db.query(
        "SELECT competition_id FROM competition WHERE enabled = 1")]
    in_window = repo.fixtures_in_window(
        db, now, now + dt.timedelta(hours=horizon), competition_ids)

    next_row = db.query_one(
        "SELECT MIN(kickoff_utc) AS k FROM fixture WHERE kickoff_utc > ? "
        "AND status IN ('SCHEDULED','TIMED')", (clock.to_iso(now),))
    next_kickoff = (clock.from_iso(next_row["k"])
                    if next_row and next_row["k"] else None)

    if len(in_window) >= minimum:
        earliest = min(clock.from_iso(row["kickoff_utc"]) for row in in_window)
        send_at = earliest - dt.timedelta(hours=lead_time)
        if send_at > now + dt.timedelta(minutes=MIN_GAP_MINUTES):
            target = send_at - dt.timedelta(minutes=SCHEDULE_SLACK_MINUTES)
            code, reason = "SEND_WINDOW_APPROACHING", (
                f"{len(in_window)} fixtures in the next {horizon:.0f}h; send "
                f"window opens {clock.to_iso(send_at)}")
        else:
            # Inside or past the window: keep ticking at the normal cadence so
            # later kickoffs in the same batch still get their chance.
            target = now + dt.timedelta(minutes=interval)
            code, reason = "IN_SEND_WINDOW", (
                f"{len(in_window)} fixtures in window; inside the send window "
                f"now")
    elif in_window:
        target = now + dt.timedelta(minutes=interval * 2)
        code, reason = "BELOW_MINIMUM", (
            f"only {len(in_window)} fixtures in the next {horizon:.0f}h, "
            f"minimum is {minimum}; more may enter the window")
    elif next_kickoff is not None:
        enters_window = next_kickoff - dt.timedelta(hours=horizon)
        target = max(now + dt.timedelta(minutes=MIN_GAP_MINUTES), enters_window)
        days = (next_kickoff.date() - now.date()).days
        code, reason = "WAITING_FOR_FIXTURES", (
            f"no fixtures within {horizon:.0f}h; next kickoff "
            f"{next_kickoff.date()} ({days} days away), enters the window "
            f"{clock.to_iso(enters_window)}")
    else:
        target = now + dt.timedelta(hours=MAX_SLEEP_HOURS)
        code, reason = "NO_FIXTURES_KNOWN", (
            "no upcoming fixtures stored; re-checking daily in case the "
            "schedule is published")

    floor = now + dt.timedelta(minutes=MIN_GAP_MINUTES)
    ceiling = now + dt.timedelta(hours=MAX_SLEEP_HOURS)
    target = min(max(target, floor), ceiling)

    decision = ScheduleDecision(
        next_run_at=clock.to_iso(target), reason=reason, code=code,
        fixtures_in_window=len(in_window),
        next_kickoff=clock.to_iso(next_kickoff) if next_kickoff else None,
        sleep_hours=round(clock.hours_between(now, target), 3))
    decision.cron = build_cron(target, now)
    return decision


def build_cron(target: dt.datetime, now: dt.datetime) -> list[str]:
    """Cron lines that will fire around `target`, plus a daily safety net.

    A single cron entry matching one minute would be fragile: GitHub routinely
    runs scheduled jobs late, and a missed minute means a missed batch. A short
    dense window around the target absorbs that, and the daily entry means the
    loop can always recover even if the dense window is wrong.
    """
    target = clock.ensure_utc(target)
    lines: list[str] = []
    if clock.hours_between(now, target) <= MAX_SLEEP_HOURS:
        start_hour = max(0, target.hour - 1)
        end_hour = min(23, target.hour + 1)
        hours = (f"{start_hour}-{end_hour}" if start_hour != end_hour
                 else str(start_hour))
        lines.append(f"*/15 {hours} {target.day} {target.month} *")
    # Safety net: a daily run keeps the cache warm, keeps the scheduled
    # workflow from being disabled for inactivity, and recovers the loop if the
    # dense window above ever fails to fire.
    lines.append("17 6 * * *")
    return lines


# --------------------------------------------------------------- state file
def export_state(db: Database, decision: ScheduleDecision,
                 path: Path | None = None) -> Path:
    """Write the small, critical state that must outlive the database.

    The notified-selection ledger is what prevents the same recommendations
    being emailed twice. The database itself lives in a GitHub Actions cache,
    which can be evicted, and can be rebuilt from scratch -- but a rebuilt
    database has no memory of what was already sent. Keeping the ledger in a
    committed file means an evicted cache costs a rebuild, not a duplicate
    email, and the git history doubles as an audit trail of what went out.
    """
    path = Path(path or STATE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    ledger = [dict(row) for row in db.query(
        "SELECT fixture_id, market_key, selection, probability_sent, sent_at "
        "FROM notified_selection UNION "
        "SELECT fixture_id, market_key, selection, probability_sent, sent_at "
        "FROM notified_ledger ORDER BY sent_at DESC LIMIT 500")]

    batches = [dict(row) for row in db.query(
        "SELECT batch_id, created_at, status, skip_reason, recommendation_count, "
        "actual_send_time FROM prediction_batch ORDER BY batch_id DESC LIMIT 20")]

    payload = {
        "version": STATE_VERSION,
        "updated_at": clock.to_iso(clock.now()),
        "schedule": decision.to_dict(),
        "notified": ledger,
        "recent_batches": batches,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    log.info("state written", context={"path": str(path),
                                       "notified": len(ledger)})
    return path


def import_state(db: Database, path: Path | None = None) -> int:
    """Restore the ledger into the FK-free table after a database rebuild."""
    path = Path(path or STATE_PATH)
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        log.error("state file is not valid JSON", context={"path": str(path)})
        return 0
    rows = payload.get("notified") or []
    db.executemany(
        "INSERT OR REPLACE INTO notified_ledger (fixture_id, market_key, "
        "selection, probability_sent, sent_at) VALUES (?,?,?,?,?)",
        [(r["fixture_id"], r["market_key"], r["selection"],
          r["probability_sent"], r["sent_at"]) for r in rows])
    log.info("state restored", context={"notified": len(rows)})
    return len(rows)


def read_next_run(path: Path | None = None) -> dt.datetime | None:
    path = Path(path or STATE_PATH)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        return clock.from_iso(payload["schedule"]["next_run_at"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def is_due(now: dt.datetime | None = None, path: Path | None = None,
           grace_minutes: float = 5.0) -> tuple[bool, str]:
    """Should this run do the work, or exit cheaply?

    The gate is what makes a coarse cron acceptable: a run that is not due
    finishes in seconds instead of ingesting and predicting.
    """
    now = clock.ensure_utc(now or clock.now())
    scheduled = read_next_run(path)
    if scheduled is None:
        return True, "no schedule recorded yet; running"
    if now + dt.timedelta(minutes=grace_minutes) >= scheduled:
        return True, f"due (scheduled {clock.to_iso(scheduled)})"
    wait = clock.hours_between(now, scheduled)
    return False, (f"not due for {wait:.1f}h "
                   f"(scheduled {clock.to_iso(scheduled)})")
