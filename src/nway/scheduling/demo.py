"""Replay a historical matchday end to end.

The live path needs a football-data.org token for fixtures and a Resend key for
delivery. This command needs neither: it copies the database, rewinds the clock
to a real historical moment, presents the fixtures of that weekend as upcoming,
and runs the *production* tick against them.

It never touches the real database, and it never sends a real email.
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

from nway import clock
from nway.config import Config
from nway.logging_setup import get_logger
from nway.scheduling.tick import TickReport, run_tick
from nway.storage.db import Database, connect

log = get_logger(__name__)


def find_busy_moment(db: Database, minimum: int = 10) -> dt.datetime | None:
    """A historical moment with at least `minimum` fixtures in the next 72 hours.

    Anchored 30 minutes before the earliest kickoff of that window, which is the
    system's own target lead time -- so the replay lands inside a real send
    window instead of an arbitrary instant.
    """
    # Start from mid-season: the opening weekends are a legitimate but
    # unrepresentative case (no season-to-date history, thinner form windows).
    rows = db.query("""
        SELECT kickoff_utc FROM fixture
        WHERE status = 'FINISHED' AND kickoff_utc >= '2025-11-01'
        ORDER BY kickoff_utc
    """)
    kickoffs = [clock.from_iso(row["kickoff_utc"]) for row in rows]
    if not kickoffs:
        return None
    for index, start in enumerate(kickoffs):
        horizon = start + dt.timedelta(hours=72)
        count = sum(1 for k in kickoffs[index:] if k <= horizon)
        if count >= minimum:
            return start - dt.timedelta(minutes=30)
    return None


def run_demo(config: Config, source_db: Database, *, as_of: dt.datetime | None = None,
             scratch_path: Path | None = None,
             provider: str = "console") -> TickReport | None:
    source_path = Path(source_db.path)
    if str(source_path) == ":memory:":
        raise RuntimeError("demo needs a file-backed database")

    scratch = scratch_path or source_path.with_name("nway_demo.db")
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(scratch) + suffix)
        candidate.unlink(missing_ok=True)
    shutil.copy2(source_path, scratch)

    db = connect(f"sqlite:///{scratch}")
    moment = as_of or find_busy_moment(db)
    if moment is None:
        log.error("no historical window with enough fixtures; ingest more history")
        return None

    window_end = moment + dt.timedelta(hours=72)
    # Present that weekend's matches as upcoming, and hide their results so the
    # as-of repository cannot see them -- the replay must not know the scores.
    db.execute(
        "UPDATE fixture SET status='TIMED' WHERE kickoff_utc > ? AND kickoff_utc <= ?",
        (clock.to_iso(moment), clock.to_iso(window_end)))
    db.execute(
        "UPDATE match_result SET knowledge_time = ? WHERE fixture_id IN "
        "(SELECT fixture_id FROM fixture WHERE kickoff_utc > ? AND kickoff_utc <= ?)",
        (clock.to_iso(window_end + dt.timedelta(hours=3)),
         clock.to_iso(moment), clock.to_iso(window_end)))
    db.execute(
        "UPDATE team_match_stats SET knowledge_time = ? WHERE fixture_id IN "
        "(SELECT fixture_id FROM fixture WHERE kickoff_utc > ? AND kickoff_utc <= ?)",
        (clock.to_iso(window_end + dt.timedelta(hours=75)),
         clock.to_iso(moment), clock.to_iso(window_end)))

    count = db.scalar(
        "SELECT COUNT(*) FROM fixture WHERE status='TIMED' AND kickoff_utc > ? "
        "AND kickoff_utc <= ?", (clock.to_iso(moment), clock.to_iso(window_end)),
        default=0)
    print(f"Replaying {clock.to_iso(moment)} — {count} fixtures in the next 72 hours")
    print(f"Scratch database: {scratch}\n")

    with clock.frozen_at(moment):
        report = run_tick(db, config, dry_run=False, skip_ingest=True,
                          provider_override=provider)
    return report
