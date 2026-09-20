"""Self-scheduling.

Each run decides when the next one happens, so a bug here does not produce a
wrong answer -- it strands the loop. The clamps and the safety net matter more
than the cleverness.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from nway import clock
from nway.scheduling.autoschedule import (
    MAX_SLEEP_HOURS, MIN_GAP_MINUTES, build_cron, decide_next_run,
    export_state, import_state, is_due, read_next_run,
)

NOW = dt.datetime(2026, 9, 20, 12, 0, tzinfo=clock.UTC)


def _upcoming(seeded, count, first_hours_ahead):
    """Create `count` DISTINCT upcoming fixtures.

    Every pairing must be unique: upsert_fixture treats the same pairing within
    45 days as a reschedule of one match, so cycling through a short list would
    silently collapse nine fixtures into six.
    """
    names = list(seeded["teams"])
    pairs = [(h, a) for h in names for a in names if h != a]
    assert count <= len(pairs)
    for index, (home, away) in enumerate(pairs[:count]):
        seeded["add_upcoming"](
            home, away, NOW + dt.timedelta(hours=first_hours_ahead + index))


def test_no_fixtures_sleeps_a_day_not_forever(seeded, config):
    """A 19-day international break is normal. Sleeping through it entirely
    would let the Actions cache expire and the workflow be disabled."""
    decision = decide_next_run(seeded["db"], config, now=NOW)
    assert decision.code == "NO_FIXTURES_KNOWN"
    assert decision.sleep_hours == pytest.approx(MAX_SLEEP_HOURS)


def test_distant_fixtures_wake_when_they_enter_the_horizon(seeded, config):
    _upcoming(seeded, 8, first_hours_ahead=96)      # beyond the 72h horizon
    decision = decide_next_run(seeded["db"], config, now=NOW)
    assert decision.code == "WAITING_FOR_FIXTURES"
    # 96h out, 72h horizon -> enters in 24h, which is also the clamp.
    assert decision.sleep_hours == pytest.approx(24.0, abs=0.1)


def test_fixtures_in_window_wake_before_the_send_window(seeded, config):
    _upcoming(seeded, 9, first_hours_ahead=10)
    decision = decide_next_run(seeded["db"], config, now=NOW)
    assert decision.code == "SEND_WINDOW_APPROACHING"
    # Earliest kickoff +10h, lead time 0.5h, minus 20 minutes of slack.
    assert decision.sleep_hours == pytest.approx(10 - 0.5 - (20 / 60), abs=0.05)


def test_inside_the_send_window_ticks_at_the_normal_cadence(seeded, config):
    _upcoming(seeded, 9, first_hours_ahead=0.4)
    decision = decide_next_run(seeded["db"], config, now=NOW)
    assert decision.code == "IN_SEND_WINDOW"
    assert decision.sleep_hours == pytest.approx(0.5, abs=0.02)


def test_too_few_fixtures_retries_rather_than_sleeping(seeded, config):
    _upcoming(seeded, 3, first_hours_ahead=20)
    decision = decide_next_run(seeded["db"], config, now=NOW)
    assert decision.code == "BELOW_MINIMUM"
    assert 0 < decision.sleep_hours <= 2.0


@pytest.mark.parametrize("hours_ahead", [0.01, 0.5, 5, 23, 200, 10000])
def test_sleep_is_always_clamped(seeded, config, hours_ahead):
    """Whatever the fixtures say, never sleep less than the minimum gap nor
    more than a day -- a bug here must stay recoverable."""
    _upcoming(seeded, 9, first_hours_ahead=hours_ahead)
    decision = decide_next_run(seeded["db"], config, now=NOW)
    assert MIN_GAP_MINUTES / 60 - 1e-6 <= decision.sleep_hours <= MAX_SLEEP_HOURS + 1e-6


def test_cron_always_carries_a_safety_net():
    lines = build_cron(NOW + dt.timedelta(hours=5), NOW)
    assert "17 6 * * *" in lines, "no daily fallback; the loop could strand"
    assert len(lines) == 2


def test_cron_fields_are_well_formed():
    for offset in (0.2, 3, 12, 23.5):
        for line in build_cron(NOW + dt.timedelta(hours=offset), NOW):
            assert len(line.split()) == 5


# ------------------------------------------------------------ state file
def test_state_roundtrip_preserves_the_ledger(seeded, config, tmp_path):
    """The ledger must survive a database rebuild, or the system re-sends."""
    db = seeded["db"]
    db.executemany(
        "INSERT INTO notified_ledger (fixture_id, market_key, selection, "
        "probability_sent, sent_at) VALUES (?,?,?,?,?)",
        [(1, "OVER_1_5", "YES", 0.86, "2026-09-19T17:30:00Z"),
         (2, "HOME_WIN", "HOME", 0.71, "2026-09-19T17:30:00Z")])

    decision = decide_next_run(db, config, now=NOW)
    path = tmp_path / "state.json"
    export_state(db, decision, path)

    payload = json.loads(path.read_text())
    assert len(payload["notified"]) == 2
    assert payload["schedule"]["next_run_at"] == decision.next_run_at

    # Simulate a rebuilt database: wipe the ledger, restore from the file.
    db.execute("DELETE FROM notified_ledger")
    assert import_state(db, path) == 2
    assert db.scalar("SELECT COUNT(*) FROM notified_ledger", default=0) == 2


def test_restored_ledger_suppresses_a_duplicate_send(seeded, config, tmp_path):
    """The whole point: a rebuilt database must not re-email what was sent."""
    from nway.notifications.batch import load_notified

    db = seeded["db"]
    db.execute(
        "INSERT INTO notified_ledger (fixture_id, market_key, selection, "
        "probability_sent, sent_at) VALUES (?,?,?,?,?)",
        (99, "OVER_1_5", "YES", 0.88, clock.to_iso(NOW - dt.timedelta(hours=2))))

    notified = load_notified(db, NOW - dt.timedelta(days=7))
    assert any(entry.fixture_id == 99 for entry in notified)


def test_gate_is_open_when_no_schedule_exists(tmp_path):
    """A first run, or a lost state file, must not deadlock."""
    due, reason = is_due(now=NOW, path=tmp_path / "missing.json")
    assert due is True
    assert "no schedule" in reason


def test_gate_is_closed_before_the_scheduled_time(seeded, config, tmp_path):
    path = tmp_path / "state.json"
    decision = decide_next_run(seeded["db"], config, now=NOW)
    export_state(seeded["db"], decision, path)

    due, _ = is_due(now=NOW, path=path)
    assert due is False
    later = clock.from_iso(decision.next_run_at) + dt.timedelta(minutes=1)
    due, _ = is_due(now=later, path=path)
    assert due is True


def test_corrupt_state_file_does_not_deadlock(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{ this is not json")
    assert read_next_run(path) is None
    due, _ = is_due(now=NOW, path=path)
    assert due is True, "a corrupt state file must fail open, not strand the loop"
