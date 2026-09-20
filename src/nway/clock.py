"""The single source of "now".

Nothing else in this package may call ``datetime.now()`` or ``utcnow()``. A
test in ``tests/unit/test_architecture.py`` enforces that.

Two reasons it matters. First, every scheduling and prediction decision has to
be made in UTC regardless of where the machine is; server-local time creeping
into a comparison is a bug that only shows up twice a year. Second, the
backtester replays the production code path by freezing this clock, so the
thing being tested is literally the thing that runs.
"""

from __future__ import annotations

import datetime as dt
from contextlib import contextmanager

UTC = dt.timezone.utc

_frozen: dt.datetime | None = None


def now() -> dt.datetime:
    """Current time, always timezone-aware UTC."""
    if _frozen is not None:
        return _frozen
    return dt.datetime.now(tz=UTC)


@contextmanager
def frozen_at(moment: dt.datetime):
    """Freeze the clock. Used by tests and by the backtester."""
    global _frozen
    previous = _frozen
    _frozen = ensure_utc(moment)
    try:
        yield _frozen
    finally:
        _frozen = previous


def ensure_utc(value: dt.datetime) -> dt.datetime:
    """Coerce to UTC, refusing naive datetimes.

    Naive datetimes are rejected rather than assumed to be UTC: an assumption
    here would silently paper over exactly the bug this module exists to
    prevent.
    """
    if value.tzinfo is None:
        raise ValueError(f"naive datetime rejected: {value!r} (must be tz-aware)")
    return value.astimezone(UTC)


def to_iso(value: dt.datetime) -> str:
    """UTC ISO-8601 with a Z suffix.

    Stored as TEXT so that lexicographic order equals chronological order,
    which keeps ``WHERE knowledge_time <= :as_of`` index-friendly in both
    SQLite and PostgreSQL.
    """
    return ensure_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def from_iso(value: str) -> dt.datetime:
    """Parse a stored timestamp back into an aware UTC datetime."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def hours_between(start: dt.datetime, end: dt.datetime) -> float:
    return (ensure_utc(end) - ensure_utc(start)).total_seconds() / 3600.0
