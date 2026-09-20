import datetime as dt

import pytest

from nway import clock


def test_now_is_always_utc_aware():
    moment = clock.now()
    assert moment.tzinfo is not None
    assert moment.utcoffset() == dt.timedelta(0)


def test_naive_datetimes_are_rejected():
    # Assuming naive means UTC would paper over exactly the bug this guards.
    with pytest.raises(ValueError):
        clock.ensure_utc(dt.datetime(2026, 1, 1, 12, 0))


def test_iso_roundtrip_is_lossless():
    moment = dt.datetime(2026, 9, 20, 17, 30, tzinfo=clock.UTC)
    assert clock.to_iso(moment) == "2026-09-20T17:30:00Z"
    assert clock.from_iso("2026-09-20T17:30:00Z") == moment


def test_iso_strings_sort_chronologically():
    # The schema relies on this: lexicographic order == chronological order,
    # which is what keeps `knowledge_time <= as_of` index-friendly.
    moments = [dt.datetime(2026, 1, 5, tzinfo=clock.UTC),
               dt.datetime(2026, 1, 15, tzinfo=clock.UTC),
               dt.datetime(2026, 2, 1, tzinfo=clock.UTC)]
    encoded = [clock.to_iso(m) for m in moments]
    assert encoded == sorted(encoded)


def test_frozen_clock_restores_previous_state():
    target = dt.datetime(2023, 3, 14, 13, 30, tzinfo=clock.UTC)
    before = clock.now()
    with clock.frozen_at(target):
        assert clock.now() == target
    assert clock.now() >= before


def test_nested_freeze_restores_outer_value():
    outer = dt.datetime(2024, 1, 1, tzinfo=clock.UTC)
    inner = dt.datetime(2025, 1, 1, tzinfo=clock.UTC)
    with clock.frozen_at(outer):
        with clock.frozen_at(inner):
            assert clock.now() == inner
        assert clock.now() == outer


def test_hours_between_handles_offsets():
    start = dt.datetime(2026, 9, 20, 12, 0, tzinfo=clock.UTC)
    end = dt.datetime(2026, 9, 20, 18, 30, tzinfo=clock.UTC)
    assert clock.hours_between(start, end) == pytest.approx(6.5)
