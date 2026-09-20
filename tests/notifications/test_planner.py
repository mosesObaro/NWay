"""Notification rules.

The count matrix from the design is asserted literally, plus the behavioural
cases that matter most: international breaks, duplicate suppression, lead-time
expiry and cooldown.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pytest

from nway import clock
from nway.notifications.planner import (
    Decision, NotificationConfig, NotifiedSelection, PreviousBatch,
    should_send_notification,
)

NOW = dt.datetime(2026, 9, 20, 10, 0, tzinfo=clock.UTC)


@dataclass
class FakeCandidate:
    prediction_id: int
    fixture_id: int
    market_key: str = "OVER_1_5"
    selection: str = "YES"
    probability: float = 0.86
    competition_slug: str = "premier_league"
    competition_name: str = "Premier League"
    home_team: str = "Alpha"
    away_team: str = "Bravo"
    kickoff_utc: dt.datetime = NOW + dt.timedelta(hours=8)
    prediction_timestamp: dt.datetime = NOW - dt.timedelta(hours=1)
    lambda_home: float = 1.8
    lambda_away: float = 1.1
    model_version: str = "test_model"

    @property
    def match_label(self) -> str:
        return f"{self.home_team} vs {self.away_team}"


@dataclass
class FakeScored:
    candidate: FakeCandidate
    passed: bool = True
    rejection_reasons: list = None
    reliability_score: float = 0.8
    ranking_score: float = 0.8
    confidence_band: str = "MEDIUM"
    selected: bool = False

    def __post_init__(self):
        if self.rejection_reasons is None:
            self.rejection_reasons = []


def make(n: int, *, hours_ahead: int = 8, passed: bool = True,
         market: str = "OVER_1_5", start_id: int = 1) -> list[FakeScored]:
    return [FakeScored(FakeCandidate(
        prediction_id=start_id + i, fixture_id=start_id + i, market_key=market,
        kickoff_utc=NOW + dt.timedelta(hours=hours_ahead + i)), passed=passed)
        for i in range(n)]


def config(**overrides) -> NotificationConfig:
    base = dict(min_recommendations=7, max_recommendations=20,
                target_lead_time_hours=0.5, cooldown_hours=12,
                scheduler_interval_minutes=30, max_prediction_age_hours=12)
    base.update(overrides)
    return NotificationConfig(**base)


def decide(scored, fixtures=None, previous=(), notified=(), cfg=None, now=NOW):
    return should_send_notification(
        now=now,
        upcoming_fixtures=fixtures if fixtures is not None else [object()] * max(1, len(scored)),
        scored=scored, previous_batches=list(previous), notified=list(notified),
        config=cfg or config())


# ----------------------------------------------------------- count matrix
@pytest.mark.parametrize("count,should_send", [
    (0, False), (3, False), (6, False), (7, True), (12, True), (20, True),
    (25, True), (100, True),
])
def test_minimum_threshold(count, should_send):
    scored = make(count)
    fixtures = [object()] * count
    # Anchor the send window on the earliest kickoff so timing is satisfied.
    now = (min((s.candidate.kickoff_utc for s in scored), default=NOW)
           - dt.timedelta(minutes=30)) if scored else NOW
    decision, qualifying = decide(scored, fixtures=fixtures, now=now)
    assert decision.should_send is should_send
    if should_send:
        assert len(qualifying) == count


def test_zero_fixtures_is_silent_and_not_an_error():
    decision, qualifying = decide([], fixtures=[])
    assert decision.should_send is False
    assert decision.decision_code == Decision.NO_FIXTURES.value
    assert qualifying == []


def test_six_qualifying_does_not_send():
    scored = make(6)
    decision, _ = decide(scored, fixtures=[object()] * 6)
    assert decision.should_send is False
    assert decision.decision_code == Decision.INSUFFICIENT_QUALIFYING.value
    assert decision.prediction_count == 6


def test_thresholds_are_never_relaxed_to_reach_seven():
    """Five strong plus two weak must NOT become seven."""
    strong = make(5)
    weak = make(2, start_id=100, passed=False)
    for item in weak:
        item.rejection_reasons = ["BELOW_PROBABILITY_FLOOR"]
    decision, _ = decide(strong + weak, fixtures=[object()] * 7)
    assert decision.should_send is False
    assert decision.prediction_count == 5


# ------------------------------------------------------------- behaviour
def test_international_break_stays_silent():
    """The real 2024-11-11 to 11-19 FIFA window: no fixtures, no email, no error."""
    for day in range(11, 20):
        now = dt.datetime(2024, 11, day, 12, 0, tzinfo=clock.UTC)
        decision, _ = should_send_notification(
            now=now, upcoming_fixtures=[], scored=[], previous_batches=[],
            notified=[], config=config())
        assert decision.should_send is False
        assert decision.decision_code == Decision.NO_FIXTURES.value


def test_already_notified_selections_are_suppressed():
    scored = make(10)
    ledger = [NotifiedSelection(
        fixture_id=item.candidate.fixture_id, market_key=item.candidate.market_key,
        selection="YES", probability_sent=item.candidate.probability,
        sent_at=NOW - dt.timedelta(hours=2)) for item in scored[:5]]
    decision, _ = decide(scored, notified=ledger)
    # 10 minus 5 duplicates leaves 5, below the minimum.
    assert decision.should_send is False
    assert decision.suppressed_duplicates == 5


def test_material_probability_move_permits_a_resend():
    scored = make(8)
    ledger = [NotifiedSelection(
        fixture_id=item.candidate.fixture_id, market_key=item.candidate.market_key,
        selection="YES",
        probability_sent=item.candidate.probability - 0.15,   # a real move
        sent_at=NOW - dt.timedelta(hours=8)) for item in scored]
    now = min(s.candidate.kickoff_utc for s in scored) - dt.timedelta(minutes=30)
    decision, qualifying = decide(scored, notified=ledger, now=now)
    assert decision.should_send is True
    assert len(qualifying) == 8


def test_immaterial_drift_is_suppressed():
    scored = make(8)
    ledger = [NotifiedSelection(
        fixture_id=item.candidate.fixture_id, market_key=item.candidate.market_key,
        selection="YES",
        probability_sent=item.candidate.probability - 0.03,   # noise
        sent_at=NOW - dt.timedelta(hours=8)) for item in scored]
    decision, _ = decide(scored, notified=ledger)
    assert decision.should_send is False
    assert decision.suppressed_duplicates == 8


def test_cooldown_blocks_a_second_batch():
    scored = make(9)
    previous = [PreviousBatch(1, "SENT", NOW - dt.timedelta(hours=3), frozenset())]
    decision, _ = decide(scored, previous=previous)
    assert decision.should_send is False
    assert decision.decision_code == Decision.COOLDOWN_ACTIVE.value


def test_expired_lead_time_drops_matches_and_rechecks_the_minimum():
    soon = make(3, hours_ahead=0)          # already kicked off
    for item in soon:
        item.candidate.kickoff_utc = NOW - dt.timedelta(minutes=5)
    later = make(5, hours_ahead=10, start_id=50)
    decision, _ = decide(soon + later, fixtures=[object()] * 8)
    # 8 candidates, but only 5 retain lead time -> below the minimum.
    assert decision.should_send is False
    assert decision.decision_code == Decision.INSUFFICIENT_QUALIFYING.value


def test_stale_predictions_trigger_a_refresh_not_a_send():
    scored = make(9)
    for item in scored:
        item.candidate.prediction_timestamp = NOW - dt.timedelta(hours=30)
    decision, _ = decide(scored)
    assert decision.should_send is False
    assert decision.decision_code == Decision.STALE_PREDICTIONS.value


def test_waits_for_the_send_window():
    """Friday 10:00 with the earliest kickoff at 18:00 must wait, not send."""
    scored = make(11, hours_ahead=8)
    decision, _ = decide(scored)
    assert decision.should_send is False
    assert decision.decision_code == Decision.WAITING_FOR_SEND_WINDOW.value
    assert decision.recommended_send_time is not None
    # Anchored on the earliest kickoff minus the target lead time.
    expected = min(s.candidate.kickoff_utc for s in scored) - dt.timedelta(minutes=30)
    assert decision.recommended_send_time == clock.to_iso(expected)


def test_sends_inside_the_window():
    scored = make(11, hours_ahead=8)
    earliest = min(s.candidate.kickoff_utc for s in scored)
    decision, qualifying = decide(scored, now=earliest - dt.timedelta(minutes=35))
    assert decision.should_send is True
    assert decision.decision_code == Decision.SEND.value
    assert len(qualifying) == 11
    assert decision.lead_time_hours == pytest.approx(35 / 60, abs=0.02)


def test_daily_cap():
    scored = make(9)
    previous = [
        PreviousBatch(1, "SENT", NOW - dt.timedelta(hours=20), frozenset()),
        PreviousBatch(2, "SENT", NOW - dt.timedelta(hours=14), frozenset()),
    ]
    cfg = config(cooldown_hours=1, max_batches_per_day=2)
    now = NOW
    previous[0].actual_send_time = now - dt.timedelta(hours=8)
    previous[1].actual_send_time = now - dt.timedelta(hours=4)
    decision, _ = decide(scored, previous=previous, cfg=cfg, now=now)
    assert decision.should_send is False
    assert decision.decision_code == Decision.DAILY_CAP_REACHED.value


def test_insufficient_new_coverage_is_rejected():
    scored = make(9)
    fixture_ids = frozenset(item.candidate.fixture_id for item in scored)
    previous = [PreviousBatch(1, "SENT", NOW - dt.timedelta(hours=20), fixture_ids)]
    cfg = config(cooldown_hours=1)
    decision, _ = decide(scored, previous=previous, cfg=cfg)
    assert decision.should_send is False
    assert decision.decision_code == Decision.INSUFFICIENT_NEW_COVERAGE.value


def test_decision_is_deterministic():
    scored = make(11, hours_ahead=8)
    now = min(s.candidate.kickoff_utc for s in scored) - dt.timedelta(minutes=35)
    first, _ = decide(scored, now=now)
    second, _ = decide(scored, now=now)
    assert first.to_dict() == second.to_dict()


def test_disabled_notifications_never_send():
    decision, _ = decide(make(15), cfg=config(enabled=False))
    assert decision.should_send is False
    assert decision.decision_code == Decision.NOTIFICATIONS_DISABLED.value
