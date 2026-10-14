"""Post-match validation.

Every prediction is settled against the real result -- not only the ones that
were recommended. Evaluating only selections measures the selection policy, not
the model, and makes it impossible to tell whether the filter is helping.

Markets settle on ninety minutes. Extra time and penalties are excluded, which
matters for UEFA knockouts, so ``went_to_et`` exists to make the exclusion
explicit and testable rather than accidental.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Callable

from nway import clock
from nway.evaluation.metrics import log_loss_binary, brier_binary
from nway.validation.resolvers import MatchFacts, RESOLVERS, can_settle, settle
from nway.logging_setup import get_logger
from nway.storage.db import Database

log = get_logger(__name__)

WIN, LOSS, VOID, UNSETTLEABLE = "WIN", "LOSS", "VOID", "UNSETTLEABLE"


@dataclass
class SettlementSummary:
    fixtures_settled: int = 0
    predictions_settled: int = 0
    voided: int = 0
    unsettleable: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"fixtures": self.fixtures_settled,
                "predictions": self.predictions_settled,
                "voided": self.voided, "unsettleable": self.unsettleable}


def settle_finished_fixtures(db: Database, now: dt.datetime | None = None,
                             buffer_hours: float = 2.5) -> SettlementSummary:
    """Settle every unsettled prediction whose fixture has finished."""
    now = clock.ensure_utc(now or clock.now())
    cutoff = clock.to_iso(now - dt.timedelta(hours=buffer_hours))
    summary = SettlementSummary()

    # Corner and card facts come from team_match_stats, which publishes days
    # after the result; a market whose facts have not arrived settles
    # UNSETTLEABLE rather than as a loss.
    rows = db.query("""
        SELECT DISTINCT f.fixture_id, f.status, r.home_goals, r.away_goals,
               r.ht_home_goals, r.ht_away_goals, r.went_to_et, r.result_source,
               h.corners AS home_corners, a.corners AS away_corners,
               h.yellow_cards AS home_yellow, a.yellow_cards AS away_yellow,
               h.red_cards AS home_red, a.red_cards AS away_red
        FROM prediction p
        JOIN fixture f ON f.fixture_id = p.fixture_id
        LEFT JOIN match_result r ON r.fixture_id = f.fixture_id
        LEFT JOIN team_match_stats h
          ON h.fixture_id = f.fixture_id AND h.is_home = 1
        LEFT JOIN team_match_stats a
          ON a.fixture_id = f.fixture_id AND a.is_home = 0
        LEFT JOIN prediction_evaluation e ON e.prediction_id = p.prediction_id
        WHERE e.prediction_id IS NULL AND f.kickoff_utc <= ?
    """, (cutoff,))

    for row in rows:
        fixture_id = row["fixture_id"]
        predictions = db.query(
            "SELECT p.prediction_id, p.market_key, p.selection, "
            "p.calibrated_probability FROM prediction p "
            "LEFT JOIN prediction_evaluation e ON e.prediction_id = p.prediction_id "
            "WHERE p.fixture_id = ? AND e.prediction_id IS NULL", (fixture_id,))
        if not predictions:
            continue

        abandoned = row["status"] in ("POSTPONED", "CANCELLED", "SUSPENDED")
        has_result = row["home_goals"] is not None
        facts = None
        if has_result:
            total_cards = None
            if row["home_yellow"] is not None and row["away_yellow"] is not None:
                total_cards = (row["home_yellow"] + row["away_yellow"]
                               + (row["home_red"] or 0) + (row["away_red"] or 0))
            facts = MatchFacts(
                home_goals=row["home_goals"], away_goals=row["away_goals"],
                ht_home_goals=row["ht_home_goals"],
                ht_away_goals=row["ht_away_goals"],
                home_corners=row["home_corners"], away_corners=row["away_corners"],
                total_cards=total_cards, went_to_et=bool(row["went_to_et"]))

        with db.transaction():
            for prediction in predictions:
                market_key = prediction["market_key"]
                if abandoned:
                    outcome, hit = VOID, None
                elif not has_result:
                    # Kicked off long ago with no result: a data problem, not a
                    # prediction that lost. Tracked separately for exactly that
                    # reason -- a rising rate here is a source failure.
                    outcome, hit = UNSETTLEABLE, None
                else:
                    won = settle(market_key, facts)
                    if won is None:
                        # Unknown market, or the facts it needs have not been
                        # published. Tracked separately from a loss: a data gap
                        # is not a wrong prediction.
                        outcome, hit = UNSETTLEABLE, None
                    else:
                        outcome, hit = (WIN if won else LOSS), int(won)

                db.execute(
                    "INSERT OR REPLACE INTO market_outcome (fixture_id, market_key, "
                    "selection, outcome, outcome_value, settled_at, settlement_source) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (fixture_id, market_key, prediction["selection"], outcome,
                     _outcome_value(market_key, facts),
                     clock.to_iso(now), row["result_source"] or "unknown"))

                probability = prediction["calibrated_probability"]
                recommended = db.scalar(
                    "SELECT COUNT(*) FROM recommendation WHERE prediction_id=? "
                    "AND selected=1", (prediction["prediction_id"],), default=0)
                notified = db.scalar(
                    "SELECT COUNT(*) FROM notified_selection n JOIN prediction p "
                    "ON p.fixture_id=n.fixture_id AND p.market_key=n.market_key "
                    "WHERE p.prediction_id=?", (prediction["prediction_id"],), default=0)

                db.insert("prediction_evaluation", {
                    "prediction_id": prediction["prediction_id"],
                    "outcome": outcome, "hit": hit,
                    "log_loss": (log_loss_binary([probability], [hit])
                                 if hit is not None else None),
                    "brier": (brier_binary([probability], [hit])
                              if hit is not None else None),
                    "was_recommended": int(bool(recommended)),
                    "was_notified": int(bool(notified)),
                    "evaluated_at": clock.to_iso(now),
                }, or_ignore=True)

                summary.predictions_settled += 1
                if outcome == VOID:
                    summary.voided += 1
                elif outcome == UNSETTLEABLE:
                    summary.unsettleable += 1
        summary.fixtures_settled += 1

    if summary.predictions_settled:
        log.info("settled predictions", context=summary.as_dict())
    return summary


def _outcome_value(market_key: str, facts: MatchFacts | None) -> float | None:
    """The realised quantity behind the market, for reporting."""
    if facts is None:
        return None
    if market_key.startswith("CORNERS_") or "CORNERS_OVER" in market_key:
        return (None if facts.total_corners is None else float(facts.total_corners))
    if market_key.startswith("CARDS_"):
        return None if facts.total_cards is None else float(facts.total_cards)
    if market_key.startswith(("HT_", "HTFT_")):
        return None if facts.ht_total is None else float(facts.ht_total)
    return float(facts.total_goals)


def settle_backfill(db: Database, now: dt.datetime | None = None) -> SettlementSummary:
    """Settle everything settleable, ignoring the kickoff buffer.

    Used after a backtest or a bulk historical prediction run.
    """
    return settle_finished_fixtures(db, now=now, buffer_hours=0.0)
