"""Model training and calibration fitting.

Both are strictly temporal. The model trains on matches knowable before
``train_end``; the calibrator is fitted on a *later, disjoint* window; and the
test period comes after both. Sharing the calibration and test folds inflates
every calibration number reported, which is the most common way a project like
this fools itself.
"""

from __future__ import annotations

import datetime as dt
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from nway import clock
from nway.calibration.calibrators import fit_calibrator
from nway.config import PROJECT_ROOT, Config
from nway.evaluation.metrics import (
    brier_multiclass, expected_calibration_error, log_loss_multiclass,
)
from nway.features.context import AsOfRepository
from nway.logging_setup import get_logger
from nway.models.baselines import BaseRateBaseline
from nway.models.dixon_coles import DixonColesModel
from nway.markets import derive_markets
from nway.prediction.pipeline import STORED_MARKETS
from nway.storage.db import Database

log = get_logger(__name__)

CALIBRATOR_DIR = PROJECT_ROOT / "data" / "calibrators"
OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}

# Which derived market a real outcome settles. Kept beside training because the
# calibrator needs the same resolution rule the settler uses.
MARKET_RESOLVERS = {
    "HOME_WIN": lambda h, a: int(h > a),
    "DRAW": lambda h, a: int(h == a),
    "AWAY_WIN": lambda h, a: int(h < a),
    "DOUBLE_CHANCE_1X": lambda h, a: int(h >= a),
    "DOUBLE_CHANCE_X2": lambda h, a: int(h <= a),
    "OVER_0_5": lambda h, a: int(h + a > 0),
    "OVER_1_5": lambda h, a: int(h + a > 1),
    "OVER_2_5": lambda h, a: int(h + a > 2),
    "OVER_3_5": lambda h, a: int(h + a > 3),
    "UNDER_2_5": lambda h, a: int(h + a <= 2),
    "BTTS": lambda h, a: int(h > 0 and a > 0),
    "HOME_CLEAN_SHEET": lambda h, a: int(a == 0),
    "AWAY_CLEAN_SHEET": lambda h, a: int(h == 0),
    "HOME_TO_SCORE": lambda h, a: int(h > 0),
    "AWAY_TO_SCORE": lambda h, a: int(a > 0),
}


@dataclass
class TrainingReport:
    model_version: str
    n_train: int
    n_calibration: int
    train_start: dt.datetime
    train_end: dt.datetime
    metrics: dict[str, Any]
    calibrators_fitted: int

    def summary(self) -> str:
        lines = [
            f"model            {self.model_version}",
            f"training matches {self.n_train}",
            f"calibration set  {self.n_calibration}",
            f"window           {self.train_start.date()} .. {self.train_end.date()}",
        ]
        for name, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"{name:<16} {value:.4f}")
        lines.append(f"calibrators      {self.calibrators_fitted}")
        return "\n".join(lines)


def _matches_between(db: Database, start: dt.datetime, end: dt.datetime,
                     competition_ids: Sequence[int] | None = None) -> list[dict[str, Any]]:
    """Completed matches in a window, read through the as-of boundary at `end`."""
    repo = AsOfRepository(db, end)
    matches = repo.results_for_training(competition_ids, since=start)
    return [m for m in matches if m["home_goals"] is not None]


def train_goal_model(db: Database, config: Config, *,
                     train_end: dt.datetime | None = None,
                     calibration_months: int = 12,
                     lookback_years: int = 9,
                     xi: float | None = None, l2: float | None = None,
                     model_version: str | None = None,
                     register: bool = True) -> TrainingReport:
    """Fit Dixon-Coles, then fit calibrators on a later, disjoint window."""
    settings = config.models.get("goals_dixon_coles", {})
    train_end = clock.ensure_utc(train_end or clock.now())
    calibration_start = train_end - dt.timedelta(days=30 * calibration_months)
    train_start = train_end - dt.timedelta(days=365 * lookback_years)

    xi = xi if xi is not None else float(
        (settings.get("time_decay") or {}).get("initial", 0.0030))
    l2 = l2 if l2 is not None else 0.5

    competition_ids = [
        row["competition_id"] for row in db.query(
            "SELECT competition_id FROM competition WHERE enabled = 1")
    ]

    # The model trains on everything up to train_end. The calibration slice is
    # the most recent stretch of that, held out for the calibrator only.
    all_matches = _matches_between(db, train_start, train_end, competition_ids)
    if len(all_matches) < 500:
        raise RuntimeError(f"only {len(all_matches)} matches available; ingest more history")

    fit_matches = [m for m in all_matches
                   if clock.from_iso(m["kickoff_utc"]) < calibration_start]
    calibration_matches = [m for m in all_matches
                           if clock.from_iso(m["kickoff_utc"]) >= calibration_start]
    if len(fit_matches) < 400:
        fit_matches, calibration_matches = all_matches, []

    model = DixonColesModel(xi=xi, l2=l2).fit(fit_matches, as_of=calibration_start
                                              if calibration_matches else train_end)

    metrics = _score(model, calibration_matches or fit_matches)
    baseline = BaseRateBaseline().fit(fit_matches)
    metrics.update(_score_baseline(baseline, calibration_matches or fit_matches))

    version = model_version or f"goals_dc_{clock.now().strftime('%Y%m%d_%H%M%S')}"
    artifact = model.to_artifact(
        version, train_start,
        calibration_start if calibration_matches else train_end,
        metrics=metrics)
    if register:
        artifact.register(db, active=True)

    fitted = 0
    if calibration_matches:
        fitted = fit_calibrators(db, model, version, calibration_matches,
                                 calibration_start, train_end, config)

    log.info("training complete", context={
        "model": version, "train": len(fit_matches),
        "calibration": len(calibration_matches),
        "log_loss": round(metrics.get("log_loss_1x2", float("nan")), 4),
        "baseline": round(metrics.get("baseline_log_loss_1x2", float("nan")), 4),
    })
    return TrainingReport(version, len(fit_matches), len(calibration_matches),
                          train_start, train_end, metrics, fitted)


def _score(model: DixonColesModel, matches: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not matches:
        return {}
    probabilities, outcomes = [], []
    for match in matches:
        markets = model.predict(match["home_team_id"], match["away_team_id"],
                                match["competition_id"])
        probabilities.append([markets.probabilities["HOME_WIN"],
                              markets.probabilities["DRAW"],
                              markets.probabilities["AWAY_WIN"]])
        outcomes.append(OUTCOME_INDEX[match["outcome"]])
    return {
        "n": len(matches),
        "log_loss_1x2": log_loss_multiclass(probabilities, outcomes),
        "brier_1x2": brier_multiclass(probabilities, outcomes),
    }


def _score_baseline(baseline: BaseRateBaseline,
                    matches: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not matches:
        return {}
    probabilities, outcomes = [], []
    for match in matches:
        rates = baseline.predict(competition_id=match["competition_id"])
        probabilities.append([rates["HOME_WIN"], rates["DRAW"], rates["AWAY_WIN"]])
        outcomes.append(OUTCOME_INDEX[match["outcome"]])
    return {
        "baseline_log_loss_1x2": log_loss_multiclass(probabilities, outcomes),
        "baseline_brier_1x2": brier_multiclass(probabilities, outcomes),
    }


def fit_calibrators(db: Database, model: DixonColesModel, model_version: str,
                    matches: Sequence[dict[str, Any]],
                    window_start: dt.datetime, window_end: dt.datetime,
                    config: Config) -> int:
    """Fit one calibrator per (market, competition) where the sample allows."""
    CALIBRATOR_DIR.mkdir(parents=True, exist_ok=True)
    min_settled = int(config.eligibility.get("min_settled_predictions", 500))

    predictions: dict[str, list[float]] = {m: [] for m in STORED_MARKETS}
    actuals: dict[str, list[int]] = {m: [] for m in STORED_MARKETS}
    by_competition: dict[tuple[str, int], tuple[list[float], list[int]]] = {}

    for match in matches:
        markets = derive_markets(
            *model.predict_lambdas(match["home_team_id"], match["away_team_id"],
                                   match["competition_id"]), model.params.rho)
        home, away = match["home_goals"], match["away_goals"]
        for market_key in STORED_MARKETS:
            probability = markets.probabilities.get(market_key)
            resolver = MARKET_RESOLVERS.get(market_key)
            if probability is None or resolver is None:
                continue
            outcome = resolver(home, away)
            predictions[market_key].append(probability)
            actuals[market_key].append(outcome)
            key = (market_key, match["competition_id"])
            bucket = by_competition.setdefault(key, ([], []))
            bucket[0].append(probability)
            bucket[1].append(outcome)

    db.execute("DELETE FROM calibrator_version WHERE model_version = ?", (model_version,))
    fitted = 0

    def store(market_key: str, competition_id: int | None,
              probabilities: list[float], outcomes: list[int]) -> bool:
        result = fit_calibrator(probabilities, outcomes, window_start, window_end)
        suffix = competition_id if competition_id is not None else "pooled"
        version = f"cal_{model_version}_{market_key}_{suffix}"
        path = CALIBRATOR_DIR / f"{version}.pkl"
        with path.open("wb") as handle:
            pickle.dump(result.calibrator, handle)
        db.insert("calibrator_version", {
            "calibrator_version": version, "model_version": model_version,
            "market_key": market_key, "competition_id": competition_id,
            "method": result.method, "fitted_at": clock.to_iso(clock.now()),
            "fit_window_start": clock.to_iso(window_start),
            "fit_window_end": clock.to_iso(window_end),
            "n_samples": result.n_samples,
            "ece_before": None if result.ece_before != result.ece_before else result.ece_before,
            "ece_after": None if result.ece_after != result.ece_after else result.ece_after,
            "artifact_path": str(path),
        }, or_ignore=True)
        return True

    # Per (market, competition) where the sample is large enough; otherwise the
    # pooled calibrator. Seven leagues split fifteen ways is how a calibrator
    # ends up fitting noise.
    for (market_key, competition_id), (probabilities, outcomes) in by_competition.items():
        if len(probabilities) >= min_settled:
            fitted += int(store(market_key, competition_id, probabilities, outcomes))

    for market_key in STORED_MARKETS:
        if predictions[market_key]:
            fitted += int(store(market_key, None, predictions[market_key],
                                actuals[market_key]))
    return fitted


def evaluate_markets(model: DixonColesModel, matches: Sequence[dict[str, Any]],
                     calibrators: dict[tuple[str, int | None], Any] | None = None
                     ) -> dict[str, dict[str, float]]:
    """Per-market held-out metrics, optionally after calibration."""
    from nway.evaluation.metrics import market_report

    calibrators = calibrators or {}
    collected: dict[str, tuple[list[float], list[int]]] = {}
    for match in matches:
        markets = derive_markets(
            *model.predict_lambdas(match["home_team_id"], match["away_team_id"],
                                   match["competition_id"]), model.params.rho)
        for market_key, resolver in MARKET_RESOLVERS.items():
            probability = markets.probabilities.get(market_key)
            if probability is None:
                continue
            calibrator = (calibrators.get((market_key, match["competition_id"]))
                          or calibrators.get((market_key, None)))
            if calibrator is not None:
                probability = float(calibrator.transform([probability])[0])
            bucket = collected.setdefault(market_key, ([], []))
            bucket[0].append(probability)
            bucket[1].append(resolver(match["home_goals"], match["away_goals"]))

    reports: dict[str, dict[str, float]] = {}
    for market_key, (probabilities, outcomes) in collected.items():
        report = market_report(market_key, probabilities, outcomes)
        reports[market_key] = {
            "n": report.n, "mean_predicted": report.mean_predicted,
            "actual_rate": report.actual_rate, "log_loss": report.log_loss,
            "brier": report.brier, "ece": report.ece, "auc": report.auc,
            "ci_low": report.ci_low, "ci_high": report.ci_high,
        }
    return reports
