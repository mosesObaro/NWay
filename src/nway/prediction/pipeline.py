"""Prediction orchestration.

fixture + as_of -> snapshot -> quality gate -> features -> goal model ->
distribution -> market derivation -> calibration -> invariants -> persist.

The whole run is one transaction: a coherence failure rolls back rather than
storing a partially consistent set of markets.

Nothing is ever overwritten. Each refresh appends a new ``prediction_run`` and
links the previous one through ``superseded_by``, so the full history of what
the system believed, and when, survives.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from typing import Any

from nway import clock
from nway.calibration.calibrators import IdentityCalibrator
from nway.config import Config
from nway.features.compute import completeness, compute_features, persist_features
from nway.features.context import FeatureContext
from nway.features.registry import feature_version
from nway.logging_setup import get_logger
from nway.models.base import ModelArtifact
from nway.models.dixon_coles import DixonColesModel
from nway.markets import (
    ALL_DERIVED_MARKETS, check_coherence, derive_card_markets,
    derive_corner_markets, derive_half_markets, derive_markets, market_selection,
)
from nway.prediction.explain import build_explanations
from nway.storage import repositories as repo
from nway.storage.db import Database

log = get_logger(__name__)

# No football outcome is certain. Even after Laplace-smoothed calibration, a
# displayed "100%" would contradict the uncertainty notice the email carries,
# so this is a hard backstop independent of any calibrator's behaviour.
MAX_PROBABILITY = 0.99
MIN_PROBABILITY = 0.01

# Every market the four grids produce. Numeric quantities (expected goals,
# expected corners) are stored on the run and as expected_value, never as
# probabilities.
#
# Taken from the market registry rather than written out by hand: six markets
# the pipeline stored were once absent from the configuration, so 282 of 705
# candidates were discarded as UNKNOWN_MARKET with no error anywhere.
STORED_MARKETS: tuple[str, ...] = ALL_DERIVED_MARKETS

REFRESH_STAGES = (
    ("FINAL", 0.0, 2.0),
    ("T8", 2.0, 12.0),
    ("T24", 12.0, 30.0),
    ("T48", 30.0, 60.0),
)


def refresh_stage(hours_to_kickoff: float) -> str:
    for name, lower, upper in REFRESH_STAGES:
        if lower <= hours_to_kickoff < upper:
            return name
    return "ADHOC"


@dataclass
class PredictionResult:
    fixture_id: int
    prediction_run_id: int | None
    lambda_home: float
    lambda_away: float
    markets: dict[str, float]
    calibrated: dict[str, float]
    data_completeness: float
    feature_staleness_hours: float
    skipped_reason: str | None = None
    explanations: dict[str, list] = field(default_factory=dict)

    @property
    def was_skipped(self) -> bool:
        return self.skipped_reason is not None


class PredictionPipeline:
    def __init__(self, db: Database, config: Config, model: DixonColesModel,
                 model_version: str, calibrators: dict[str, Any] | None = None,
                 corner_model: Any = None, card_model: Any = None,
                 half_model: Any = None) -> None:
        self.db = db
        self.config = config
        self.model = model
        self.model_version = model_version
        self.calibrators = calibrators or {}
        # Side grids. Absent models simply mean those markets are not produced,
        # which is how the Champions League ends up with goals and halves only.
        self.corner_model = corner_model
        self.card_model = card_model
        self.half_model = half_model
        self.artifact: ModelArtifact | None = None

    # -- calibration lookup ---------------------------------------------
    def _grid_of(self, market_key: str) -> str:
        try:
            return self.config.market(market_key).grid
        except KeyError:
            return "GOAL_MATRIX"

    def _calibrator(self, market_key: str, competition_id: int):
        for key in ((market_key, competition_id), (market_key, None)):
            if key in self.calibrators:
                return self.calibrators[key]
        return IdentityCalibrator()

    # -- main entry point -------------------------------------------------
    def predict_fixture(self, fixture: dict[str, Any], as_of: dt.datetime,
                        snapshot_id: int | None = None,
                        persist: bool = True) -> PredictionResult:
        as_of = clock.ensure_utc(as_of)
        kickoff = clock.from_iso(fixture["kickoff_utc"])
        hours_to_kickoff = clock.hours_between(as_of, kickoff)

        if self.artifact is not None:
            self.artifact.assert_usable_at(as_of)
            scope = "mens_club"
            slug = fixture.get("competition_slug")
            if slug:
                try:
                    scope = self.config.competition(slug).model_scope
                except KeyError:
                    scope = "mens_club"
            self.artifact.assert_scope(scope)

        context = FeatureContext.build(self.db, fixture, as_of, self.config)
        features = compute_features(context)
        data_completeness = completeness(features)

        newest = context.repo.max_knowledge_time
        # Publication lag, not data age. See AsOfRepository.data_lag_hours:
        # gating on age refuses every first matchday after an international
        # break, when the source is in fact fully caught up.
        staleness = max(context.repo.data_lag_hours(fixture["home_team_id"]),
                        context.repo.data_lag_hours(fixture["away_team_id"]))
        feature_age_hours = max(
            0.0, clock.hours_between(clock.from_iso(newest), as_of))

        minimum = float(self.config.recommendations.get("eligibility", {})
                        .get("min_data_completeness", 0.85))
        if data_completeness < minimum:
            # Decline to predict rather than predicting from incomplete inputs.
            log.info("skipped fixture: incomplete data", context={
                "fixture_id": fixture["fixture_id"],
                "completeness": data_completeness})
            return PredictionResult(
                fixture["fixture_id"], None, 0.0, 0.0, {}, {},
                data_completeness, staleness,
                skipped_reason="INSUFFICIENT_DATA_COMPLETENESS")

        # A competition the model was never fitted on has no goal rate and no
        # home advantage of its own, so every fixture in it would receive the
        # same league-average pair -- identical probabilities for Roma v Real
        # Madrid and Feyenoord v Como. Refuse rather than emit that.
        if hasattr(self.model, "is_fitted_for") and \
                not self.model.is_fitted_for(fixture["competition_id"]):
            log.info("skipped fixture: competition not in training data",
                     context={"fixture_id": fixture["fixture_id"],
                              "competition": fixture.get("competition_slug")})
            return PredictionResult(
                fixture["fixture_id"], None, 0.0, 0.0, {}, {},
                data_completeness, staleness,
                skipped_reason="COMPETITION_NOT_FITTED")

        # Without a rating a team receives the competition average, which
        # looks like a prediction and is not one. Cup competitions draw on
        # clubs from outside the configured leagues.
        ratings = getattr(self.model.params, "attack", {})
        unrated = [side for side, key in (("home", "home_team_id"),
                                          ("away", "away_team_id"))
                   if ratings and fixture[key] not in ratings]
        if unrated:
            log.info("skipped fixture: unrated team", context={
                "fixture_id": fixture["fixture_id"], "sides": unrated})
            return PredictionResult(
                fixture["fixture_id"], None, 0.0, 0.0, {}, {},
                data_completeness, staleness, skipped_reason="UNRATED_TEAM")

        lam_home, lam_away = self.model.predict_lambdas(
            fixture["home_team_id"], fixture["away_team_id"],
            fixture["competition_id"])
        markets = derive_markets(lam_home, lam_away, self.model.params.rho)

        slug = fixture.get("competition_slug") or ""
        available = set(self.config.grids_enabled_for(slug)) if slug else {
            "GOAL_MATRIX", "HALF_MATRIX", "CORNER_MATRIX", "CARD_DISTRIBUTION"}

        probabilities = dict(markets.probabilities)
        grid_parameters: dict[str, Any] = {
            "GOAL_MATRIX": {"lambda_home": round(lam_home, 6),
                            "lambda_away": round(lam_away, 6),
                            "rho": round(self.model.params.rho, 6)},
        }

        # Half-time: a binomial split of the SAME full-time grid, so the
        # full-time marginal is preserved exactly and no half market can
        # exceed its full-time counterpart.
        if "HALF_MATRIX" in available and self.half_model is not None:
            share = self.half_model.share_for(fixture["competition_id"])
            half = derive_half_markets(markets.matrix, share)
            probabilities.update(half.probabilities)
            grid_parameters["HALF_MATRIX"] = {"share": round(share, 6)}

        if "CORNER_MATRIX" in available and self.corner_model is not None \
                and self.corner_model.is_fitted_for(fixture["competition_id"]):
            parameters = self.corner_model.predict_parameters(
                fixture["home_team_id"], fixture["away_team_id"],
                fixture["competition_id"], supremacy=lam_home - lam_away)
            corners = derive_corner_markets(*parameters)
            probabilities.update(corners.probabilities)
            grid_parameters["CORNER_MATRIX"] = {
                "mean_total": round(parameters[0], 6),
                "dispersion": round(parameters[1], 6),
                "home_share": round(parameters[2], 6),
                "concentration": round(parameters[3], 6)}

        if "CARD_DISTRIBUTION" in available and self.card_model is not None \
                and self.card_model.is_fitted_for(fixture["competition_id"]):
            mean_cards, dispersion, referee_known = \
                self.card_model.predict_parameters(
                    fixture["home_team_id"], fixture["away_team_id"],
                    fixture["competition_id"], fixture.get("referee_id"))
            cards = derive_card_markets(mean_cards, dispersion)
            probabilities.update(cards.probabilities)
            grid_parameters["CARD_DISTRIBUTION"] = {
                "mean_total": round(mean_cards, 6),
                "dispersion": round(dispersion, 6),
                "referee_known": referee_known}

        # One coherence pass over the merged set: ladders, complements, trios
        # and cross-grid impossibilities (a half cannot exceed the match, a
        # team total cannot exceed the match total).
        class _Merged:
            pass
        merged = _Merged()
        merged.probabilities = probabilities
        merged.matrix = markets.matrix
        merged.lambda_home = lam_home
        merged.lambda_away = lam_away
        check_coherence(merged)

        calibrated: dict[str, float] = {}
        calibrator_versions: dict[str, str | None] = {}
        for market_key in STORED_MARKETS:
            raw = probabilities.get(market_key)
            if raw is None:
                continue
            calibrator = self._calibrator(market_key, fixture["competition_id"])
            value = float(calibrator.transform([raw])[0])
            calibrated[market_key] = min(MAX_PROBABILITY, max(MIN_PROBABILITY, value))
            calibrator_versions[market_key] = getattr(calibrator, "version", None)

        league_baseline = context.repo.competition_baseline(fixture["competition_id"])
        explanations: dict[str, list] = {}
        for market_key, probability in calibrated.items():
            try:
                market = self.config.market(market_key)
                base_rate = market.base_rate
            except KeyError:
                base_rate = None
            explanations[market_key] = build_explanations(
                market_key=market_key, probability=probability, features=features,
                lambdas=(lam_home, lam_away), base_rate=base_rate,
                league_baseline=league_baseline)

        result = PredictionResult(
            fixture_id=fixture["fixture_id"], prediction_run_id=None,
            lambda_home=lam_home, lambda_away=lam_away,
            markets=dict(markets.probabilities), calibrated=calibrated,
            data_completeness=data_completeness,
            feature_staleness_hours=round(staleness, 2),
            explanations=explanations)

        if not persist:
            return result

        with self.db.transaction():
            persist_features(self.db, fixture["fixture_id"], as_of, features, newest)
            run_id = self._persist_run(
                fixture, as_of, kickoff, hours_to_kickoff, lam_home, lam_away,
                data_completeness, staleness, snapshot_id, grid_parameters)
            result.prediction_run_id = run_id
            self._persist_markets(run_id, fixture, probabilities, calibrated,
                                  calibrator_versions, explanations)
        return result

    # feature_age_hours is recorded for auditing only; the gate uses lag.

    # -- persistence -----------------------------------------------------
    def _persist_run(self, fixture, as_of, kickoff, hours_to_kickoff,
                     lam_home, lam_away, data_completeness, staleness,
                     snapshot_id, grid_parameters=None) -> int:
        as_of_iso = clock.to_iso(as_of)
        existing = self.db.query_one(
            "SELECT prediction_run_id FROM prediction_run WHERE fixture_id=? "
            "AND prediction_timestamp=? AND model_version=?",
            (fixture["fixture_id"], as_of_iso, self.model_version))
        if existing:
            # Idempotency: a second tick inside the same quantised minute
            # reuses the run instead of creating a twin.
            return existing["prediction_run_id"]

        previous = self.db.query_one(
            "SELECT prediction_run_id FROM prediction_run WHERE fixture_id=? "
            "AND superseded_by IS NULL ORDER BY prediction_timestamp DESC LIMIT 1",
            (fixture["fixture_id"],))

        run_id = self.db.insert("prediction_run", {
            "fixture_id": fixture["fixture_id"],
            "prediction_timestamp": as_of_iso,
            "kickoff_utc": clock.to_iso(kickoff),
            "hours_to_kickoff": round(hours_to_kickoff, 3),
            "refresh_stage": refresh_stage(hours_to_kickoff),
            "model_version": self.model_version,
            "feature_version": feature_version(),
            "snapshot_id": snapshot_id,
            "config_hash": self.config.config_hash,
            "lambda_home": round(lam_home, 5), "lambda_away": round(lam_away, 5),
            "grid_parameters": json.dumps(grid_parameters or {}),
            "data_completeness": data_completeness,
            "feature_staleness_hours": round(staleness, 2),
            "created_at": clock.to_iso(clock.now()),
        })
        if previous:
            # Never overwrite: the old snapshot stays, linked forward.
            self.db.execute(
                "UPDATE prediction_run SET superseded_by=? WHERE prediction_run_id=?",
                (run_id, previous["prediction_run_id"]))
        return run_id

    def _persist_markets(self, run_id, fixture, probabilities, calibrated,
                         calibrator_versions, explanations) -> None:
        now = clock.to_iso(clock.now())
        expected_for = {
            "HOME_WIN": "EXPECTED_HOME_GOALS", "AWAY_WIN": "EXPECTED_AWAY_GOALS",
        }
        for market_key, probability in calibrated.items():
            _, selection = market_selection(market_key)
            expected_value = None
            if market_key in expected_for:
                expected_value = probabilities.get(expected_for[market_key])
            elif market_key.startswith("CORNERS_"):
                expected_value = probabilities.get("EXPECTED_CORNERS")
            elif market_key.startswith("CARDS_"):
                expected_value = probabilities.get("EXPECTED_CARDS")
            elif market_key.startswith("HT_"):
                expected_value = probabilities.get("EXPECTED_HT_GOALS")
            prediction_id = self.db.insert("prediction", {
                "prediction_run_id": run_id, "fixture_id": fixture["fixture_id"],
                "market_key": market_key, "selection": selection,
                "raw_probability": round(probabilities[market_key], 6),
                "calibrated_probability": round(probability, 6),
                "calibrator_version": calibrator_versions.get(market_key),
                "expected_value": expected_value,
                "uncertainty": None,
                "is_derived_from": self._grid_of(market_key),
                "created_at": now,
            }, or_ignore=True)
            if not prediction_id:
                continue
            for line in explanations.get(market_key, []):
                self.db.insert("prediction_explanation", {
                    "prediction_id": prediction_id, "direction": line.direction,
                    "feature_key": line.feature_key, "feature_value": line.feature_value,
                    "reference_value": line.reference_value,
                    "contribution": line.contribution,
                    "template_key": line.template_key, "rank": line.rank,
                })


def load_pipeline(db: Database, config: Config,
                  model_version: str | None = None) -> PredictionPipeline:
    """Load the active goal model and its calibrators."""
    from nway.models.base import active_model_version

    version = model_version or active_model_version(db, "GOALS")
    if not version:
        raise RuntimeError("no active GOALS model; run `nway train` first")
    artifact = ModelArtifact.load(version)
    model = DixonColesModel.from_artifact(artifact)

    pipeline = PredictionPipeline(
        db, config, model, version, load_calibrators(db, version),
        corner_model=_load_side_model(db, "CORNERS"),
        card_model=_load_side_model(db, "CARDS"),
        half_model=_load_side_model(db, "HALVES"))
    pipeline.artifact = artifact
    return pipeline


def _load_side_model(db: Database, target: str) -> Any:
    """Load an active side model, or None when none has been trained.

    None means those markets are simply not produced -- which is the correct
    behaviour before the first training run, and permanently for a competition
    whose statistics no source supplies.
    """
    from nway.models.base import active_model_version
    from nway.models.cards import CardModel
    from nway.models.corners import CornerModel
    from nway.models.halves import HalfShareModel

    loaders = {"CORNERS": CornerModel, "CARDS": CardModel,
               "HALVES": HalfShareModel}
    version = active_model_version(db, target)
    if not version:
        return None
    try:
        return loaders[target].from_artifact(ModelArtifact.load(version))
    except (FileNotFoundError, KeyError) as exc:
        log.warning("side model unavailable", context={
            "target": target, "version": version, "error": str(exc)})
        return None


def load_calibrators(db: Database, model_version: str) -> dict[tuple[str, int | None], Any]:
    import pickle
    from pathlib import Path

    calibrators: dict[tuple[str, int | None], Any] = {}
    for row in db.query(
            "SELECT * FROM calibrator_version WHERE model_version = ?", (model_version,)):
        path = Path(row["artifact_path"])
        if not path.exists():
            continue
        with path.open("rb") as handle:
            calibrator = pickle.load(handle)
        calibrator.version = row["calibrator_version"]
        calibrators[(row["market_key"], row["competition_id"])] = calibrator
    return calibrators


def snapshot_for(db: Database, as_of: dt.datetime) -> int:
    return repo.create_snapshot(db, as_of)
