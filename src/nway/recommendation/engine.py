"""Recommendation engine.

Deliberately separate from the models: it reads stored, calibrated predictions
plus historical reliability, and never imports from ``nway.models``. A test in
tests/unit/test_architecture.py enforces that. Prediction quality and selection
policy change for different reasons and at different rates.

    predictions -> eligibility gates -> reliability scoring -> ranking
                -> diversity and correlation caps -> selection (7..20)

Two decisions here carry most of the weight, and both are measured rather than
chosen by feel (research/threshold_study.py):

  * Probability floors are per market, ``max(0.65, base + 0.25*(1-base))``. A
    flat floor is incoherent across markets -- Over 1.5 comes in 77.1% of the
    time, so a flat 72% floor there would qualify predictions weaker than
    assuming the league average.
  * Ranking is by a pessimistic estimate of the realised hit rate, not by raw
    probability, so a well-evidenced 82% outranks a thinly-evidenced 85%.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from scipy.stats import beta as beta_dist

from nway.markets import derive_markets

from nway import clock
from nway.config import Config, market_floor
from nway.logging_setup import get_logger
from nway.storage.db import Database

log = get_logger(__name__)


@dataclass
class Candidate:
    """A stored prediction being considered for recommendation."""
    prediction_id: int
    prediction_run_id: int
    fixture_id: int
    competition_id: int
    competition_slug: str
    competition_name: str
    market_key: str
    selection: str
    probability: float
    raw_probability: float
    home_team: str
    away_team: str
    kickoff_utc: dt.datetime
    prediction_timestamp: dt.datetime
    data_completeness: float
    feature_staleness_hours: float
    lambda_home: float | None
    lambda_away: float | None
    model_version: str
    fixture_status: str
    kickoff_is_confirmed: bool

    @property
    def match_label(self) -> str:
        return f"{self.home_team} vs {self.away_team}"


@dataclass
class ScoredRecommendation:
    candidate: Candidate
    reliability_score: float
    ranking_score: float
    confidence_band: str
    passed: bool
    rejection_reasons: list[str] = field(default_factory=list)
    n_history: int = 0
    historical_hit_rate: float | None = None
    selected: bool = False

    @property
    def fixture_id(self) -> int:
        return self.candidate.fixture_id

    @property
    def market_key(self) -> str:
        return self.candidate.market_key


class RecommendationEngine:
    def __init__(self, db: Database, config: Config) -> None:
        self.db = db
        self.config = config
        recommendations = config.recommendations or {}
        self.eligibility = recommendations.get("eligibility", {}) or {}
        self.reliability = recommendations.get("reliability", {}) or {}
        self.diversity = recommendations.get("diversity", {}) or {}
        self.bands = recommendations.get("confidence_bands", {}) or {}
        self.min_count = int(recommendations.get("min_count", 7))
        self.max_count = int(recommendations.get("max_count", 20))
        thresholds = config.thresholds or {}
        self.absolute_floor = float(thresholds.get("absolute_floor", 0.65))
        self.lift_k = float(thresholds.get("lift_k", 0.25))
        self._history_cache: dict[tuple[str, int, int], tuple[int, int]] = {}

    # -- floors ----------------------------------------------------------
    def floor_for(self, market_key: str) -> float:
        market = self.config.market(market_key)
        if market.min_probability is not None:
            return float(market.min_probability)
        if market.base_rate is not None:
            return market_floor(market.base_rate, self.absolute_floor, self.lift_k)
        return self.absolute_floor

    # -- candidate loading -----------------------------------------------
    def load_candidates(self, window_start: dt.datetime, window_end: dt.datetime,
                        as_of: dt.datetime) -> list[Candidate]:
        """Latest prediction per (fixture, market) inside the rolling window."""
        rows = self.db.query("""
            SELECT p.prediction_id, p.prediction_run_id, p.fixture_id, p.market_key,
                   p.selection, p.calibrated_probability, p.raw_probability,
                   r.prediction_timestamp, r.data_completeness,
                   r.feature_staleness_hours, r.lambda_home, r.lambda_away,
                   r.model_version, f.kickoff_utc, f.status, f.kickoff_is_confirmed,
                   f.competition_id, c.slug AS competition_slug,
                   c.name AS competition_name,
                   h.canonical_name AS home_team, a.canonical_name AS away_team
            FROM prediction p
            JOIN prediction_run r ON r.prediction_run_id = p.prediction_run_id
            JOIN fixture f ON f.fixture_id = p.fixture_id
            JOIN competition c ON c.competition_id = f.competition_id
            JOIN team h ON h.team_id = f.home_team_id
            JOIN team a ON a.team_id = f.away_team_id
            WHERE f.kickoff_utc > ? AND f.kickoff_utc <= ?
              AND r.superseded_by IS NULL
              AND r.prediction_timestamp <= ?
            ORDER BY f.kickoff_utc, p.fixture_id, p.market_key
        """, (clock.to_iso(window_start), clock.to_iso(window_end), clock.to_iso(as_of)))

        return [Candidate(
            prediction_id=row["prediction_id"],
            prediction_run_id=row["prediction_run_id"],
            fixture_id=row["fixture_id"], competition_id=row["competition_id"],
            competition_slug=row["competition_slug"],
            competition_name=row["competition_name"],
            market_key=row["market_key"], selection=row["selection"],
            probability=row["calibrated_probability"],
            raw_probability=row["raw_probability"],
            home_team=row["home_team"], away_team=row["away_team"],
            kickoff_utc=clock.from_iso(row["kickoff_utc"]),
            prediction_timestamp=clock.from_iso(row["prediction_timestamp"]),
            data_completeness=row["data_completeness"],
            feature_staleness_hours=row["feature_staleness_hours"],
            lambda_home=row["lambda_home"], lambda_away=row["lambda_away"],
            model_version=row["model_version"], fixture_status=row["status"],
            kickoff_is_confirmed=bool(row["kickoff_is_confirmed"]),
        ) for row in rows]

    # -- eligibility -------------------------------------------------------
    def check_eligibility(self, candidate: Candidate, now: dt.datetime) -> list[str]:
        """Hard gates. Reasons are recorded even for candidates never selected,
        because on a system that stays silent most of the time, "why was there
        no email?" is the question that gets asked constantly."""
        reasons: list[str] = []

        try:
            market = self.config.market(candidate.market_key)
        except KeyError:
            return ["UNKNOWN_MARKET"]

        if not market.recommendable:
            reasons.append("MARKET_NOT_RECOMMENDABLE")
        elif not market.enabled_for(candidate.competition_slug):
            reasons.append("MARKET_DISABLED_FOR_COMPETITION")

        if candidate.probability < self.floor_for(candidate.market_key):
            reasons.append("BELOW_PROBABILITY_FLOOR")

        if candidate.fixture_status not in ("SCHEDULED", "TIMED"):
            reasons.append(f"FIXTURE_{candidate.fixture_status}")

        if self.eligibility.get("require_confirmed_kickoff", True) and \
                not candidate.kickoff_is_confirmed:
            reasons.append("KICKOFF_UNCONFIRMED")

        completeness_floor = float(self.eligibility.get("min_data_completeness", 0.85))
        if candidate.data_completeness < completeness_floor:
            reasons.append("INCOMPLETE_DATA")

        staleness_limit = float(self.eligibility.get("max_feature_staleness_hours", 96))
        if candidate.feature_staleness_hours > staleness_limit:
            reasons.append("STALE_FEATURES")

        lead_time = clock.hours_between(now, candidate.kickoff_utc)
        if lead_time < float(self.eligibility.get("min_lead_time_hours", 0.25)):
            reasons.append("INSUFFICIENT_LEAD_TIME")

        n_history, hits = self.market_history(
            candidate.market_key, candidate.competition_id, candidate.probability)
        min_settled = int(self.eligibility.get("min_settled_predictions", 500))
        if n_history < min_settled and not self._allow_cold_start():
            reasons.append("INSUFFICIENT_SETTLED_HISTORY")

        if self._blocking_quality_issue(candidate.fixture_id):
            reasons.append("BLOCKING_DATA_QUALITY")

        return reasons

    def _allow_cold_start(self) -> bool:
        """At launch there is no settled history, so the sample gate would
        block everything forever. When explicitly allowed, the Beta prior does
        the work instead and confidence bands stay capped at MEDIUM."""
        return bool(self.eligibility.get("allow_cold_start", True))

    def _blocking_quality_issue(self, fixture_id: int) -> bool:
        return bool(self.db.scalar(
            "SELECT COUNT(*) FROM data_quality_check WHERE severity='BLOCKING' "
            "AND resolved_at IS NULL AND entity_kind='FIXTURE' AND entity_id=?",
            (fixture_id,), default=0))

    # -- reliability -------------------------------------------------------
    def market_history(self, market_key: str, competition_id: int,
                       probability: float) -> tuple[int, int]:
        """Settled (n, hits) for this market, competition and probability bucket."""
        width = float(self.reliability.get("probability_bucket_width", 0.05))
        bucket = int(probability / width)
        cache_key = (market_key, competition_id, bucket)
        if cache_key in self._history_cache:
            return self._history_cache[cache_key]

        low, high = bucket * width, (bucket + 1) * width
        row = self.db.query_one("""
            SELECT COUNT(*) AS n, COALESCE(SUM(e.hit), 0) AS hits
            FROM prediction_evaluation e
            JOIN prediction p ON p.prediction_id = e.prediction_id
            JOIN fixture f ON f.fixture_id = p.fixture_id
            WHERE p.market_key = ? AND f.competition_id = ?
              AND p.calibrated_probability >= ? AND p.calibrated_probability < ?
              AND e.hit IS NOT NULL
        """, (market_key, competition_id, low, high))
        result = (int(row["n"] or 0), int(row["hits"] or 0))
        self._history_cache[cache_key] = result
        return result

    def market_prior(self, market_key: str) -> float:
        """Prior centre: the market's own pooled realised rate, else its base rate."""
        row = self.db.query_one("""
            SELECT COUNT(*) AS n, COALESCE(SUM(e.hit), 0) AS hits
            FROM prediction_evaluation e
            JOIN prediction p ON p.prediction_id = e.prediction_id
            WHERE p.market_key = ? AND e.hit IS NOT NULL
        """, (market_key,))
        if row and row["n"] and row["n"] >= 100:
            return float(row["hits"]) / float(row["n"])
        try:
            base_rate = self.config.market(market_key).base_rate
        except KeyError:
            base_rate = None
        return float(base_rate) if base_rate is not None else 0.5

    def reliability_score(self, candidate: Candidate) -> tuple[float, int, float | None]:
        """Lower bound of a Beta posterior over the realised hit rate.

        A single quantity that handles three requirements at once: a thin
        sample widens the posterior and drags the bound down; miscalibration
        shows up as a realised rate below the predicted probability, which
        lowers the bound directly; and a long, well-behaved track record is
        rewarded without any hand-tuned weighting.
        """
        n, hits = self.market_history(
            candidate.market_key, candidate.competition_id, candidate.probability)
        pseudo = float(self.reliability.get("prior_pseudo_observations", 50))
        quantile = float(self.reliability.get("posterior_quantile", 0.10))

        # The prior is centred on the model's own claim, shrunk toward the
        # market's realised behaviour where that is known.
        prior_centre = candidate.probability
        pooled = self.market_prior(candidate.market_key)
        if pooled is not None:
            prior_centre = 0.5 * candidate.probability + 0.5 * pooled
        prior_centre = min(max(prior_centre, 0.02), 0.98)

        alpha = prior_centre * pseudo + hits
        beta = (1 - prior_centre) * pseudo + (n - hits)
        score = float(beta_dist.ppf(quantile, alpha, beta))
        observed = (hits / n) if n else None
        return score, n, observed

    def confidence_band(self, reliability: float, n_history: int) -> str:
        high = self.bands.get("HIGH", {}) or {}
        medium = self.bands.get("MEDIUM", {}) or {}
        if (reliability >= float(high.get("min_posterior", 0.75))
                and n_history >= int(high.get("min_samples", 1000))):
            return "HIGH"
        if reliability >= float(medium.get("min_posterior", 0.65)):
            # Without a real track record, MEDIUM is the ceiling. Calling a
            # cold-start prediction HIGH would be a claim the data cannot back.
            return "MEDIUM"
        return "LOW"

    def score(self, candidates: Sequence[Candidate],
              now: dt.datetime) -> list[ScoredRecommendation]:
        scored: list[ScoredRecommendation] = []
        freshness_floor = float(self.reliability.get("freshness_factor_floor", 0.80))
        completeness_floor = float(self.reliability.get("completeness_factor_floor", 0.80))
        staleness_limit = max(1.0, float(
            self.eligibility.get("max_feature_staleness_hours", 96)))

        for candidate in candidates:
            reasons = self.check_eligibility(candidate, now)
            reliability, n_history, observed = self.reliability_score(candidate)

            # Tie-breakers only. They cannot rescue a candidate that failed a gate.
            freshness = 1.0 - (1.0 - freshness_floor) * min(
                1.0, candidate.feature_staleness_hours / staleness_limit)
            completeness = completeness_floor + (1 - completeness_floor) * min(
                1.0, candidate.data_completeness)
            ranking = reliability * freshness * completeness

            band = self.confidence_band(reliability, n_history)
            if band == "LOW" and not self.bands.get("LOW", {}).get("recommendable", False):
                reasons.append("LOW_CONFIDENCE_BAND")

            scored.append(ScoredRecommendation(
                candidate=candidate, reliability_score=round(reliability, 6),
                ranking_score=round(ranking, 6), confidence_band=band,
                passed=not reasons, rejection_reasons=reasons,
                n_history=n_history, historical_hit_rate=observed))
        scored.sort(key=lambda s: -s.ranking_score)
        return scored

    # -- selection ---------------------------------------------------------
    def select(self, scored: Sequence[ScoredRecommendation],
               min_count: int | None = None,
               max_count: int | None = None) -> list[ScoredRecommendation]:
        """Greedy selection under diversity caps.

        Caps are computed against the ACTUAL batch size rather than the 20
        maximum, then relaxed by one slot before a viable batch is abandoned.
        Measured: unconstrained selection produces batches that are 81% double
        chance. Those are the highest probabilities available and, as a
        product, close to worthless -- one goal-model drift takes the whole
        batch down together.

        There is deliberately no code path that lowers a threshold to reach the
        minimum. Thresholds are an input to this function, never an output.
        """
        min_count = self.min_count if min_count is None else min_count
        max_count = self.max_count if max_count is None else max_count
        eligible = [s for s in scored if s.passed]
        if not eligible:
            return []

        unconstrained = self._greedy(eligible, {}, max_count)
        if len(unconstrained) < min_count:
            return unconstrained

        shares = self.diversity.get("family_share_caps", {}) or {}
        minimum_allowance = int(self.diversity.get("min_family_allowance", 2))
        size = len(unconstrained)
        caps = {family: max(minimum_allowance, math.ceil(share * size))
                for family, share in shares.items()}

        selected = self._greedy(eligible, caps, max_count)
        if len(selected) < min_count and self.diversity.get(
                "relax_caps_before_skipping", True):
            relaxed = {family: value + 1 for family, value in caps.items()}
            selected = self._greedy(eligible, relaxed, max_count)
        return selected

    def _greedy(self, eligible: Sequence[ScoredRecommendation],
                family_caps: dict[str, int], max_count: int
                ) -> list[ScoredRecommendation]:
        max_per_fixture = int(self.diversity.get("max_selections_per_fixture", 1))
        max_correlation = float(self.diversity.get("max_within_fixture_correlation", 0.30))
        competition_share = float(self.diversity.get("max_share_per_competition", 0.50))
        competition_cap = max(2, math.ceil(competition_share * max_count))

        selected: list[ScoredRecommendation] = []
        per_fixture: Counter[int] = Counter()
        per_family: Counter[str] = Counter()
        per_competition: Counter[int] = Counter()
        chosen_by_fixture: dict[int, list[str]] = {}

        for item in eligible:
            if len(selected) >= max_count:
                break
            candidate = item.candidate
            if per_fixture[candidate.fixture_id] >= max_per_fixture:
                continue
            family = self.config.market(candidate.market_key).family
            if family in family_caps and per_family[family] >= family_caps[family]:
                continue
            if per_competition[candidate.competition_id] >= competition_cap:
                continue
            if self._too_correlated(candidate, chosen_by_fixture, max_correlation):
                continue

            item.selected = True
            selected.append(item)
            per_fixture[candidate.fixture_id] += 1
            per_family[family] += 1
            per_competition[candidate.competition_id] += 1
            chosen_by_fixture.setdefault(candidate.fixture_id, []).append(
                candidate.market_key)
        return selected

    def _too_correlated(self, candidate: Candidate,
                        chosen: dict[int, list[str]], limit: float) -> bool:
        """Exact correlation from the fixture's own goal matrix.

        Because every goal market is derived from one distribution, the joint
        probability of two selections is a sum over matrix cells -- computed,
        not assumed.
        """
        already = chosen.get(candidate.fixture_id)
        if not already:
            return False
        if candidate.lambda_home is None or candidate.lambda_away is None:
            # No stored goal matrix means correlation cannot be computed, so
            # refuse the second selection rather than guess it is independent.
            return True
        markets = derive_markets(candidate.lambda_home, candidate.lambda_away)
        for market_key in already:
            correlation = markets.correlation(candidate.market_key, market_key)
            if correlation is None or abs(correlation) > limit:
                return True
        return False

    # -- persistence -------------------------------------------------------
    def persist(self, scored: Sequence[ScoredRecommendation],
                evaluated_at: dt.datetime, batch_id: int | None = None) -> None:
        now = clock.to_iso(evaluated_at)
        rows = [(
            item.candidate.prediction_id, item.candidate.fixture_id, now,
            int(item.passed),
            json.dumps(item.rejection_reasons) if item.rejection_reasons else None,
            item.reliability_score, item.ranking_score, item.confidence_band,
            f"fixture:{item.candidate.fixture_id}", int(item.selected),
            batch_id if item.selected else None,
        ) for item in scored]
        self.db.executemany(
            "INSERT INTO recommendation (prediction_id, fixture_id, evaluated_at, "
            "passed_eligibility, rejection_reasons, reliability_score, ranking_score, "
            "confidence_band, correlation_group, selected, batch_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)

    @staticmethod
    def rejection_summary(scored: Sequence[ScoredRecommendation]) -> dict[str, int]:
        summary: Counter[str] = Counter()
        for item in scored:
            for reason in item.rejection_reasons:
                summary[reason] += 1
        return dict(summary)
