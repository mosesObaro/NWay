"""Card model: negative-binomial total cards with referee shrinkage.

Measured over 21,919 matches: mean 4.26, variance 5.69, var/mean 1.336. Poisson
would understate the spread exactly at the 3.5 to 5.5 lines.

    log mu = mu_c + tendency_h + tendency_a + referee_effect

Per-competition intercepts are mandatory, not tidiness: measured league means
run from 3.18 cards a match in the Eredivisie to 5.39 in the Primeira Liga, a
70% spread. One pooled intercept would be wrong in both directions at once.

Referee identity is the strongest single predictor of cards and the one the
free sources mostly withhold: football-data.co.uk carries it for the Premier
League only, 15.8% of matches here. Rather than impute an average, each
referee's effect is shrunk toward zero in proportion to how little has been
seen of them:

    effect = (n / (n + k)) * mean(log(observed / expected))

A referee with four matches barely moves the rate; one with eighty moves it
most of the way. Where the referee is unknown the effect is exactly zero and a
``referee_known`` flag records that, so a missing referee is visibly missing
rather than silently average.

Only total cards are modelled. A home/away split would need per-side referee
behaviour, and on 15.8% coverage that is guessing with a decimal point.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from nway import clock
from nway.logging_setup import get_logger
from nway.models.base import ModelArtifact

log = get_logger(__name__)

MIN_MATCHES = 500
REFEREE_SHRINKAGE = 12.0      # pseudo-matches; a referee needs ~12 to count half


@dataclass
class CardParams:
    tendency: dict[int, float] = field(default_factory=dict)
    intercept: dict[int, float] = field(default_factory=dict)
    referee: dict[int, float] = field(default_factory=dict)
    referee_matches: dict[int, int] = field(default_factory=dict)
    dispersion: float = 12.0
    competition_defaults: dict[int, float] = field(default_factory=dict)


class CardModel:
    def __init__(self, xi: float = 0.0025, l2: float = 2.0,
                 max_iterations: int = 400,
                 referee_shrinkage: float = REFEREE_SHRINKAGE) -> None:
        self.xi = xi
        self.l2 = l2
        self.max_iterations = max_iterations
        self.referee_shrinkage = referee_shrinkage
        self.params = CardParams()

    def fit(self, matches: Sequence[dict[str, Any]],
            as_of: dt.datetime) -> "CardModel":
        usable = [m for m in matches if m.get("total_cards") is not None]
        if len(usable) < MIN_MATCHES:
            raise ValueError(f"not enough card matches to fit: {len(usable)}")

        teams = sorted({m["home_team_id"] for m in usable}
                       | {m["away_team_id"] for m in usable})
        competitions = sorted({m["competition_id"] for m in usable})
        team_index = {t: i for i, t in enumerate(teams)}
        comp_index = {c: i for i, c in enumerate(competitions)}
        n_teams, n_comps = len(teams), len(competitions)

        home_idx = np.array([team_index[m["home_team_id"]] for m in usable])
        away_idx = np.array([team_index[m["away_team_id"]] for m in usable])
        comp_idx = np.array([comp_index[m["competition_id"]] for m in usable])
        cards = np.array([m["total_cards"] for m in usable], dtype=float)

        kickoffs = [clock.from_iso(m["kickoff_utc"]) for m in usable]
        age_days = np.array([max(0.0, (clock.ensure_utc(as_of) - k).total_seconds()
                                 / 86400.0) for k in kickoffs])
        weights = np.exp(-self.xi * age_days)

        def unpack(theta: np.ndarray):
            tendency = np.zeros(n_teams)
            tendency[:-1] = theta[:n_teams - 1]
            tendency[-1] = -tendency[:-1].sum()
            intercept = theta[n_teams - 1:n_teams - 1 + n_comps]
            return tendency, intercept, theta[n_teams - 1 + n_comps]

        def negative_log_likelihood(theta: np.ndarray) -> float:
            tendency, intercept, log_dispersion = unpack(theta)
            dispersion = float(np.exp(np.clip(log_dispersion, -2.0, 8.0)))
            mu = np.exp(np.clip(intercept[comp_idx] + tendency[home_idx]
                                + tendency[away_idx], -3.0, 3.6))
            nb = (gammaln(cards + dispersion) - gammaln(dispersion)
                  - gammaln(cards + 1.0)
                  + dispersion * np.log(dispersion / (dispersion + mu))
                  + cards * np.log(mu / (dispersion + mu)))
            penalty = self.l2 * np.sum(tendency ** 2)
            return float(-np.sum(weights * nb) + penalty)

        theta0 = np.concatenate([np.zeros(n_teams - 1),
                                 np.full(n_comps, np.log(4.26)),
                                 np.array([np.log(12.0)])])
        result = minimize(negative_log_likelihood, theta0, method="L-BFGS-B",
                          options={"maxiter": self.max_iterations, "maxfun": 300000})
        tendency, intercept, log_dispersion = unpack(result.x)

        self.params = CardParams(
            tendency={t: float(tendency[i]) for t, i in team_index.items()},
            intercept={c: float(intercept[i]) for c, i in comp_index.items()},
            dispersion=float(np.exp(np.clip(log_dispersion, -2.0, 8.0))),
        )

        # Referee effects, from the residual of observed against expected.
        residuals: dict[int, list[float]] = defaultdict(list)
        for row, match in enumerate(usable):
            referee_id = match.get("referee_id")
            if not referee_id:
                continue
            expected = float(np.exp(
                intercept[comp_idx[row]] + tendency[home_idx[row]]
                + tendency[away_idx[row]]))
            observed = max(0.5, cards[row])
            residuals[referee_id].append(float(np.log(observed / expected)))

        for referee_id, values in residuals.items():
            n = len(values)
            raw = float(np.mean(values))
            # Shrunk toward zero: a referee seen four times barely moves the
            # rate, one seen eighty times moves it most of the way.
            self.params.referee[referee_id] = raw * (n / (n + self.referee_shrinkage))
            self.params.referee_matches[referee_id] = n

        by_comp: dict[int, list[float]] = defaultdict(list)
        for match in usable:
            by_comp[match["competition_id"]].append(match["total_cards"])
        self.params.competition_defaults = {
            comp: float(np.mean(values)) for comp, values in by_comp.items() if values}

        log.info("fitted card model", context={
            "matches": len(usable), "competitions": n_comps,
            "dispersion": round(self.params.dispersion, 2),
            "referees": len(self.params.referee),
            "converged": bool(result.success)})
        return self

    def is_fitted_for(self, competition_id: int) -> bool:
        return competition_id in self.params.intercept

    def predict_parameters(self, home_team_id: int, away_team_id: int,
                           competition_id: int,
                           referee_id: int | None = None) -> tuple[float, float, bool]:
        """(mean_total_cards, dispersion, referee_known)."""
        params = self.params
        if competition_id not in params.intercept:
            default = params.competition_defaults.get(competition_id, 4.26)
            return default, params.dispersion, False

        referee_effect = 0.0
        referee_known = False
        if referee_id and referee_id in params.referee:
            referee_effect = params.referee[referee_id]
            referee_known = True

        mu = float(np.exp(min(3.6,
                              params.intercept[competition_id]
                              + params.tendency.get(home_team_id, 0.0)
                              + params.tendency.get(away_team_id, 0.0)
                              + referee_effect)))
        return mu, params.dispersion, referee_known

    def to_artifact(self, model_version: str, train_start: dt.datetime,
                    train_end: dt.datetime, metrics: dict[str, Any] | None = None,
                    model_scope: str = "mens_club") -> ModelArtifact:
        return ModelArtifact(
            model_version=model_version, model_family="NEG_BINOMIAL",
            target="CARDS", trained_at=clock.now(),
            train_window_start=train_start, train_window_end=train_end,
            hyperparameters={"xi": self.xi, "l2": self.l2,
                             "referee_shrinkage": self.referee_shrinkage},
            parameters={
                "tendency": self.params.tendency,
                "intercept": self.params.intercept,
                "referee": self.params.referee,
                "referee_matches": self.params.referee_matches,
                "dispersion": self.params.dispersion,
                "competition_defaults": self.params.competition_defaults,
            },
            metrics=metrics or {}, model_scope=model_scope)

    @staticmethod
    def from_artifact(artifact: ModelArtifact) -> "CardModel":
        model = CardModel(xi=artifact.hyperparameters.get("xi", 0.0025),
                          l2=artifact.hyperparameters.get("l2", 2.0))
        stored = artifact.parameters
        model.params = CardParams(
            tendency={int(k): v for k, v in stored["tendency"].items()},
            intercept={int(k): v for k, v in stored["intercept"].items()},
            referee={int(k): v for k, v in stored.get("referee", {}).items()},
            referee_matches={int(k): v for k, v
                             in stored.get("referee_matches", {}).items()},
            dispersion=stored.get("dispersion", 12.0),
            competition_defaults={int(k): v for k, v
                                  in stored.get("competition_defaults", {}).items()},
        )
        return model
