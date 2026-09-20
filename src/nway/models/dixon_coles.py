"""Dixon-Coles goal model.

Time-weighted maximum likelihood over attack and defence ratings, with
per-competition intercepts and home advantage, L2 shrinkage toward the
competition mean, and the low-score dependence correction.

    log lambda_home = mu_c + home_adv_c + attack_i + defence_j
    log lambda_away = mu_c              + attack_j + defence_i

Why this family and not a three-class classifier: modelling goals first gives
one coherent joint distribution, so every goal market is a summation over it
and cannot contradict any other. Why Poisson rather than negative binomial for
goals: the measured variance-to-mean ratio of total goals across 21,545
matches is 1.00 (corners are 1.18 and cards 1.34, which is why those get a
different family).

Two structural details that matter empirically:

  * ``crowd_present`` -- home advantage collapsed in 2020/21 (40.2% home wins
    against 46.1% in 2017/18, mean goal difference +0.155 against +0.356). An
    indicator absorbs that into a parameter instead of letting it contaminate
    every team rating.
  * shrinkage -- without it a promoted side with four matches played gets an
    extreme rating, and August is exactly when the fixture calendar is dense
    and the notifier is most active.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

from nway import clock
from nway.logging_setup import get_logger
from nway.models.base import ModelArtifact
from nway.markets import MarketProbabilities, derive_markets

log = get_logger(__name__)

# Seasons played behind closed doors. Home advantage was materially different.
CLOSED_DOORS_START = dt.date(2020, 3, 12)
CLOSED_DOORS_END = dt.date(2021, 6, 30)


@dataclass
class DixonColesParams:
    attack: dict[int, float] = field(default_factory=dict)
    defence: dict[int, float] = field(default_factory=dict)
    intercept: dict[int, float] = field(default_factory=dict)
    home_advantage: dict[int, float] = field(default_factory=dict)
    rho: float = -0.03
    crowd_effect: float = 0.0
    xi: float = 0.0030
    l2: float = 0.5
    league_defaults: dict[int, tuple[float, float]] = field(default_factory=dict)


class DixonColesModel:
    def __init__(self, xi: float = 0.0030, l2: float = 0.5,
                 max_iterations: int = 300) -> None:
        self.xi = xi
        self.l2 = l2
        self.max_iterations = max_iterations
        self.params = DixonColesParams(xi=xi, l2=l2)
        self.teams: list[int] = []

    # -- fitting ---------------------------------------------------------
    def fit(self, matches: Sequence[dict[str, Any]],
            as_of: dt.datetime) -> "DixonColesModel":
        """Fit on matches that were completed and knowable before ``as_of``."""
        usable = [m for m in matches if m.get("home_goals") is not None]
        if len(usable) < 50:
            raise ValueError(f"not enough matches to fit: {len(usable)}")

        teams = sorted({m["home_team_id"] for m in usable}
                       | {m["away_team_id"] for m in usable})
        competitions = sorted({m["competition_id"] for m in usable})
        self.teams = teams
        team_index = {t: i for i, t in enumerate(teams)}
        competition_index = {c: i for i, c in enumerate(competitions)}

        n_teams, n_competitions = len(teams), len(competitions)
        home_idx = np.array([team_index[m["home_team_id"]] for m in usable])
        away_idx = np.array([team_index[m["away_team_id"]] for m in usable])
        comp_idx = np.array([competition_index[m["competition_id"]] for m in usable])
        home_goals = np.array([m["home_goals"] for m in usable], dtype=float)
        away_goals = np.array([m["away_goals"] for m in usable], dtype=float)

        kickoffs = [clock.from_iso(m["kickoff_utc"]) for m in usable]
        age_days = np.array(
            [max(0.0, (clock.ensure_utc(as_of) - k).total_seconds() / 86400.0)
             for k in kickoffs])
        weights = np.exp(-self.xi * age_days)
        closed = np.array(
            [1.0 if CLOSED_DOORS_START <= k.date() <= CLOSED_DOORS_END else 0.0
             for k in kickoffs])

        # Parameter vector: attack (n-1, last implied by sum-to-zero),
        # defence (n-1), intercepts, home advantages, rho, crowd effect.
        def unpack(theta: np.ndarray):
            offset = 0
            attack = np.zeros(n_teams)
            attack[:-1] = theta[offset:offset + n_teams - 1]
            attack[-1] = -attack[:-1].sum()
            offset += n_teams - 1
            defence = np.zeros(n_teams)
            defence[:-1] = theta[offset:offset + n_teams - 1]
            defence[-1] = -defence[:-1].sum()
            offset += n_teams - 1
            intercept = theta[offset:offset + n_competitions]
            offset += n_competitions
            home_adv = theta[offset:offset + n_competitions]
            offset += n_competitions
            rho = theta[offset]
            crowd = theta[offset + 1]
            return attack, defence, intercept, home_adv, rho, crowd

        def negative_log_likelihood(theta: np.ndarray) -> float:
            attack, defence, intercept, home_adv, rho, crowd = unpack(theta)
            rho = float(np.clip(rho, -0.2, 0.2))
            log_home = (intercept[comp_idx] + home_adv[comp_idx] + crowd * closed
                        + attack[home_idx] + defence[away_idx])
            log_away = intercept[comp_idx] + attack[away_idx] + defence[home_idx]
            lam_home = np.exp(np.clip(log_home, -4.0, 2.2))
            lam_away = np.exp(np.clip(log_away, -4.0, 2.2))

            log_likelihood = (home_goals * np.log(lam_home) - lam_home
                              + away_goals * np.log(lam_away) - lam_away)

            # Dixon-Coles correction, applied only to the four low-score cells.
            tau = np.ones_like(lam_home)
            both_zero = (home_goals == 0) & (away_goals == 0)
            zero_one = (home_goals == 0) & (away_goals == 1)
            one_zero = (home_goals == 1) & (away_goals == 0)
            one_one = (home_goals == 1) & (away_goals == 1)
            tau[both_zero] = 1.0 - lam_home[both_zero] * lam_away[both_zero] * rho
            tau[zero_one] = 1.0 + lam_home[zero_one] * rho
            tau[one_zero] = 1.0 + lam_away[one_zero] * rho
            tau[one_one] = 1.0 - rho
            tau = np.clip(tau, 1e-6, None)
            log_likelihood = log_likelihood + np.log(tau)

            penalty = self.l2 * (np.sum(attack ** 2) + np.sum(defence ** 2))
            return float(-np.sum(weights * log_likelihood) + penalty)

        theta0 = np.concatenate([
            np.zeros(n_teams - 1), np.zeros(n_teams - 1),
            np.full(n_competitions, math.log(1.35)),
            np.full(n_competitions, 0.25),
            np.array([-0.03, -0.10]),
        ])
        result = minimize(negative_log_likelihood, theta0, method="L-BFGS-B",
                          options={"maxiter": self.max_iterations, "maxfun": 200000})

        attack, defence, intercept, home_adv, rho, crowd = unpack(result.x)
        self.params = DixonColesParams(
            attack={t: float(attack[i]) for t, i in team_index.items()},
            defence={t: float(defence[i]) for t, i in team_index.items()},
            intercept={c: float(intercept[i]) for c, i in competition_index.items()},
            home_advantage={c: float(home_adv[i]) for c, i in competition_index.items()},
            rho=float(np.clip(rho, -0.2, 0.2)), crowd_effect=float(crowd),
            xi=self.xi, l2=self.l2,
        )

        # A fallback pair of rates per competition, used when a team has no
        # rating at all (a newly promoted side the model has never seen).
        by_competition: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for match in usable:
            by_competition[match["competition_id"]].append(
                (match["home_goals"], match["away_goals"]))
        self.params.league_defaults = {
            competition: (
                sum(h for h, _ in pairs) / len(pairs),
                sum(a for _, a in pairs) / len(pairs),
            ) for competition, pairs in by_competition.items() if pairs
        }

        log.info("fitted dixon-coles", context={
            "matches": len(usable), "teams": n_teams,
            "competitions": n_competitions, "rho": round(self.params.rho, 4),
            "crowd_effect": round(self.params.crowd_effect, 4),
            "converged": bool(result.success),
        })
        return self

    # -- prediction ------------------------------------------------------
    def is_fitted_for(self, competition_id: int) -> bool:
        """Was this competition present in the training data?

        Goal rate and home advantage are fitted per competition, so a
        competition the model never saw has neither. Without this check
        predict_lambdas silently returns the same league-average pair for
        every fixture in it -- identical probabilities for Roma v Real Madrid
        and Feyenoord v Como, which looks like a prediction and is not one.
        """
        return competition_id in self.params.intercept

    def predict_lambdas(self, home_team_id: int, away_team_id: int,
                        competition_id: int) -> tuple[float, float]:
        params = self.params
        default_home, default_away = params.league_defaults.get(
            competition_id, (1.55, 1.26))
        if competition_id not in params.intercept:
            # Callers should gate on is_fitted_for; this remains only so the
            # function is total, and returns an explicitly average pair.
            return default_home, default_away

        intercept = params.intercept[competition_id]
        home_adv = params.home_advantage.get(competition_id, 0.25)
        # An unrated team gets 0.0, i.e. exactly average for its competition.
        attack_home = params.attack.get(home_team_id, 0.0)
        attack_away = params.attack.get(away_team_id, 0.0)
        defence_home = params.defence.get(home_team_id, 0.0)
        defence_away = params.defence.get(away_team_id, 0.0)

        lam_home = math.exp(min(2.2, intercept + home_adv + attack_home + defence_away))
        lam_away = math.exp(min(2.2, intercept + attack_away + defence_home))
        return max(0.12, lam_home), max(0.12, lam_away)

    def predict(self, home_team_id: int, away_team_id: int,
                competition_id: int) -> MarketProbabilities:
        lam_home, lam_away = self.predict_lambdas(
            home_team_id, away_team_id, competition_id)
        return derive_markets(lam_home, lam_away, self.params.rho)

    def team_rating(self, team_id: int) -> dict[str, float]:
        return {
            "attack": self.params.attack.get(team_id, 0.0),
            "defence": self.params.defence.get(team_id, 0.0),
        }

    # -- persistence -----------------------------------------------------
    def to_artifact(self, model_version: str, train_start: dt.datetime,
                    train_end: dt.datetime, metrics: dict[str, Any] | None = None,
                    feature_version: str | None = None) -> ModelArtifact:
        return ModelArtifact(
            model_version=model_version, model_family="DIXON_COLES", target="GOALS",
            trained_at=clock.now(), train_window_start=train_start,
            train_window_end=train_end,
            hyperparameters={"xi": self.xi, "l2": self.l2,
                             "max_iterations": self.max_iterations},
            parameters={
                "attack": self.params.attack, "defence": self.params.defence,
                "intercept": self.params.intercept,
                "home_advantage": self.params.home_advantage,
                "rho": self.params.rho, "crowd_effect": self.params.crowd_effect,
                "league_defaults": self.params.league_defaults,
            },
            metrics=metrics or {}, feature_version=feature_version,
        )

    @staticmethod
    def from_artifact(artifact: ModelArtifact) -> "DixonColesModel":
        model = DixonColesModel(
            xi=artifact.hyperparameters.get("xi", 0.0030),
            l2=artifact.hyperparameters.get("l2", 0.5))
        stored = artifact.parameters
        model.params = DixonColesParams(
            attack={int(k): v for k, v in stored["attack"].items()},
            defence={int(k): v for k, v in stored["defence"].items()},
            intercept={int(k): v for k, v in stored["intercept"].items()},
            home_advantage={int(k): v for k, v in stored["home_advantage"].items()},
            rho=stored["rho"], crowd_effect=stored.get("crowd_effect", 0.0),
            xi=model.xi, l2=model.l2,
            league_defaults={int(k): tuple(v)
                             for k, v in stored.get("league_defaults", {}).items()},
        )
        model.teams = sorted(model.params.attack)
        return model
