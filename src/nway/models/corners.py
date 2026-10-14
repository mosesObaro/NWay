"""Corner model: negative-binomial total with a beta-binomial home share.

Fitted jointly, because the construction in ``nway.markets.corners`` makes the
likelihood exact: a match contributes the NB probability of its total and the
beta-binomial probability of the home side's share of that total.

    log mu_home = mu_c + home_adv_c + attack_h + concede_a + beta * supremacy
    log mu_away = mu_c              + attack_a + concede_h - beta * supremacy
    mu_total    = mu_home + mu_away
    share       = mu_home / mu_total

Supremacy is the goal model's expected margin: a side expected to dominate
takes more corners. Measured correlation between corner difference and goal
supremacy is +0.105 -- small but real, and free to include since the goal model
has already produced the rates.

Dispersion and share concentration are fitted, not assumed. Measured moments
the fit has to reproduce: total corners mean 9.87, variance 11.68; home mean
5.41, variance 8.95; correlation between sides -0.276.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln

from nway import clock
from nway.logging_setup import get_logger
from nway.models.base import ModelArtifact

log = get_logger(__name__)

MIN_MATCHES = 500


@dataclass
class CornerParams:
    attack: dict[int, float] = field(default_factory=dict)
    concede: dict[int, float] = field(default_factory=dict)
    intercept: dict[int, float] = field(default_factory=dict)
    home_advantage: dict[int, float] = field(default_factory=dict)
    supremacy: float = 0.0
    dispersion: float = 50.0
    concentration: float = 7.0
    competition_defaults: dict[int, tuple[float, float]] = field(default_factory=dict)


class CornerModel:
    """Negative-binomial corners with a beta-binomial split."""

    def __init__(self, xi: float = 0.0030, l2: float = 1.0,
                 max_iterations: int = 400) -> None:
        self.xi = xi
        self.l2 = l2
        self.max_iterations = max_iterations
        self.params = CornerParams()

    def fit(self, matches: Sequence[dict[str, Any]],
            as_of: dt.datetime) -> "CornerModel":
        usable = [m for m in matches
                  if m.get("home_corners") is not None
                  and m.get("away_corners") is not None]
        if len(usable) < MIN_MATCHES:
            raise ValueError(f"not enough corner matches to fit: {len(usable)}")

        teams = sorted({m["home_team_id"] for m in usable}
                       | {m["away_team_id"] for m in usable})
        competitions = sorted({m["competition_id"] for m in usable})
        team_index = {t: i for i, t in enumerate(teams)}
        comp_index = {c: i for i, c in enumerate(competitions)}
        n_teams, n_comps = len(teams), len(competitions)

        home_idx = np.array([team_index[m["home_team_id"]] for m in usable])
        away_idx = np.array([team_index[m["away_team_id"]] for m in usable])
        comp_idx = np.array([comp_index[m["competition_id"]] for m in usable])
        home_corners = np.array([m["home_corners"] for m in usable], dtype=float)
        away_corners = np.array([m["away_corners"] for m in usable], dtype=float)
        totals = home_corners + away_corners
        supremacy = np.array([float(m.get("supremacy") or 0.0) for m in usable])

        kickoffs = [clock.from_iso(m["kickoff_utc"]) for m in usable]
        age_days = np.array([max(0.0, (clock.ensure_utc(as_of) - k).total_seconds()
                                 / 86400.0) for k in kickoffs])
        weights = np.exp(-self.xi * age_days)

        def unpack(theta: np.ndarray):
            offset = 0
            attack = np.zeros(n_teams)
            attack[:-1] = theta[offset:offset + n_teams - 1]
            attack[-1] = -attack[:-1].sum()
            offset += n_teams - 1
            concede = np.zeros(n_teams)
            concede[:-1] = theta[offset:offset + n_teams - 1]
            concede[-1] = -concede[:-1].sum()
            offset += n_teams - 1
            intercept = theta[offset:offset + n_comps]
            offset += n_comps
            home_adv = theta[offset:offset + n_comps]
            offset += n_comps
            return (attack, concede, intercept, home_adv,
                    theta[offset], theta[offset + 1], theta[offset + 2])

        def negative_log_likelihood(theta: np.ndarray) -> float:
            (attack, concede, intercept, home_adv, beta_sup,
             log_dispersion, log_concentration) = unpack(theta)
            dispersion = float(np.exp(np.clip(log_dispersion, -2.0, 8.0)))
            concentration = float(np.exp(np.clip(log_concentration, -1.0, 6.0)))

            mu_home = np.exp(np.clip(
                intercept[comp_idx] + home_adv[comp_idx] + attack[home_idx]
                + concede[away_idx] + beta_sup * supremacy, -3.0, 3.6))
            mu_away = np.exp(np.clip(
                intercept[comp_idx] + attack[away_idx] + concede[home_idx]
                - beta_sup * supremacy, -3.0, 3.6))
            mu_total = mu_home + mu_away
            share = np.clip(mu_home / mu_total, 1e-4, 1 - 1e-4)

            # Negative binomial on the total.
            nb = (gammaln(totals + dispersion) - gammaln(dispersion)
                  - gammaln(totals + 1.0)
                  + dispersion * np.log(dispersion / (dispersion + mu_total))
                  + totals * np.log(mu_total / (dispersion + mu_total)))

            # Beta-binomial on the home share of that total.
            alpha = share * concentration
            beta_p = (1.0 - share) * concentration
            bb = (gammaln(totals + 1.0) - gammaln(home_corners + 1.0)
                  - gammaln(away_corners + 1.0)
                  + betaln(home_corners + alpha, away_corners + beta_p)
                  - betaln(alpha, beta_p))

            penalty = self.l2 * (np.sum(attack ** 2) + np.sum(concede ** 2))
            return float(-np.sum(weights * (nb + bb)) + penalty)

        theta0 = np.concatenate([
            np.zeros(n_teams - 1), np.zeros(n_teams - 1),
            np.full(n_comps, np.log(4.9)), np.full(n_comps, 0.2),
            np.array([0.05, np.log(50.0), np.log(7.0)]),
        ])
        result = minimize(negative_log_likelihood, theta0, method="L-BFGS-B",
                          options={"maxiter": self.max_iterations, "maxfun": 300000})

        (attack, concede, intercept, home_adv, beta_sup,
         log_dispersion, log_concentration) = unpack(result.x)
        self.params = CornerParams(
            attack={t: float(attack[i]) for t, i in team_index.items()},
            concede={t: float(concede[i]) for t, i in team_index.items()},
            intercept={c: float(intercept[i]) for c, i in comp_index.items()},
            home_advantage={c: float(home_adv[i]) for c, i in comp_index.items()},
            supremacy=float(beta_sup),
            dispersion=float(np.exp(np.clip(log_dispersion, -2.0, 8.0))),
            concentration=float(np.exp(np.clip(log_concentration, -1.0, 6.0))),
        )

        by_comp: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for match in usable:
            by_comp[match["competition_id"]].append(
                (match["home_corners"], match["away_corners"]))
        self.params.competition_defaults = {
            comp: (sum(h for h, _ in rows) / len(rows),
                   sum(a for _, a in rows) / len(rows))
            for comp, rows in by_comp.items() if rows
        }

        log.info("fitted corner model", context={
            "matches": len(usable), "teams": n_teams, "competitions": n_comps,
            "dispersion": round(self.params.dispersion, 2),
            "concentration": round(self.params.concentration, 2),
            "supremacy": round(self.params.supremacy, 4),
            "converged": bool(result.success)})
        return self

    def is_fitted_for(self, competition_id: int) -> bool:
        return competition_id in self.params.intercept

    def predict_parameters(self, home_team_id: int, away_team_id: int,
                           competition_id: int,
                           supremacy: float = 0.0) -> tuple[float, float, float, float]:
        """(mean_total, dispersion, home_share, concentration)."""
        params = self.params
        if competition_id not in params.intercept:
            home, away = params.competition_defaults.get(competition_id, (5.41, 4.45))
            total = home + away
            return total, params.dispersion, home / total, params.concentration

        intercept = params.intercept[competition_id]
        home_adv = params.home_advantage.get(competition_id, 0.2)
        mu_home = np.exp(min(3.6, intercept + home_adv
                             + params.attack.get(home_team_id, 0.0)
                             + params.concede.get(away_team_id, 0.0)
                             + params.supremacy * supremacy))
        mu_away = np.exp(min(3.6, intercept
                             + params.attack.get(away_team_id, 0.0)
                             + params.concede.get(home_team_id, 0.0)
                             - params.supremacy * supremacy))
        total = float(mu_home + mu_away)
        return (total, params.dispersion, float(mu_home) / total,
                params.concentration)

    def to_artifact(self, model_version: str, train_start: dt.datetime,
                    train_end: dt.datetime, metrics: dict[str, Any] | None = None,
                    model_scope: str = "mens_club") -> ModelArtifact:
        return ModelArtifact(
            model_version=model_version, model_family="NEG_BINOMIAL_BETA_BINOMIAL",
            target="CORNERS", trained_at=clock.now(),
            train_window_start=train_start, train_window_end=train_end,
            hyperparameters={"xi": self.xi, "l2": self.l2},
            parameters={
                "attack": self.params.attack, "concede": self.params.concede,
                "intercept": self.params.intercept,
                "home_advantage": self.params.home_advantage,
                "supremacy": self.params.supremacy,
                "dispersion": self.params.dispersion,
                "concentration": self.params.concentration,
                "competition_defaults": self.params.competition_defaults,
            },
            metrics=metrics or {}, model_scope=model_scope)

    @staticmethod
    def from_artifact(artifact: ModelArtifact) -> "CornerModel":
        model = CornerModel(xi=artifact.hyperparameters.get("xi", 0.0030),
                            l2=artifact.hyperparameters.get("l2", 1.0))
        stored = artifact.parameters
        model.params = CornerParams(
            attack={int(k): v for k, v in stored["attack"].items()},
            concede={int(k): v for k, v in stored["concede"].items()},
            intercept={int(k): v for k, v in stored["intercept"].items()},
            home_advantage={int(k): v for k, v in stored["home_advantage"].items()},
            supremacy=stored.get("supremacy", 0.0),
            dispersion=stored.get("dispersion", 50.0),
            concentration=stored.get("concentration", 7.0),
            competition_defaults={int(k): tuple(v) for k, v
                                  in stored.get("competition_defaults", {}).items()},
        )
        return model
