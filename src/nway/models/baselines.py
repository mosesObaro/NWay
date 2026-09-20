"""Baseline models.

These exist so that every later claim has something to beat. Measured over
21,544 matches from the seven target leagues (research/feasibility_study.py):

    uniform (1/3 each)    log loss 1.0986   Brier 0.6667
    pooled base rate      log loss 1.0711   Brier 0.6481
    market closing odds   log loss 0.9563   Brier 0.5670

The goal model must beat the pooled base rate. The market figure is a ceiling,
not a target: a held-out score below ~0.95 is treated as a leakage alarm.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from nway.markets import MarketProbabilities, derive_markets


@dataclass
class UniformBaseline:
    """The floor. Cannot be beaten by accident."""
    name: str = "baseline_uniform"

    def predict(self, *_args: Any, **_kwargs: Any) -> dict[str, float]:
        return {"HOME_WIN": 1 / 3, "DRAW": 1 / 3, "AWAY_WIN": 1 / 3}


@dataclass
class BaseRateBaseline:
    """Competition-and-season frequency of each outcome.

    This is the bar a real model has to clear. It is not a straw man: knowing
    that home teams win 43.8% of the time is genuinely most of what is knowable
    about a football match.
    """
    name: str = "baseline_base_rate"
    rates: dict[int, dict[str, float]] = None  # competition_id -> outcome rates
    pooled: dict[str, float] = None

    def fit(self, matches: Sequence[dict[str, Any]]) -> "BaseRateBaseline":
        counts: dict[int, dict[str, int]] = defaultdict(
            lambda: {"H": 0, "D": 0, "A": 0})
        pooled: dict[str, int] = {"H": 0, "D": 0, "A": 0}
        for match in matches:
            outcome = match["outcome"]
            if outcome not in pooled:
                continue
            counts[match["competition_id"]][outcome] += 1
            pooled[outcome] += 1
        total = sum(pooled.values()) or 1
        self.pooled = {
            "HOME_WIN": pooled["H"] / total,
            "DRAW": pooled["D"] / total,
            "AWAY_WIN": pooled["A"] / total,
        }
        self.rates = {}
        for competition_id, tally in counts.items():
            subtotal = sum(tally.values()) or 1
            # Below 200 matches a competition's own rate is noisier than the
            # pooled rate, so fall back rather than fit noise.
            if subtotal < 200:
                continue
            self.rates[competition_id] = {
                "HOME_WIN": tally["H"] / subtotal,
                "DRAW": tally["D"] / subtotal,
                "AWAY_WIN": tally["A"] / subtotal,
            }
        return self

    def predict(self, competition_id: int | None = None, **_kwargs: Any) -> dict[str, float]:
        if competition_id is not None and self.rates and competition_id in self.rates:
            return dict(self.rates[competition_id])
        return dict(self.pooled or {"HOME_WIN": 0.438, "DRAW": 0.248, "AWAY_WIN": 0.313})


@dataclass
class NaiveGoalsBaseline:
    """Independent Poisson from each team's mean goals for and against.

    Isolates how much Dixon-Coles' additions -- opponent adjustment, time
    decay, shrinkage, the low-score correction -- are actually worth.
    """
    name: str = "baseline_naive_goals"
    team_scored: dict[int, float] = None
    team_conceded: dict[int, float] = None
    league_home: float = 1.55
    league_away: float = 1.26

    def fit(self, matches: Sequence[dict[str, Any]]) -> "NaiveGoalsBaseline":
        scored: dict[int, list[int]] = defaultdict(list)
        conceded: dict[int, list[int]] = defaultdict(list)
        home_goals, away_goals = [], []
        for match in matches:
            scored[match["home_team_id"]].append(match["home_goals"])
            conceded[match["home_team_id"]].append(match["away_goals"])
            scored[match["away_team_id"]].append(match["away_goals"])
            conceded[match["away_team_id"]].append(match["home_goals"])
            home_goals.append(match["home_goals"])
            away_goals.append(match["away_goals"])
        self.team_scored = {k: sum(v) / len(v) for k, v in scored.items() if v}
        self.team_conceded = {k: sum(v) / len(v) for k, v in conceded.items() if v}
        if home_goals:
            self.league_home = sum(home_goals) / len(home_goals)
            self.league_away = sum(away_goals) / len(away_goals)
        return self

    def predict_lambdas(self, home_team_id: int, away_team_id: int,
                        competition_id: int | None = None) -> tuple[float, float]:
        scored = self.team_scored or {}
        conceded = self.team_conceded or {}
        league_mean = (self.league_home + self.league_away) / 2
        home_attack = scored.get(home_team_id, self.league_home)
        away_defence = conceded.get(away_team_id, self.league_away)
        away_attack = scored.get(away_team_id, self.league_away)
        home_defence = conceded.get(home_team_id, self.league_home)
        lam_home = max(0.15, (home_attack + away_defence) / 2 * (self.league_home / league_mean))
        lam_away = max(0.15, (away_attack + home_defence) / 2 * (self.league_away / league_mean))
        return lam_home, lam_away

    def predict(self, home_team_id: int, away_team_id: int,
                competition_id: int | None = None, **_kwargs: Any) -> MarketProbabilities:
        return derive_markets(*self.predict_lambdas(home_team_id, away_team_id))


@dataclass
class FormBaseline:
    """Recent points-per-match difference mapped to outcome probabilities.

    Tests whether raw recent form carries signal on its own, which is the
    assumption most casual prediction systems rest on.
    """
    name: str = "baseline_form"
    window: int = 5
    base: dict[str, float] = None

    def fit(self, matches: Sequence[dict[str, Any]]) -> "FormBaseline":
        self.base = BaseRateBaseline().fit(matches).pooled
        return self

    def predict(self, home_form: float | None, away_form: float | None,
                **_kwargs: Any) -> dict[str, float]:
        base = dict(self.base or {"HOME_WIN": 0.438, "DRAW": 0.248, "AWAY_WIN": 0.313})
        if home_form is None or away_form is None:
            return base
        # Points per match runs 0..3; the shift is deliberately gentle because
        # form alone is a weak signal and an aggressive mapping just produces
        # overconfident predictions.
        shift = max(-0.18, min(0.18, (home_form - away_form) * 0.06))
        probabilities = {
            "HOME_WIN": base["HOME_WIN"] + shift,
            "DRAW": base["DRAW"] - abs(shift) * 0.35,
            "AWAY_WIN": base["AWAY_WIN"] - shift + abs(shift) * 0.35,
        }
        clipped = {k: max(0.01, v) for k, v in probabilities.items()}
        total = sum(clipped.values())
        return {k: v / total for k, v in clipped.items()}
