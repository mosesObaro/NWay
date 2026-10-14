"""First-half goal share, per competition.

The only parameter the half markets need. Everything else is derived from the
authoritative full-time grid by a binomial split (see nway.markets.halves), so
coherence between half and full-time markets holds by construction rather than
by fitting a second model that would disagree with the first.

Measured across 21,919 matches: 0.4429 pooled, from 0.4359 (Serie A) to 0.4512
(Primeira Liga). The spread is small but systematic, and shrinkage toward the
pooled value keeps a thin competition from chasing noise.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from nway import clock
from nway.logging_setup import get_logger
from nway.models.base import ModelArtifact

log = get_logger(__name__)

POOLED_SHARE = 0.4429
SHRINKAGE_MATCHES = 400.0     # pseudo-matches pulling a competition to pooled


@dataclass
class HalfParams:
    share: dict[int, float] = field(default_factory=dict)
    pooled: float = POOLED_SHARE
    matches: dict[int, int] = field(default_factory=dict)


class HalfShareModel:
    def __init__(self, shrinkage: float = SHRINKAGE_MATCHES) -> None:
        self.shrinkage = shrinkage
        self.params = HalfParams()

    def fit(self, matches: Sequence[dict[str, Any]],
            as_of: dt.datetime) -> "HalfShareModel":
        usable = [m for m in matches
                  if m.get("ht_home_goals") is not None
                  and m.get("ht_away_goals") is not None
                  and m.get("home_goals") is not None]
        if not usable:
            raise ValueError("no half-time data available")

        first: dict[int, float] = defaultdict(float)
        full: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        for match in usable:
            comp = match["competition_id"]
            first[comp] += match["ht_home_goals"] + match["ht_away_goals"]
            full[comp] += match["home_goals"] + match["away_goals"]
            counts[comp] += 1

        pooled_first = sum(first.values())
        pooled_full = sum(full.values())
        pooled = (pooled_first / pooled_full) if pooled_full else POOLED_SHARE

        shares: dict[int, float] = {}
        for comp, total_goals in full.items():
            if total_goals <= 0:
                continue
            observed = first[comp] / total_goals
            n = counts[comp]
            weight = n / (n + self.shrinkage)
            shares[comp] = weight * observed + (1 - weight) * pooled

        self.params = HalfParams(share=shares, pooled=float(pooled),
                                 matches=dict(counts))
        log.info("fitted half-share model", context={
            "matches": len(usable), "competitions": len(shares),
            "pooled_share": round(pooled, 4)})
        return self

    def share_for(self, competition_id: int) -> float:
        return self.params.share.get(competition_id, self.params.pooled)

    def to_artifact(self, model_version: str, train_start: dt.datetime,
                    train_end: dt.datetime,
                    model_scope: str = "mens_club") -> ModelArtifact:
        return ModelArtifact(
            model_version=model_version, model_family="HALF_SHARE",
            target="HALVES", trained_at=clock.now(),
            train_window_start=train_start, train_window_end=train_end,
            hyperparameters={"shrinkage": self.shrinkage},
            parameters={"share": self.params.share, "pooled": self.params.pooled,
                        "matches": self.params.matches},
            model_scope=model_scope)

    @staticmethod
    def from_artifact(artifact: ModelArtifact) -> "HalfShareModel":
        model = HalfShareModel(
            shrinkage=artifact.hyperparameters.get("shrinkage", SHRINKAGE_MATCHES))
        stored = artifact.parameters
        model.params = HalfParams(
            share={int(k): v for k, v in stored["share"].items()},
            pooled=stored.get("pooled", POOLED_SHARE),
            matches={int(k): v for k, v in stored.get("matches", {}).items()})
        return model
