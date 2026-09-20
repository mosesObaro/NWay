"""Model protocol and artefact handling."""

from __future__ import annotations

import datetime as dt
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from nway import clock
from nway.config import PROJECT_ROOT
from nway.storage.db import Database

ARTIFACT_DIR = PROJECT_ROOT / "data" / "models"


class LeakageError(RuntimeError):
    """A model was asked to predict at a time before its training window ended."""


@dataclass
class ModelArtifact:
    model_version: str
    model_family: str
    target: str
    trained_at: dt.datetime
    train_window_start: dt.datetime
    train_window_end: dt.datetime
    hyperparameters: dict[str, Any]
    parameters: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_version: str | None = None

    def assert_usable_at(self, as_of: dt.datetime) -> None:
        """A model may not predict a moment that its training data already saw.

        This is checked at load time rather than left to review, because it is
        the single easiest leak to introduce: a model trained through May 2026
        quietly producing March 2026 backtest predictions looks like a very
        good model.
        """
        if self.train_window_end > clock.ensure_utc(as_of):
            raise LeakageError(
                f"model {self.model_version} trained through "
                f"{clock.to_iso(self.train_window_end)} cannot predict at "
                f"{clock.to_iso(as_of)}")

    def path(self) -> Path:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        return ARTIFACT_DIR / f"{self.model_version}.pkl"

    def save(self) -> Path:
        path = self.path()
        with path.open("wb") as handle:
            pickle.dump(self, handle)
        return path

    @staticmethod
    def load(model_version: str) -> "ModelArtifact":
        path = ARTIFACT_DIR / f"{model_version}.pkl"
        with path.open("rb") as handle:
            return pickle.load(handle)

    def register(self, db: Database, active: bool = True) -> None:
        path = self.save()
        if active:
            db.execute("UPDATE model_version SET is_active = 0 WHERE target = ?",
                       (self.target,))
        db.execute(
            "INSERT OR REPLACE INTO model_version (model_version, model_family, target, "
            "trained_at, train_window_start, train_window_end, feature_version, "
            "hyperparameters, artifact_path, metrics, is_active) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (self.model_version, self.model_family, self.target,
             clock.to_iso(self.trained_at), clock.to_iso(self.train_window_start),
             clock.to_iso(self.train_window_end), self.feature_version,
             json.dumps(self.hyperparameters), str(path),
             json.dumps(self.metrics), int(active)))


class GoalModel(Protocol):
    """Anything that turns a fixture into two goal rates."""

    def predict_lambdas(self, home_team_id: int, away_team_id: int,
                        competition_id: int) -> tuple[float, float]: ...


def active_model_version(db: Database, target: str = "GOALS") -> str | None:
    return db.scalar(
        "SELECT model_version FROM model_version WHERE target=? AND is_active=1 "
        "ORDER BY trained_at DESC LIMIT 1", (target,))
