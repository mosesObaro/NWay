"""Configuration loading.

Every operational parameter lives in ``config/*.yaml``; every secret lives in
``.env``. Nothing is hard-coded. The resolved configuration is hashed and the
hash is stored on each prediction, so a prediction made under different
thresholds is traceable to the configuration that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")
_SECRET_HINT = re.compile(r"(token|key|password|secret)", re.IGNORECASE)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"


def load_dotenv(path: Path | None = None) -> None:
    """Read .env into os.environ without overwriting real environment vars."""
    path = path or (PROJECT_ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _expand(value: Any) -> Any:
    """Substitute ${ENV_VAR} references, leaving unset ones as None."""
    if isinstance(value, str):
        match = _ENV_PATTERN.fullmatch(value.strip())
        if match:
            return os.environ.get(match.group(1))
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


@dataclass(frozen=True)
class Competition:
    slug: str
    name: str
    kind: str
    structure: str
    enabled: bool
    providers: dict[str, str]
    country: str | None = None
    stats_history_from: str | None = None
    teams: int | None = None
    blocked_reason: str | None = None

    @property
    def is_uefa(self) -> bool:
        return self.kind == "UEFA_CLUB"


@dataclass(frozen=True)
class Market:
    key: str
    family: str
    mvp: bool
    base_rate: float | None = None
    min_probability: float | None = None
    derived_from: str | None = None
    model: str | None = None
    enabled_competitions: tuple[str, ...] = ()
    correlated_with: tuple[str, ...] = ()
    requires_stats: tuple[str, ...] = ()
    recommendable: bool = True
    notes: str | None = None
    blocked_reason: str | None = None
    disabled_reason: str | None = None

    def enabled_for(self, competition_slug: str) -> bool:
        if not self.recommendable:
            return False
        return "ALL" in self.enabled_competitions or competition_slug in self.enabled_competitions


@dataclass(frozen=True)
class Config:
    """The resolved, immutable configuration."""

    competitions: tuple[Competition, ...]
    markets: tuple[Market, ...]
    thresholds: dict[str, Any]
    eligibility: dict[str, Any]
    recommendations: dict[str, Any]
    notifications: dict[str, Any]
    sources: dict[str, Any]
    http: dict[str, Any]
    models: dict[str, Any]
    calibration: dict[str, Any]
    validation: dict[str, Any]
    benchmarks: dict[str, Any]
    config_hash: str
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    # -- lookups ---------------------------------------------------------
    def competition(self, slug: str) -> Competition:
        for comp in self.competitions:
            if comp.slug == slug:
                return comp
        raise KeyError(f"unknown competition: {slug}")

    def enabled_competitions(self) -> tuple[Competition, ...]:
        return tuple(c for c in self.competitions if c.enabled)

    def market(self, key: str) -> Market:
        for market in self.markets:
            if market.key == key:
                return market
        raise KeyError(f"unknown market: {key}")

    def recommendable_markets(self) -> tuple[Market, ...]:
        return tuple(m for m in self.markets if m.recommendable and m.enabled_competitions)

    def mvp_markets(self) -> tuple[Market, ...]:
        return tuple(m for m in self.markets if m.mvp)

    def provider_code(self, competition_slug: str, provider: str) -> str | None:
        return self.competition(competition_slug).providers.get(provider)

    def competition_for_provider_code(self, provider: str, code: str) -> Competition | None:
        for comp in self.competitions:
            if comp.providers.get(provider) == code:
                return comp
        return None


def _redact(value: Any) -> Any:
    """Strip secrets before hashing or logging the configuration."""
    if isinstance(value, dict):
        return {
            k: ("***" if _SECRET_HINT.search(str(k)) else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def load_config(config_dir: Path | None = None) -> Config:
    load_dotenv()
    directory = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    documents: dict[str, Any] = {}
    for name in ("competitions", "markets", "models", "notifications",
                 "recommendations", "sources"):
        path = directory / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"missing configuration file: {path}")
        documents[name] = _expand(yaml.safe_load(path.read_text()) or {})

    competitions: list[Competition] = []
    groups = documents["competitions"].get("competitions", {})
    for group in ("domestic", "uefa"):
        for entry in groups.get(group, []) or []:
            competitions.append(Competition(
                slug=entry["slug"], name=entry["name"], kind=entry["kind"],
                structure=entry["structure"], enabled=bool(entry.get("enabled", False)),
                providers=dict(entry.get("providers", {})),
                country=entry.get("country"),
                stats_history_from=entry.get("stats_history_from"),
                teams=entry.get("teams"),
                blocked_reason=entry.get("blocked_reason"),
            ))

    markets: list[Market] = []
    for entry in documents["markets"].get("markets", []) or []:
        markets.append(Market(
            key=entry["key"], family=entry["family"], mvp=bool(entry.get("mvp", False)),
            base_rate=entry.get("base_rate"), min_probability=entry.get("min_probability"),
            derived_from=entry.get("derived_from"), model=entry.get("model"),
            enabled_competitions=tuple(entry.get("enabled_competitions", []) or ()),
            correlated_with=tuple(entry.get("correlated_with", []) or ()),
            requires_stats=tuple(entry.get("requires_stats", []) or ()),
            recommendable=bool(entry.get("recommendable", True)),
            notes=entry.get("notes"), blocked_reason=entry.get("blocked_reason"),
            disabled_reason=entry.get("disabled_reason"),
        ))

    payload = _redact(documents)
    config_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    return Config(
        competitions=tuple(competitions),
        markets=tuple(markets),
        thresholds=documents["markets"].get("thresholds", {}),
        eligibility=documents["markets"].get("eligibility", {}),
        recommendations=documents["recommendations"].get("recommendations", {}),
        notifications=documents["notifications"].get("notifications", {}),
        sources=documents["sources"].get("sources", {}),
        http=documents["sources"].get("http", {}),
        models=documents["models"].get("models", {}),
        calibration=documents["models"].get("calibration", {}),
        validation=documents["models"].get("validation", {}),
        benchmarks=documents["models"].get("benchmarks", {}),
        config_hash=config_hash,
        raw=documents,
    )


def market_floor(base_rate: float, absolute_floor: float, lift_k: float) -> float:
    """floor = max(absolute, base + k*(1-base)).

    Both terms are needed. The relative term makes the floor mean the same
    thing in every market -- the model must be ``lift_k`` of the way from the
    market's own base rate to certainty. Over 1.5 comes in 77.1% of the time,
    so a flat 72% floor there would qualify predictions *weaker* than assuming
    the league average. The absolute term stops a low-base-rate market (an away
    win at 49%) from counting as high confidence merely by beating its base
    rate. Derivation: research/threshold_study.py.
    """
    return max(absolute_floor, base_rate + lift_k * (1.0 - base_rate))
