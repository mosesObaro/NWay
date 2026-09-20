"""Architectural rules, enforced rather than documented.

Each of these encodes a decision from docs/SYSTEM_DESIGN.md that is easy to
erode silently during ordinary refactoring.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "nway"


def _python_files():
    return sorted(SRC.rglob("*.py"))


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_recommendation_never_imports_models():
    """Selection policy and prediction quality must change independently.

    Market derivation lives in nway.markets precisely so the engine can compute
    exact within-fixture correlation without reaching into the model package.
    """
    for path in (SRC / "recommendation").rglob("*.py"):
        offending = [name for name in _imports(path) if name.startswith("nway.models")]
        assert not offending, f"{path.name} imports {offending}"


def test_planner_never_imports_a_delivery_provider():
    """The planner decides what and when; providers decide how."""
    planner = SRC / "notifications" / "planner.py"
    offending = [name for name in _imports(planner) if "delivery" in name]
    assert not offending, f"planner imports {offending}"


def test_only_clock_module_reads_wall_time():
    """Every scheduling decision must flow through the injectable clock.

    Without this the backtester cannot replay the production path, and
    server-local time creeps into comparisons that must be UTC.
    """
    pattern = re.compile(r"datetime\.(now|utcnow)\s*\(|\bdt\.datetime\.now\s*\(")
    # logging_setup stamps log lines, which is an observation of the real world
    # rather than a scheduling decision -- a frozen backtest clock must NOT make
    # its log lines claim to have been written in 2023.
    exempt = {"clock.py", "logging_setup.py"}
    offenders = []
    for path in _python_files():
        if path.name in exempt:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line):
                offenders.append(f"{path.relative_to(SRC)}:{number}")
    assert not offenders, f"wall-clock reads outside clock.py: {offenders}"


def test_no_scheduling_query_filters_on_matchday():
    """Fixtures drive scheduling; matchday is reporting metadata only."""
    for module in ("storage/repositories.py", "scheduling/tick.py",
                   "notifications/planner.py", "recommendation/engine.py"):
        text = (SRC / module).read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("*"):
                continue
            assert not re.search(r"(WHERE|AND)\s+\S*matchday", stripped, re.I), \
                f"{module} filters on matchday: {stripped!r}"


def test_every_market_in_config_has_a_derivation_or_a_model():
    from nway.config import load_config

    config = load_config()
    for market in config.markets:
        assert market.derived_from or market.model or not market.mvp, \
            f"{market.key} has neither a derivation nor a model"


def test_enabled_competitions_reference_real_competitions():
    from nway.config import load_config

    config = load_config()
    slugs = {competition.slug for competition in config.competitions}
    for market in config.markets:
        for slug in market.enabled_competitions:
            assert slug == "ALL" or slug in slugs, \
                f"{market.key} enables unknown competition {slug}"


def test_competition_provider_codes_are_unique_per_provider():
    """Guards the CL/UCL trap: football-data.org uses UCL for CONFERENCE League."""
    from nway.config import load_config

    config = load_config()
    seen: dict[tuple[str, str], str] = {}
    for competition in config.competitions:
        for provider, code in competition.providers.items():
            key = (provider, code)
            assert key not in seen, (
                f"{provider} code {code} maps to both {seen[key]} "
                f"and {competition.slug}")
            seen[key] = competition.slug


def test_champions_and_conference_league_codes_are_not_confused():
    from nway.config import load_config

    config = load_config()
    champions = config.competition("champions_league")
    conference = config.competition("conference_league")
    assert champions.providers["football_data_org"] == "CL"
    assert conference.providers["football_data_org"] == "UCL"


def test_market_floors_match_the_documented_rule():
    """min_probability must equal max(absolute, base + k*(1-base))."""
    from nway.config import load_config, market_floor

    config = load_config()
    absolute = float(config.thresholds["absolute_floor"])
    lift = float(config.thresholds["lift_k"])
    for market in config.markets:
        if market.base_rate is None or market.min_probability is None:
            continue
        expected = market_floor(market.base_rate, absolute, lift)
        assert abs(market.min_probability - expected) < 0.002, (
            f"{market.key}: configured {market.min_probability}, "
            f"rule gives {expected:.3f}")


def test_over_1_5_floor_exceeds_its_base_rate_meaningfully():
    """A 72% 'high confidence' Over 1.5 would be weaker than the league average."""
    from nway.config import load_config

    config = load_config()
    market = config.market("OVER_1_5")
    assert market.min_probability > market.base_rate + 0.04
