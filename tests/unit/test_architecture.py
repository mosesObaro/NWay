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


def test_every_mvp_market_names_the_grid_it_comes_from():
    """`grid` replaced the old free-text `derived_from`: it is the unit the
    diversity caps work on, so it has to be a real grid name."""
    from nway.config import load_config

    valid = {"GOAL_MATRIX", "HALF_MATRIX", "CORNER_MATRIX", "CARD_DISTRIBUTION"}
    for market in load_config().markets:
        if not market.mvp:
            continue
        assert market.grid in valid, \
            f"{market.key} declares an unknown grid {market.grid!r}"


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


def test_every_stored_market_is_declared_in_config():
    """The pipeline stores a fixed market list; the registry must cover it.

    Six of the fifteen were undeclared, so 282 of 705 candidates were thrown
    out as UNKNOWN_MARKET on 2026-10-10 -- silently, because an unknown market
    is just another rejection reason.
    """
    from nway.config import load_config
    from nway.prediction.pipeline import STORED_MARKETS

    declared = {market.key for market in load_config().markets}
    missing = sorted(set(STORED_MARKETS) - declared)
    assert not missing, f"markets predicted but not declared: {missing}"


def test_every_declared_family_has_a_diversity_cap():
    """A family without a cap is unbounded in a batch."""
    from nway.config import load_config

    config = load_config()
    caps = (config.recommendations.get("diversity", {})
            .get("family_share_caps", {}) or {})
    families = {m.family for m in config.markets
                if m.recommendable and m.enabled_competitions}
    uncapped = sorted(families - set(caps))
    assert not uncapped, f"recommendable families with no cap: {uncapped}"


def test_every_market_declares_its_grid():
    """The diversity caps work on the grid, so an unlabelled market would
    escape them."""
    from nway.config import load_config
    from nway.markets.registry import MARKET_GRID

    for market in load_config().markets:
        if market.key in ("EXPECTED_GOALS", "GOAL_MARGIN", "ANYTIME_SCORER"):
            continue
        assert market.grid == MARKET_GRID[market.key], (
            f"{market.key} claims grid {market.grid} but is derived from "
            f"{MARKET_GRID[market.key]}")


def test_corner_and_card_markets_are_gated_off_for_the_champions_league():
    """football-data.org supplies full-time and half-time goals only, so there
    are no corner or card statistics for this competition at any tier."""
    from nway.config import load_config

    config = load_config()
    for market in config.markets:
        if market.grid in ("CORNER_MATRIX", "CARD_DISTRIBUTION"):
            assert not market.enabled_for("champions_league"), (
                f"{market.key} is enabled for a competition with no statistics")
    grids = config.grids_enabled_for("champions_league")
    assert set(grids) == {"GOAL_MATRIX", "HALF_MATRIX"}


def test_goal_and_half_markets_are_enabled_for_every_competition():
    from nway.config import load_config

    config = load_config()
    for slug in ("premier_league", "champions_league", "eredivisie"):
        grids = config.grids_enabled_for(slug)
        assert "GOAL_MATRIX" in grids and "HALF_MATRIX" in grids


def test_market_floors_still_match_the_documented_rule():
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


def test_base_rates_carry_their_sample_size():
    """A base rate without an n cannot be judged."""
    from nway.config import load_config

    for market in load_config().markets:
        if market.base_rate is None:
            continue
        assert market.base_rate_n and market.base_rate_n > 1000, (
            f"{market.key} base rate rests on {market.base_rate_n} matches")
