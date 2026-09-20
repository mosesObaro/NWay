"""Competitions outside the model's scope must not get predictions.

UEFA Nations League and the Women's Champions League are registered but
disabled. If either is ever enabled, the failure mode to avoid is not an error
-- it is a confident-looking probability produced from nothing, because team
ratings are per-team and no national team or women's club has one.
"""

from __future__ import annotations

import datetime as dt

import pytest

from nway import clock
from nway.models.base import LeakageError, ModelArtifact, ScopeError


def artifact(scope: str = "mens_club") -> ModelArtifact:
    return ModelArtifact(
        model_version="test", model_family="DIXON_COLES", target="GOALS",
        trained_at=dt.datetime(2026, 6, 1, tzinfo=clock.UTC),
        train_window_start=dt.datetime(2017, 8, 1, tzinfo=clock.UTC),
        train_window_end=dt.datetime(2026, 5, 31, tzinfo=clock.UTC),
        hyperparameters={}, parameters={}, model_scope=scope)


def test_model_serves_its_own_scope():
    artifact().assert_scope("mens_club")


@pytest.mark.parametrize("scope", ["international", "womens_club"])
def test_model_refuses_a_scope_it_was_not_fitted_on(scope):
    with pytest.raises(ScopeError) as excinfo:
        artifact().assert_scope(scope)
    assert "train a separate model" in str(excinfo.value)


def test_scope_and_time_guards_are_independent():
    """Both must hold; passing one says nothing about the other."""
    art = artifact()
    art.assert_usable_at(dt.datetime(2026, 7, 1, tzinfo=clock.UTC))
    with pytest.raises(ScopeError):
        art.assert_scope("international")
    with pytest.raises(LeakageError):
        art.assert_usable_at(dt.datetime(2026, 3, 1, tzinfo=clock.UTC))


# ------------------------------------------------------------ registry
def test_both_requested_competitions_are_registered(config):
    slugs = {c.slug for c in config.competitions}
    assert "nations_league" in slugs
    assert "womens_champions_league" in slugs


@pytest.mark.parametrize("slug", ["nations_league", "womens_champions_league"])
def test_blocked_competitions_are_disabled_with_a_stated_reason(config, slug):
    competition = config.competition(slug)
    assert competition.enabled is False
    assert competition.blocked_reason, "a disabled competition must say why"
    assert len(competition.blocked_reason) > 80, \
        "the reason should be specific enough to act on"


@pytest.mark.parametrize("slug,scope", [
    ("nations_league", "international"),
    ("womens_champions_league", "womens_club"),
    ("premier_league", "mens_club"),
    ("champions_league", "mens_club"),
])
def test_model_scope_is_declared_per_competition(config, slug, scope):
    assert config.competition(slug).model_scope == scope


def test_disabled_competitions_are_not_ingested_or_predicted(config):
    enabled = {c.slug for c in config.enabled_competitions()}
    assert "nations_league" not in enabled
    assert "womens_champions_league" not in enabled


def test_registry_reads_every_group_not_a_fixed_pair(config):
    """Adding a category of competition must be a configuration change."""
    groups = {c.group for c in config.competitions}
    assert {"domestic", "uefa", "uefa_women", "international"} <= groups
