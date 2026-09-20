"""The workflow rewrites its own cron. A mistake there strands the scheduler."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "rewrite_cron", ROOT / ".github" / "scripts" / "rewrite_cron.py")
rewrite_cron = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rewrite_cron)

TEMPLATE = """on:
  schedule:
    # >>> nway:dynamic-schedule
    - cron: "17 6 * * *"
    # <<< nway:dynamic-schedule
  workflow_dispatch:
"""


@pytest.fixture
def workflow(tmp_path):
    path = tmp_path / "predict.yml"
    path.write_text(TEMPLATE)
    return path


def test_replaces_only_the_fenced_block(workflow):
    rewrite_cron.main([str(workflow), "*/15 16-18 9 10 *"])
    text = workflow.read_text()
    assert '- cron: "*/15 16-18 9 10 *"' in text
    assert "workflow_dispatch:" in text, "content outside the markers was lost"
    assert text.count("nway:dynamic-schedule") == 2


def test_safety_net_is_always_present(workflow):
    rewrite_cron.main([str(workflow), "*/15 16-18 9 10 *"])
    assert rewrite_cron.SAFETY_NET in workflow.read_text()


def test_malformed_crons_are_rejected_and_the_net_survives(workflow):
    """Values reach this script from a computed output; a malformed one must
    not be able to empty the schedule."""
    rewrite_cron.main([str(workflow), "not a cron", "; rm -rf /", "$(whoami)"])
    text = workflow.read_text()
    assert "rm -rf" not in text
    assert "whoami" not in text
    assert rewrite_cron.SAFETY_NET in text


def test_rewriting_is_idempotent(workflow):
    rewrite_cron.main([str(workflow), "*/15 16-18 9 10 *"])
    first = workflow.read_text()
    rewrite_cron.main([str(workflow), "*/15 16-18 9 10 *"])
    assert workflow.read_text() == first


def test_indentation_is_preserved(workflow):
    rewrite_cron.main([str(workflow), "*/15 16-18 9 10 *"])
    for line in workflow.read_text().splitlines():
        if "- cron:" in line:
            assert line.startswith("    - cron:")


def test_missing_markers_fail_loudly(tmp_path):
    path = tmp_path / "no-markers.yml"
    path.write_text("on:\n  schedule:\n    - cron: \"0 * * * *\"\n")
    assert rewrite_cron.main([str(path), "*/15 * * * *"]) == 1


def test_the_real_workflow_still_has_its_markers():
    """Guards against a refactor silently removing the fence."""
    text = (ROOT / ".github" / "workflows" / "predict.yml").read_text()
    assert rewrite_cron.BEGIN in text and rewrite_cron.END in text
