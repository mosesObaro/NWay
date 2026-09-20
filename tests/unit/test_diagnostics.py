"""The setup check must not report success on an unconfigured machine."""

from __future__ import annotations

import os

import pytest

from nway.diagnostics import FAIL, WARN, _is_set, check_email_addresses, render


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("NWAY_FOOTBALL_DATA_ORG_TOKEN", "NWAY_RESEND_API_KEY",
                 "NWAY_EMAIL_FROM", "NWAY_EMAIL_TO"):
        monkeypatch.delenv(name, raising=False)


def test_todo_placeholders_count_as_unset(monkeypatch):
    """A freshly copied .env is not a configured one."""
    monkeypatch.setenv("NWAY_RESEND_API_KEY", "TODO_paste_your_resend_key")
    assert _is_set("NWAY_RESEND_API_KEY") is None
    monkeypatch.setenv("NWAY_RESEND_API_KEY", "re_realkey123456")
    assert _is_set("NWAY_RESEND_API_KEY") == "re_realkey123456"


def test_blank_and_whitespace_count_as_unset(monkeypatch):
    monkeypatch.setenv("NWAY_EMAIL_TO", "   ")
    assert _is_set("NWAY_EMAIL_TO") is None


def test_missing_email_addresses_fail(clean_env, config):
    check = check_email_addresses(config)
    assert check.status == FAIL
    assert "NWAY_EMAIL_FROM" in check.detail


def test_configured_email_addresses_pass(clean_env, config, monkeypatch):
    monkeypatch.setenv("NWAY_EMAIL_FROM", "NWay <onboarding@resend.dev>")
    monkeypatch.setenv("NWAY_EMAIL_TO", "someone@example.com")
    assert check_email_addresses(config).status not in (FAIL,)


def test_render_never_prints_a_secret(monkeypatch, config):
    """The report says whether a key works, never what it is."""
    from nway.diagnostics import Check

    secret = "re_supersecretkey0123456789"
    monkeypatch.setenv("NWAY_RESEND_API_KEY", secret)
    output = render([Check("Resend API key", WARN, "HTTP 500", None)])
    assert secret not in output
