"""Resend delivery.

Resend sits behind Cloudflare, which rejects the default urllib signature with
HTTP 403 and a body of "error code: 1010". That is a bot block, not an
authentication failure, and it reads exactly like a bad API key -- so it gets
its own handling and its own tests.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from nway.notifications.delivery.base import EmailMessage
from nway.notifications.delivery.resend import USER_AGENT, ResendProvider


def message() -> EmailMessage:
    return EmailMessage(subject="s", html="<p>h</p>", text="t",
                        to="to@example.com", sender="from@example.com",
                        idempotency_key="abc123")


class _Response:
    status = 200

    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def test_every_request_identifies_itself(monkeypatch):
    """Without this header Cloudflare blocks the send before Resend sees it."""
    captured = {}

    def fake_urlopen(request, **_kwargs):
        captured["headers"] = dict(request.headers)
        return _Response({"id": "msg_1"})

    monkeypatch.setenv("NWAY_RESEND_API_KEY", "re_testkey123456")
    monkeypatch.setattr(
        "nway.notifications.delivery.resend.urllib.request.urlopen", fake_urlopen)

    result = ResendProvider().send(message())
    assert result.success and result.message_id == "msg_1"
    # urllib title-cases header names.
    assert captured["headers"].get("User-agent") == USER_AGENT


def test_idempotency_key_is_sent(monkeypatch):
    """A retry after an ambiguous timeout must not deliver the batch twice."""
    captured = {}

    def fake_urlopen(request, **_kwargs):
        captured["headers"] = dict(request.headers)
        return _Response({"id": "msg_1"})

    monkeypatch.setenv("NWAY_RESEND_API_KEY", "re_testkey123456")
    monkeypatch.setattr(
        "nway.notifications.delivery.resend.urllib.request.urlopen", fake_urlopen)

    ResendProvider().send(message())
    assert captured["headers"].get("Idempotency-key") == "abc123"


def test_cloudflare_block_is_named_not_mistaken_for_a_bad_key(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "url", 403, "Forbidden", {}, io.BytesIO(b"error code: 1010\n"))

    monkeypatch.setenv("NWAY_RESEND_API_KEY", "re_testkey123456")
    monkeypatch.setattr(
        "nway.notifications.delivery.resend.urllib.request.urlopen", blocked)

    result = ResendProvider().send(message())
    assert result.success is False
    assert "Cloudflare" in result.error
    assert "User-Agent" in result.error


def test_cloudflare_block_is_not_retried(monkeypatch):
    """Retrying an identical blocked request cannot succeed."""
    calls = {"n": 0}

    def blocked(*_args, **_kwargs):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            "url", 403, "Forbidden", {}, io.BytesIO(b"error code: 1010\n"))

    monkeypatch.setenv("NWAY_RESEND_API_KEY", "re_testkey123456")
    monkeypatch.setattr(
        "nway.notifications.delivery.resend.urllib.request.urlopen", blocked)
    monkeypatch.setattr("nway.notifications.delivery.resend.time.sleep",
                        lambda _s: None)

    ResendProvider().send(message())
    assert calls["n"] == 1


def test_missing_key_fails_without_a_network_call(monkeypatch):
    monkeypatch.delenv("NWAY_RESEND_API_KEY", raising=False)
    monkeypatch.setattr(
        "nway.notifications.delivery.resend.urllib.request.urlopen",
        lambda *_a, **_k: pytest.fail("should not call the network"))
    result = ResendProvider().send(message())
    assert result.success is False
    assert "NWAY_RESEND_API_KEY" in result.error


def test_server_errors_are_retried(monkeypatch):
    calls = {"n": 0}

    def flaky(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(
                "url", 500, "Server Error", {}, io.BytesIO(b"{}"))
        return _Response({"id": "msg_ok"})

    monkeypatch.setenv("NWAY_RESEND_API_KEY", "re_testkey123456")
    monkeypatch.setattr(
        "nway.notifications.delivery.resend.urllib.request.urlopen", flaky)
    monkeypatch.setattr("nway.notifications.delivery.resend.time.sleep",
                        lambda _s: None)

    result = ResendProvider().send(message())
    assert result.success and calls["n"] == 3
