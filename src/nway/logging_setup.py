"""Structured logging with secret redaction."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone

_SECRET_VALUES: list[str] = []
_SECRET_KEYS = re.compile(r"(token|api_key|apikey|password|secret|authorization)", re.I)


def register_secret(value: str | None) -> None:
    """Values registered here are masked wherever they appear in a log line."""
    if value and len(value) >= 8:
        _SECRET_VALUES.append(value)


def _scrub(text: str) -> str:
    for secret in _SECRET_VALUES:
        text = text.replace(secret, "***")
    return text


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": _scrub(record.getMessage()),
        }
        for key, value in getattr(record, "context", {}).items():
            payload[key] = "***" if _SECRET_KEYS.search(key) else value
        if record.exc_info:
            payload["exception"] = _scrub(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    COLOURS = {"DEBUG": "\033[90m", "INFO": "\033[36m",
               "WARNING": "\033[33m", "ERROR": "\033[31m", "CRITICAL": "\033[35m"}

    def format(self, record: logging.LogRecord) -> str:
        colour = self.COLOURS.get(record.levelname, "")
        reset = "\033[0m" if colour else ""
        stamp = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
        context = getattr(record, "context", {})
        extra = ""
        if context:
            shown = {k: ("***" if _SECRET_KEYS.search(k) else v) for k, v in context.items()}
            extra = "  " + " ".join(f"{k}={v}" for k, v in shown.items())
        return (f"{stamp} {colour}{record.levelname:<7}{reset} "
                f"{_scrub(record.getMessage())}{_scrub(extra)}")


def setup_logging(level: str | None = None, json_output: bool = False) -> None:
    resolved = (level or os.environ.get("NWAY_LOG_LEVEL") or "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_output else HumanFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)


def get_logger(name: str) -> logging.LoggerAdapter:
    class _Adapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            context = kwargs.pop("context", None) or {}
            extra = kwargs.setdefault("extra", {})
            extra["context"] = context
            return msg, kwargs

    return _Adapter(logging.getLogger(name), {})
