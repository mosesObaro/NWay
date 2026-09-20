"""Email delivery abstraction.

The planner decides *what* to send and *when*; providers decide *how*. Keeping
them apart means switching from Resend to SMTP is a configuration change, and
the whole notification path can be tested without a network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class EmailMessage:
    subject: str
    html: str
    text: str
    to: str
    sender: str
    reply_to: str | None = None
    idempotency_key: str | None = None


@dataclass
class DeliveryResult:
    success: bool
    provider: str
    message_id: str | None = None
    error: str | None = None
    attempt: int = 1
    status_code: int | None = None


class EmailProvider(Protocol):
    name: str

    def send(self, message: EmailMessage) -> DeliveryResult: ...


def build_provider(config, force: str | None = None) -> EmailProvider:
    """Construct the configured provider.

    ``force`` exists for --dry-run, which must never touch a real provider.
    """
    from nway.notifications.delivery.console import ConsoleProvider
    from nway.notifications.delivery.file import FileProvider
    from nway.notifications.delivery.resend import ResendProvider
    from nway.notifications.delivery.smtp import SmtpProvider

    email_config = (config.notifications or {}).get("email", {}) or {}
    name = (force or email_config.get("provider") or "console").lower()
    providers = {
        "resend": ResendProvider, "smtp": SmtpProvider,
        "console": ConsoleProvider, "file": FileProvider,
    }
    if name not in providers:
        raise ValueError(f"unknown email provider: {name}")
    return providers[name](email_config)
