"""SMTP delivery — the provider-independent fallback."""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage as MimeMessage

from nway.logging_setup import get_logger, register_secret
from nway.notifications.delivery.base import DeliveryResult, EmailMessage

log = get_logger(__name__)


class SmtpProvider:
    name = "smtp"

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.host = os.environ.get("NWAY_SMTP_HOST", "")
        self.port = int(os.environ.get("NWAY_SMTP_PORT", "587"))
        self.username = os.environ.get("NWAY_SMTP_USERNAME", "")
        self.password = os.environ.get("NWAY_SMTP_PASSWORD", "")
        register_secret(self.password)

    def send(self, message: EmailMessage) -> DeliveryResult:
        if not self.host:
            return DeliveryResult(False, self.name, error="NWAY_SMTP_HOST is not set")

        mime = MimeMessage()
        mime["Subject"] = message.subject
        mime["From"] = message.sender
        mime["To"] = message.to
        if message.reply_to:
            mime["Reply-To"] = message.reply_to
        # Plain text first: a client that cannot render HTML shows the part it
        # can, and the text part is generated from data, not stripped tags.
        mime.set_content(message.text)
        mime.add_alternative(message.html, subtype="html")

        try:
            with smtplib.SMTP(self.host, self.port, timeout=30) as server:
                server.starttls(context=ssl.create_default_context())
                if self.username:
                    server.login(self.username, self.password)
                server.send_message(mime)
            return DeliveryResult(True, self.name, message_id=mime.get("Message-ID"))
        except Exception as exc:  # noqa: BLE001
            log.error("smtp send failed", context={"error": str(exc)})
            return DeliveryResult(False, self.name, error=str(exc))
