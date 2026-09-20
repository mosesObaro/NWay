"""File delivery for tests and inspection. Writes both parts to disk."""

from __future__ import annotations

from pathlib import Path

from nway import clock
from nway.config import PROJECT_ROOT
from nway.notifications.delivery.base import DeliveryResult, EmailMessage

OUTBOX = PROJECT_ROOT / "data" / "outbox"


class FileProvider:
    name = "file"

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.directory = Path(self.config.get("outbox") or OUTBOX)

    def send(self, message: EmailMessage) -> DeliveryResult:
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = clock.now().strftime("%Y%m%dT%H%M%SZ")
        (self.directory / f"{stamp}.html").write_text(message.html)
        (self.directory / f"{stamp}.txt").write_text(
            f"To: {message.to}\nSubject: {message.subject}\n\n{message.text}")
        return DeliveryResult(True, self.name, message_id=stamp)
