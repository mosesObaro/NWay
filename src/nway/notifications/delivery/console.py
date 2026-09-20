"""Console delivery for development. Prints the text part; sends nothing."""

from __future__ import annotations

from nway.notifications.delivery.base import DeliveryResult, EmailMessage


class ConsoleProvider:
    name = "console"

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    def send(self, message: EmailMessage) -> DeliveryResult:
        print("=" * 72)
        print(f"To:      {message.to}")
        print(f"From:    {message.sender}")
        print(f"Subject: {message.subject}")
        print("=" * 72)
        print(message.text)
        return DeliveryResult(True, self.name, message_id="console")
