"""Resend delivery.

POST https://api.resend.com/emails with a Bearer key; the response carries the
message id. Retries use exponential backoff with jitter, and 4xx responses
other than 429 are not retried because they will not succeed on a second
attempt.

Every attempt is written to ``notification_log`` by the caller, and the
duplicate-prevention ledger is written only on confirmed success -- so a
failure leaves the next tick free to retry the same content rather than losing
the batch.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request

from nway.logging_setup import get_logger, register_secret
from nway.notifications.delivery.base import DeliveryResult, EmailMessage

log = get_logger(__name__)

API_URL = "https://api.resend.com/emails"


class ResendProvider:
    name = "resend"

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.api_key = os.environ.get("NWAY_RESEND_API_KEY", "").strip()
        register_secret(self.api_key)
        self.max_attempts = int(self.config.get("max_attempts", 3))
        self.backoff = self.config.get("backoff_seconds") or [30, 120, 600]
        self.timeout = int(self.config.get("timeout_seconds", 30))

    def send(self, message: EmailMessage) -> DeliveryResult:
        if not self.api_key:
            return DeliveryResult(
                False, self.name,
                error="NWAY_RESEND_API_KEY is not set; see .env.example")

        payload = {
            "from": message.sender,
            "to": [message.to],
            "subject": message.subject,
            "html": message.html,
            "text": message.text,
        }
        if message.reply_to:
            payload["reply_to"] = message.reply_to

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Resend honours an idempotency key, which is a second line of defence
        # behind our own ledger: a retry after an ambiguous timeout cannot
        # deliver the same batch twice.
        if message.idempotency_key:
            headers["Idempotency-Key"] = message.idempotency_key

        last_error: str | None = None
        last_status: int | None = None
        for attempt in range(1, self.max_attempts + 1):
            request = urllib.request.Request(
                API_URL, data=json.dumps(payload).encode("utf-8"),
                headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8") or "{}")
                message_id = body.get("id")
                log.info("email sent", context={
                    "provider": self.name, "message_id": message_id,
                    "attempt": attempt})
                return DeliveryResult(True, self.name, message_id=message_id,
                                      attempt=attempt, status_code=response.status)
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8")[:400]
                except Exception:  # noqa: BLE001
                    pass
                last_status, last_error = exc.code, f"HTTP {exc.code}: {detail}"
                # A 4xx other than rate limiting will not succeed on a retry.
                if exc.code < 500 and exc.code != 429:
                    log.error("resend rejected the message", context={
                        "status": exc.code, "detail": detail})
                    return DeliveryResult(False, self.name, error=last_error,
                                          attempt=attempt, status_code=exc.code)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = str(exc)

            if attempt < self.max_attempts:
                delay = self.backoff[min(attempt - 1, len(self.backoff) - 1)]
                delay += random.uniform(0, delay * 0.2)
                log.warning("retrying send", context={
                    "attempt": attempt, "sleep_s": round(delay, 1),
                    "error": last_error})
                time.sleep(delay)

        return DeliveryResult(False, self.name, error=last_error,
                              attempt=self.max_attempts, status_code=last_status)
