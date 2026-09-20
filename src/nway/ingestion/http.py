"""Polite HTTP client.

Token-bucket rate limiting, exponential backoff with jitter, conditional
requests, an identifying User-Agent with a contact address, and persistence of
every raw response before anything parses it.

That last point is the important one: parsing happens against stored bytes, so
a parser bug is a re-run rather than a lost week of rate-limited fetches, and
the backtest replays exactly what was received.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nway import clock
from nway.config import PROJECT_ROOT
from nway.logging_setup import get_logger

log = get_logger(__name__)


class RateLimiter:
    """Token bucket. Blocks rather than failing when the budget is spent."""

    def __init__(self, requests: int, per_seconds: float) -> None:
        self.capacity = max(1, requests)
        self.per_seconds = max(1e-6, per_seconds)
        self.tokens = float(self.capacity)
        self.updated = time.monotonic()

    def acquire(self) -> None:
        while True:
            now = time.monotonic()
            elapsed = now - self.updated
            self.updated = now
            self.tokens = min(
                float(self.capacity),
                self.tokens + elapsed * (self.capacity / self.per_seconds),
            )
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            deficit = (1.0 - self.tokens) * (self.per_seconds / self.capacity)
            time.sleep(min(deficit, 5.0))


@dataclass
class Response:
    url: str
    status: int
    body: bytes
    headers: dict[str, str]
    from_cache: bool = False
    not_modified: bool = False

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))

    def text(self, encoding: str = "utf-8") -> str:
        return self.body.decode(encoding, errors="replace")


class HttpClient:
    """One client per source, configured from config/sources.yaml."""

    def __init__(self, *, source: str, user_agent: str, rate_limit: dict[str, Any],
                 timeout: int = 30, max_retries: int = 4,
                 backoff_base: float = 2.0, cache_dir: Path | None = None,
                 use_conditional: bool = True, db=None) -> None:
        self.source = source
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.use_conditional = use_conditional
        self.db = db
        self.limiter = RateLimiter(
            int(rate_limit.get("requests", 1)),
            float(rate_limit.get("per_seconds", 1)),
        )
        self.cache_dir = Path(cache_dir or (PROJECT_ROOT / "data" / "raw" / source))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.cache_dir / "_conditional.json"
        self._meta = self._load_meta()
        self.daily_quota_remaining: int | None = None

    # -- conditional-request bookkeeping ---------------------------------
    def _load_meta(self) -> dict[str, dict[str, str]]:
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_meta(self) -> None:
        self.meta_path.write_text(json.dumps(self._meta, indent=2, sort_keys=True))

    # -- fetching --------------------------------------------------------
    def get(self, url: str, *, headers: dict[str, str] | None = None,
            params: dict[str, Any] | None = None,
            allow_conditional: bool = True) -> Response:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        request_headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip"}
        request_headers.update(headers or {})
        cached = self._meta.get(url, {})
        if self.use_conditional and allow_conditional:
            if cached.get("etag"):
                request_headers["If-None-Match"] = cached["etag"]
            elif cached.get("last_modified"):
                request_headers["If-Modified-Since"] = cached["last_modified"]

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self.limiter.acquire()
            try:
                response = self._attempt(url, request_headers)
            except urllib.error.HTTPError as exc:
                if exc.code == 304:
                    body = self._read_cached_body(cached.get("content_hash"))
                    return Response(url, 304, body, dict(exc.headers or {}),
                                    from_cache=True, not_modified=True)
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    delay = self._backoff(attempt, exc.headers.get("Retry-After"))
                    log.warning("retrying after HTTP error",
                                context={"source": self.source, "status": exc.code,
                                         "attempt": attempt, "sleep_s": round(delay, 1)})
                    time.sleep(delay)
                    last_error = exc
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt < self.max_retries:
                    delay = self._backoff(attempt, None)
                    log.warning("retrying after network error",
                                context={"source": self.source, "error": str(exc),
                                         "attempt": attempt, "sleep_s": round(delay, 1)})
                    time.sleep(delay)
                    last_error = exc
                    continue
                raise

            self._remember(url, response)
            self._persist(url, response)
            return response

        raise RuntimeError(f"exhausted retries for {url}") from last_error

    def _attempt(self, url: str, headers: dict[str, str]) -> Response:
        request = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=self.timeout) as raw:
            body = raw.read()
            response_headers = {k.lower(): v for k, v in raw.headers.items()}
            if response_headers.get("content-encoding") == "gzip":
                body = gzip.decompress(body)
            available = response_headers.get("x-requests-available")
            if available is not None:
                try:
                    self.daily_quota_remaining = int(available)
                except ValueError:
                    pass
            return Response(url, raw.status, body, response_headers)

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 120.0)
            except ValueError:
                pass
        return min(self.backoff_base ** attempt + random.uniform(0, 1.0), 120.0)

    def _remember(self, url: str, response: Response) -> None:
        digest = hashlib.sha256(response.body).hexdigest()
        self._meta[url] = {
            "etag": response.headers.get("etag", ""),
            "last_modified": response.headers.get("last-modified", ""),
            "content_hash": digest,
            "fetched_at": clock.to_iso(clock.now()),
        }
        self._save_meta()

    def _body_path(self, content_hash: str) -> Path:
        return self.cache_dir / f"{content_hash[:16]}.bin.gz"

    def _read_cached_body(self, content_hash: str | None) -> bytes:
        if not content_hash:
            return b""
        path = self._body_path(content_hash)
        return gzip.decompress(path.read_bytes()) if path.exists() else b""

    def _persist(self, url: str, response: Response) -> None:
        """Store the raw body and register it, so parsing is always replayable."""
        digest = hashlib.sha256(response.body).hexdigest()
        path = self._body_path(digest)
        if not path.exists():
            path.write_bytes(gzip.compress(response.body))
        if self.db is None:
            return
        parsed = urllib.parse.urlparse(url)
        self.db.insert("raw_response", {
            "source": self.source,
            "endpoint": parsed.path,
            "request_params": parsed.query or None,
            "http_status": response.status,
            "content_hash": digest,
            "body_path": str(path.relative_to(PROJECT_ROOT)),
            "fetched_at": clock.to_iso(clock.now()),
        }, or_ignore=True)


def build_client(config, source_name: str, db=None) -> HttpClient:
    source = config.sources.get(source_name, {})
    http = config.http or {}
    return HttpClient(
        source=source_name,
        user_agent=http.get("user_agent") or "NWay-FootballPredictor/0.1",
        rate_limit=source.get("rate_limit", {"requests": 1, "per_seconds": 2}),
        timeout=int(http.get("timeout_seconds", 30)),
        max_retries=int(http.get("max_retries", 4)),
        backoff_base=float(http.get("backoff_base_seconds", 2)),
        use_conditional=bool(http.get("use_conditional_requests", True)),
        db=db,
    )
