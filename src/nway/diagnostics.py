"""Environment check.

Answers one question: is this machine configured well enough to run live?

Each check reports what it found, and when something is missing it says where
to get it rather than only that it is absent. Secrets are never printed -- only
whether they are present and whether they work.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from nway.config import PROJECT_ROOT, Config

OK, WARN, FAIL = "ok", "warn", "fail"
MARK = {OK: "\033[32m✓\033[0m", WARN: "\033[33m!\033[0m", FAIL: "\033[31m✗\033[0m"}


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str | None = None


def _is_set(name: str) -> str | None:
    """Return the value only if it is present and not still a TODO placeholder."""
    value = (os.environ.get(name) or "").strip()
    if not value or value.upper().startswith("TODO"):
        return None
    return value


def check_env_file() -> Check:
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return Check(".env file", FAIL, "not found",
                     "cp .env.example .env   then fill in the values")
    remaining = [
        line.split("=", 1)[0].strip()
        for line in path.read_text().splitlines()
        if "=" in line and not line.strip().startswith("#")
        and line.split("=", 1)[1].strip().upper().startswith("TODO")
    ]
    if remaining:
        return Check(".env file", WARN, f"{len(remaining)} value(s) still TODO: "
                     + ", ".join(remaining), f"edit {path}")
    return Check(".env file", OK, str(path))


def check_football_data_token() -> Check:
    token = _is_set("NWAY_FOOTBALL_DATA_ORG_TOKEN")
    if not token:
        return Check("football-data.org token", FAIL, "not set",
                     "free token: https://www.football-data.org/client/register")
    request = urllib.request.Request(
        "https://api.football-data.org/v4/competitions/PL/matches?dateFrom="
        "2026-09-20&dateTo=2026-09-21",
        headers={"X-Auth-Token": token, "User-Agent": "NWay/0.1 (setup check)"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode())
            remaining = response.headers.get("X-Requests-Available", "?")
        return Check("football-data.org token", OK,
                     f"valid — {payload.get('count', 0)} matches in the probe "
                     f"window, {remaining} requests left this period")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return Check("football-data.org token", FAIL,
                         f"rejected (HTTP {exc.code})",
                         "check the token was copied whole, with no spaces")
        return Check("football-data.org token", WARN, f"HTTP {exc.code}",
                     "the token may be fine; the API may be rate-limiting")
    except Exception as exc:  # noqa: BLE001
        return Check("football-data.org token", WARN, f"could not reach API: {exc}",
                     "check your network connection")


def check_resend() -> Check:
    key = _is_set("NWAY_RESEND_API_KEY")
    if not key:
        return Check("Resend API key", FAIL, "not set",
                     "create one at https://resend.com/api-keys")
    # Listing domains verifies the key without sending anything.
    request = urllib.request.Request(
        "https://api.resend.com/domains",
        headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode() or "{}")
        domains = payload.get("data") or []
        verified = [d.get("name") for d in domains if d.get("status") == "verified"]
        if verified:
            return Check("Resend API key", OK,
                         f"valid — verified domain(s): {', '.join(verified)}")
        return Check(
            "Resend API key", OK,
            "valid — no verified domain yet, so Resend will only accept "
            "onboarding@resend.dev as sender and only deliver to your own "
            "Resend account address")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return Check("Resend API key", FAIL, f"rejected (HTTP {exc.code})",
                         "check the key starts with re_ and was copied whole")
        return Check("Resend API key", WARN, f"HTTP {exc.code}", None)
    except Exception as exc:  # noqa: BLE001
        return Check("Resend API key", WARN, f"could not reach Resend: {exc}", None)


def check_email_addresses(config: Config) -> Check:
    sender = _is_set("NWAY_EMAIL_FROM")
    recipient = _is_set("NWAY_EMAIL_TO")
    if not sender or not recipient:
        missing = [n for n, v in (("NWAY_EMAIL_FROM", sender),
                                  ("NWAY_EMAIL_TO", recipient)) if not v]
        return Check("email addresses", FAIL, f"missing: {', '.join(missing)}",
                     "set them in .env")
    provider = ((config.notifications or {}).get("email", {}) or {}).get("provider")
    return Check("email addresses", OK,
                 f"{sender} -> {recipient}  (provider: {provider})")


def check_database(config: Config) -> Check:
    from nway.storage.db import Database, _database_path

    path = _database_path()
    if not Path(path).exists():
        return Check("database", FAIL, f"{path} does not exist",
                     "nway init-db   then   nway ingest --history")
    db = Database()
    db.migrate()
    fixtures = db.scalar("SELECT COUNT(*) FROM fixture", default=0)
    stats = db.scalar("SELECT COUNT(*) FROM team_match_stats", default=0)
    db.close()
    if fixtures < 1000:
        return Check("database", WARN, f"only {fixtures} fixtures loaded",
                     "nway ingest --history --seasons 2017/18..2025/26")
    return Check("database", OK, f"{fixtures:,} fixtures, {stats:,} team-match rows")


def check_model() -> Check:
    from nway.storage.db import Database

    db = Database()
    db.migrate()
    row = db.query_one(
        "SELECT model_version, metrics FROM model_version WHERE is_active = 1 "
        "ORDER BY trained_at DESC LIMIT 1")
    db.close()
    if not row:
        return Check("trained model", FAIL, "no active model",
                     "nway train --through 2026-06-30")
    metrics = json.loads(row["metrics"] or "{}")
    achieved = metrics.get("log_loss_1x2")
    baseline = metrics.get("baseline_log_loss_1x2")
    if achieved and baseline and achieved >= baseline:
        return Check("trained model", WARN,
                     f"{row['model_version']} does not beat the base rate "
                     f"({achieved:.4f} vs {baseline:.4f})",
                     "retrain with more history")
    detail = row["model_version"]
    if achieved:
        detail += f" — log loss {achieved:.4f}"
        if baseline:
            detail += f" (base rate {baseline:.4f})"
    return Check("trained model", OK, detail)


def check_competitions(config: Config) -> Check:
    enabled = config.enabled_competitions()
    blocked = [c.slug for c in config.competitions if not c.enabled]
    detail = f"{len(enabled)} enabled: " + ", ".join(c.slug for c in enabled)
    if blocked:
        detail += f"  |  blocked (paid tiers): {', '.join(blocked)}"
    return Check("competitions", OK, detail)


def run_checks(config: Config) -> list[Check]:
    return [
        check_env_file(),
        check_football_data_token(),
        check_resend(),
        check_email_addresses(config),
        check_database(config),
        check_model(),
        check_competitions(config),
    ]


def render(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = ["", "NWay environment check", "=" * 60]
    for check in checks:
        lines.append(f"  {MARK[check.status]} {check.name:<{width}}  {check.detail}")
        if check.fix and check.status != OK:
            lines.append(f"      {'':<{width}}  -> {check.fix}")
    failures = [c for c in checks if c.status == FAIL]
    warnings = [c for c in checks if c.status == WARN]
    lines.append("")
    if failures:
        lines.append(f"{len(failures)} blocking issue(s). "
                     f"Fix those, then run `nway check` again.")
    elif warnings:
        lines.append("Ready to run, with caveats above. Try: nway tick --dry-run")
    else:
        lines.append("All good. Try: nway tick --dry-run")
    lines.append("")
    return "\n".join(lines)
