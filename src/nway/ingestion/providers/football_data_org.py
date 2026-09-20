"""football-data.org — live fixtures, kickoff times and results.

The free tier (TIER_ONE) was verified on 2026-09-20 to cover all seven target
domestic leagues plus the Champions League. Europa League is TIER_TWO and
Conference League TIER_FOUR, both paid.

Naming trap, and the reason competition codes live in configuration: this
provider uses ``CL`` for the CHAMPIONS League and ``UCL`` for the CONFERENCE
League. Assuming UCL means Champions ingests the wrong competition silently.

Timestamps here are genuine UTC (``utcDate``) and are the system's scheduling
source of truth.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Any

from nway import clock
from nway.logging_setup import get_logger, register_secret

log = get_logger(__name__)

BASE_URL = "https://api.football-data.org/v4"
SOURCE = "football_data_org"

# Provider status -> our status vocabulary.
STATUS_MAP = {
    "SCHEDULED": "SCHEDULED", "TIMED": "TIMED", "IN_PLAY": "IN_PLAY",
    "PAUSED": "IN_PLAY", "FINISHED": "FINISHED", "POSTPONED": "POSTPONED",
    "SUSPENDED": "SUSPENDED", "CANCELLED": "CANCELLED", "AWARDED": "AWARDED",
}


@dataclass
class OrgFixture:
    provider_id: str
    competition_code: str
    season_label: str
    season_start: str
    season_end: str
    home_name: str
    away_name: str
    home_provider_id: str | None
    away_provider_id: str | None
    kickoff_utc: dt.datetime
    status: str
    matchday: int | None
    stage: str | None
    home_goals: int | None
    away_goals: int | None
    ht_home_goals: int | None
    ht_away_goals: int | None
    last_updated: dt.datetime | None

    @property
    def is_finished(self) -> bool:
        return self.status == "FINISHED"

    @property
    def kickoff_is_confirmed(self) -> bool:
        # A provisional kickoff shows as SCHEDULED; TIMED means the clock is set.
        return self.status in ("TIMED", "IN_PLAY", "FINISHED")


def auth_headers() -> dict[str, str]:
    token = os.environ.get("NWAY_FOOTBALL_DATA_ORG_TOKEN", "").strip()
    if token:
        register_secret(token)
        return {"X-Auth-Token": token}
    return {}


def _season_label(season: dict[str, Any]) -> str:
    start = (season or {}).get("startDate", "")
    end = (season or {}).get("endDate", "")
    if not start:
        return "unknown"
    start_year = int(start[:4])
    end_year = int(end[:4]) if end else start_year
    if end_year > start_year:
        return f"{start_year}/{str(end_year)[-2:]}"
    return str(start_year)


def _parse_match(payload: dict[str, Any], competition_code: str) -> OrgFixture | None:
    try:
        score = payload.get("score", {}) or {}
        full_time = score.get("fullTime", {}) or {}
        half_time = score.get("halfTime", {}) or {}
        season = payload.get("season", {}) or {}
        last_updated = payload.get("lastUpdated")
        return OrgFixture(
            provider_id=str(payload["id"]),
            competition_code=competition_code,
            season_label=_season_label(season),
            season_start=season.get("startDate", ""),
            season_end=season.get("endDate", ""),
            home_name=(payload.get("homeTeam") or {}).get("name") or "",
            away_name=(payload.get("awayTeam") or {}).get("name") or "",
            home_provider_id=str((payload.get("homeTeam") or {}).get("id") or "") or None,
            away_provider_id=str((payload.get("awayTeam") or {}).get("id") or "") or None,
            kickoff_utc=clock.from_iso(payload["utcDate"]),
            status=STATUS_MAP.get(payload.get("status", ""), "SCHEDULED"),
            matchday=payload.get("matchday"),
            stage=payload.get("stage"),
            home_goals=full_time.get("home"),
            away_goals=full_time.get("away"),
            ht_home_goals=half_time.get("home"),
            ht_away_goals=half_time.get("away"),
            last_updated=clock.from_iso(last_updated) if last_updated else None,
        )
    except (KeyError, ValueError, TypeError) as exc:
        log.warning("unparseable match payload", context={"error": str(exc)})
        return None


def fetch_competition_matches(client, competition_code: str,
                              date_from: dt.date | None = None,
                              date_to: dt.date | None = None) -> list[OrgFixture]:
    """Fetch matches for one competition, optionally bounded by date."""
    params: dict[str, Any] = {}
    if date_from:
        params["dateFrom"] = date_from.isoformat()
    if date_to:
        params["dateTo"] = date_to.isoformat()
    url = f"{BASE_URL}/competitions/{competition_code}/matches"
    response = client.get(url, headers=auth_headers(), params=params or None)
    if response.not_modified or not response.body:
        return []
    payload = response.json()
    fixtures = []
    for match in payload.get("matches", []) or []:
        parsed = _parse_match(match, competition_code)
        if parsed and parsed.home_name and parsed.away_name:
            fixtures.append(parsed)
    log.info("fetched competition matches",
             context={"competition": competition_code, "count": len(fixtures),
                      "quota_left": client.daily_quota_remaining})
    return fixtures


def fetch_competitions(client) -> list[dict[str, Any]]:
    """The competition list is readable without a token; useful for diagnostics."""
    response = client.get(f"{BASE_URL}/competitions", headers=auth_headers())
    return response.json().get("competitions", []) if response.body else []
