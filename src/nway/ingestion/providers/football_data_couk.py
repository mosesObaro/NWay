"""football-data.co.uk — historical match statistics.

Free CSV, one file per division per season. Verified 2026-09-20: all seven
target leagues carry 100%-complete shots, shots on target, corners, fouls,
cards and half-time scores.

Four traps, each of which silently corrupts an unguarded ingestion:
  * requests to the ``www`` host return 302 to the apex domain, so a client
    that does not follow redirects writes empty files and reports success;
  * the files are latin-1, not UTF-8;
  * the first header cell carries a UTF-8 BOM, parsing as "\\ufeffDiv";
  * dates are dd/mm/yy in older files and dd/mm/yyyy in newer ones, and the
    kickoff times are LOCAL, so they must never be used for scheduling.

The site updates roughly twice weekly, so rows become *knowable* well after the
match is played. ``assumed_publication_lag_hours`` models that; using kickoff
time as knowledge_time would leak statistics into predictions made before they
were available.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
from dataclasses import dataclass
from typing import Iterator

from nway import clock
from nway.logging_setup import get_logger

log = get_logger(__name__)

BASE_URL = "https://football-data.co.uk/mmz4281"
SOURCE = "football_data_couk"


@dataclass
class CoukMatch:
    division: str
    date: dt.date
    local_time: str | None
    home_name: str
    away_name: str
    home_goals: int
    away_goals: int
    outcome: str
    ht_home_goals: int | None
    ht_away_goals: int | None
    referee: str | None
    home_shots: int | None
    away_shots: int | None
    home_shots_on_target: int | None
    away_shots_on_target: int | None
    home_corners: int | None
    away_corners: int | None
    home_fouls: int | None
    away_fouls: int | None
    home_yellow: int | None
    away_yellow: int | None
    home_red: int | None
    away_red: int | None

    def kickoff_utc(self) -> dt.datetime:
        """Approximate kickoff.

        The file's time column is local, so this is only ever used as a
        fallback for historical rows that never appear in the live feed. The
        live provider supplies true UTC for anything the scheduler touches.
        """
        hour, minute = 15, 0
        if self.local_time and ":" in self.local_time:
            try:
                hh, mm = self.local_time.split(":")
                hour, minute = int(hh), int(mm)
            except ValueError:
                pass
        return dt.datetime(self.date.year, self.date.month, self.date.day,
                           hour, minute, tzinfo=clock.UTC)

    def knowledge_time(self, lag_hours: float) -> dt.datetime:
        """When this row became knowable to us, not when it happened."""
        return self.kickoff_utc() + dt.timedelta(hours=lag_hours)


def _int(value: str | None) -> int | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def season_code(label: str) -> str:
    """'2025/26' -> '2526'."""
    start, _, end = label.partition("/")
    return f"{start[-2:]}{end[-2:]}"


def parse_date(value: str) -> dt.date | None:
    text = (value or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_csv(body: bytes, division: str) -> Iterator[CoukMatch]:
    """Parse one season file. Rows that cannot be parsed are skipped, loudly."""
    text = body.decode("latin-1")
    if text.startswith("﻿"):
        text = text[1:]
    reader = csv.DictReader(io.StringIO(text))
    skipped = 0
    for row in reader:
        # Trailing blank lines and summary rows are common in these files.
        if not (row.get("HomeTeam") or "").strip():
            continue
        home_goals, away_goals = _int(row.get("FTHG")), _int(row.get("FTAG"))
        if home_goals is None or away_goals is None:
            skipped += 1
            continue
        date = parse_date(row.get("Date", ""))
        if date is None:
            skipped += 1
            continue
        outcome = (row.get("FTR") or "").strip().upper()
        if outcome not in ("H", "D", "A"):
            outcome = "H" if home_goals > away_goals else "A" if away_goals > home_goals else "D"
        yield CoukMatch(
            division=division, date=date,
            local_time=(row.get("Time") or "").strip() or None,
            home_name=row["HomeTeam"].strip(), away_name=row["AwayTeam"].strip(),
            home_goals=home_goals, away_goals=away_goals, outcome=outcome,
            ht_home_goals=_int(row.get("HTHG")), ht_away_goals=_int(row.get("HTAG")),
            referee=(row.get("Referee") or "").strip() or None,
            home_shots=_int(row.get("HS")), away_shots=_int(row.get("AS")),
            home_shots_on_target=_int(row.get("HST")),
            away_shots_on_target=_int(row.get("AST")),
            home_corners=_int(row.get("HC")), away_corners=_int(row.get("AC")),
            home_fouls=_int(row.get("HF")), away_fouls=_int(row.get("AF")),
            home_yellow=_int(row.get("HY")), away_yellow=_int(row.get("AY")),
            home_red=_int(row.get("HR")), away_red=_int(row.get("AR")),
        )
    if skipped:
        log.warning("skipped unparseable rows",
                    context={"division": division, "skipped": skipped})


def season_url(season_label: str, division: str) -> str:
    return f"{BASE_URL}/{season_code(season_label)}/{division}.csv"


def fetch_season(client, season_label: str, division: str) -> list[CoukMatch]:
    url = season_url(season_label, division)
    response = client.get(url)
    if not response.body:
        log.warning("empty response", context={"url": url})
        return []
    matches = list(parse_csv(response.body, division))
    log.info("parsed season file",
             context={"division": division, "season": season_label,
                      "matches": len(matches),
                      "cached": response.not_modified})
    return matches
