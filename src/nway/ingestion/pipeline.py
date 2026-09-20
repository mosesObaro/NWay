"""Ingestion orchestration: provider payloads -> canonical rows.

Two distinct jobs:

  * ``ingest_history`` loads deep historical statistics from
    football-data.co.uk. This is the training corpus.
  * ``ingest_live`` refreshes fixtures, kickoff times and results from
    football-data.org. This is what the scheduler runs.

They meet in the same ``fixture`` table. Where both cover a match, the live
provider's UTC kickoff wins -- the CSV's time column is local and approximate,
and letting it overwrite a true kickoff would corrupt the scheduling key.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from nway import clock
from nway.config import Config
from nway.entities.resolution import EntityResolver
from nway.ingestion import http as http_mod
from nway.ingestion.providers import football_data_couk as couk
from nway.ingestion.providers import football_data_org as fdo
from nway.logging_setup import get_logger
from nway.storage import repositories as repo
from nway.storage.db import Database

log = get_logger(__name__)


@dataclass
class IngestSummary:
    fixtures_created: int = 0
    fixtures_updated: int = 0
    results_written: int = 0
    stats_written: int = 0
    unresolved_names: int = 0
    errors: int = 0

    def merge(self, other: "IngestSummary") -> "IngestSummary":
        return IngestSummary(
            self.fixtures_created + other.fixtures_created,
            self.fixtures_updated + other.fixtures_updated,
            self.results_written + other.results_written,
            self.stats_written + other.stats_written,
            self.unresolved_names + other.unresolved_names,
            self.errors + other.errors,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "fixtures_created": self.fixtures_created,
            "fixtures_updated": self.fixtures_updated,
            "results": self.results_written,
            "stats": self.stats_written,
            "unresolved": self.unresolved_names,
            "errors": self.errors,
        }


def _sxg_proxy(shots: int | None, on_target: int | None,
               weight_on_target: float = 0.30, weight_off: float = 0.03) -> float | None:
    """Shot-based expected-goals PROXY. Not xG, and never stored as xG.

    No free, terms-compliant xG source exists (FBref's terms prohibit building
    tools on scraped data; Understat's robots.txt disallows all crawlers), so
    this stands in. It captures shot volume and accuracy but knows nothing
    about shot quality -- it cannot tell a tap-in from a thirty-yard effort.
    The default weights are a documented starting point; Phase 5 refits them
    per competition by Poisson regression of goals on the two shot counts.
    """
    if shots is None or on_target is None:
        return None
    off_target = max(0, shots - on_target)
    return round(weight_on_target * on_target + weight_off * off_target, 4)


def ingest_history(db: Database, config: Config, seasons: list[str],
                   competitions: list[str] | None = None) -> IngestSummary:
    """Load historical statistics for the given seasons."""
    client = http_mod.build_client(config, couk.SOURCE, db=db)
    resolver = EntityResolver(db)
    lag_hours = float(
        config.sources.get(couk.SOURCE, {}).get("assumed_publication_lag_hours", 72))
    summary = IngestSummary()

    targets = [
        c for c in config.competitions
        if c.providers.get(couk.SOURCE)
        and (competitions is None or c.slug in competitions)
    ]

    for competition in targets:
        division = competition.providers[couk.SOURCE]
        competition_id = repo.upsert_competition(
            db, competition.slug, competition.name, competition.kind,
            competition.structure, competition.country, competition.enabled)

        for season_label in seasons:
            try:
                matches = couk.fetch_season(client, season_label, division)
            except Exception as exc:  # noqa: BLE001 — one bad season must not stop the run
                log.error("season fetch failed",
                          context={"division": division, "season": season_label,
                                   "error": str(exc)})
                summary.errors += 1
                continue
            if not matches:
                continue

            dates = [m.date for m in matches]
            season_id = repo.upsert_season(
                db, competition_id, season_label,
                min(dates).isoformat(), max(dates).isoformat(), is_current=False)

            with db.transaction():
                for match in matches:
                    result = _write_historical_match(
                        db, resolver, match, competition_id, season_id, lag_hours)
                    if result is None:
                        summary.unresolved_names += 1
                        continue
                    created, wrote_stats = result
                    summary.fixtures_created += int(created)
                    summary.fixtures_updated += int(not created)
                    summary.results_written += 1
                    summary.stats_written += 2 if wrote_stats else 0

            log.info("ingested season",
                     context={"competition": competition.slug, "season": season_label,
                              "matches": len(matches)})
    return summary


def _write_historical_match(db: Database, resolver: EntityResolver, match: couk.CoukMatch,
                            competition_id: int, season_id: int,
                            lag_hours: float) -> tuple[bool, bool] | None:
    home_id = resolver.resolve_team(couk.SOURCE, match.home_name)
    away_id = resolver.resolve_team(couk.SOURCE, match.away_name)
    if home_id is None or away_id is None:
        return None

    kickoff = match.kickoff_utc()
    knowledge = match.knowledge_time(lag_hours)
    referee_id = repo.get_or_create_referee(db, match.referee) if match.referee else None

    existing = db.query_one(
        "SELECT fixture_id, kickoff_utc, status FROM fixture WHERE competition_id=? "
        "AND season_id=? AND home_team_id=? AND away_team_id=?",
        (competition_id, season_id, home_id, away_id))
    # Never let the CSV's local, approximate time overwrite a real UTC kickoff.
    kickoff_value = existing["kickoff_utc"] if existing else clock.to_iso(kickoff)

    fixture_id, changes = repo.upsert_fixture(
        db, competition_id=competition_id, season_id=season_id,
        home_team_id=home_id, away_team_id=away_id,
        kickoff_utc=kickoff_value, status="FINISHED",
        knowledge_time=clock.to_iso(knowledge), referee_id=referee_id,
        source=couk.SOURCE)
    created = "created" in changes

    db.execute(
        "INSERT OR REPLACE INTO match_result (fixture_id, home_goals, away_goals, "
        "ht_home_goals, ht_away_goals, outcome, went_to_et, went_to_pens, "
        "result_source, event_time, knowledge_time) VALUES (?,?,?,?,?,?,0,0,?,?,?)",
        (fixture_id, match.home_goals, match.away_goals, match.ht_home_goals,
         match.ht_away_goals, match.outcome, couk.SOURCE,
         clock.to_iso(kickoff + dt.timedelta(hours=2)),
         clock.to_iso(kickoff + dt.timedelta(hours=2))))

    has_stats = match.home_shots is not None or match.home_corners is not None
    if has_stats:
        for is_home, team_id in ((1, home_id), (0, away_id)):
            goals = match.home_goals if is_home else match.away_goals
            conceded = match.away_goals if is_home else match.home_goals
            ht = match.ht_home_goals if is_home else match.ht_away_goals
            shots = match.home_shots if is_home else match.away_shots
            sot = match.home_shots_on_target if is_home else match.away_shots_on_target
            db.execute(
                "INSERT OR REPLACE INTO team_match_stats (fixture_id, team_id, is_home, "
                "goals, goals_conceded, ht_goals, shots, shots_on_target, corners, fouls, "
                "yellow_cards, red_cards, possession, xg, sxg_proxy, source, event_time, "
                "knowledge_time) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,?,?,?,?)",
                (fixture_id, team_id, is_home, goals, conceded, ht, shots, sot,
                 match.home_corners if is_home else match.away_corners,
                 match.home_fouls if is_home else match.away_fouls,
                 match.home_yellow if is_home else match.away_yellow,
                 match.home_red if is_home else match.away_red,
                 _sxg_proxy(shots, sot), couk.SOURCE,
                 clock.to_iso(kickoff + dt.timedelta(hours=2)),
                 clock.to_iso(match.knowledge_time(lag_hours))))
    return created, has_stats


def ingest_live(db: Database, config: Config,
                discovery_days: int | None = None) -> IngestSummary:
    """Refresh fixtures and results for every enabled competition."""
    source_config = config.sources.get(fdo.SOURCE, {})
    if not source_config.get("enabled", True):
        return IngestSummary()

    client = http_mod.build_client(config, fdo.SOURCE, db=db)
    resolver = EntityResolver(db)
    summary = IngestSummary()
    horizon = int(discovery_days or source_config.get("discovery_horizon_days", 30))

    today = clock.now().date()
    # Look back far enough to pick up results and postponements we may have
    # missed while the machine was asleep.
    date_from = today - dt.timedelta(days=14)
    date_to = today + dt.timedelta(days=horizon)

    for competition in config.enabled_competitions():
        code = competition.providers.get(fdo.SOURCE)
        if not code:
            continue
        try:
            fixtures = fdo.fetch_competition_matches(client, code, date_from, date_to)
        except Exception as exc:  # noqa: BLE001
            log.error("live fetch failed",
                      context={"competition": competition.slug, "error": str(exc)})
            repo.record_quality_check(
                db, "API_FAILURE", "WARN",
                f"{competition.slug}: {exc}", "COMPETITION", None)
            summary.errors += 1
            continue

        competition_id = repo.upsert_competition(
            db, competition.slug, competition.name, competition.kind,
            competition.structure, competition.country, competition.enabled)

        with db.transaction():
            for fixture in fixtures:
                outcome = _write_live_fixture(
                    db, resolver, fixture, competition_id, competition.country)
                if outcome is None:
                    summary.unresolved_names += 1
                    continue
                created, wrote_result = outcome
                summary.fixtures_created += int(created)
                summary.fixtures_updated += int(not created)
                summary.results_written += int(wrote_result)

    if resolver.pending_count():
        repo.record_quality_check(
            db, "UNRESOLVED_ENTITIES", "WARN",
            f"{resolver.pending_count()} provider names await review")
    return summary


def _write_live_fixture(db: Database, resolver: EntityResolver, fixture: fdo.OrgFixture,
                        competition_id: int, country: str | None) -> tuple[bool, bool] | None:
    home_id = resolver.resolve_team(
        fdo.SOURCE, fixture.home_name, fixture.home_provider_id, country)
    away_id = resolver.resolve_team(
        fdo.SOURCE, fixture.away_name, fixture.away_provider_id, country)
    if home_id is None or away_id is None:
        return None

    season_id = repo.upsert_season(
        db, competition_id, fixture.season_label,
        fixture.season_start or fixture.kickoff_utc.date().isoformat(),
        fixture.season_end or fixture.kickoff_utc.date().isoformat(),
        is_current=True)

    # knowledge_time is when WE learned it: the fetch time, but never earlier
    # than the provider's own lastUpdated.
    knowledge = clock.now()
    if fixture.last_updated and fixture.last_updated > knowledge:
        knowledge = fixture.last_updated

    fixture_id, changes = repo.upsert_fixture(
        db, competition_id=competition_id, season_id=season_id,
        home_team_id=home_id, away_team_id=away_id,
        kickoff_utc=clock.to_iso(fixture.kickoff_utc), status=fixture.status,
        knowledge_time=clock.to_iso(knowledge), matchday=fixture.matchday,
        stage=fixture.stage, kickoff_is_confirmed=fixture.kickoff_is_confirmed,
        source=fdo.SOURCE)
    created = "created" in changes

    if "kickoff_utc" in changes:
        log.warning("kickoff moved",
                    context={"fixture_id": fixture_id,
                             "match": f"{fixture.home_name} v {fixture.away_name}"})
    if "status" in changes and fixture.status in ("POSTPONED", "CANCELLED", "SUSPENDED"):
        repo.record_quality_check(
            db, "FIXTURE_" + fixture.status, "INFO",
            f"{fixture.home_name} v {fixture.away_name}", "FIXTURE", fixture_id)

    wrote_result = False
    if fixture.is_finished and fixture.home_goals is not None:
        outcome = ("H" if fixture.home_goals > fixture.away_goals
                   else "A" if fixture.away_goals > fixture.home_goals else "D")
        db.execute(
            "INSERT OR REPLACE INTO match_result (fixture_id, home_goals, away_goals, "
            "ht_home_goals, ht_away_goals, outcome, went_to_et, went_to_pens, "
            "result_source, event_time, knowledge_time) VALUES (?,?,?,?,?,?,0,0,?,?,?)",
            (fixture_id, fixture.home_goals, fixture.away_goals,
             fixture.ht_home_goals, fixture.ht_away_goals, outcome, fdo.SOURCE,
             clock.to_iso(fixture.kickoff_utc + dt.timedelta(hours=2)),
             clock.to_iso(knowledge)))
        wrote_result = True
    return created, wrote_result
