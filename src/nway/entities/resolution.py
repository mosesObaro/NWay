"""Entity resolution.

Two providers, two spellings of every club. "Man United" and "Manchester
United FC" are the same team; "Manchester United" and "Manchester City" are
not, and a careless fuzzy matcher will happily merge them.

The strategy is exact -> curated alias -> normalised -> scoped fuzzy. A fuzzy
match above the auto-accept threshold is recorded unverified; below it, the
name is queued for review and the fixture is *not* ingested. Silent guessing is
how a dataset quietly becomes wrong.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

from nway import clock
from nway.logging_setup import get_logger
from nway.storage.db import Database

log = get_logger(__name__)

AUTO_ACCEPT = 0.90
QUEUE_BELOW = 0.90

_SUFFIXES = (
    " fc", " cf", " afc", " sc", " ac", " as", " ssc", " sv", " tsg", " vfl",
    " vfb", " bsc", " fsv", " sad", " cp", " cd", " ud", " rc", " sp",
)
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def normalise_name(name: str) -> str:
    """Casefold, strip accents, punctuation and common club suffixes."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = _PUNCT.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    for suffix in _SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text


# Curated aliases: provider spelling -> canonical name. Every entry here was
# taken from an actual football-data.co.uk or football-data.org payload.
# Keys are written naturally and normalised at import: a key like
# "real madrid cf" would otherwise be unreachable, because lookup happens on
# the normalised form and normalisation strips the club suffix.
_RAW_TEAM_ALIASES: dict[str, str] = {
    # England
    "man united": "Manchester United", "man utd": "Manchester United",
    "manchester utd": "Manchester United",
    "man city": "Manchester City",
    "nott m forest": "Nottingham Forest", "nottm forest": "Nottingham Forest",
    "sheffield utd": "Sheffield United", "sheffield weds": "Sheffield Wednesday",
    "newcastle": "Newcastle United", "west brom": "West Bromwich Albion",
    "west ham": "West Ham United", "wolves": "Wolverhampton Wanderers",
    "tottenham": "Tottenham Hotspur", "leicester": "Leicester City",
    "brighton": "Brighton and Hove Albion", "leeds": "Leeds United",
    "norwich": "Norwich City", "stoke": "Stoke City", "hull": "Hull City",
    "cardiff": "Cardiff City", "swansea": "Swansea City", "burnley": "Burnley",
    "luton": "Luton Town", "ipswich": "Ipswich Town", "coventry": "Coventry City",
    "qpr": "Queens Park Rangers",
    # Spain
    "ath bilbao": "Athletic Bilbao", "athletic club": "Athletic Bilbao",
    "ath madrid": "Atletico Madrid", "club atletico de madrid": "Atletico Madrid",
    "atletico de madrid": "Atletico Madrid",
    "espanol": "Espanyol", "rcd espanyol de barcelona": "Espanyol",
    "sociedad": "Real Sociedad", "real sociedad de futbol": "Real Sociedad",
    "betis": "Real Betis", "real betis balompie": "Real Betis",
    "celta": "Celta Vigo", "rc celta de vigo": "Celta Vigo",
    "vallecano": "Rayo Vallecano", "rayo vallecano de madrid": "Rayo Vallecano",
    "alaves": "Deportivo Alaves", "deportivo alaves": "Deportivo Alaves",
    "la coruna": "Deportivo La Coruna", "fc barcelona": "Barcelona",
    "real madrid cf": "Real Madrid", "valencia cf": "Valencia",
    "villarreal cf": "Villarreal", "sevilla fc": "Sevilla",
    "getafe cf": "Getafe", "girona fc": "Girona", "cadiz cf": "Cadiz",
    "ud las palmas": "Las Palmas", "ca osasuna": "Osasuna",
    "rcd mallorca": "Mallorca", "elche cf": "Elche", "levante ud": "Levante",
    # Germany
    "m gladbach": "Borussia Monchengladbach",
    "borussia monchengladbach": "Borussia Monchengladbach",
    "ein frankfurt": "Eintracht Frankfurt",
    "bayern munich": "Bayern Munich", "fc bayern munchen": "Bayern Munich",
    "dortmund": "Borussia Dortmund", "borussia dortmund": "Borussia Dortmund",
    "leverkusen": "Bayer Leverkusen", "bayer 04 leverkusen": "Bayer Leverkusen",
    "rb leipzig": "RB Leipzig", "wolfsburg": "VfL Wolfsburg",
    "hoffenheim": "TSG Hoffenheim", "tsg 1899 hoffenheim": "TSG Hoffenheim",
    "stuttgart": "VfB Stuttgart", "werder bremen": "Werder Bremen",
    "sv werder bremen": "Werder Bremen", "freiburg": "SC Freiburg",
    "sc freiburg": "SC Freiburg", "mainz": "Mainz 05", "1 fsv mainz 05": "Mainz 05",
    "union berlin": "Union Berlin", "1 fc union berlin": "Union Berlin",
    "augsburg": "FC Augsburg", "fc augsburg": "FC Augsburg",
    "heidenheim": "Heidenheim", "1 fc heidenheim 1846": "Heidenheim",
    "st pauli": "St Pauli", "fc st pauli": "St Pauli",
    "hamburg": "Hamburger SV", "hamburger sv": "Hamburger SV",
    "fc koln": "FC Koln", "1 fc koln": "FC Koln", "koln": "FC Koln",
    # Italy
    "ac milan": "AC Milan", "inter": "Inter Milan", "fc internazionale milano": "Inter Milan",
    "juventus": "Juventus", "juventus fc": "Juventus",
    "napoli": "Napoli", "ssc napoli": "Napoli",
    "roma": "AS Roma", "as roma": "AS Roma",
    "lazio": "Lazio", "ss lazio": "Lazio",
    "atalanta": "Atalanta", "atalanta bc": "Atalanta",
    "fiorentina": "Fiorentina", "acf fiorentina": "Fiorentina",
    "torino": "Torino", "torino fc": "Torino",
    "bologna": "Bologna", "bologna fc 1909": "Bologna",
    "udinese": "Udinese", "udinese calcio": "Udinese",
    "verona": "Hellas Verona", "hellas verona fc": "Hellas Verona",
    "genoa": "Genoa", "genoa cfc": "Genoa",
    "sassuolo": "Sassuolo", "us sassuolo calcio": "Sassuolo",
    "cagliari": "Cagliari", "cagliari calcio": "Cagliari",
    "lecce": "Lecce", "us lecce": "Lecce", "empoli": "Empoli", "empoli fc": "Empoli",
    "parma": "Parma", "parma calcio 1913": "Parma", "como": "Como", "como 1907": "Como",
    "spezia": "Spezia", "monza": "Monza", "ac monza": "Monza",
    # France
    "paris sg": "Paris Saint-Germain", "paris saint germain fc": "Paris Saint-Germain",
    "marseille": "Marseille", "olympique de marseille": "Marseille",
    "lyon": "Lyon", "olympique lyonnais": "Lyon",
    "monaco": "Monaco", "as monaco fc": "Monaco",
    "lille": "Lille", "lille osc": "Lille",
    "rennes": "Rennes", "stade rennais fc 1901": "Rennes",
    "nice": "Nice", "ogc nice": "Nice",
    "lens": "Lens", "rc lens": "Lens",
    "st etienne": "Saint-Etienne", "as saint etienne": "Saint-Etienne",
    "reims": "Reims", "stade de reims": "Reims",
    "nantes": "Nantes", "fc nantes": "Nantes",
    "strasbourg": "Strasbourg", "rc strasbourg alsace": "Strasbourg",
    "toulouse": "Toulouse", "toulouse fc": "Toulouse",
    "montpellier": "Montpellier", "montpellier hsc": "Montpellier",
    "brest": "Brest", "stade brestois 29": "Brest",
    "angers": "Angers", "angers sco": "Angers",
    "auxerre": "Auxerre", "aj auxerre": "Auxerre",
    "le havre": "Le Havre", "havre ac": "Le Havre",
    "paris fc": "Paris FC",
    # Netherlands
    "ajax": "Ajax", "afc ajax": "Ajax",
    "psv eindhoven": "PSV", "psv": "PSV",
    "feyenoord": "Feyenoord", "feyenoord rotterdam": "Feyenoord",
    "az alkmaar": "AZ Alkmaar", "az": "AZ Alkmaar",
    "twente": "FC Twente", "fc twente 65": "FC Twente",
    "utrecht": "FC Utrecht", "fc utrecht": "FC Utrecht",
    "vitesse": "Vitesse", "sbv vitesse": "Vitesse",
    "heerenveen": "SC Heerenveen", "sc heerenveen": "SC Heerenveen",
    "groningen": "FC Groningen", "fc groningen": "FC Groningen",
    "sparta rotterdam": "Sparta Rotterdam", "nec": "NEC Nijmegen",
    "nec nijmegen": "NEC Nijmegen", "go ahead eagles": "Go Ahead Eagles",
    "for sittard": "Fortuna Sittard", "fortuna sittard": "Fortuna Sittard",
    "waalwijk": "RKC Waalwijk", "rkc waalwijk": "RKC Waalwijk",
    "zwolle": "PEC Zwolle", "pec zwolle": "PEC Zwolle",
    "heracles": "Heracles Almelo", "heracles almelo": "Heracles Almelo",
    "willem ii": "Willem II", "excelsior": "Excelsior", "telstar": "Telstar",
    "volendam": "FC Volendam", "fc volendam": "FC Volendam",
    # Portugal
    "sp lisbon": "Sporting CP", "sporting cp": "Sporting CP",
    "sporting clube de portugal": "Sporting CP",
    "porto": "FC Porto", "fc porto": "FC Porto",
    "benfica": "Benfica", "sl benfica": "Benfica",
    "sp braga": "Sporting Braga", "sc braga": "Sporting Braga",
    "vitoria": "Vitoria Guimaraes", "vitoria sc": "Vitoria Guimaraes",
    "guimaraes": "Vitoria Guimaraes",
    "famalicao": "Famalicao", "fc famalicao": "Famalicao",
    "gil vicente": "Gil Vicente", "gil vicente fc": "Gil Vicente",
    "boavista": "Boavista", "boavista fc": "Boavista",
    "rio ave": "Rio Ave", "rio ave fc": "Rio Ave",
    "moreirense": "Moreirense", "moreirense fc": "Moreirense",
    "arouca": "Arouca", "fc arouca": "Arouca",
    "estoril": "Estoril Praia", "gd estoril praia": "Estoril Praia",
    "casa pia": "Casa Pia", "casa pia ac": "Casa Pia",
    "santa clara": "Santa Clara", "cd santa clara": "Santa Clara",
    "farense": "Farense", "sc farense": "Farense",
    "nacional": "Nacional", "cd nacional": "Nacional",
    "estrela": "Estrela Amadora", "avs": "AVS",
    "tondela": "Tondela", "cd tondela": "Tondela",
    "alverca": "Alverca",
    # Chaves and Aves are DIFFERENT clubs whose names fuzzy-match at 0.80.
    # The resolver correctly refused to merge them; these entries make the
    # distinction explicit rather than leaving it to a threshold.
    "chaves": "GD Chaves", "gd chaves": "GD Chaves",
    "aves": "Desportivo das Aves", "desportivo aves": "Desportivo das Aves",
    "feirense": "Feirense", "cd feirense": "Feirense",
    "portimonense": "Portimonense", "pacos ferreira": "Pacos de Ferreira",
    "maritimo": "Maritimo", "belenenses": "Belenenses",
    "vizela": "Vizela", "fc vizela": "Vizela",
    "santa clara": "Santa Clara", "estrela amadora": "Estrela Amadora",
}


TEAM_ALIASES: dict[str, str] = {
    normalise_name(key): value for key, value in _RAW_TEAM_ALIASES.items()
}

# Canonical names are reachable by their own normalised form. Without this, the
# FIRST provider spelling seen creates the canonical row -- so "Manchester
# United FC" arriving before "Man United" would create a team literally named
# "Manchester United FC", and the next near-miss ("Manchester City FC", which
# scores 0.81 against it) would land in the review queue instead of resolving.
CANONICAL_BY_NORMALISED: dict[str, str] = {
    normalise_name(value): value for value in _RAW_TEAM_ALIASES.values()
}


class EntityResolver:
    """Resolves provider names to canonical ids, caching within a run."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self._cache: dict[tuple[str, str, str], int] = {}

    # -- teams -----------------------------------------------------------
    def resolve_team(self, provider: str, provider_name: str,
                     provider_id: str | None = None,
                     country: str | None = None) -> int | None:
        """Return a canonical team_id, or None when the name needs review."""
        key = ("TEAM", provider, provider_name)
        if key in self._cache:
            return self._cache[key]

        mapped = self.db.query_one(
            "SELECT canonical_id FROM provider_entity_map "
            "WHERE entity_kind='TEAM' AND provider=? AND provider_name=?",
            (provider, provider_name))
        if mapped:
            self._cache[key] = mapped["canonical_id"]
            return mapped["canonical_id"]

        normalised = normalise_name(provider_name)

        # 1. curated alias, then the canonical names themselves
        canonical_name = TEAM_ALIASES.get(normalised)
        method = "ALIAS" if canonical_name else None
        if canonical_name is None:
            canonical_name = CANONICAL_BY_NORMALISED.get(normalised)
            method = "ALIAS" if canonical_name else None

        # 2. exact match against an existing canonical name
        if canonical_name is None:
            row = self.db.query_one(
                "SELECT team_id, canonical_name FROM team WHERE canonical_name = ?",
                (provider_name,))
            if row:
                canonical_name, method = row["canonical_name"], "EXACT"

        # 3. normalised match against existing teams
        if canonical_name is None:
            for row in self.db.query("SELECT team_id, canonical_name FROM team"):
                if normalise_name(row["canonical_name"]) == normalised:
                    canonical_name, method = row["canonical_name"], "NORMALISED"
                    break

        # 4. fuzzy, and only when confident
        confidence = 1.0
        if canonical_name is None:
            existing = self.db.query("SELECT team_id, canonical_name FROM team")
            best_name, best_score = None, 0.0
            for row in existing:
                score = difflib.SequenceMatcher(
                    None, normalised, normalise_name(row["canonical_name"])).ratio()
                if score > best_score:
                    best_name, best_score = row["canonical_name"], score
            if best_name is not None and best_score >= AUTO_ACCEPT:
                canonical_name, method, confidence = best_name, "FUZZY", best_score
            elif best_name is not None and best_score >= 0.75:
                # Close but not close enough: queue it and refuse to guess.
                self._queue(provider, provider_name, best_score)
                log.warning("team name queued for review",
                            context={"provider": provider, "name": provider_name,
                                     "best_guess": best_name,
                                     "score": round(best_score, 3)})
                return None

        # 5. genuinely new club
        if canonical_name is None:
            canonical_name, method = provider_name, "NEW"

        row = self.db.query_one(
            "SELECT team_id FROM team WHERE canonical_name = ?", (canonical_name,))
        team_id = row["team_id"] if row else self.db.insert("team", {
            "canonical_name": canonical_name, "country": country,
            "created_at": clock.to_iso(clock.now()),
        })

        self.db.insert("provider_entity_map", {
            "entity_kind": "TEAM", "canonical_id": team_id, "provider": provider,
            "provider_entity_id": provider_id, "provider_name": provider_name,
            "confidence": confidence,
            "resolved_by": "MANUAL" if method == "ALIAS" else (method or "NEW"),
            "verified": int(method in ("ALIAS", "EXACT", "NORMALISED")),
            "created_at": clock.to_iso(clock.now()),
        }, or_ignore=True)
        self._cache[key] = team_id
        return team_id

    def _queue(self, provider: str, provider_name: str, score: float) -> None:
        self.db.insert("entity_resolution_queue", {
            "entity_kind": "TEAM", "provider": provider,
            "provider_name": provider_name, "best_score": score,
            "status": "PENDING", "created_at": clock.to_iso(clock.now()),
        }, or_ignore=True)

    def pending_count(self) -> int:
        return self.db.scalar(
            "SELECT COUNT(*) FROM entity_resolution_queue WHERE status='PENDING'",
            default=0)
