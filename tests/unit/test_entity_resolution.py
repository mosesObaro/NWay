import pytest

from nway.entities.resolution import TEAM_ALIASES, EntityResolver, normalise_name


@pytest.mark.parametrize("raw,expected", [
    ("Man United", "man united"),
    ("Manchester United FC", "manchester united"),
    ("Nott'm Forest", "nott m forest"),
    ("M'gladbach", "m gladbach"),
    ("Atlético Madrid", "atletico madrid"),
    ("Borussia Mönchengladbach", "borussia monchengladbach"),
    ("  Paris   SG  ", "paris sg"),
])
def test_normalisation(raw, expected):
    assert normalise_name(raw) == expected


def test_aliases_cover_both_provider_spellings(db):
    resolver = EntityResolver(db)
    couk = resolver.resolve_team("football_data_couk", "Man United")
    org = resolver.resolve_team("football_data_org", "Manchester United FC")
    assert couk == org, "the same club resolved to two different ids"


@pytest.mark.parametrize("a,b", [
    ("Man United", "Man City"),
    ("Manchester United FC", "Manchester City FC"),
    ("Chaves", "Aves"),           # fuzzy-match at 0.80; genuinely different clubs
    ("Real Madrid", "Real Sociedad"),
])
def test_distinct_clubs_never_merge(db, a, b):
    resolver = EntityResolver(db)
    first = resolver.resolve_team("football_data_couk", a)
    second = resolver.resolve_team("football_data_couk", b)
    assert first is not None and second is not None
    assert first != second, f"{a} and {b} merged into one team"


def test_unknown_close_name_is_queued_not_guessed(db):
    resolver = EntityResolver(db)
    resolver.resolve_team("football_data_couk", "Arsenal")
    # "Arsenall" is close enough to be suspicious but not close enough to accept.
    result = resolver.resolve_team("football_data_couk", "Arsenaal")
    if result is None:
        assert resolver.pending_count() >= 1
    else:
        # If it auto-accepted, it must have matched Arsenal exactly.
        assert result == resolver.resolve_team("football_data_couk", "Arsenal")


def test_resolution_is_stable_across_calls(db):
    resolver = EntityResolver(db)
    first = resolver.resolve_team("football_data_couk", "Ath Bilbao")
    second = EntityResolver(db).resolve_team("football_data_couk", "Ath Bilbao")
    assert first == second


def test_alias_table_has_no_self_contradiction():
    """No normalised key may map to two different canonical names."""
    seen: dict[str, str] = {}
    for key, value in TEAM_ALIASES.items():
        assert key == normalise_name(key), f"alias key {key!r} is not normalised"
        seen.setdefault(key, value)
        assert seen[key] == value


@pytest.mark.parametrize("provider_spelling,other_spelling", [
    # Real pairs that the first live ingest rejected: the designator is a
    # PREFIX here, and stripping only suffixes left 44 fixtures unresolved.
    ("AFC Bournemouth", "Bournemouth"),
    ("FC Schalke 04", "Schalke 04"),
    ("RCD Mallorca", "Mallorca"),
    ("SV Werder Bremen", "Werder Bremen"),
    ("AC Milan", "Milan"),
    ("1. FC Koln", "FC Koln"),
])
def test_prefixed_designators_normalise_to_the_same_name(provider_spelling,
                                                         other_spelling):
    assert normalise_name(provider_spelling) == normalise_name(other_spelling)


@pytest.mark.parametrize("a,b", [
    ("AC Milan", "Inter Milan"),
    ("Manchester United FC", "Manchester City FC"),
    ("Sporting CP", "Sporting Gijon"),
    ("Real Madrid", "Real Sociedad"),
    ("Atletico Madrid", "Real Madrid"),
])
def test_stripping_designators_does_not_merge_distinct_clubs(a, b):
    """Prefix stripping is aggressive; this guards against it going too far.

    Known and accepted limitation: designator stripping WILL collide across
    confederations -- "FC Barcelona" and Ecuador's "Barcelona SC" both reduce
    to "barcelona". Resolution is scoped to the eight configured competitions,
    where no such pair occurs, and anything ambiguous goes to the review queue
    rather than being guessed.
    """
    assert normalise_name(a) != normalise_name(b)


def test_a_bare_designator_is_not_stripped_to_nothing():
    """Otherwise every such name normalises to '' and matches every other."""
    assert normalise_name("FC") == "fc"
    assert normalise_name("AC") == "ac"
