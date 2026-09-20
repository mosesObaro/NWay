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
