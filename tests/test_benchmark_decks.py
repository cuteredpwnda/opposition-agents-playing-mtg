import hashlib
import json
from collections import Counter

import pytest

from scripts.fetch_benchmark_decks import audit_deck, parse_forge, render_deck
from scripts.run_paper_pilot import REPO_ROOT, build_matchups, parse_args
from src.integrations.phase_rs import load_deck_data


def test_forge_sections_printings_and_aliases():
    deck = parse_forge(
        "[metadata]\nName=Ignored\n[Main]\n58 Island|M11\n2 Opt\n"
        "[Sideboard]\n1 Ghirapur AEther Grid\n"
    )
    assert deck["main"] == Counter({"Island": 58, "Opt": 2})
    assert deck["sideboard"] == Counter({"Ghirapur Aether Grid": 1})
    rendered = render_deck(deck, "https://example.org/source")
    assert "Mainboard\n58 Island\n2 Opt\nSideboard\n1 Ghirapur Aether Grid\n" in rendered
    assert "Ignored" not in rendered


@pytest.mark.parametrize("text", [
    "[Main]\n59 Island", "[Main]\n60 Island\n[Sideboard]\n16 Opt",
    "[Main]\nno-count Island", "[Main]\n0 Island",
])
def test_invalid_source_is_rejected(text):
    with pytest.raises(ValueError):
        parse_forge(text)


def test_legality_uses_non_token_identity_and_includes_sideboard():
    deck = {"main": Counter({"Llanowar Elves": 60}), "sideboard": Counter({"Opt": 1})}
    cards = [
        {"name": "Llanowar Elves", "layout": "normal", "legalities": {"modern": "legal"}},
        {"name": "Llanowar Elves", "layout": "token", "legalities": {"modern": "not_legal"}},
        {"name": "Opt", "layout": "normal", "legalities": {"modern": "not_legal"}},
    ]
    assert [failure["name"] for failure in audit_deck(deck, cards)] == ["Opt"]


def test_source_above_minimum_is_preserved_not_trimmed():
    assert sum(parse_forge("[Main]\n61 Island")["main"].values()) == 61


def test_committed_source_manifest_counts_hashes_and_pool():
    directory = REPO_ROOT / "data" / "decks" / "benchmark" / "sourced"
    manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
    assert manifest["source_license"] == "GPL-3.0"
    assert len(manifest["decks"]) == 4
    for record in manifest["decks"]:
        path = directory / record["file"]
        assert hashlib.sha256(path.read_text("utf-8").encode()).hexdigest() == record["deck_sha256"]
        deck = load_deck_data(path)
        assert len(deck["main_deck"]) == record["mainboard_count"] >= 60
        assert len(deck["sideboard"]) == record["sideboard_count"] <= 15
        assert record["modern_oracle_eligible"] is True
        assert record["native_qualification"] == "pending"
    assert len(build_matchups(parse_args(["--deck-pool", "modern-expanded"]))) == 7
    assert len(build_matchups(parse_args([
        "--deck-pool", "modern-expanded", "--matchups", "round-robin",
    ]))) == 49
