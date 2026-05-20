"""Bridge our ``data/decks/*.txt`` decklists into phase-server's ``DeckData``.

phase-server's ``DeckData`` (see ``crates/engine/src/starter_decks.rs``) is
flat — ``main_deck`` is a list of card-name strings, repeated by count.

This module also exposes the names of phase-rs's built-in starter decks so
smoke tests can run without depending on our deck files at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.integrations.decklist_loader import Decklist, DecklistLoader

# Names hard-coded in crates/engine/src/starter_decks.rs. Kept here so a
# missing card database (Decklist file path) isn't a blocker for the demo.
#
# Phase-RS starter decks (built-in):
STARTER_DECK_NAMES: tuple[str, ...] = (
    # Standard (built-in phase-rs starters)
    "Red Deck Wins",
    "White Weenie",
    "Blue Control",
    "Green Stompy",
    "Azorius Flyers",
    # Extended Standard (from MTGGoldfish, playable in phase-rs)
    "Golgari Midrange",
    "Gruul Aggro",
    "Mono Red Aggro",
    "Boros Convoke",
    "Esper Pixie",
    # Pioneer
    "Izzet Murktide",
    "Mono Blue Tempo",
    # Modern
    "Grixis Murktide",
    "Rhinos",
    # Commander (legal 1v1 and multiplayer)
    "Atraxa Praetors",
    "Krenko Goblins",
    "Urza Artifacts",
    "Meren Midrange",
)


def decklist_to_deck_data(deck: Decklist) -> dict[str, Any]:
    """Flatten a ``Decklist`` into phase-server's ``DeckData`` JSON.

    ``main_deck`` becomes a list of card names with each name repeated by its
    count. ``sideboard`` and ``commander`` follow the same shape.
    """
    main_deck: list[str] = []
    for name, count in deck.mainboard.items():
        main_deck.extend([name] * count)
    sideboard: list[str] = []
    for name, count in deck.sideboard.items():
        sideboard.extend([name] * count)
    return {
        "main_deck": main_deck,
        "sideboard": sideboard,
        "commander": list(deck.commander),
    }


def load_deck_data(path: str | Path) -> dict[str, Any]:
    """Read a ``data/decks/*.txt`` decklist and convert it to ``DeckData``."""
    text = Path(path).read_text(encoding="utf-8")
    deck = DecklistLoader().from_text(text)
    return decklist_to_deck_data(deck)


def starter_deck_request(name: str = "Red Deck Wins") -> dict[str, Any]:
    """A ``DeckData`` placeholder whose ``main_deck`` is the starter-deck name.

    phase-server's ``CreateGameWithSettings`` requires a ``DeckData``; for AI
    seats the server can use a built-in starter via the ``aiDeckName`` field.
    The host seat still needs *some* deck, so we send a deck composed of the
    starter cards — but we don't have them client-side without a card-data
    download. For the first smoke test, prefer:

      1. Pass an explicit ``DeckData`` you own (via ``decklist_to_deck_data``).
      2. Or run the server with ``--data-dir`` pointing at the bundled
         ``external/phase-rs/data/`` and let the AI seat use its starter.

    This helper returns an empty stub appropriate only for *protocol-shape*
    tests against fixtures — it is **not** valid for a live game.
    """
    if name not in STARTER_DECK_NAMES:
        raise ValueError(
            f"unknown starter deck {name!r}; valid: {', '.join(STARTER_DECK_NAMES)}"
        )
    return {"main_deck": [], "sideboard": [], "commander": []}
