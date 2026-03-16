"""
Decklist loader — parse decklists from text, Moxfield, Archidekt.

Reference: Section 11.2 of PLAN.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Decklist:
    """A parsed decklist."""

    mainboard: dict[str, int] = field(default_factory=dict)  # card_name → count
    sideboard: dict[str, int] = field(default_factory=dict)
    commander: list[str] = field(default_factory=list)

    @property
    def all_cards(self) -> dict[str, int]:
        result = dict(self.mainboard)
        for name, count in self.sideboard.items():
            result[name] = result.get(name, 0) + count
        for name in self.commander:
            result[name] = result.get(name, 0) + 1
        return result


class DecklistLoader:
    """Load decklists from text or API sources."""

    # Pattern: "1 Sol Ring" or "1x Sol Ring" or "1 Sol Ring (CMR) 123"
    _LINE_RE = re.compile(r"^(\d+)x?\s+(.+?)(?:\s+\([A-Z0-9]+\)\s*\d+)?$")

    def from_text(self, decklist_text: str) -> Decklist:
        """Parse a plaintext decklist.

        Format: "1 Sol Ring\\n1 Command Tower\\n..."
        Sections separated by blank lines or "Sideboard" header.
        """
        deck = Decklist()
        current = deck.mainboard
        is_commander_section = False

        for line in decklist_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith("sideboard"):
                current = deck.sideboard
                is_commander_section = False
                continue
            if lower.startswith("commander"):
                is_commander_section = True
                continue

            match = self._LINE_RE.match(line)
            if match:
                count = int(match.group(1))
                name = match.group(2).strip()
                if is_commander_section:
                    deck.commander.append(name)
                else:
                    current[name] = current.get(name, 0) + count

        return deck

    async def from_archidekt(self, deck_id: int) -> Decklist:
        """Load from Archidekt API. GET https://archidekt.com/api/decks/{id}/"""
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://archidekt.com/api/decks/{deck_id}/",
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        deck = Decklist()
        for entry in data.get("cards", []):
            card = entry.get("card", {})
            name = card.get("oracleCard", {}).get("name", "")
            qty = entry.get("quantity", 1)
            categories = entry.get("categories", [])
            if "Commander" in categories:
                deck.commander.append(name)
            elif "Sideboard" in categories:
                deck.sideboard[name] = qty
            else:
                deck.mainboard[name] = qty
        return deck
