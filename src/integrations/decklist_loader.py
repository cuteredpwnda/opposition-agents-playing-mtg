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
                # Blank line ends a Commander section (commanders are 1-2 cards
                # separated by a blank line from the mainboard).
                if is_commander_section:
                    is_commander_section = False
                continue
            lower = line.lower()
            if lower.startswith("sideboard"):
                current = deck.sideboard
                is_commander_section = False
                continue
            if lower.startswith("mainboard") or lower.startswith("deck"):
                current = deck.mainboard
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

    # ------------------------------------------------------------------
    # Moxfield
    # ------------------------------------------------------------------

    _MOXFIELD_URL_RE = re.compile(
        r"moxfield\.com/decks/(?P<id>[A-Za-z0-9_-]+)"
    )

    async def from_moxfield(self, deck_id_or_url: str) -> Decklist:
        """Load a public deck from Moxfield.

        Accepts either the public deck ID (e.g. ``"abc123"``) or a full URL
        (``https://www.moxfield.com/decks/abc123``). Uses the v3 API which
        returns the full board layout at
        ``https://api2.moxfield.com/v3/decks/all/<publicId>``.
        """
        import httpx

        m = self._MOXFIELD_URL_RE.search(deck_id_or_url)
        deck_id = m.group("id") if m else deck_id_or_url

        url = f"https://api2.moxfield.com/v3/decks/all/{deck_id}"
        headers = {
            "Accept": "application/json",
            # Moxfield's CDN refuses the default httpx UA; any browser-like UA works.
            "User-Agent": "OppositionAgentsMTG/1.0 (+research)",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        deck = Decklist()
        boards = data.get("boards", {}) or {}

        def _add(board_key: str, target: dict[str, int] | list[str]):
            board = boards.get(board_key, {}) or {}
            cards = board.get("cards", {}) or {}
            for entry in cards.values():
                card = entry.get("card", {}) or {}
                name = card.get("name") or entry.get("name") or ""
                qty = int(entry.get("quantity", 1) or 1)
                if not name:
                    continue
                if isinstance(target, list):
                    for _ in range(qty):
                        target.append(name)
                else:
                    target[name] = target.get(name, 0) + qty

        _add("mainboard", deck.mainboard)
        _add("sideboard", deck.sideboard)
        _add("commanders", deck.commander)
        # Some Moxfield decks also use "companions" / "signatureSpells" — fold in.
        _add("companions", deck.commander)
        return deck

    # ------------------------------------------------------------------
    # EDHREC (read-only: average decks, recommendations)
    # ------------------------------------------------------------------

    async def from_edhrec_average(self, commander_slug: str) -> Decklist:
        """Load EDHREC's "average deck" for a commander.

        ``commander_slug`` is the EDHREC URL slug — e.g. ``"krenko-mob-boss"``
        for ``https://edhrec.com/commanders/krenko-mob-boss``. Uses the public
        JSON endpoint ``https://json.edhrec.com/v2/decks/<slug>.json`` (not
        always available — falls back to the commander page's recommended-card
        list).
        """
        import httpx

        slug = commander_slug.strip().lower().replace(" ", "-")
        urls = [
            f"https://json.edhrec.com/v2/decks/{slug}.json",
            f"https://json.edhrec.com/pages/decks/{slug}.json",
        ]
        headers = {
            "Accept": "application/json",
            "User-Agent": "OppositionAgentsMTG/1.0 (+research)",
        }
        data: dict | None = None
        async with httpx.AsyncClient(timeout=30.0) as client:
            for url in urls:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    break
        if data is None:
            raise RuntimeError(f"EDHREC has no average-deck JSON for '{slug}'")

        deck = Decklist()
        # EDHREC payload shape: data["deck"] is a list of card names (strings)
        # for the average deck; commander is in data["container"]["json_dict"]
        # ["card_lists"][0]["cardviews"]... For the simple endpoint, look at
        # data["deck"] and data["commanders"].
        for name in data.get("commanders", []) or []:
            if isinstance(name, str):
                deck.commander.append(name)
        for entry in data.get("deck", []) or []:
            if isinstance(entry, str):
                deck.mainboard[entry] = deck.mainboard.get(entry, 0) + 1
            elif isinstance(entry, dict):
                n = entry.get("name") or entry.get("sanitized")
                q = int(entry.get("count", 1) or 1)
                if n:
                    deck.mainboard[n] = deck.mainboard.get(n, 0) + q

        if not (deck.commander or deck.mainboard):
            raise RuntimeError(
                f"EDHREC payload for '{slug}' had no parseable card list"
            )
        return deck

    # ------------------------------------------------------------------
    # Auto-dispatch from a URL
    # ------------------------------------------------------------------

    async def from_url(self, url: str) -> Decklist:
        """Auto-detect Moxfield / Archidekt / EDHREC URL and dispatch."""
        u = url.strip()
        if "moxfield.com" in u:
            return await self.from_moxfield(u)
        if "archidekt.com" in u:
            m = re.search(r"archidekt\.com/decks/(\d+)", u)
            if not m:
                raise ValueError(f"can't extract Archidekt deck id from: {url}")
            return await self.from_archidekt(int(m.group(1)))
        if "edhrec.com" in u:
            m = re.search(r"edhrec\.com/commanders/([A-Za-z0-9_-]+)", u)
            if not m:
                raise ValueError(f"can't extract EDHREC slug from: {url}")
            return await self.from_edhrec_average(m.group(1))
        raise ValueError(f"unsupported decklist URL: {url}")
