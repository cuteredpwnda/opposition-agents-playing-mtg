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

        Accepts the public deck ID (``"abc123"``) or full URL
        (``https://www.moxfield.com/decks/abc123``).

        .. note::

            Moxfield's API sits behind Cloudflare and rejects plain ``httpx``
            requests with HTTP 403. To make this work, install
            `curl_cffi <https://pypi.org/project/curl-cffi/>`_::

                pip install curl_cffi

            which performs full browser TLS-fingerprint impersonation. If
            ``curl_cffi`` is not installed, this method raises
            :class:`RuntimeError` with a hint to either install it or paste
            the deck's exported text into a local file and use
            :meth:`from_text` instead.
        """
        m = self._MOXFIELD_URL_RE.search(deck_id_or_url)
        deck_id = m.group("id") if m else deck_id_or_url
        url = f"https://api2.moxfield.com/v3/decks/all/{deck_id}"

        try:
            from curl_cffi import requests as cffi_requests  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Moxfield requires the 'curl_cffi' package to bypass "
                "Cloudflare. Install it with `pip install curl_cffi`, or "
                "use Moxfield's Export -> Text feature and load the result "
                "with DecklistLoader.from_text(text)."
            ) from exc

        # curl_cffi is sync, so run in a thread to keep the API async-friendly.
        import asyncio

        def _fetch():
            r = cffi_requests.get(url, impersonate="chrome", timeout=30)
            r.raise_for_status()
            return r.json()

        data = await asyncio.to_thread(_fetch)

        deck = Decklist()
        boards = data.get("boards", {}) or {}

        def _add(board_key: str, target):
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
        _add("companions", deck.commander)
        return deck

    # ------------------------------------------------------------------
    # EDHREC (read-only: average decks, recommendations)
    # ------------------------------------------------------------------

    # Map bracket selectors → EDHREC URL slug components.
    # 1=Exhibition, 2=Core, 3=Upgraded, 4=Optimized, 5=cEDH
    # plus the meta budget/expensive views.
    _EDHREC_BRACKET_SLUGS = {
        None: None,
        1: "exhibition", 2: "core", 3: "upgraded",
        4: "optimized", 5: "cedh",
        "exhibition": "exhibition", "core": "core",
        "upgraded": "upgraded", "optimized": "optimized",
        "cedh": "cedh", "budget": "budget", "expensive": "expensive",
    }

    async def from_edhrec_average(
        self,
        commander_slug: str,
        bracket: int | str | None = None,
    ) -> Decklist:
        """Build an EDHREC-derived "average" 100-card deck for a commander.

        EDHREC publishes per-commander aggregate stats at
        ``/pages/commanders/<slug>.json`` (or the bracket-filtered variants
        ``.../<slug>/{exhibition,core,upgraded,optimized,cedh,budget,expensive}.json``).
        Each payload exposes:

        * ``creature``, ``instant``, ``sorcery``, ``artifact``,
          ``enchantment``, ``planeswalker``, ``nonbasic``, ``basic`` —
          average card counts per category.
        * ``container.json_dict.cardlists`` — ranked card recommendations
          grouped by category (Creatures, Instants, Sorceries, Utility
          Artifacts, Enchantments, Mana Artifacts, Utility Lands, Lands,
          Planeswalkers, …) with inclusion counts.

        We pick the top-N most-included cards in each category to match the
        average composition, then fill with basics matching the commander's
        color identity, returning a 100-card singleton list.

        Args:
            commander_slug: EDHREC URL slug, e.g. ``"krenko-mob-boss"``.
            bracket: Optional WotC Commander Bracket filter — accepts
                ``1..5`` (Exhibition .. cEDH), the names ``"exhibition"``,
                ``"core"``, ``"upgraded"``, ``"optimized"``, ``"cedh"``,
                or the special slugs ``"budget"`` / ``"expensive"``.
                ``None`` (default) uses the unfiltered average.
        """
        import httpx

        if bracket not in self._EDHREC_BRACKET_SLUGS:
            raise ValueError(
                f"unknown bracket {bracket!r}; "
                f"expected one of {sorted(k for k in self._EDHREC_BRACKET_SLUGS if k is not None)}"
            )
        bracket_slug = self._EDHREC_BRACKET_SLUGS[bracket]

        slug = commander_slug.strip().lower().replace(" ", "-")
        url = (
            f"https://json.edhrec.com/pages/commanders/{slug}/{bracket_slug}.json"
            if bracket_slug
            else f"https://json.edhrec.com/pages/commanders/{slug}.json"
        )
        headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; OppositionAgentsMTG/1.0; +research)",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"EDHREC returned {resp.status_code} for "
                    f"'{slug}' (bracket={bracket})"
                )
            data = resp.json()

        # Average counts per category (rounded ints); fall back to sane defaults.
        def _n(key: str, default: int) -> int:
            v = data.get(key)
            return int(v) if isinstance(v, (int, float)) else default

        n_creature = _n("creature", 30)
        n_instant = _n("instant", 8)
        n_sorcery = _n("sorcery", 8)
        n_artifact = _n("artifact", 10)        # split across utility + mana
        n_enchant = _n("enchantment", 6)
        n_planeswalker = _n("planeswalker", 0)
        n_nonbasic = _n("nonbasic", 6)
        n_basic = _n("basic", 30)

        cardlists = (
            data.get("container", {}).get("json_dict", {}).get("cardlists", [])
            or []
        )
        by_tag: dict[str, list[dict]] = {
            cl.get("tag", ""): cl.get("cardviews", []) or []
            for cl in cardlists
        }

        def pick_top(tag: str, n: int, exclude: set[str]) -> list[str]:
            """Pick the top-N most-included cards from a cardlist tag."""
            out: list[str] = []
            for cv in by_tag.get(tag, []):
                if n <= 0:
                    break
                name = cv.get("name")
                if not name or name in exclude:
                    continue
                out.append(name)
                exclude.add(name)
                n -= 1
            return out

        deck = Decklist()
        # Commander
        cmd_name = (
            data.get("container", {}).get("json_dict", {}).get("card", {}).get("name")
            or data.get("header")
            or slug.replace("-", " ").title()
        )
        deck.commander.append(cmd_name)
        chosen: set[str] = {cmd_name}

        for tag, n in [
            ("creatures", n_creature),
            ("instants", n_instant),
            ("sorceries", n_sorcery),
            ("manaartifacts", max(0, n_artifact // 2)),
            ("utilityartifacts", max(0, n_artifact - n_artifact // 2)),
            ("enchantments", n_enchant),
            ("planeswalkers", n_planeswalker),
            ("utilitylands", max(0, n_nonbasic // 2)),
            ("lands", max(0, n_nonbasic - n_nonbasic // 2)),
        ]:
            for name in pick_top(tag, n, chosen):
                deck.mainboard[name] = 1

        # Top-up shortfalls from "topcards" then "highsynergycards"
        target_nonbasic = (
            n_creature + n_instant + n_sorcery + n_artifact + n_enchant
            + n_planeswalker + n_nonbasic
        )
        have_nonbasic = sum(deck.mainboard.values())
        if have_nonbasic < target_nonbasic:
            for tag in ("topcards", "highsynergycards"):
                for name in pick_top(tag, target_nonbasic - have_nonbasic, chosen):
                    deck.mainboard[name] = 1
                    have_nonbasic += 1
                    if have_nonbasic >= target_nonbasic:
                        break
                if have_nonbasic >= target_nonbasic:
                    break

        # Fill basics. Distribute n_basic across the commander's color identity.
        commander_card = data.get("container", {}).get("json_dict", {}).get("card", {})
        ci = commander_card.get("color_identity") or []
        if isinstance(ci, str):
            ci = list(ci)
        ci = [c for c in ci if c in "WUBRG"] or ["C"]
        basic_for = {"W": "Plains", "U": "Island", "B": "Swamp",
                     "R": "Mountain", "G": "Forest", "C": "Wastes"}
        # Make total exactly 100: 1 commander + nonbasics + basics
        basics_needed = 100 - 1 - sum(deck.mainboard.values())
        if basics_needed < 0:
            basics_needed = max(15, n_basic)
        per = basics_needed // len(ci)
        rem = basics_needed - per * len(ci)
        for i, color in enumerate(ci):
            land = basic_for.get(color, "Wastes")
            deck.mainboard[land] = (
                deck.mainboard.get(land, 0) + per + (1 if i < rem else 0)
            )

        return deck

    # ------------------------------------------------------------------
    # Auto-dispatch from a URL
    # ------------------------------------------------------------------

    async def from_url(self, url: str, bracket: int | str | None = None) -> Decklist:
        """Auto-detect Moxfield / Archidekt / EDHREC URL and dispatch.

        ``bracket`` is forwarded to :meth:`from_edhrec_average` and ignored
        for the other providers.
        """
        u = url.strip()
        if "moxfield.com" in u:
            return await self.from_moxfield(u)
        if "archidekt.com" in u:
            m = re.search(r"archidekt\.com/decks/(\d+)", u)
            if not m:
                raise ValueError(f"can't extract Archidekt deck id from: {url}")
            return await self.from_archidekt(int(m.group(1)))
        if "edhrec.com" in u:
            # Allow URL-embedded bracket: /commanders/<slug>/<bracket>
            m = re.search(
                r"edhrec\.com/commanders/([A-Za-z0-9_-]+)(?:/([A-Za-z0-9_-]+))?",
                u,
            )
            if not m:
                raise ValueError(f"can't extract EDHREC slug from: {url}")
            slug = m.group(1)
            url_bracket = m.group(2)
            if url_bracket and bracket is None:
                bracket = url_bracket
            return await self.from_edhrec_average(slug, bracket=bracket)
        raise ValueError(f"unsupported decklist URL: {url}")
