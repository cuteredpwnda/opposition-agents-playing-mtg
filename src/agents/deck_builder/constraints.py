"""Deck-construction hard constraints for Commander format.

G1 — Color identity, singleton, deck size, mana-curve bins, format legality.

All checks are synchronous and work purely against Scryfall card dicts
(no network calls, no KG).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Basic land names always pass colour-identity checks (CR 903.5d).
_BASIC_LANDS: frozenset[str] = frozenset(
    {
        "Plains",
        "Island",
        "Swamp",
        "Mountain",
        "Forest",
        "Wastes",
        "Snow-Covered Plains",
        "Snow-Covered Island",
        "Snow-Covered Swamp",
        "Snow-Covered Mountain",
        "Snow-Covered Forest",
    }
)

# CMC bucket upper bounds for mana-curve reporting.
_CURVE_BINS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def commander_color_identity(commander_cards: list[dict[str, Any]]) -> set[str]:
    """Return the union of colour identities across all commanders."""
    result: set[str] = set()
    for card in commander_cards:
        ci = card.get("color_identity") or []
        result.update(c.upper() for c in ci)
    return result


def card_color_identity(card: dict[str, Any]) -> set[str]:
    """Return the colour identity of a single card as a set of letters."""
    ci = card.get("color_identity") or []
    return {c.upper() for c in ci}


def is_basic_land(card: dict[str, Any]) -> bool:
    return card.get("name", "") in _BASIC_LANDS


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ConstraintResult:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------------------
# Main constraint set
# ---------------------------------------------------------------------------


@dataclass
class ConstraintSet:
    """All hard constraints for a Commander deck build.

    Usage::

        cs = ConstraintSet(commander_ci={"R"}, format="commander")
        result = cs.accept(card_dict)
        if result:
            cs.add(card_dict)

    After each :meth:`accept` + :meth:`add` round, :meth:`needs_more`
    tells the caller whether to keep filling.
    """

    commander_ci: set[str]
    format: str = "commander"
    target_size: int = 99        # mainboard cards (excluding commander)
    allow_non_singleton: bool = False

    _mainboard: list[str] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the tracked mainboard."""
        self._mainboard = []

    def current_names(self) -> set[str]:
        return set(self._mainboard)

    def current_count(self) -> int:
        return len(self._mainboard)

    def needs_more(self) -> bool:
        return len(self._mainboard) < self.target_size

    # ------------------------------------------------------------------
    # Hard constraint checks
    # ------------------------------------------------------------------

    def accept(self, card: dict[str, Any]) -> ConstraintResult:
        """Return whether *card* may be legally added to the current deck."""
        name = card.get("name", "")

        # 1. Colour identity — basics always pass (CR 903.5d).
        if not is_basic_land(card):
            ci = card_color_identity(card)
            if not ci.issubset(self.commander_ci):
                return ConstraintResult(
                    False,
                    f"{name}: colour identity {ci} ⊄ {self.commander_ci}",
                )

        # 2. Singleton rule (Commander rule 903.5b).
        if not self.allow_non_singleton and name in self._mainboard:
            return ConstraintResult(False, f"{name}: already in deck (singleton)")

        # 3. Format legality.
        legality = (card.get("legalities") or {}).get(self.format, "")
        if legality == "banned":
            return ConstraintResult(False, f"{name}: banned in {self.format}")
        if legality == "not_legal":
            return ConstraintResult(False, f"{name}: not legal in {self.format}")

        return ConstraintResult(True)

    def add(self, card: dict[str, Any]) -> None:
        """Append card name to the tracked mainboard (call after accept)."""
        self._mainboard.append(card["name"])

    # ------------------------------------------------------------------
    # Soft statistics (non-blocking, informational only)
    # ------------------------------------------------------------------

    def mana_curve(self, card_dicts: list[dict[str, Any]]) -> dict[str, int]:
        """Return a CMC histogram over non-land cards.

        Keys are ``"<=1"``, ``"<=2"``, …, ``"<=6"``, ``">6"``.
        """
        buckets: dict[str, int] = {f"<={b}": 0 for b in _CURVE_BINS}
        buckets[f">{_CURVE_BINS[-1]}"] = 0
        for card in card_dicts:
            if "Land" in (card.get("type_line") or ""):
                continue
            cmc = float(card.get("cmc") or 0)
            placed = False
            for b in _CURVE_BINS:
                if cmc <= b:
                    buckets[f"<={b}"] += 1
                    placed = True
                    break
            if not placed:
                buckets[f">{_CURVE_BINS[-1]}"] += 1
        return buckets


# ---------------------------------------------------------------------------
# Standalone validation helper
# ---------------------------------------------------------------------------


def validate_full_deck(
    commander_cards: list[dict[str, Any]],
    mainboard_cards: list[dict[str, Any]],
    format: str = "commander",
) -> list[str]:
    """Validate a complete deck and return a list of violation strings.

    An empty list means the deck is fully legal.
    """
    errors: list[str] = []
    cmd_ci = commander_color_identity(commander_cards)
    cs = ConstraintSet(commander_ci=cmd_ci, format=format)

    for card in mainboard_cards:
        result = cs.accept(card)
        if not result.ok:
            errors.append(result.reason)
        else:
            cs.add(card)

    total = len(commander_cards) + len(mainboard_cards)
    if total != 100:
        errors.append(f"Deck has {total} cards (Commander requires exactly 100)")

    return errors
