"""Standard benchmark archetype decks.

A small but illustrative deck zoo used by `BenchmarkSuite` and the
training pipeline so that agents can be evaluated against a fixed,
reproducible set of strategies. Each builder returns a 60-card deck
in the simplified card-dict format expected by `GameRunner`.

The archetypes loosely mirror the MTG meta categories:
- ``mono_red_aggro``  : creature-heavy fast clock
- ``mono_blue_control`` : counter / draw control
- ``mono_green_ramp``  : ramp into big creatures
- ``mono_white_weenie`` : small efficient creatures
- ``mono_black_midrange`` : removal + value creatures
- ``izzet_burn``       : direct-damage burn

The dictionaries are intentionally lightweight; the rules engine only
reads the keys ``name``, ``type_line``, ``mana_cost``, ``cmc``,
``oracle_text``, ``power``, ``toughness``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

Card = Dict[str, Any]
DeckBuilder = Callable[[], List[Card]]


def _card(
    name: str,
    type_line: str,
    cmc: int,
    mana_cost: str = "",
    oracle_text: str = "",
    power: str | None = None,
    toughness: str | None = None,
) -> Card:
    return {
        "name": name,
        "type_line": type_line,
        "mana_cost": mana_cost,
        "cmc": cmc,
        "oracle_text": oracle_text,
        "power": power,
        "toughness": toughness,
    }


def _land(name: str, mana_letter: str) -> Card:
    return _card(
        name,
        type_line="Land",
        cmc=0,
        oracle_text=f"{{T}}: Add {{{mana_letter}}}",
    )


def _fill_to_60(deck: List[Card]) -> List[Card]:
    if len(deck) >= 60:
        return deck[:60]
    deck.extend(deck[: 60 - len(deck)])
    return deck


# ---------------------------------------------------------------------------
# Archetype builders
# ---------------------------------------------------------------------------

def mono_red_aggro() -> List[Card]:
    deck: List[Card] = []
    for i in range(22):
        deck.append(_land(f"Mountain_{i}", "R"))
    for i in range(20):
        deck.append(_card(f"GoblinChampion_{i}", "Creature — Goblin", 1, "{R}",
                          "Haste", "2", "1"))
    for i in range(10):
        deck.append(_card(f"FireBlast_{i}", "Instant", 1, "{R}",
                          "Deal 3 damage to any target.", None, None))
    for i in range(8):
        deck.append(_card(f"BurnRush_{i}", "Sorcery", 2, "{1}{R}",
                          "Deal 4 damage divided as you choose.", None, None))
    return _fill_to_60(deck)


def mono_blue_control() -> List[Card]:
    deck: List[Card] = []
    for i in range(24):
        deck.append(_land(f"Island_{i}", "U"))
    for i in range(8):
        deck.append(_card(f"Counterspell_{i}", "Instant", 2, "{U}{U}",
                          "Counter target spell.", None, None))
    for i in range(8):
        deck.append(_card(f"DrawTwo_{i}", "Sorcery", 2, "{1}{U}",
                          "Draw two cards.", None, None))
    for i in range(8):
        deck.append(_card(f"BounceWizard_{i}", "Creature — Wizard", 2, "{1}{U}",
                          "When this enters, return target creature to its owner's hand.",
                          "1", "3"))
    for i in range(12):
        deck.append(_card(f"AirElemental_{i}", "Creature — Elemental", 4,
                          "{2}{U}{U}", "Flying", "4", "4"))
    return _fill_to_60(deck)


def mono_green_ramp() -> List[Card]:
    deck: List[Card] = []
    for i in range(24):
        deck.append(_land(f"Forest_{i}", "G"))
    for i in range(10):
        deck.append(_card(f"ManaDork_{i}", "Creature — Elf", 1, "{G}",
                          "{T}: Add {G}.", "1", "1"))
    for i in range(8):
        deck.append(_card(f"RampSpell_{i}", "Sorcery", 3, "{2}{G}",
                          "Search your library for two basic land cards and put them onto the battlefield tapped.",
                          None, None))
    for i in range(10):
        deck.append(_card(f"BeastRanger_{i}", "Creature — Beast", 5, "{3}{G}{G}",
                          "Trample", "5", "5"))
    for i in range(8):
        deck.append(_card(f"GroveTitan_{i}", "Creature — Giant", 7, "{5}{G}{G}",
                          "Trample, vigilance.", "7", "7"))
    return _fill_to_60(deck)


def mono_white_weenie() -> List[Card]:
    deck: List[Card] = []
    for i in range(22):
        deck.append(_land(f"Plains_{i}", "W"))
    for i in range(20):
        deck.append(_card(f"SoldierRecruit_{i}", "Creature — Soldier", 1, "{W}",
                          "First strike.", "2", "1"))
    for i in range(10):
        deck.append(_card(f"AnthemEffect_{i}", "Enchantment", 2, "{1}{W}",
                          "Creatures you control get +1/+1.", None, None))
    for i in range(8):
        deck.append(_card(f"Disenchant_{i}", "Instant", 2, "{1}{W}",
                          "Destroy target artifact or enchantment.", None, None))
    return _fill_to_60(deck)


def mono_black_midrange() -> List[Card]:
    deck: List[Card] = []
    for i in range(22):
        deck.append(_land(f"Swamp_{i}", "B"))
    for i in range(8):
        deck.append(_card(f"DarkConfidant_{i}", "Creature — Human", 2, "{1}{B}",
                          "Reveal the top card of your library at upkeep; lose life equal to its cmc and put it in your hand.",
                          "2", "1"))
    for i in range(10):
        deck.append(_card(f"FatalRemoval_{i}", "Instant", 2, "{1}{B}",
                          "Destroy target creature.", None, None))
    for i in range(8):
        deck.append(_card(f"Inquisition_{i}", "Sorcery", 1, "{B}",
                          "Target opponent reveals their hand; you choose a nonland card with cmc 3 or less; that player discards that card.",
                          None, None))
    for i in range(12):
        deck.append(_card(f"PhyrexianTactician_{i}", "Creature — Knight", 4, "{2}{B}{B}",
                          "Menace, lifelink.", "4", "4"))
    return _fill_to_60(deck)


def izzet_burn() -> List[Card]:
    deck: List[Card] = []
    for i in range(11):
        deck.append(_land(f"Mountain_{i}", "R"))
    for i in range(11):
        deck.append(_land(f"Island_{i}", "U"))
    for i in range(12):
        deck.append(_card(f"LightningBolt_{i}", "Instant", 1, "{R}",
                          "Deal 3 damage to any target.", None, None))
    for i in range(8):
        deck.append(_card(f"ShockWizard_{i}", "Creature — Wizard", 2, "{1}{R}",
                          "When this enters, deal 2 damage to any target.", "2", "2"))
    for i in range(10):
        deck.append(_card(f"Snap_{i}", "Instant", 2, "{1}{U}",
                          "Return target creature to its owner's hand.", None, None))
    for i in range(8):
        deck.append(_card(f"FireblastFinisher_{i}", "Sorcery", 4, "{2}{U}{R}",
                          "Deal 5 damage to any target; you may copy this spell.", None, None))
    return _fill_to_60(deck)


ARCHETYPES: Dict[str, DeckBuilder] = {
    "mono_red_aggro": mono_red_aggro,
    "mono_blue_control": mono_blue_control,
    "mono_green_ramp": mono_green_ramp,
    "mono_white_weenie": mono_white_weenie,
    "mono_black_midrange": mono_black_midrange,
    "izzet_burn": izzet_burn,
}


def get_archetype(name: str) -> List[Card]:
    """Return a fresh deck for the given archetype name."""
    if name not in ARCHETYPES:
        raise KeyError(
            f"Unknown archetype '{name}'. Available: {sorted(ARCHETYPES)}"
        )
    return ARCHETYPES[name]()


def list_archetypes() -> List[str]:
    return sorted(ARCHETYPES)
