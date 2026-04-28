"""Mana-rock activation: Sol Ring, Mind Stone, Birds of Paradise."""

from __future__ import annotations

from src.engine.game_state import (
    CardInstance,
    GameState,
    Phase,
    PlayerState,
    Zone,
)
from src.engine.mana import (
    auto_tap_for_cost,
    permanent_mana_production,
    potential_mana,
)


def _state(*pids: str) -> GameState:
    return GameState(
        format="commander",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[PlayerState(player_id=p) for p in pids],
    )


def _battlefield(state: GameState, owner: str, name: str, type_line: str,
                 oracle: str, summoning_sick: bool = False) -> CardInstance:
    card = CardInstance(
        instance_id=f"{owner}_{name.lower().replace(' ', '_')}",
        card_data={"name": name, "type_line": type_line, "oracle_text": oracle},
        zone=Zone.BATTLEFIELD,
        owner_id=owner,
        controller_id=owner,
        summoning_sick=summoning_sick,
    )
    state.cards.append(card)
    return card


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_sol_ring_produces_two_colorless():
    s = _state("A")
    sol = _battlefield(s, "A", "Sol Ring", "Artifact",
                       "{T}: Add {C}{C}.")
    assert permanent_mana_production(sol) == {"C": 2}


def test_mind_stone_produces_one_colorless_only():
    # Mind Stone has two abilities; only the {T}: Add {C} mana ability is exposed.
    s = _state("A")
    rock = _battlefield(s, "A", "Mind Stone", "Artifact",
                        "{T}: Add {C}.\n{1}, {T}, Sacrifice CARDNAME: Draw a card.")
    assert permanent_mana_production(rock) == {"C": 1}


def test_birds_of_paradise_any_color_treated_as_colorless():
    s = _state("A")
    bop = _battlefield(s, "A", "Birds of Paradise", "Creature — Bird",
                       "Flying\n{T}: Add one mana of any color.")
    # Models as {"C": 1} for affordability; cast-time will spread colour.
    assert permanent_mana_production(bop) == {"C": 1}


def test_treasure_token_rejected_due_to_sacrifice_cost():
    # Treasure: "{T}, Sacrifice CARDNAME: Add one mana of any color."
    # That's not a plain {T} ability — it costs sacrifice.
    s = _state("A")
    treasure = _battlefield(
        s, "A", "Treasure", "Token Artifact — Treasure",
        "{T}, Sacrifice CARDNAME: Add one mana of any color.",
    )
    assert permanent_mana_production(treasure) is None


# ---------------------------------------------------------------------------
# Aggregation into potential_mana / auto_tap_for_cost
# ---------------------------------------------------------------------------

def test_potential_mana_includes_sol_ring():
    s = _state("A")
    p = s.players[0]
    _battlefield(s, "A", "Sol Ring", "Artifact", "{T}: Add {C}{C}.")
    assert potential_mana(s, p)["C"] == 2


def test_summoning_sick_creature_does_not_contribute():
    s = _state("A")
    p = s.players[0]
    _battlefield(s, "A", "Llanowar Elves", "Creature — Elf Druid",
                 "{T}: Add {G}.", summoning_sick=True)
    assert potential_mana(s, p)["G"] == 0


def test_haste_creature_overrides_summoning_sickness():
    s = _state("A")
    p = s.players[0]
    _battlefield(s, "A", "Hasty Druid", "Creature — Elf Druid",
                 "Haste\n{T}: Add {G}.", summoning_sick=True)
    assert potential_mana(s, p)["G"] == 1


def test_auto_tap_uses_sol_ring_for_generic_cost():
    s = _state("A")
    p = s.players[0]
    sol = _battlefield(s, "A", "Sol Ring", "Artifact", "{T}: Add {C}{C}.")

    # Cost: {2} generic. Sol Ring should satisfy.
    assert auto_tap_for_cost(s, p, {"generic": 2}) is True
    assert sol.tapped is True
    # Pool should hold 2 colorless after tap.
    assert p.mana_pool["C"] == 2


def test_auto_tap_combines_basic_land_and_dork():
    s = _state("A")
    p = s.players[0]
    forest = _battlefield(s, "A", "Forest", "Basic Land — Forest",
                          "({T}: Add {G}.)")
    elf = _battlefield(s, "A", "Llanowar Elves", "Creature — Elf Druid",
                       "{T}: Add {G}.", summoning_sick=False)

    # {1}{G} → tap elf for G, forest for generic.
    assert auto_tap_for_cost(s, p, {"G": 1, "generic": 1}) is True
    assert forest.tapped and elf.tapped
