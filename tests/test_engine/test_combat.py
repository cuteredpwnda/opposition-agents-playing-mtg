"""Tests for the combat module."""

import pytest

from src.engine.combat import declare_attackers, resolve_combat_damage
from src.engine.game_state import (
    CardInstance,
    CombatState,
    GameState,
    Phase,
    PlayerState,
)


def _make_creature(name: str, power: int, toughness: int, owner: str) -> CardInstance:
    return CardInstance(
        instance_id=f"{owner}_{name}",
        owner=owner,
        controller=owner,
        name=name,
        oracle_text="",
        type_line="Creature",
        mana_cost="",
        cmc=0,
        is_creature=True,
        power=power,
        toughness=toughness,
    )


class TestCombat:
    def test_declare_attackers_taps(self):
        creature = _make_creature("Grizzly Bears", 2, 2, "p1")
        gs = GameState(
            players={"p1": PlayerState(), "p2": PlayerState()},
            active_player="p1",
            priority_player="p1",
            phase=Phase.COMBAT_ATTACKERS,
            turn_number=2,
            cards_in_zone={("p1", "battlefield"): [creature]},
            combat=CombatState(),
        )
        gs = declare_attackers(gs, [creature.instance_id], "p2")
        assert creature.tapped is True
        assert creature.instance_id in gs.combat.attackers

    def test_unblocked_damage(self):
        attacker = _make_creature("Grizzly Bears", 2, 2, "p1")
        attacker.tapped = True
        gs = GameState(
            players={"p1": PlayerState(), "p2": PlayerState(life=20)},
            active_player="p1",
            priority_player="p1",
            phase=Phase.COMBAT_DAMAGE,
            turn_number=2,
            cards_in_zone={
                ("p1", "battlefield"): [attacker],
                ("p2", "battlefield"): [],
            },
            combat=CombatState(
                attackers={attacker.instance_id: "p2"},
                blockers={},
            ),
        )
        gs = resolve_combat_damage(gs)
        assert gs.players["p2"].life == 18  # 20 - 2
