"""Tests for the combat module."""

import pytest

from src.engine.combat import declare_attackers, resolve_combat_damage
from src.engine.game_state import (
    CardInstance,
    CombatState,
    GameState,
    Phase,
    PlayerState,
    Zone,
)


def _make_creature(name: str, power: int, toughness: int, owner: str) -> CardInstance:
    c = CardInstance(
        instance_id=f"{owner}_{name}",
        card_data={
            "name": name,
            "oracle_text": "",
            "type_line": "Creature",
            "mana_cost": "",
            "cmc": 0,
            "power": str(power),
            "toughness": str(toughness),
        },
        zone=Zone.BATTLEFIELD,
        owner_id=owner,
        controller_id=owner,
    )
    c.summoning_sick = False
    return c


class TestCombat:
    def test_declare_attackers_taps(self):
        creature = _make_creature("Grizzly Bears", 2, 2, "p1")
        gs = GameState(
            players=[PlayerState(player_id="p1"), PlayerState(player_id="p2")],
            active_player="p1",
            priority_player="p1",
            phase=Phase.COMBAT_ATTACKERS,
            turn_number=2,
            cards=[creature],
            combat=CombatState(),
        )
        declare_attackers(gs, {creature.instance_id: "p2"})
        assert creature.tapped is True
        assert creature.instance_id in gs.combat.attackers

    def test_unblocked_damage(self):
        attacker = _make_creature("Grizzly Bears", 2, 2, "p1")
        attacker.tapped = True
        gs = GameState(
            players=[PlayerState(player_id="p1"), PlayerState(player_id="p2", life_total=20)],
            active_player="p1",
            priority_player="p1",
            phase=Phase.COMBAT_DAMAGE,
            turn_number=2,
            cards=[attacker],
            combat=CombatState(
                attackers={attacker.instance_id: "p2"},
                blockers={},
            ),
        )
        resolve_combat_damage(gs)
        p2 = next(p for p in gs.players if p.player_id == "p2")
        assert p2.life_total == 18  # 20 - 2
