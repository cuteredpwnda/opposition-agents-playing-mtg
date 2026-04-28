"""Heuristic agent: counterspell awareness."""

from __future__ import annotations

import asyncio

from src.agents.heuristic_agent import HeuristicAgent
from src.engine.game_state import (
    Action,
    ActionType,
    CardInstance,
    GameState,
    Phase,
    PlayerState,
    StackItem,
    Zone,
)


def _state(*cards: CardInstance) -> GameState:
    state = GameState(
        format="commander",
        turn_number=1,
        active_player_index=0,
        priority_player_index=1,
        phase=Phase.MAIN_1,
        players=[PlayerState(player_id="A"), PlayerState(player_id="B")],
    )
    state.cards = list(cards)
    return state


def _counterspell(controller: str = "B") -> CardInstance:
    return CardInstance(
        instance_id=f"{controller}_counter",
        card_data={
            "name": "Counterspell",
            "type_line": "Instant",
            "oracle_text": "Counter target spell.",
        },
        zone=Zone.HAND,
        owner_id=controller,
        controller_id=controller,
    )


def _opp_spell() -> CardInstance:
    return CardInstance(
        instance_id="A_threat",
        card_data={
            "name": "Lightning Bolt",
            "type_line": "Instant",
            "oracle_text": "Lightning Bolt deals 3 damage to any target.",
        },
        zone=Zone.STACK,
        owner_id="A",
        controller_id="A",
    )


def test_counter_held_when_stack_empty():
    counter = _counterspell()
    state = _state(counter)
    actions = [
        Action(action_type=ActionType.CAST_SPELL, player_id="B",
               card_instance_id=counter.instance_id),
        Action(action_type=ActionType.PASS_PRIORITY, player_id="B"),
    ]
    agent = HeuristicAgent(player_id="B", seed=0)
    chosen = asyncio.run(agent.decide_action(state, actions))
    assert chosen.action_type == ActionType.PASS_PRIORITY


def test_counter_used_when_opp_spell_on_stack():
    counter = _counterspell()
    threat = _opp_spell()
    state = _state(counter, threat)
    state.stack.append(StackItem(
        source_card_id=threat.instance_id,
        controller_id="A",
        is_spell=True,
        card_data=threat.card_data,
    ))
    actions = [
        Action(action_type=ActionType.CAST_SPELL, player_id="B",
               card_instance_id=counter.instance_id),
        Action(action_type=ActionType.PASS_PRIORITY, player_id="B"),
    ]
    agent = HeuristicAgent(player_id="B", seed=0)
    chosen = asyncio.run(agent.decide_action(state, actions))
    assert chosen.action_type == ActionType.CAST_SPELL
    assert chosen.card_instance_id == counter.instance_id


def test_counter_not_used_against_own_spell():
    counter = _counterspell(controller="B")
    own_spell = CardInstance(
        instance_id="B_dork",
        card_data={"name": "Llanowar Elves", "type_line": "Creature — Elf",
                   "oracle_text": "{T}: Add {G}."},
        zone=Zone.STACK,
        owner_id="B",
        controller_id="B",
    )
    state = _state(counter, own_spell)
    state.stack.append(StackItem(
        source_card_id=own_spell.instance_id,
        controller_id="B",
        is_spell=True,
        card_data=own_spell.card_data,
    ))
    actions = [
        Action(action_type=ActionType.CAST_SPELL, player_id="B",
               card_instance_id=counter.instance_id),
        Action(action_type=ActionType.PASS_PRIORITY, player_id="B"),
    ]
    agent = HeuristicAgent(player_id="B", seed=0)
    chosen = asyncio.run(agent.decide_action(state, actions))
    assert chosen.action_type == ActionType.PASS_PRIORITY
