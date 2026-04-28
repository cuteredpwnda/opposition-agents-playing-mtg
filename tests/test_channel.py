"""Channel keyword (Kamigawa: Neon Dynasty)."""

from __future__ import annotations

from src.engine.channel import parse_channel, execute_channel
from src.engine.game_state import (
    Action,
    ActionType,
    CardInstance,
    GameState,
    Phase,
    PlayerState,
    Zone,
)
from src.engine.rules_engine import RulesEngine


def _make_boseiju() -> CardInstance:
    return CardInstance(
        instance_id="A_boseiju",
        card_data={
            "name": "Boseiju, Who Endures",
            "type_line": "Legendary Land",
            "oracle_text": (
                "Tap: Add G.\n"
                "Channel — {1}{G}, Discard Boseiju, Who Endures: "
                "Destroy target nonbasic land."
            ),
        },
        zone=Zone.HAND,
        owner_id="A",
        controller_id="A",
    )


def test_parse_channel_extracts_cost_and_effect():
    card = _make_boseiju()
    parsed = parse_channel(card)
    assert parsed is not None
    cost, effect = parsed
    assert cost == "{1}{G}"
    assert effect.lower().startswith("destroy target nonbasic land")


def test_parse_channel_returns_none_for_non_channel_card():
    card = CardInstance(
        instance_id="A_dummy",
        card_data={
            "name": "Dummy",
            "type_line": "Instant",
            "oracle_text": "Draw a card.",
        },
        zone=Zone.HAND,
        owner_id="A",
        controller_id="A",
    )
    assert parse_channel(card) is None


def test_channel_appears_as_legal_action_when_mana_available():
    state = GameState(
        format="commander",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[PlayerState(player_id="A"), PlayerState(player_id="B")],
    )
    state.cards = [_make_boseiju()]
    state.players[0].mana_pool = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 1, "C": 1}
    actions = RulesEngine().get_legal_actions(state, "A")
    assert any(
        a.action_type == ActionType.SPECIAL_ACTION
        and (a.metadata or {}).get("special") == "channel"
        for a in actions
    ), [a for a in actions]


def test_execute_channel_discards_card_and_pushes_stack_item():
    state = GameState(
        format="commander",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[PlayerState(player_id="A"), PlayerState(player_id="B")],
    )
    boseiju = _make_boseiju()
    target_land = CardInstance(
        instance_id="B_temple",
        card_data={
            "name": "Temple of Mystery",
            "type_line": "Land",
            "oracle_text": "",
        },
        zone=Zone.BATTLEFIELD,
        owner_id="B",
        controller_id="B",
    )
    state.cards = [boseiju, target_land]
    state.players[0].mana_pool = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 1, "C": 1}
    execute_channel(state, boseiju, state.players[0])
    assert boseiju.zone == Zone.GRAVEYARD
    assert state.players[0].mana_pool.get("G", 0) == 0
    assert any(
        item.source_card_id == boseiju.instance_id and not item.is_spell
        for item in state.stack
    )
