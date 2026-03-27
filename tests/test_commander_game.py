import pytest

from src.orchestrator.game_runner import GameRunner, GameConfig
from src.agents.random_agent import RandomAgent
from src.training.deck_utils import create_mock_deck


@pytest.mark.asyncio
async def test_commander_game_initial_state():
    # Four-player commander game with random agents
    players = ["P1", "P2", "P3", "P4"]
    agents = {pid: RandomAgent(player_id=pid) for pid in players}

    runner = GameRunner(GameConfig(format="commander", starting_life=40, max_turns=25))
    deck = create_mock_deck() * 2
    if len(deck) < 100:
        deck = (deck * 2)[:100]

    result = await runner.run_game(
        agents=agents,
        decks={pid: deck.copy() for pid in players},
    )

    assert result is not None
    assert result.turns >= 1
    assert result.winner in {None, *players}


def test_commander_color_identity_and_tax():
    from src.engine.rules_engine import RulesEngine
    from src.engine.game_state import GameState, PlayerState, CardInstance, Phase, Zone, ActionType

    player = PlayerState(
        player_id="P1",
        name="P1",
        life_total=40,
        mana_pool={"W": 0, "U": 0, "B": 0, "R": 1, "G": 0, "C": 0},
    )

    commander_card = CardInstance(
        instance_id="cmd1",
        card_data={
            "name": "Rubinia Soulsinger",
            "type_line": "Legendary Creature — Human Wizard",
            "mana_cost": "{R}",
            "color_identity": ["R"],
        },
        zone=Zone.COMMAND_ZONE,
        owner_id="P1",
        controller_id="P1",
    )

    green_card = CardInstance(
        instance_id="g1",
        card_data={
            "name": "Llanowar Elves",
            "type_line": "Creature — Elf Druid",
            "mana_cost": "{G}",
            "color_identity": ["G"],
        },
        zone=Zone.HAND,
        owner_id="P1",
        controller_id="P1",
    )

    game_state = GameState(
        format="commander",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[player],
        cards=[commander_card, green_card],
        stack=[],
        triggered_abilities=[],
        combat=None,
        game_log=[],
    )
    game_state.commanders = {"P1": "cmd1"}

    engine = RulesEngine()

    # With commander's color identity = R, green card should not be legal to cast.
    actions = engine.get_legal_actions(game_state, "P1")
    cast_actions = [a for a in actions if a.action_type == ActionType.CAST_SPELL]
    assert any(a.card_instance_id == "cmd1" for a in cast_actions)
    assert not any(a.card_instance_id == "g1" for a in cast_actions)

    # Cast commander and assert commander tax increments.
    spell_action = next(a for a in cast_actions if a.card_instance_id == "cmd1")
    game_state = engine.execute_action(game_state, spell_action)
    assert player.commander_tax == 1
    cmd_card_after = next(c for c in game_state.cards if c.instance_id == "cmd1")
    assert cmd_card_after.zone == Zone.STACK

    # After commander has been cast once, the next command cast requires extra cost.
    commander_card2 = CardInstance(
        instance_id="cmd1",
        card_data={
            "name": "Rubinia Soulsinger",
            "type_line": "Legendary Creature — Human Wizard",
            "mana_cost": "{R}",
            "color_identity": ["R"],
        },
        zone=Zone.COMMAND_ZONE,
        owner_id="P1",
        controller_id="P1",
    )
    green_card2 = CardInstance(
        instance_id="g1",
        card_data={
            "name": "Llanowar Elves",
            "type_line": "Creature — Elf Druid",
            "mana_cost": "{G}",
            "color_identity": ["G"],
        },
        zone=Zone.HAND,
        owner_id="P1",
        controller_id="P1",
    )
    player2 = PlayerState(
        player_id="P1",
        name="P1",
        life_total=40,
        mana_pool={"W": 0, "U": 0, "B": 0, "R": 1, "G": 0, "C": 0},
        commander_tax=1,
    )
    new_state = GameState(
        format="commander",
        turn_number=1,
        active_player_index=0,
        priority_player_index=0,
        phase=Phase.MAIN_1,
        players=[player2],
        cards=[commander_card2, green_card2],
        stack=[],
        triggered_abilities=[],
        combat=None,
        game_log=[],
    )
    new_state.commanders = {"P1": "cmd1"}
    actions2 = engine.get_legal_actions(new_state, "P1")
    cast_actions2 = [a for a in actions2 if a.action_type == ActionType.CAST_SPELL]
    assert not any(a.card_instance_id == "cmd1" for a in cast_actions2)

