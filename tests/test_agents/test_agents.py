"""Tests for agent base class and random agent."""

import pytest

from src.agents.base_agent import MTGAgent
from src.agents.random_agent import RandomAgent
from src.engine.game_state import Action, ActionType, GameState, Phase, PlayerState


class TestRandomAgent:
    @pytest.mark.asyncio
    async def test_picks_legal_action(self):
        agent = RandomAgent(player_id="p1")
        gs = GameState(
            players={"p1": PlayerState(), "p2": PlayerState()},
            active_player="p1",
            priority_player="p1",
            phase=Phase.MAIN_1,
            turn_number=1,
        )
        actions = [
            Action(action_type=ActionType.PASS_PRIORITY, player="p1"),
            Action(action_type=ActionType.PLAY_LAND, player="p1"),
        ]
        chosen = await agent.decide_action(gs, actions)
        assert chosen in actions

    def test_score_attack_target(self):
        agent = RandomAgent(player_id="p1")

        game_state = GameState(
            players=[
                PlayerState(player_id="p1", life_total=40),
                PlayerState(player_id="p2", life_total=10, commander_damage_received={"p1": 5}),
                PlayerState(player_id="p3", life_total=20, commander_damage_received={"p1": 0}),
            ],
            active_player_index=0,
            priority_player_index=0,
            phase=Phase.MAIN_1,
            turn_number=1,
        )

        score2 = agent._score_attack_target(game_state, "p2")
        score3 = agent._score_attack_target(game_state, "p3")

        assert score2 > score3
