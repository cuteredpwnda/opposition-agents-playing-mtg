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
