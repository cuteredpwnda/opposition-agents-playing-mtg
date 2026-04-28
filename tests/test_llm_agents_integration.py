"""Integration test: LLM agents in a priority loop (no full game).

This tests that:
1. OllamaAgent instantiates correctly
2. decide_action() and decide_mulligan() are callable
3. The priority loop accepts agent decisions
4. Fallback works when Ollama is unavailable
"""

from __future__ import annotations

import pytest
import asyncio

from src.agents.llm_agent import OllamaAgent
from src.agents.random_agent import RandomAgent
from src.engine.game_state import GameState, PlayerState, CardInstance, Zone, ActionType, Action, Phase


@pytest.mark.asyncio
async def test_ollama_agent_decide_action_signature():
    """OllamaAgent.decide_action(game_state, legal_actions) works."""
    agent = OllamaAgent("p1", model="llama3.2:1b")
    
    # Create minimal game state
    player1 = PlayerState(player_id="p1", name="Player 1")
    player2 = PlayerState(player_id="p2", name="Player 2")
    game = GameState(players=[player1, player2])
    
    # Create some minimal legal actions
    legal = [
        Action(action_type=ActionType.PASS_PRIORITY, player_id="p1"),
    ]
    
    # Call should not raise
    result = await agent.decide_action(game, legal)
    assert isinstance(result, Action)


@pytest.mark.asyncio
async def test_ollama_agent_with_fallback():
    """OllamaAgent uses fallback when Ollama unavailable."""
    agent = OllamaAgent("p1", model="nonexistent:model")
    assert agent._fallback_agent is not None
    assert isinstance(agent._fallback_agent, RandomAgent)
    
    # Fallback should work
    player1 = PlayerState(player_id="p1", name="Player 1")
    player2 = PlayerState(player_id="p2", name="Player 2")
    game = GameState(players=[player1, player2])
    legal = [
        Action(action_type=ActionType.PASS_PRIORITY, player_id="p1"),
    ]
    
    result = await agent.decide_action(game, legal)
    assert result is not None
    assert result in legal  # Should pick one of the legal actions


def test_ollama_agent_mulligan_decision_sync():
    """Mulligan decisions work (sync, with fallback)."""
    agent = OllamaAgent("p1", model="llama3.2:1b")
    
    # Create a sample hand (empty hand should return True/False)
    hand = []
    result = agent.decide_mulligan(hand, mulligans_taken=0, max_mulligans=3)
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_two_agents_in_sequence():
    """Two agents can make decisions in sequence."""
    agent1 = OllamaAgent("p1", model="qwen2.5-coder:1.5b")
    agent2 = OllamaAgent("p2", model="llama3.2:1b")
    
    player1 = PlayerState(player_id="p1", name="Player 1")
    player2 = PlayerState(player_id="p2", name="Player 2")
    game = GameState(players=[player1, player2])
    
    legal_p1 = [Action(action_type=ActionType.PASS_PRIORITY, player_id="p1")]
    legal_p2 = [Action(action_type=ActionType.PASS_PRIORITY, player_id="p2")]
    
    # Both agents can decide
    action1 = await agent1.decide_action(game, legal_p1)
    action2 = await agent2.decide_action(game, legal_p2)
    
    assert action1 in legal_p1
    assert action2 in legal_p2
