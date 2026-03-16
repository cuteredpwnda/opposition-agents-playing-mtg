#!/usr/bin/env python
"""Debug test to trace action selection."""

import pytest

from src.agents.random_agent import RandomAgent
from src.orchestrator.game_runner import GameRunner, GameConfig
from src.engine.game_state import ActionType


class DebugRandomAgent(RandomAgent):
    """Random agent with logging."""
    
    async def decide_action(self, game_state, legal_actions):
        """Choose with logging."""
        action = await super().decide_action(game_state, legal_actions)
        
        # Count action types in legal actions
        types = {}
        for a in legal_actions:
            t = a.action_type.value
            types[t] = types.get(t, 0) + 1
        
        print(f"\n    [{self.name}] Legal: {types}, Chose: {action.action_type.value}")
        
        return action


@pytest.mark.asyncio
async def test_action_selection_debug():
    """Debug first turn action selection."""
    config = GameConfig(format="standard", starting_life=20, max_turns=3)
    runner = GameRunner(config)
    
    agents = {
        "Alice": DebugRandomAgent(player_id="Alice", name="Alice"),
        "Bob": DebugRandomAgent(player_id="Bob", name="Bob"),
    }
    
    # Simple deck: balanced mana
    deck = [
        {
            "name": "Mountain",
            "mana_cost": "",
            "type_line": "Land — Mountain",
            "oracle_text": "{T}: Add {R}.",
            "power": None,
            "toughness": None,
            "cmc": 0,
            "keywords": [],
            "set": "DOM",
        }
        for _ in range(20)
    ] + [
        {
            "name": "Goblin",
            "mana_cost": "{R}",
            "type_line": "Creature — Goblin",
            "oracle_text": "",
            "power": "1",
            "toughness": "1",
            "cmc": 1,
            "keywords": [],
            "set": "DOM",
        }
        for _ in range(20)
    ]
    
    decks = {"Alice": deck.copy(), "Bob": deck.copy()}
    game_state = runner._setup_game(agents, decks)
    
    print(f"\nStarting first turn...\n")
    game_state = await runner._play_turn(game_state, agents)
    
    print(f"\n\nAlice's hand after turn 1:")
    alice_hand = [c for c in game_state.cards if c.owner_id == "Alice" and c.zone.value == "hand"]
    for card in alice_hand:
        print(f"  {card.name} ({card.mana_cost})")
    
    print(f"\nAlice's battlefield after turn 1:")
    alice_bf = [c for c in game_state.cards if c.owner_id == "Alice" and c.zone.value == "battlefield"]
    for card in alice_bf:
        print(f"  {card.name}")
    
    # Assert we actually have battlefield permanents
    assert len(alice_bf) > 0, "Alice should have cast lands or creatures"
