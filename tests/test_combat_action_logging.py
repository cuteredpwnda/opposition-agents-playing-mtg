#!/usr/bin/env python
"""Test to show action selection during combat."""

import pytest

from src.agents.random_agent import RandomAgent
from src.orchestrator.game_runner import GameRunner, GameConfig
from src.engine.game_state import ActionType, Zone, Phase


class LoggingRandomAgent(RandomAgent):
    """Random agent that logs action choices."""
    
    async def decide_action(self, game_state, legal_actions):
        """Log actions and choose."""
        action = await super().decide_action(game_state, legal_actions)
        
        if game_state.phase == Phase.COMBAT_ATTACKERS:
            action_types = {}
            for a in legal_actions:
                t = a.action_type.name
                action_types[t] = action_types.get(t, 0) + 1
            
            print(f"[{self.player_id}] COMBAT phase - Legal: {action_types} → Chose: {action.action_type.name}")
        
        return action


@pytest.mark.asyncio
async def test_combat_action_selection():
    """Show action selection during COMBAT_ATTACKERS phase."""
    config = GameConfig(format="standard", starting_life=20, max_turns=10)
    runner = GameRunner(config)
    
    agents = {
        "Alice": LoggingRandomAgent(player_id="Alice", name="Alice"),
        "Bob": LoggingRandomAgent(player_id="Bob", name="Bob"),
    }
    
    # Simple deck
    deck = [
        {"name": "Mountain", "mana_cost": "", "type_line": "Land",
         "oracle_text": "{T}: Add {R}.", "power": None, "toughness": None,
         "cmc": 0, "keywords": [], "set": "DOM"}
        for _ in range(10)
    ] + [
        {"name": "Goblin", "mana_cost": "{R}", "type_line": "Creature — Goblin",
         "oracle_text": "", "power": "1", "toughness": "1", "cmc": 1,
         "keywords": [], "set": "DOM"}
        for _ in range(20)
    ]
    
    decks = {"Alice": deck.copy(), "Bob": deck.copy()}
    game_state = runner._setup_game(agents, decks)
    
    # Play 5 turns
    for turn_num in range(1, 6):
        if game_state.game_over:
            break
        
        print(f"\n=== TURN {turn_num} ===")
        
        # Show creatures before turn
        alice_creatures = [c for c in game_state.cards if c.owner_id == "Alice" and c.zone == Zone.BATTLEFIELD and c.is_creature() and not c.summoning_sick]
        bob_creatures = [c for c in game_state.cards if c.owner_id == "Bob" and c.zone == Zone.BATTLEFIELD and c.is_creature() and not c.summoning_sick]
        print(f"Ready to attack: Alice={len(alice_creatures)}, Bob={len(bob_creatures)}")
        
        game_state = await runner._play_turn(game_state, agents)
        
        # Show life totals
        print(f"Life totals after turn: Alice={game_state.players[0].life_total}, Bob={game_state.players[1].life_total}")
