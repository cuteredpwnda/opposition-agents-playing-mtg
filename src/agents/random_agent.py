"""
Random agent — picks legal actions with bias toward playing threats. 
Includes heuristic fallback for when no strategy available.

Reference: mtg-player (MIT) — https://github.com/theRealMarkCastillo/mtg-player
Reference: open-mtg (MIT) — https://github.com/hlynurd/open-mtg (MCTS patterns)
"""

from __future__ import annotations

import random

from src.agents.base_agent import MTGAgent
from src.engine.game_state import Action, GameState, ActionType


class RandomAgent(MTGAgent):
    """Baseline agent that picks legal actions with strategic bias."""

    async def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        """Choose an action with bias: prefer CAST_SPELL > PLAY_LAND > ACTIVATE_ABILITY > PASS"""
        if not legal_actions:
            return Action(action_type=ActionType.PASS_PRIORITY, player_id=self.player_id)
        
        # Weight actions by type
        weighted = []
        for action in legal_actions:
            if action.action_type == ActionType.CAST_SPELL:
                weight = 10  # Strongly prefer casting creatures/spells
            elif action.action_type == ActionType.DECLARE_ATTACKERS:
                weight = 8   # High priority for attacking

                # Promote attacking low-life or high-commander-damage opponents
                if action.targets:
                    target_id = action.targets[0]
                    weight += self._score_attack_target(game_state, target_id)

            elif action.action_type == ActionType.PLAY_LAND:
                weight = 5   # Prefer playing lands
            elif action.action_type == ActionType.ACTIVATE_ABILITY:
                weight = 4   # Tap lands for mana (increased priority)
            elif action.action_type == ActionType.PASS_PRIORITY:
                weight = 1   # Pass is last resort
            elif action.action_type == ActionType.CONCEDE:
                weight = 0   # Avoid conceding unless forced
            else:
                weight = 1
            
            weighted.extend([action] * weight)
        
        chosen = random.choice(weighted)

        # Log action score data
        if chosen.action_type == ActionType.DECLARE_ATTACKERS and chosen.targets:
            chosen.metadata["decision_mode"] = "random_weighted"
            chosen.metadata["target_score"] = self._score_attack_target(game_state, chosen.targets[0])

        return chosen
    
    def get_heuristic_score(self, game_state: GameState) -> float:
        """Heuristic evaluation: life total + creatures on battlefield.
        
        Used as fallback when no strategic info available.
        Range: -100 (losing) to +100 (winning).
        """
        from src.engine.game_state import Zone
        
        player = next((p for p in game_state.players if p.player_id == self.player_id), None)
        if not player:
            return 0.0
        
        my_creatures = len([c for c in game_state.cards 
                           if c.owner_id == self.player_id 
                           and c.zone == Zone.BATTLEFIELD 
                           and c.is_creature()])
        
        opp_creatures = len([c for c in game_state.cards 
                            if c.owner_id != self.player_id 
                            and c.zone == Zone.BATTLEFIELD 
                            and c.is_creature()])
        
        # Simple heuristic: (my_life - opp_life) + (my_creatures * 3 - opp_creatures * 3)
        my_opponent = next((p for p in game_state.players if p.player_id != self.player_id), None)
        if not my_opponent:
            return float(player.life_total)
        
        life_diff = player.life_total - my_opponent.life_total
        board_advantage = (my_creatures - opp_creatures) * 3
        
        return float(life_diff + board_advantage)

    def _score_attack_target(self, game_state: GameState, target_id: str) -> float:
        target = next((p for p in game_state.players if p.player_id == target_id), None)
        if target is None:
            return 0.0

        score = max(0, 40 - target.life_total)
        commander_dmg = 0
        if hasattr(target, "commander_damage_received"):
            commander_dmg = sum(target.commander_damage_received.values())

        score += commander_dmg * 2
        return float(score)
