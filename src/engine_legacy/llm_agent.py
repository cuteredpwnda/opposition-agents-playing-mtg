"""
LLM-powered agent for strategic MTG decision making.

Uses Claude AI (via Anthropic API) to evaluate game state and make strategic decisions.
Provides fallback to simple heuristics if LLM is unavailable.
"""

from __future__ import annotations

import os
import json
import re
from typing import Optional
from dataclasses import dataclass

from src.engine.game_state import GameState, Zone, CardInstance
from src.engine.agent_strategies import Strategy

# Try to import Anthropic client
try:
    from anthropic import Anthropic
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


@dataclass
class LLMDecision:
    """Result of LLM decision making."""
    action_type: str  # "play", "attack", "pass"
    card_name: Optional[str] = None
    reasoning: Optional[str] = None
    confidence: float = 0.8


class LLMAgent:
    """Agent powered by Claude LLM for strategic decision making."""
    
    def __init__(self, 
                 player_id: str,
                 strategy: Strategy = Strategy.AGGRESSIVE,
                 api_key: Optional[str] = None):
        """Initialize LLM agent.
        
        Args:
            player_id: Player identifier
            strategy: Preferred play strategy
            api_key: Anthropic API key (reads from ANTHROPIC_API_KEY if not provided)
        """
        self.player_id = player_id
        self.strategy = strategy
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        
        if self.api_key and LLM_AVAILABLE:
            self.client = Anthropic(api_key=self.api_key)
    
    def _format_game_state(self, game: GameState) -> str:
        """Format game state into readable text for LLM.
        
        Args:
            game: Current game state
            
        Returns:
            Natural language description of game state
        """
        our_player = next(
            (p for p in game.players if p.player_id == self.player_id),
            None
        )
        opp_player = next(
            (p for p in game.players if p.player_id != self.player_id),
            None
        )
        
        if not our_player or not opp_player:
            return "Game state unavailable"
        
        # Format our resources
        our_hand = [
            c for c in game.cards
            if c.zone == Zone.HAND and c.controller_id == self.player_id
        ]
        our_creatures = [
            c for c in game.cards
            if c.zone == Zone.BATTLEFIELD
            and c.controller_id == self.player_id
            and "Creature" in c.card_data.get("type_line", "")
        ]
        our_lands = [
            c for c in game.cards
            if c.zone == Zone.BATTLEFIELD
            and c.controller_id == self.player_id
            and "Land" in c.card_data.get("type_line", "")
        ]
        
        # Format opponent resources
        opp_creatures = [
            c for c in game.cards
            if c.zone == Zone.BATTLEFIELD
            and c.controller_id == opp_player.player_id
            and "Creature" in c.card_data.get("type_line", "")
        ]
        
        description = f"""
CURRENT GAME STATE (Turn {game.turn_number}):

OUR PLAYER ({self.player_id}):
- Life Total: {our_player.life_total} HP
- Hand: {len(our_hand)} cards
  Cards: {', '.join([c.card_data.get('name', 'Unknown') for c in our_hand[:5]])}
  {('...' + str(len(our_hand) - 5) + ' more cards') if len(our_hand) > 5 else ''}
- Battlefield: {len(our_creatures)} creatures, {len(our_lands)} lands
  Creatures: {', '.join([f"{c.card_data.get('name', 'Unknown')} ({c.power}/{c.toughness})" for c in our_creatures[:3]])}
  {('...' + str(len(our_creatures) - 3) + ' more creatures') if len(our_creatures) > 3 else ''}

OPPONENT ({opp_player.player_id}):
- Life Total: {opp_player.life_total} HP
- Creatures: {len(opp_creatures)} on battlefield
  {', '.join([f"{c.card_data.get('name', 'Unknown')} ({c.power}/{c.toughness})" for c in opp_creatures[:3]])}
  {('...' + str(len(opp_creatures) - 3) + ' more creatures') if len(opp_creatures) > 3 else ''}

STRATEGY: Play {"aggressively - prioritize attacking and dealing damage" if self.strategy == Strategy.AGGRESSIVE else "defensively - prioritize board control and survival" if self.strategy == Strategy.CONTROL else "combo-focused" if self.strategy == Strategy.COMBO else "reactively - respond to threats"}
"""
        return description
    
    def decide_main_phase_play(self, game: GameState, game_id: str) -> Optional[LLMDecision]:
        """Use LLM to decide what to play from hand.
        
        Args:
            game: Current game state
            game_id: Game identifier
            
        Returns:
            LLMDecision with card to play or pass
        """
        if not self.client:
            return None
        
        our_hand = [
            c for c in game.cards
            if c.zone == Zone.HAND and c.controller_id == self.player_id
        ]
        
        if not our_hand:
            return LLMDecision("pass", reasoning="No cards in hand")
        
        game_state_desc = self._format_game_state(game)
        
        prompt = f"""{game_state_desc}

You are playing Magic: The Gathering. Based on your {self.strategy.value} strategy, which card should you play from your hand?

Available cards in hand:
{json.dumps([{
    'name': c.card_data.get('name', 'Unknown'),
    'cost': c.card_data.get('mana_cost', '0'),
    'type': c.card_data.get('type_line', 'Unknown'),
    'oracle': c.card_data.get('oracle_text', '')[:100]
} for c in our_hand], indent=2)}

Respond in JSON format:
{{
    "action": "play" or "pass",
    "card_name": "card name if playing, otherwise null",
    "reasoning": "brief explanation of your decision"
}}

Make your best strategic decision."""
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            response_text = response.content[0].text
            
            # Try to extract JSON
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                decision_data = json.loads(json_match.group())
                
                if decision_data.get("action") == "play" and decision_data.get("card_name"):
                    # Find the card in hand
                    card = next(
                        (c for c in our_hand if c.card_data.get("name") == decision_data.get("card_name")),
                        None
                    )
                    if card:
                        return LLMDecision(
                            action_type="play",
                            card_name=card.card_data.get("name"),
                            reasoning=decision_data.get("reasoning", "LLM decided to play this card"),
                            confidence=0.85
                        )
                
                return LLMDecision(
                    action_type="pass",
                    reasoning=decision_data.get("reasoning", "LLM decided to pass"),
                    confidence=0.85
                )
            
            # Fallback parsing if no JSON found
            if "pass" in response_text.lower():
                return LLMDecision("pass", reasoning="LLM recommended passing")
            
        except Exception as e:
            print(f"LLM decision error: {e}")
            return None
        
        return None
    
    def decide_attack(self, game: GameState, game_id: str) -> list[str]:
        """Use LLM to decide which creatures to attack with.
        
        Args:
            game: Current game state
            game_id: Game identifier
            
        Returns:
            List of creature names to attack with
        """
        if not self.client:
            return []
        
        our_creatures = [
            c for c in game.cards
            if c.zone == Zone.BATTLEFIELD
            and c.controller_id == self.player_id
            and "Creature" in c.card_data.get("type_line", "")
            and not c.tapped
        ]
        
        if not our_creatures:
            return []
        
        opp_info = ""
        opp_player = next((p for p in game.players if p.player_id != self.player_id), None)
        if opp_player:
            opp_info = f"Opponent life total: {opp_player.life_total}\n"
        
        prompt = f"""You are playing Magic: The Gathering with a {self.strategy.value} strategy.

{self.strategy.value} strategy means: {"Attack aggressively to deal damage" if self.strategy == Strategy.AGGRESSIVE else "Attack cautiously to maintain board control"}

{opp_info}

Your creatures that can attack:
{json.dumps([{
    'name': c.card_data.get('name', 'Unknown'),
    'power': c.power or 0,
    'toughness': c.toughness or 0
} for c in our_creatures], indent=2)}

Should you attack? If yes, with which creatures? Respond in JSON:
{{
    "should_attack": true/false,
    "creatures": ["creature name 1", "creature name 2"],
    "reasoning": "brief explanation"
}}"""
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            
            if json_match:
                decision_data = json.loads(json_match.group())
                if decision_data.get("should_attack"):
                    return decision_data.get("creatures", [])
            
        except Exception as e:
            print(f"LLM attack decision error: {e}")
        
        return []
