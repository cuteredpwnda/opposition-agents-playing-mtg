"""
Agent Strategies and Decision-Making Using Neo4j Knowledge Graph.

This module provides strategic decision-making capabilities for agents by querying
the Neo4j knowledge graph to evaluate board states, identify threats, and determine
optimal play sequences.

Strategy Types:
1. Aggressive: Maximize board presence and attack threats
2. Control: Manage opponent threats while building advantage
3. Combo: Build toward specific ability combinations
4. Reactive: Optimize based on opponent's board state
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from enum import Enum

from src.engine_legacy.game_state import GameState, Zone
from src.engine_legacy.knowledge_graph import MTGKnowledgeGraph


class Strategy(str, Enum):
    """Agent strategy types."""
    AGGRESSIVE = "aggressive"
    CONTROL = "control"
    COMBO = "combo"
    REACTIVE = "reactive"


class PlayEvaluation(str, Enum):
    """Evaluation of a play's quality."""
    CRITICAL = "critical"  # Must do now
    STRONG = "strong"  # Good play
    NEUTRAL = "neutral"  # Okay play
    WEAK = "weak"  # Suboptimal
    TRAP = "trap"  # Likely bad


@dataclass
class PlayRecommendation:
    """Recommendation for the agent's next action."""
    
    action_id: str  # Card instance ID or ability ID
    action_type: str  # "play_card", "activate_ability", "attack", "block"
    evaluation: PlayEvaluation
    reasoning: str
    confidence: float  # 0.0 to 1.0
    alternatives: list[str] = None  # Alternative action IDs


class AgentStrategist:
    """Uses Neo4j KG to make strategic decisions."""
    
    def __init__(self, knowledge_graph: MTGKnowledgeGraph, player_id: str, strategy: Strategy = Strategy.AGGRESSIVE):
        """Initialize strategist for a player.
        
        Args:
            knowledge_graph: Neo4j knowledge graph instance
            player_id: This player's ID
            strategy: Agent's play strategy
        """
        self.kg = knowledge_graph
        self.player_id = player_id
        self.strategy = strategy
    
    def evaluate_board_state(self, game_id: str) -> dict:
        """Evaluate the current board position using KG queries.
        
        Returns:
            Dict with board metrics (threats, resources, advantage)
        """
        # Query board position
        board = self.kg.query_board_evaluation(game_id)
        
        # Query our resources
        our_resources = self.kg.query_player_resources(game_id, self.player_id)
        
        # Query threats to us
        threats = self.kg.query_threats(game_id, self.player_id)
        
        # Query our creatures
        our_creatures = self.kg.query_cards_on_battlefield(game_id, player_id=self.player_id)
        
        return {
            "board": board,
            "our_resources": our_resources,
            "threats": threats,
            "our_creatures": our_creatures,
            "threat_count": len(threats),
            "total_threat_power": sum(t.get("power", 0) for t in threats) if threats else 0,
            "our_creature_count": len(our_creatures) if our_creatures else 0
        }
    
    def should_attack(self, board_state: dict) -> bool:
        """Determine if we should attack based on board state.
        
        Strategy-dependent logic:
        - AGGRESSIVE: Attack if we have power advantage
        - CONTROL: Attack if safe and beneficial  
        - COMBO: Attack only if enables combo setup
        - REACTIVE: Attack if opponent threat is low
        """
        our_board = board_state.get("board", {})
        our_creatures = board_state.get("our_creatures", [])
        threats = board_state.get("threats", [])
        
        if not our_creatures:
            return False
        
        our_power = sum(c.get("power", 0) for c in our_creatures)
        threat_power = board_state.get("total_threat_power", 0)
        
        if self.strategy == Strategy.AGGRESSIVE:
            # Attack if we have advantage or equal board
            return our_power >= threat_power or len(our_creatures) >= len(threats)
        
        elif self.strategy == Strategy.CONTROL:
            # Attack cautiously, only if threat is minimal
            return threat_power == 0 or our_power >= threat_power * 1.5
        
        elif self.strategy == Strategy.COMBO:
            # Attack only if we have card advantage (hand full)
            hand_size = board_state.get("our_resources", {}).get("hand_size", 0)
            return hand_size >= 4 and our_power >= threat_power
        
        elif self.strategy == Strategy.REACTIVE:
            # Attack if opponent has no threats
            return len(threats) == 0
        
        return False
    
    def should_block(self, board_state: dict, incoming_threat: dict) -> bool:
        """Determine if we should block an attacker.
        
        Args:
            board_state: Current board evaluation
            incoming_threat: Creature attacking us
            
        Returns:
            True if we should block with a creature
        """
        our_creatures = board_state.get("our_creatures", [])
        threat_power = incoming_threat.get("power", 0)
        
        if not our_creatures:
            return False
        
        # Find if we have a blocker
        blocker = next(
            (c for c in our_creatures if c.get("toughness", 0) >= threat_power),
            None
        )
        
        if not blocker:
            return False
        
        # Strategy-specific block logic
        if self.strategy == Strategy.AGGRESSIVE:
            # Don't block unless necessary to preserve board
            return False
        
        elif self.strategy == Strategy.CONTROL:
            # Block to remove threats
            return True
        
        elif self.strategy == Strategy.COMBO:
            # Block if it's cheap (doesn't disrupt combo setup)
            return not blocker.get("tapped", False)
        
        elif self.strategy == Strategy.REACTIVE:
            # Block available threats
            return True
        
        return False
    
    def evaluate_play(self, card_name: str, board_state: dict) -> PlayRecommendation:
        """Evaluate if we should play a card from hand.
        
        Args:
            card_name: Name of card in hand
            board_state: Current board evaluation
            
        Returns:
            PlayRecommendation with evaluation and reasoning
        """
        # Base evaluation on strategy
        threats = board_state.get("threats", [])
        our_power = sum(c.get("power", 0) for c in board_state.get("our_creatures", []))
        threat_power = board_state.get("total_threat_power", 0)
        
        if self.strategy == Strategy.AGGRESSIVE:
            if "lord" in card_name.lower() or "anthem" in card_name.lower():
                return PlayRecommendation(
                    action_id=card_name,
                    action_type="play_card",
                    evaluation=PlayEvaluation.STRONG,
                    reasoning="Lord effects boost board presence",
                    confidence=0.85
                )
            elif "creature" in card_name.lower().split():
                return PlayRecommendation(
                    action_id=card_name,
                    action_type="play_card",
                    evaluation=PlayEvaluation.STRONG,
                    reasoning="Build board to attack",
                    confidence=0.75
                )
        
        elif self.strategy == Strategy.CONTROL:
            # Recognize common removal patterns
            removal_keywords = ["removal", "bolt", "shock", "kill", "destroy", "exile", "bounce"]
            is_removal = any(kw in card_name.lower() for kw in removal_keywords)
            
            if is_removal:
                if threats:
                    return PlayRecommendation(
                        action_id=card_name,
                        action_type="play_card",
                        evaluation=PlayEvaluation.CRITICAL,
                        reasoning="Remove major threat immediately",
                        confidence=0.95
                    )
                else:
                    return PlayRecommendation(
                        action_id=card_name,
                        action_type="play_card",
                        evaluation=PlayEvaluation.STRONG,
                        reasoning="Hold removal for future threats",
                        confidence=0.70
                    )
            elif "draw" in card_name.lower():
                return PlayRecommendation(
                    action_id=card_name,
                    action_type="play_card",
                    evaluation=PlayEvaluation.STRONG,
                    reasoning="Maintain card advantage",
                    confidence=0.80
                )
        
        # Default: play if mana available
        return PlayRecommendation(
            action_id=card_name,
            action_type="play_card",
            evaluation=PlayEvaluation.NEUTRAL,
            reasoning="No strategic reason to hold",
            confidence=0.50
        )
    
    def get_next_action(self, game_id: str, game: GameState) -> Optional[PlayRecommendation]:
        """Get the next recommended action for this player.
        
        Evaluates:
        1. Board state
        2. Hand options
        3. Ability activations
        4. Attack/block decisions
        
        Args:
            game_id: Game ID in knowledge graph
            game: Current GameState
            
        Returns:
            PlayRecommendation or None if no action
        """
        try:
            board_state = self.evaluate_board_state(game_id)
        except Exception as e:
            # If KG query fails, fall back to simple evaluation
            return None
        
        # Prioritize plays based on strategy
        if self.strategy == Strategy.CONTROL and board_state.get("threat_count", 0) > 0:
            # Look for removal spells
            return PlayRecommendation(
                action_id="seek_removal",
                action_type="search_hand",
                evaluation=PlayEvaluation.CRITICAL,
                reasoning="Control strategy: neutralize threats first",
                confidence=0.90
            )
        
        if self.should_attack(board_state):
            return PlayRecommendation(
                action_id="declare_attackers",
                action_type="attack",
                evaluation=PlayEvaluation.STRONG,
                reasoning=f"Board advantage detected: {board_state.get('our_creature_count', 0)} creatures",
                confidence=0.75
            )
        
        # Default: continue game flow
        return None
    
    def get_strategy_brief(self) -> str:
        """Get a text description of this agent's strategy."""
        descriptions = {
            Strategy.AGGRESSIVE: "Maximize board presence and attack damage. Prioritize creatures and lords.",
            Strategy.CONTROL: "Remove threats and manage board. Prioritize removal and card draw.",
            Strategy.COMBO: "Build toward ability combinations. Hold resources for setup turns.",
            Strategy.REACTIVE: "Respond to opponent's plays. Adapt strategy based on threats."
        }
        return descriptions.get(self.strategy, "Unknown strategy")
