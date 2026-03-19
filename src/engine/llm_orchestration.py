"""
LLM Orchestration via Ollama for MTG Agent Decision-Making.

This module provides LLM-powered reasoning for agents:
- Connects to Ollama API (localhost:11434 by default)
- Generates strategic decisions for main phase, combat, blocks
- Populates Neo4j KG with reasoning chain
- Supports any Ollama model (default: mistral)
"""

from __future__ import annotations

import json
import requests
from typing import Optional
from dataclasses import dataclass

from src.engine.game_state import GameState, Zone, CardInstance
from src.engine.knowledge_graph import MTGKnowledgeGraph


@dataclass
class LLMDecision:
    """LLM-powered game decision."""
    action: str  # "play_card", "attack", "pass", etc.
    card_or_target: str  # Card name or target
    reasoning: str  # Why this decision
    confidence: float  # 0.0-1.0 confidence


class OllamaConnector:
    """Manages connection to Ollama API for LLM inference."""
    
    def __init__(self, 
                 model: str = "mistral",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 60):
        """Initialize Ollama connector.
        
        Args:
            model: Ollama model name (mistral, llama2, neural-chat, etc.)
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.is_available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Warning: Ollama not available at {self.base_url}: {e}")
            return False
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate text from Ollama.
        
        Args:
            prompt: Input prompt for model
            stream: Whether to stream response
            
        Returns:
            Generated text
        """
        if not self.is_available:
            return ""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": stream,
                    "temperature": 0.7,
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                full_response = ""
                if stream:
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            full_response += data.get("response", "")
                else:
                    data = response.json()
                    full_response = data.get("response", "")
                return full_response.strip()
            else:
                print(f"Ollama error: {response.status_code}")
                return ""
        except Exception as e:
            print(f"Ollama request failed: {e}")
            return ""


class MTGAgentLLM:
    """LLM-powered MTG agent strategist using Ollama."""
    
    def __init__(self, 
                 player_id: str,
                 strategy: str = "balanced",
                 ollama_model: str = "mistral",
                 knowledge_graph: Optional[MTGKnowledgeGraph] = None):
        """Initialize LLM agent.
        
        Args:
            player_id: This player's ID
            strategy: Play strategy (aggressive, control, combo, balanced)
            ollama_model: Ollama model to use
            knowledge_graph: Optional Neo4j KG for context
        """
        self.player_id = player_id
        self.strategy = strategy
        self.kg = knowledge_graph
        self.llm = OllamaConnector(model=ollama_model)
        self.decision_history = []
    
    def get_hand_summary(self, game: GameState) -> str:
        """Get text summary of player's hand.
        
        Args:
            game: Current game state
            
        Returns:
            Text description of hand
        """
        hand = [
            c for c in game.cards
            if c.zone == Zone.HAND and c.controller_id == self.player_id
        ]
        
        if not hand:
            return "Empty hand"
        
        summary = []
        for card in hand:
            name = card.card_data.get("name", "Unknown")
            mana_cost = card.card_data.get("mana_cost", "0")
            card_type = card.card_data.get("type_line", "")
            summary.append(f"- {name} {mana_cost} ({card_type})")
        
        return "\n".join(summary)
    
    def get_board_summary(self, game: GameState, opponent_id: str) -> str:
        """Get text summary of board state.
        
        Args:
            game: Current game state
            opponent_id: Opponent's player ID
            
        Returns:
            Text description of board
        """
        our_creatures = [
            c for c in game.cards
            if c.zone == Zone.BATTLEFIELD
            and c.controller_id == self.player_id
            and "Creature" in c.card_data.get("type_line", "")
        ]
        
        opp_creatures = [
            c for c in game.cards
            if c.zone == Zone.BATTLEFIELD
            and c.controller_id == opponent_id
            and "Creature" in c.card_data.get("type_line", "")
        ]
        
        our_life = next((p.life_total for p in game.players if p.player_id == self.player_id), 20)
        opp_life = next((p.life_total for p in game.players if p.player_id == opponent_id), 20)
        
        summary = f"Our creatures ({len(our_creatures)}): "
        if our_creatures:
            summary += ", ".join([c.card_data.get("name", "?") for c in our_creatures])
        else:
            summary += "None"
        
        summary += f"\nOpponent creatures ({len(opp_creatures)}): "
        if opp_creatures:
            summary += ", ".join([c.card_data.get("name", "?") for c in opp_creatures])
        else:
            summary += "None"
        
        summary += f"\nLife totals: You {our_life}, Opponent {opp_life}"
        
        return summary
    
    def decide_main_phase_play(self, game: GameState, opponent_id: str) -> Optional[LLMDecision]:
        """Use LLM to decide what to play in main phase.
        
        Args:
            game: Current game state
            opponent_id: Opponent's player ID
            
        Returns:
            LLMDecision or None if passing
        """
        if not self.llm.is_available:
            return None
        
        hand = self.get_hand_summary(game)
        board = self.get_board_summary(game, opponent_id)
        
        prompt = f"""You are a Magic: The Gathering player with a {self.strategy} strategy.

Your hand:
{hand}

Current board state:
{board}

Considering your {self.strategy} strategy, what should you play?

Format your response as:
ACTION: [play_card/pass]
CARD: [card name if playing]
REASONING: [brief explanation]

RESPONSE:"""
        
        response = self.llm.generate(prompt)
        
        if not response:
            return None
        
        return self._parse_llm_response(response, "play")
    
    def decide_combat(self, game: GameState, opponent_id: str) -> Optional[LLMDecision]:
        """Use LLM to decide combat strategy.
        
        Args:
            game: Current game state
            opponent_id: Opponent's player ID
            
        Returns:
            LLMDecision or None if not attacking
        """
        if not self.llm.is_available:
            return None
        
        board = self.get_board_summary(game, opponent_id)
        
        our_creatures = [
            c for c in game.cards
            if c.zone == Zone.BATTLEFIELD
            and c.controller_id == self.player_id
            and "Creature" in c.card_data.get("type_line", "")
            and not c.tapped
        ]
        
        creature_list = ", ".join([c.card_data.get("name", "?") for c in our_creatures])
        
        prompt = f"""You are a Magic: The Gathering player with a {self.strategy} strategy.

Current board state:
{board}

Your untapped creatures: {creature_list}

Considering your {self.strategy} strategy, which creatures should you attack with?

Format your response as:
ACTION: [attack/pass]
TARGETS: [creature names, comma-separated or "all" or "none"]
REASONING: [brief explanation]

RESPONSE:"""
        
        response = self.llm.generate(prompt)
        
        if not response:
            return None
        
        return self._parse_llm_response(response, "combat")
    
    def _parse_llm_response(self, response: str, context: str) -> Optional[LLMDecision]:
        """Parse LLM response into structured decision.
        
        Args:
            response: Raw LLM response
            context: "play" or "combat"
            
        Returns:
            LLMDecision or None
        """
        try:
            lines = response.split("\n")
            action = ""
            target = ""
            reasoning = ""
            
            for line in lines:
                if "ACTION:" in line:
                    action = line.split("ACTION:")[-1].strip().lower()
                elif "CARD:" in line or "TARGETS:" in line:
                    target = line.split(":")[-1].strip()
                elif "REASONING:" in line:
                    reasoning = line.split("REASONING:")[-1].strip()
            
            if not action:
                return None
            
            decision = LLMDecision(
                action=action,
                card_or_target=target,
                reasoning=reasoning,
                confidence=0.7  # Default confidence for LLM decisions
            )
            
            self.decision_history.append(decision)
            return decision
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            return None
    
    def record_decision_to_kg(self, 
                             game_id: str,
                             decision: LLMDecision,
                             game_state: GameState) -> None:
        """Record LLM decision to knowledge graph.
        
        Args:
            game_id: Game ID
            decision: The decision made
            game_state: Current game state
        """
        if not self.kg:
            return
        
        try:
            # Record decision node
            self.kg.query(f"""
                CREATE (d:Decision {{
                    decision_id: '{game_id}_{len(self.decision_history)}',
                    game_id: '{game_id}',
                    player_id: '{self.player_id}',
                    action: '{decision.action}',
                    target: '{decision.card_or_target}',
                    reasoning: '{decision.reasoning}',
                    confidence: {decision.confidence},
                    timestamp: datetime.now()
                }})
            """)
            
            # Link to game
            self.kg.query(f"""
                MATCH (g:Game {{game_id: '{game_id}'}})
                CREATE (g)-[:HAS_DECISION]->(d:Decision {{decision_id: '{game_id}_{len(self.decision_history)}'}})
            """)
        except Exception as e:
            print(f"Failed to record decision to KG: {e}")
    
    def get_decision_summary(self) -> str:
        """Get summary of decisions made.
        
        Returns:
            Text summary
        """
        if not self.decision_history:
            return f"Agent {self.player_id}: No decisions recorded"
        
        summary = f"Agent {self.player_id} ({self.strategy}):\n"
        for i, decision in enumerate(self.decision_history, 1):
            summary += f"  {i}. {decision.action}: {decision.card_or_target}\n"
        
        return summary
