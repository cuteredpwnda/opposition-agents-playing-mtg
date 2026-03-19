"""
Game Execution Bridge for Agent Strategic Play.

This module connects agents to the game engine, allowing them to:
1. Observe game state via Neo4j knowledge graph
2. Evaluate strategic options using LLM (Ollama)
3. Execute plays (card plays, attacks, blocks)
4. React to opponent plays
5. Populate KG with reasoning chains

This is the execution layer that turns strategic decisions into actual game actions.
"""

from __future__ import annotations

from typing import Optional
from enum import Enum

from src.engine.game_state import GameState, Zone, CardInstance
from src.engine.knowledge_graph import MTGKnowledgeGraph
from src.engine.agent_strategies import AgentStrategist, Strategy, PlayRecommendation
from src.engine.llm_orchestration import MTGAgentLLM
from src.engine.llm_agent import LLMAgent, LLMDecision


class GamePhaseAction(str, Enum):
    """Actions an agent can take during a game phase."""
    PLAY_CARD = "play_card"
    ACTIVATE_ABILITY = "activate_ability"
    DECLARE_ATTACK = "declare_attack"
    DECLARE_BLOCK = "declare_block"
    PASS = "pass"
    RESPOND = "respond"


class AgentGamePlayer:
    """Connects an agent strategist to game execution."""
    
    def __init__(self, 
                 player_id: str,
                 strategy: Strategy = Strategy.AGGRESSIVE,
                 knowledge_graph: Optional[MTGKnowledgeGraph] = None,
                 use_llm: bool = False,
                 llm_model: str = "phi"):
        """Initialize an agent game player.
        
        Args:
            player_id: This player's ID
            strategy: Agent's play strategy
            knowledge_graph: Optional Neo4j KG for advanced queries
            use_llm: If True, use LLM for strategic decisions
            llm_model: Ollama model to use (default: phi for speed)
        """
        self.player_id = player_id
        self.strategy = strategy
        self.kg = knowledge_graph
        self.strategist = AgentStrategist(knowledge_graph, player_id, strategy) if knowledge_graph else None
        
        # Initialize LLM agent for decision-making
        strategy_map = {
            Strategy.AGGRESSIVE: "aggressive",
            Strategy.CONTROL: "control",
            Strategy.COMBO: "combo",
            Strategy.REACTIVE: "balanced"
        }
        self.llm_agent = MTGAgentLLM(
            player_id=player_id,
            strategy=strategy_map.get(strategy, "balanced"),
            ollama_model=llm_model,
            knowledge_graph=knowledge_graph
        ) if use_llm else None
    
    def get_plays_from_hand(self, game: GameState) -> list[CardInstance]:
        """Get list of cards from hand that can be played.
        
        Args:
            game: Current game state
            
        Returns:
            List of playable cards from hand
        """
        hand_cards = [
            c for c in game.cards
            if c.zone == Zone.HAND and c.controller_id == self.player_id
        ]
        return hand_cards
    
    def can_play_card(self, card: CardInstance, game: GameState) -> bool:
        """Check if a card can be played given current mana.
        
        Args:
            card: Card to check
            game: Current game state
            
        Returns:
            True if card can be cast
        """
        # Get player
        player = next((p for p in game.players if p.player_id == self.player_id), None)
        if not player:
            return False
        
        # For now, simplified check - just verify mana availability
        # Real implementation would check:
        # 1. Mana cost matches available mana
        # 2. Card type restrictions (e.g., sorcery only in main phase)
        # 3. Special restrictions (e.g., legendary limit, etc.)
        
        mana_cost_str = card.card_data.get("mana_cost", "")
        if not mana_cost_str:
            return True  # No cost = free to play
        
        # Placeholder: always true for testing
        # Real implementation would compare costs to player.mana_pool
        return True
    
    def decide_play_action(self, game: GameState, game_id: str) -> Optional[PlayRecommendation]:
        """Decide what to play during main phase.
        
        Uses strategist to evaluate best play if KG is available.
        
        Args:
            game: Current game state
            game_id: Game ID for KG queries
            
        Returns:
            PlayRecommendation or None if passing
        """
        if not self.strategist:
            return None
        
        try:
            # Get strategic recommendation
            action = self.strategist.get_next_action(game_id, game)
            return action
        except Exception as e:
            print(f"Strategy evaluation failed: {e}")
            return None
    
    def decide_attack_targets(self, game: GameState, game_id: str) -> list[str]:
        """Decide which creatures to attack with.
        
        Args:
            game: Current game state
            game_id: Game ID for KG queries
            
        Returns:
            List of creature IDs to attack with
        """
        if not self.strategist:
            return []
        
        try:
            board_state = self.strategist.evaluate_board_state(game_id)
            
            if not self.strategist.should_attack(board_state):
                return []
            
            # Get attackable creatures (not tapped, not summoning sick)
            creatures = board_state.get("our_creatures", [])
            attackers = [
                c.get("name") for c in creatures
                if not c.get("tapped", False) and not c.get("summoning_sick", False)
            ]
            
            return attackers
        except Exception as e:
            print(f"Attack evaluation failed: {e}")
            return []
    
    def decide_blocks(self, 
                      attacker: CardInstance,
                      game: GameState,
                      game_id: str) -> Optional[CardInstance]:
        """Decide if/how to block an attacker.
        
        Args:
            attacker: Attacking creature
            game: Current game state
            game_id: Game ID for KG queries
            
        Returns:
            Blocking creature or None
        """
        if not self.strategist:
            return None
        
        try:
            board_state = self.strategist.evaluate_board_state(game_id)
            
            if not self.strategist.should_block(board_state, {
                "power": attacker.power or 0,
                "toughness": attacker.toughness or 0
            }):
                return None
            
            # Find a suitable blocker
            our_creatures = [
                c for c in game.cards
                if c.zone == Zone.BATTLEFIELD
                and c.controller_id == self.player_id
                and not c.tapped
                and (c.toughness or 0) >= (attacker.power or 0)
            ]
            
            if our_creatures:
                return our_creatures[0]  # Pick first valid blocker
            
            return None
        except Exception as e:
            print(f"Block decision failed: {e}")
            return None
    
    def respond_to_spell(self, spell: CardInstance, game: GameState) -> bool:
        """Decide if agent wants to respond to opposing spell.
        
        Args:
            spell: Spell being cast
            game: Current game state
            
        Returns:
            True if agent wants to respond
        """
        # Placeholder for interrupt logic (counterspells, removal, etc.)
        # Real implementation would check for available response cards
        return False
    
    def get_priority(self, game: GameState) -> bool:
        """Decide what to do when having priority.
        
        Args:
            game: Current game state
            
        Returns:
            True if passing priority
        """
        # Default: pass after declaring actions
        # Real implementation would hold priority for:
        # - Responding to opponent spells
        # - Activating instant-speed abilities
        # - Casting cards with flash
        return True
    
    def get_strategy_summary(self) -> str:
        """Get text summary of this agent's strategy."""
        if not self.strategist:
            return f"Agent {self.player_id} (no strategy)"
        
        return f"Agent {self.player_id}: {self.strategist.get_strategy_brief()}"


class GameCoordinator:
    """Manages multi-agent game execution with strategic play."""
    
    def __init__(self, 
                 agent1: AgentGamePlayer,
                 agent2: AgentGamePlayer,
                 knowledge_graph: Optional[MTGKnowledgeGraph] = None):
        """Initialize game coordinator.
        
        Args:
            agent1: First agent player
            agent2: Second agent player
            knowledge_graph: Optional Neo4j KG for game analysis
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.kg = knowledge_graph
        self.game_id = None
        self.play_history = []
    
    def setup_game(self, game: GameState, game_id: str) -> None:
        """Initialize game and KG for execution.
        
        Args:
            game: Initial game state
            game_id: Unique game identifier
        """
        self.game_id = game_id
        
        if self.kg:
            self.kg.build_from_game_state(game, game_id)
    
    def get_active_agent(self, game: GameState) -> AgentGamePlayer:
        """Get current active agent.
        
        Args:
            game: Current game state
            
        Returns:
            The agent whose turn it is
        """
        return self.agent1 if game.active_player_index == 0 else self.agent2
    
    def execute_main_phase_plays(self, game: GameState) -> list[str]:
        """Let active agent play cards from hand.
        
        Args:
            game: Current game state
            
        Returns:
            List of action descriptions
        """
        actions = []
        agent = self.get_active_agent(game)
        opponent_id = self.agent2.player_id if agent.player_id == self.agent1.player_id else self.agent1.player_id
        
        # Try LLM first if available
        if agent.llm_agent and agent.llm_agent.llm.is_available:
            decision = agent.llm_agent.decide_main_phase_play(game, opponent_id)
            if decision:
                actions.append(f"{agent.player_id}: [LLM] {decision.action} - {decision.reasoning}")
                if self.kg and self.game_id:
                    agent.llm_agent.record_decision_to_kg(self.game_id, decision, game)
                return actions
        
        # Fall back to strategist if available
        if agent.strategist and self.game_id:
            recommendation = agent.decide_play_action(game, self.game_id)
            if recommendation:
                actions.append(f"{agent.player_id}: [KG] {recommendation.action_type} - {recommendation.reasoning}")
                self.play_history.append(recommendation)
            else:
                actions.append(f"{agent.player_id}: [KG] Pass")
            return actions
        
        # Fall back to simple heuristic
        hand_cards = agent.get_plays_from_hand(game)
        if hand_cards:
            playable = [c for c in hand_cards if agent.can_play_card(c, game)]
            if playable:
                playable.sort(key=lambda c: c.card_data.get("cmc", 0))
                card = playable[0]
                actions.append(f"{agent.player_id}: Play {card.card_data.get('name', 'card')}")
            else:
                actions.append(f"{agent.player_id}: Pass")
        else:
            actions.append(f"{agent.player_id}: Pass")
        
        return actions
    
    def execute_combat_phase(self, game: GameState) -> list[str]:
        """Let active agent declare attacks.
        
        Args:
            game: Current game state
            
        Returns:
            List of attack descriptions
        """
        actions = []
        agent = self.get_active_agent(game)
        
        # Try LLM first if available
        if agent.llm_agent and self.game_id:
            attackers = agent.llm_agent.decide_attack(game, self.game_id)
            if attackers and len(attackers) > 0:
                actions.append(f"{agent.player_id}: [LLM] Attacking with {len(attackers)} creatures")
            else:
                actions.append(f"{agent.player_id}: [LLM] No attacks")
            return actions
        
        # Fall back to strategist if available
        if agent.strategist and self.game_id:
            attackers = agent.decide_attack_targets(game, self.game_id)
            if attackers:
                actions.append(f"{agent.player_id}: [KG] Attacking with {len(attackers)} creatures")
            else:
                actions.append(f"{agent.player_id}: [KG] No attacks")
            return actions
        
        # Fall back to simple heuristic
        our_creatures = [
            c for c in game.cards
            if c.zone == Zone.BATTLEFIELD
            and c.controller_id == agent.player_id
            and not c.tapped
        ]
        
        if our_creatures:
            actions.append(f"{agent.player_id}: [Heuristic] Attacking with {len(our_creatures)} creatures")
        else:
            actions.append(f"{agent.player_id}: [Heuristic] No attacks")
        
        return actions
    
    def get_game_summary(self) -> str:
        """Get text summary of current game.
        
        Returns:
            Multi-line summary of agents and game state
        """
        summary = "=== Agent Game Summary ===\n"
        summary += self.agent1.get_strategy_summary() + "\n"
        summary += self.agent2.get_strategy_summary() + "\n"
        summary += f"Plays recorded: {len(self.play_history)}\n"
        return summary
