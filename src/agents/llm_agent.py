"""
LLM-powered MTG agent using LangChain tool-calling.

Reference: mtg-player (MIT) — https://github.com/Leamas2006/mtg-player
Uses the ReAct tool-calling pattern with LangChain ChatModel.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import MTGAgent
from src.engine.game_state import Action, GameState
from src.knowledge.knowledge_graph import MTGKnowledgeGraph

SYSTEM_PROMPT = """\
You are an expert Magic: The Gathering player competing in a game.
You have access to tools for querying the knowledge graph, evaluating
board positions, checking combat math, and calling the judge.

Given the current game state and your legal actions, choose the best play.
Consider:
- Your win condition and current game plan
- Opponent's likely archetype and hand contents
- Combo availability (check the knowledge graph)
- Mana efficiency and tempo
- Information gain vs. immediate value (active inference)

Respond with EXACTLY one action from the legal actions list.
"""


class LLMAgent(MTGAgent):
    """Agent that uses an LLM with tool-calling to make decisions.

    The LLM receives the game state as context and has access to
    LangChain tools for KG queries, combat math, and judge calls.
    """

    def __init__(
        self,
        player_id: str,
        llm: BaseChatModel | None = None,
        kg: MTGKnowledgeGraph | None = None,
        tools: list | None = None,
    ):
        super().__init__(player_id)
        # Lazy init to avoid requiring LLM credentials at import time
        self.llm = llm
        self.kg = kg
        self.tools = tools or []
        self.llm_with_tools = None
        if self.llm and self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        elif self.llm:
            self.llm_with_tools = self.llm

    async def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        """Choose an action from legal actions."""
        if not self.llm:
            # Fallback: pick first action if no LLM available
            return legal_actions[0] if legal_actions else None
        
        context = self._build_context(game_state, legal_actions)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]

        try:
            response = await self.llm_with_tools.ainvoke(messages)
            chosen_index = self._parse_action_index(response, legal_actions)
            return legal_actions[chosen_index]
        except Exception as e:
            # Fallback on error: return first legal action
            return legal_actions[0] if legal_actions else None

    def _build_context(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> str:
        """Format game state and legal actions for the LLM."""
        from src.engine.game_state import Zone
        
        lines = [
            f"Turn {game_state.turn_number}, Phase: {game_state.phase.value}",
            f"Active player: {game_state.players[game_state.active_player_index].name}",
            f"You are: {self.player_id}",
            "",
        ]

        # Show each player's state
        for player in game_state.players:
            prefix = "(YOU) " if player.player_id == self.player_id else ""
            lines.append(f"{prefix}Player {player.name}:")
            lines.append(f"  Life: {player.life_total}")
            
            # Count cards by zone
            hand = [c for c in game_state.cards if c.zone == Zone.HAND and c.owner_id == player.player_id]
            bf = [c for c in game_state.cards if c.zone == Zone.BATTLEFIELD and c.owner_id == player.player_id]
            lines.append(f"  Hand size: {len(hand)}")
            if bf:
                lines.append(f"  Battlefield: {', '.join(c.name for c in bf[:5])}")
                if len(bf) > 5:
                    lines.append(f"    ... and {len(bf)-5} more")

        # Show own hand
        hand = [c for c in game_state.cards if c.zone == Zone.HAND and c.owner_id == self.player_id]
        lines.append(f"\nYour hand ({len(hand)} cards):")
        for card in hand[:10]:  # Show first 10 cards
            lines.append(f"  - {card.name} ({card.mana_cost})")
        if len(hand) > 10:
            lines.append(f"  ... and {len(hand)-10} more")

        lines.append(f"\nLegal actions ({len(legal_actions)} total):")
        for i, action in enumerate(legal_actions[:15]):  # Show first 15 actions
            card_name = ""
            if action.card_instance_id:
                card = next((c for c in game_state.cards if c.instance_id == action.card_instance_id), None)
                if card:
                    card_name = f" {card.name}"
            lines.append(f"  [{i}] {action.action_type.value}{card_name}")
        if len(legal_actions) > 15:
            lines.append(f"  ... and {len(legal_actions)-15} more actions")

        lines.append("\nRespond with the best action index number (e.g., '5').")
        return "\n".join(lines)
        return "\n".join(lines)

    @staticmethod
    def _parse_action_index(response, legal_actions: list[Action]) -> int:
        """Extract the chosen action index from the LLM response."""
        # Handle both AIMessage and string responses
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Try to find first integer in response
        for token in content.split():
            stripped = token.strip("[]().,")
            if stripped.isdigit():
                idx = int(stripped)
                if 0 <= idx < len(legal_actions):
                    return idx
        
        # Default to first non-pass action if available, else 0
        for i, action in enumerate(legal_actions):
            if action.action_type.value != "pass_priority":
                return i
        return 0
