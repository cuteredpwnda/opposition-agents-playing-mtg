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
        llm: BaseChatModel,
        kg: MTGKnowledgeGraph,
        tools: list | None = None,
    ):
        super().__init__(player_id)
        self.llm = llm
        self.kg = kg
        self.tools = tools or []
        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm

    async def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        context = self._build_context(game_state, legal_actions)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]

        response = await self.llm_with_tools.ainvoke(messages)
        chosen_index = self._parse_action_index(response.content, legal_actions)
        return legal_actions[chosen_index]

    def _build_context(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> str:
        """Format game state and legal actions for the LLM."""
        lines = [
            f"Turn {game_state.turn_number}, Phase: {game_state.phase.value}",
            f"Active player: {game_state.active_player}",
            f"Priority: {game_state.priority_player}",
            f"You are: {self.player_id}",
            "",
        ]

        for pid, pstate in game_state.players.items():
            prefix = "(YOU) " if pid == self.player_id else ""
            lines.append(f"{prefix}Player {pid}:")
            lines.append(f"  Life: {pstate.life}")
            lines.append(f"  Hand size: {len(game_state.cards_in_zone.get((pid, 'hand'), []))}")
            bf = game_state.cards_in_zone.get((pid, "battlefield"), [])
            if bf:
                lines.append(f"  Battlefield: {', '.join(c.name for c in bf)}")

        # Show own hand
        hand = game_state.cards_in_zone.get((self.player_id, "hand"), [])
        lines.append(f"\nYour hand: {', '.join(c.name for c in hand)}")

        lines.append("\nLegal actions:")
        for i, a in enumerate(legal_actions):
            card_name = a.card.name if a.card else ""
            lines.append(f"  [{i}] {a.action_type.value} {card_name}")

        lines.append("\nRespond with the action index number.")
        return "\n".join(lines)

    @staticmethod
    def _parse_action_index(response: str, legal_actions: list[Action]) -> int:
        """Extract the chosen action index from the LLM response."""
        # Try to find first integer in response
        for token in response.split():
            stripped = token.strip("[]().,")
            if stripped.isdigit():
                idx = int(stripped)
                if 0 <= idx < len(legal_actions):
                    return idx
        # Default to first action (pass priority) if parsing fails
        return 0
