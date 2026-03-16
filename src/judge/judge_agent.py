"""
LLM-based judge agent with RAG over MTG Comprehensive Rules.

Called when complex stack interactions, ambiguous card text, state-based
action edge cases, or layer system conflicts arise.

Reference: Section 10.2 of PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.engine.game_state import GameState
from src.knowledge.knowledge_graph import MTGKnowledgeGraph


@dataclass
class Ruling:
    """A judge ruling on a game situation."""

    outcome: str
    rule_numbers: list[str]
    resolution_steps: list[str]
    confidence: str  # "high" | "medium" | "low"
    reasoning: str


JUDGE_SYSTEM_PROMPT = """\
You are an MTG Level 3 Judge. You have access to the MTG Comprehensive Rules,
card-specific rulings from Scryfall, and a knowledge graph of card interactions.

When ruling on a situation:
1. Identify the specific rules that apply (cite rule numbers)
2. Determine the correct resolution order
3. State your ruling clearly
4. Indicate your confidence level (high/medium/low)

Be precise and reference exact rule numbers (e.g., CR 704.5c, CR 117.3a).
"""


class JudgeAgent:
    """LLM-based rules arbiter with RAG over the Comprehensive Rules.

    Has access to:
    1. MTG Comprehensive Rules vectorstore (FAISS)
    2. Scryfall rulings API (per-card official rulings)
    3. Knowledge graph interaction edges
    """

    def __init__(
        self,
        llm: BaseChatModel,
        rules_vectorstore: Any,  # FAISS or similar
        kg: MTGKnowledgeGraph,
        scryfall_client: Any | None = None,
    ):
        self.llm = llm
        self.rules_rag = rules_vectorstore
        self.kg = kg
        self.scryfall = scryfall_client

    async def rule_on(
        self, situation: str, game_state: GameState | None = None
    ) -> Ruling:
        """Produce a ruling for a given game situation."""
        # 1. Retrieve relevant comprehensive rules sections
        relevant_rules = self.rules_rag.similarity_search(situation, k=10)

        # 2. Card-specific rulings from Scryfall (if client available)
        card_rulings: list[dict] = []
        if self.scryfall:
            cards = self._extract_card_names(situation)
            for card in cards:
                try:
                    rulings = await self.scryfall.get_rulings(card)
                    card_rulings.extend(rulings)
                except Exception:
                    pass

        # 3. Build LLM prompt
        prompt = self._build_prompt(
            situation, relevant_rules, card_rulings, game_state
        )
        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = await self.llm.ainvoke(messages)
        return self._parse_ruling(response.content)

    def _build_prompt(
        self,
        situation: str,
        rules: list,
        card_rulings: list[dict],
        game_state: GameState | None,
    ) -> str:
        sections = [f"SITUATION: {situation}"]

        if rules:
            rule_text = "\n".join(
                getattr(r, "page_content", str(r)) for r in rules[:10]
            )
            sections.append(f"RELEVANT COMPREHENSIVE RULES:\n{rule_text}")

        if card_rulings:
            rulings_text = "\n".join(
                f"- {r.get('comment', r)}" for r in card_rulings[:10]
            )
            sections.append(f"CARD-SPECIFIC RULINGS:\n{rulings_text}")

        if game_state:
            sections.append(
                f"GAME STATE: Turn {game_state.turn_number}, "
                f"Phase {game_state.phase.value}, "
                f"Stack depth {len(game_state.stack)}"
            )

        sections.append(
            "Provide:\n"
            "1. Your ruling (what happens)\n"
            "2. The specific rule numbers that apply\n"
            "3. Step-by-step resolution order\n"
            "4. Confidence level (high/medium/low)"
        )
        return "\n\n".join(sections)

    def _parse_ruling(self, text: str) -> Ruling:
        """Parse LLM response into a structured Ruling."""
        # Simple extraction — refinable with structured output later
        confidence = "medium"
        for level in ["high", "medium", "low"]:
            if level in text.lower():
                confidence = level
                break

        return Ruling(
            outcome=text[:500],
            rule_numbers=[],  # TODO: regex extract CR numbers
            resolution_steps=[],
            confidence=confidence,
            reasoning=text,
        )

    @staticmethod
    def _extract_card_names(text: str) -> list[str]:
        """Extract likely card names from a situation description.

        Heuristic: card names are typically capitalized multi-word phrases.
        """
        # Stub — proper NER or KG lookup would be more robust
        return []
