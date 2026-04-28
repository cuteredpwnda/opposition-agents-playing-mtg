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
        local_rulings: Any | None = None,  # LocalRulingsCache
    ):
        self.llm = llm
        self.rules_rag = rules_vectorstore
        self.kg = kg
        self.scryfall = scryfall_client
        # Optional zero-latency local cache; if set + populated, judge skips
        # the per-ruling Scryfall HTTP call. See src/judge/errata_loader.py.
        self.local_rulings = local_rulings

    async def rule_on(
        self, situation: str, game_state: GameState | None = None
    ) -> Ruling:
        """Produce a ruling for a given game situation."""
        # 1. Retrieve relevant comprehensive rules sections
        relevant_rules = self.rules_rag.similarity_search(situation, k=10)

        # 2. Card-specific rulings — prefer local cache, fall back to API
        card_rulings: list[dict] = []
        cards = self._extract_card_names(situation)
        if self.local_rulings and getattr(self.local_rulings, "is_loaded", lambda: False)():
            for card in cards:
                card_rulings.extend(self.local_rulings.get_rulings(card))
        elif self.scryfall:
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
        import re
        
        # Simple extraction — refinable with structured output later
        confidence = "medium"
        for level in ["high", "medium", "low"]:
            if level in text.lower():
                confidence = level
                break

        # Extract CR rule numbers using regex
        rule_pattern = r'\b(\d{3}(?:\.\d+)?(?:[a-z])?)\b'
        rule_numbers = list(set(re.findall(rule_pattern, text)))
        
        # Extract resolution steps (lines starting with numbers or bullets)
        resolution_steps = []
        for line in text.split('\n'):
            if re.match(r'^\s*[\d*•\-]\s*', line):
                resolution_steps.append(line.strip())

        return Ruling(
            outcome=text[:500],
            rule_numbers=rule_numbers,
            resolution_steps=resolution_steps[:5],  # Limit to 5 steps
            confidence=confidence,
            reasoning=text,
        )

    @staticmethod
    def _extract_card_names(text: str) -> list[str]:
        """Extract likely card names from a situation description.

        Heuristic: card names are typically capitalized multi-word phrases.
        Uses regex + rule-based matching for robustness.
        """
        import re
        
        # 1. Extract quoted names (highest confidence)
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
        if quoted:
            return quoted
        
        # 2. Extract capitalized noun phrases (looser heuristic)
        # Look for sequences of capitalized words
        words = text.split()
        card_names = []
        current_phrase = []
        
        for word in words:
            # Remove punctuation for analysis
            clean_word = re.sub(r'[^a-zA-Z\s]', '', word)
            
            if clean_word and clean_word[0].isupper():
                current_phrase.append(clean_word)
            else:
                if current_phrase and len(''.join(current_phrase)) > 3:
                    potential_card = ' '.join(current_phrase)
                    if len(potential_card.split()) <= 3:  # Most cards are 1-3 words
                        card_names.append(potential_card)
                current_phrase = []
        
        # Return unique names, deduplicated, limited to 5
        return list(dict.fromkeys(card_names))[:5]
