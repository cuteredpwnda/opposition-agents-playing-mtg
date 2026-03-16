"""
Opponent model — infers archetype & predicts opponent hand/strategy.

Reference: Section 9.1 of PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.engine.game_state import Action, GameState
from src.knowledge.knowledge_graph import MTGKnowledgeGraph


@dataclass
class ThreatAssessment:
    """Current assessment of what the opponent is threatening."""

    archetype: str
    confidence: float
    predicted_hand: dict[str, float]
    probability_has_counterspell: float
    probability_has_removal: float
    probability_has_boardwipe: float


class OpponentModel:
    """Tracks an opponent's observed behavior and infers their strategy."""

    def __init__(self, opponent_id: str, kg: MTGKnowledgeGraph):
        self.opponent_id = opponent_id
        self.kg = kg
        self.cards_seen: list[str] = []
        self.actions_taken: list[Action] = []
        self.archetype_probabilities: dict[str, float] = {}
        self.predicted_hand: dict[str, float] = {}

    async def observe_card(self, card_name: str) -> None:
        """Update archetype probabilities when a card is revealed."""
        self.cards_seen.append(card_name)
        results = await self.kg.infer_archetype_from_cards(self.cards_seen)
        self.archetype_probabilities = {
            r["archetype"]: r["confidence"] for r in results
        }
        await self._update_hand_predictions()

    def observe_behavior(self, action: Action, game_state: GameState) -> None:
        """Update model based on play patterns (mana usage, attack lines)."""
        self.actions_taken.append(action)

    async def get_threat_assessment(self) -> ThreatAssessment:
        """What is the opponent threatening?"""
        if not self.archetype_probabilities:
            return ThreatAssessment(
                archetype="unknown",
                confidence=0.0,
                predicted_hand={},
                probability_has_counterspell=0.15,
                probability_has_removal=0.2,
                probability_has_boardwipe=0.1,
            )
        top = max(self.archetype_probabilities, key=self.archetype_probabilities.get)
        return ThreatAssessment(
            archetype=top,
            confidence=self.archetype_probabilities[top],
            predicted_hand=dict(self.predicted_hand),
            probability_has_counterspell=self._prob_has_counter(),
            probability_has_removal=self._prob_has("removal", base=0.2),
            probability_has_boardwipe=self._prob_has("boardwipe", base=0.1),
        )

    async def _update_hand_predictions(self) -> None:
        """Predict likely hand contents from top archetype."""
        if not self.archetype_probabilities:
            return
        top = max(self.archetype_probabilities, key=self.archetype_probabilities.get)
        staples = await self.kg.get_archetype_signature_cards(top)
        unseen = [s["card"] for s in staples if s["card"] not in self.cards_seen]
        self.predicted_hand = {card: 0.15 for card in unseen[:10]}

    def _prob_has_counter(self) -> float:
        """Estimate P(opponent has counterspell)."""
        if not self.archetype_probabilities:
            return 0.15
        top = max(self.archetype_probabilities, key=self.archetype_probabilities.get)
        # Control decks run 8-12 counters, aggro runs 0-2
        if "control" in top.lower():
            return 0.6
        if "midrange" in top.lower():
            return 0.2
        return 0.1

    def _prob_has(self, card_type: str, base: float) -> float:
        """Generic probability estimate for a card type."""
        return base  # Stub — expand with archetype-specific priors
