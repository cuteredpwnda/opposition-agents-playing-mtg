"""
Active inference module — Bayesian belief tracking + expected free energy.

Implements Karl Friston's Free Energy Principle for MTG decision-making
under uncertainty. Agents maintain probabilistic beliefs over hidden game
information (opponent's hand, deck composition, strategy intent) and
select actions that minimize expected free energy — balancing epistemic
value (information gain) with pragmatic value (goal achievement).

Reference: Section 7.2 of PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.engine.game_state import Action, GameState
from src.knowledge.knowledge_graph import MTGKnowledgeGraph


@dataclass
class OpponentBelief:
    """Probabilistic belief state about one opponent."""

    hand_distribution: dict[str, float] = field(default_factory=dict)
    deck_composition: dict[str, float] = field(default_factory=dict)
    strategy_intent: dict[str, float] = field(default_factory=dict)


class ActiveInferenceModule:
    """Maintains probabilistic beliefs about hidden game information
    and selects actions that minimize expected free energy.

    G(π) = epistemic_value + pragmatic_value
    """

    def __init__(self, kg: MTGKnowledgeGraph):
        self.kg = kg
        self.beliefs: dict[str, OpponentBelief] = {}

    async def initialize_beliefs(
        self, opponent_id: str, deck_archetype: str | None = None
    ) -> None:
        """Initialize prior beliefs from known archetype (or uniform if unknown)."""
        belief = OpponentBelief()
        if deck_archetype:
            archetype_cards = await self.kg.get_archetype_signature_cards(
                deck_archetype
            )
            for card_info in archetype_cards:
                name = card_info.get("card", "")
                belief.hand_distribution[name] = 0.1  # prior estimate
        self.beliefs[opponent_id] = belief

    def update_beliefs(
        self,
        observation: dict[str, Any],
        game_state: GameState,
    ) -> None:
        """Bayesian update of beliefs given a new game observation.

        Observations include cards played, mana left open, cards drawn,
        cards revealed, tutor usage, etc.
        """
        for opp_id, belief in self.beliefs.items():
            # If we saw a card played, it's no longer unknown in hand
            if card_played := observation.get("card_played"):
                belief.hand_distribution.pop(card_played, None)

            # Blue mana open → higher P(counterspell)
            if observation.get("blue_mana_open", 0) >= 2:
                for card_name in list(belief.hand_distribution):
                    if "counter" in card_name.lower():
                        belief.hand_distribution[card_name] = min(
                            belief.hand_distribution.get(card_name, 0) + 0.15,
                            0.95,
                        )

    def compute_expected_free_energy(
        self, action: Action, game_state: GameState
    ) -> float:
        """G(π) = -(epistemic_value + pragmatic_value). Lower = better."""
        epistemic = self._compute_epistemic_value(action, game_state)
        pragmatic = self._compute_pragmatic_value(action, game_state)
        return -(epistemic + pragmatic)

    def rank_actions(
        self, actions: list[Action], game_state: GameState
    ) -> list[tuple[Action, float]]:
        """Rank actions by expected free energy (lower = better)."""
        scored = [
            (a, self.compute_expected_free_energy(a, game_state)) for a in actions
        ]
        scored.sort(key=lambda x: x[1])
        return scored

    def should_hold_open_mana(self, game_state: GameState) -> dict[str, Any]:
        """Decide whether to hold mana open for reactive plays.

        Returns recommendation with color requirements and reason.
        """
        # Check if we have counterspells or instants that benefit from open mana
        # This is a heuristic stub — full implementation uses belief distributions
        return {"hold": False, "reason": "no reactive plays available"}

    # -- Private helpers ---------------------------------------------------

    def _compute_epistemic_value(
        self, action: Action, game_state: GameState
    ) -> float:
        """How much will this action reduce uncertainty?

        High for: Thoughtseize, Gitaxian Probe, probing attacks.
        """
        if action.card_instance_id is None:
            return 0.0
        
        # Find the card in game state
        card = next((c for c in game_state.cards if c.instance_id == action.card_instance_id), None)
        if card is None:
            return 0.0
        
        text = card.oracle_text.lower()
        
        # Discard/reveal effects have high epistemic value
        if any(kw in text for kw in ["look at", "reveal", "target player discards"]):
            return 0.4  # Strong information gathering
        
        # Draw effects
        if "draw" in text:
            return 0.2
        
        # Thoughtseize/Duress-like effects
        if "sacrifice" in text and "hand" in text:
            return 0.35
        
        # Default: no epistemic value
        return 0.0

    def _compute_pragmatic_value(
        self, action: Action, game_state: GameState
    ) -> float:
        """How much does this action advance the win condition?

        High for: combo pieces, lethal damage, key threats.
        """
        if action.card_instance_id is None:
            return 0.0
        
        # Find the card in game state
        card = next((c for c in game_state.cards if c.instance_id == action.card_instance_id), None)
        if card is None:
            return 0.0
        
        text = card.oracle_text.lower()
        
        # Win/infinite combos are highest priority
        if any(kw in text for kw in ["win", "infinite", "draw"]):
            return 0.5
        
        # Creatures with high power deal threat
        if card.is_creature() and card.power:
            try:
                power_val = int(card.power)
                return min(power_val / 20.0, 0.4)
            except (ValueError, TypeError):
                pass
        
        # Removal/interaction / counters
        if any(kw in text for kw in ["destroy", "return", "counter", "remove"]):
            return 0.25
        
        # Mana acceleration/ramp
        if any(kw in text for kw in ["add {", "ramp", "search", "fetch"]):
            return 0.15
        
        # Default: value based on mana cost
        cmc = card.cmc or 0.0
        return min(cmc / 10.0, 0.3)
