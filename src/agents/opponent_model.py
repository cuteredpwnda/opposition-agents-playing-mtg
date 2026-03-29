"""
Opponent model — infers archetype & predicts opponent hand/strategy.

Uses public decklist information with observed game state to model opponent's
belief state via process-of-elimination.

Reference: Section 9.1 of PLAN.md & Ontology Extension Layer 7 (BeliefState).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import Any

from src.engine.game_state import Action, GameState
from src.knowledge.knowledge_graph import MTGKnowledgeGraph


@dataclass
class CardInformation:
    """Tracking for a single card in opponent's deck."""
    name: str
    copies_in_deck: int
    copies_seen: int = 0      # Observed in open zones (graveyard, exile, battlefield)
    copies_drawn: int = 0      # Inferred drawn (in hand or played)
    copies_remaining: int = 0  # Computed: copies_in_deck - copies_seen - copies_drawn

    @property
    def probability_in_hand(self) -> float:
        """P(card in opponent hand) given observations."""
        if self.copies_remaining <= 0:
            return 0.0
        # If we know opponent has N cards in hand and M remaining cards total
        # P(this card in hand) = (copies_remaining * hand_size) / total_remaining
        # This is computed by OpponentModel with full context
        return 0.0  # Set by OpponentModel

    @property
    def probability_in_library(self) -> float:
        """P(card in opponent library) given observations."""
        if self.copies_remaining <= 0:
            return 0.0
        # Similar calculation as hand probability
        return 0.0


@dataclass
class ThreatAssessment:
    """Current assessment of what the opponent is threatening."""

    archetype: str
    confidence: float
    predicted_hand: dict[str, float]  # Card name -> P(in hand)
    probability_has_counterspell: float
    probability_has_removal: float
    probability_has_boardwipe: float
    cards_remaining_in_deck: int  # Library size
    cards_in_hand_count: int  # Known hand size


class OpponentModel:
    """
    Tracks an opponent's observed behavior and infers their strategy.
    
    If opponent decklist is public (standard in Commander, known in constructed),
    uses process-of-elimination to exactly infer library composition.
    """

    def __init__(
        self, 
        opponent_id: str, 
        kg: MTGKnowledgeGraph,
        known_decklist: dict[str, int] | None = None
    ):
        """
        Initialize opponent model.
        
        Args:
            opponent_id: Player ID of opponent
            kg: MTG Knowledge Graph for archetype/card queries
            known_decklist: Card name -> count mapping. If provided, enables
                          process-of-elimination belief tracking.
        """
        self.opponent_id = opponent_id
        self.kg = kg
        
        # Decklist tracking (CRITICAL FOR ACCURATE OPPONENT MODELING)
        self.known_decklist = known_decklist or {}  # Public information
        self.card_info: dict[str, CardInformation] = {}
        
        if self.known_decklist:
            for card_name, count in self.known_decklist.items():
                self.card_info[card_name] = CardInformation(
                    name=card_name,
                    copies_in_deck=count,
                    copies_remaining=count
                )
        
        # Observation tracking
        self.cards_seen_in_open_zones: list[str] = []  # Graveyard, exile, battlefield, revealed hand
        self.actions_taken: list[Action] = []
        self.cards_played: list[str] = []  # Cards opponent has cast/played
        
        # Inferred state
        self.archetype_probabilities: dict[str, float] = {}
        self.predicted_hand: dict[str, float] = {}
        self.cards_in_hand_count: int = 0  # Hand size (observable)
        self.cards_in_library_count: int = 0  # Library size (observable)


    async def observe_card(self, card_name: str, zone: str = "graveyard") -> None:
        """
        Update archetype probabilities when a card enters an open zone.
        Use process-of-elimination if decklist known.
        
        Args:
            card_name: Name of card observed
            zone: Zone it was observed in (graveyard, exile, battlefield, etc.)
        """
        self.cards_seen_in_open_zones.append(card_name)
        
        # Update card tracking if we have full decklist
        if card_name in self.card_info:
            info = self.card_info[card_name]
            info.copies_seen += 1
            info.copies_remaining = max(
                0,
                info.copies_in_deck - info.copies_seen - info.copies_drawn,
            )
            self._update_card_probabilities()
        
        # Infer archetype from cards seen
        results = await self.kg.infer_archetype_from_cards(self.cards_seen_in_open_zones)
        self.archetype_probabilities = {
            r["archetype"]: r["confidence"] for r in results
        }
        await self._update_hand_predictions()

    def observe_card_played(self, card_name: str) -> None:
        """
        Record that opponent cast/played a card (enters hand or graveyard).
        This is process-of-elimination: if card was in hand and left, it was played.
        """
        self.cards_played.append(card_name)
        
        if card_name in self.card_info:
            # If we haven't seen it in open zone yet, it's likely being cast from hand
            info = self.card_info[card_name]
            info.copies_drawn += 1
            info.copies_remaining = max(
                0,
                info.copies_in_deck - info.copies_seen - info.copies_drawn,
            )
            self._update_card_probabilities()

    def observe_game_state(self, game_state: GameState) -> None:
        """
        Update opponent model from current game state.
        Observes: opponent hand size, library size, graveyard, exile, battlefield.
        """
        # Compute observable zone counts from GameState cards
        self.cards_in_hand_count = len(
            [c for c in game_state.cards if c.controller_id == self.opponent_id and c.zone == "hand"]
        )
        self.cards_in_library_count = len(
            [c for c in game_state.cards if c.controller_id == self.opponent_id and c.zone == "library"]
        )

        # Process visible cards from open zones
        for zone_name in ["graveyard", "exile", "battlefield"]:
            for card_instance in [
                c for c in game_state.cards
                if c.controller_id == self.opponent_id and c.zone == zone_name
            ]:
                card_name = card_instance.card_data.get("name", "Unknown")
                if card_name not in self.cards_seen_in_open_zones:
                    self.observe_card(card_name, zone=zone_name)


    def _update_card_probabilities(self) -> None:
        """
        Recompute P(card in hand) and P(card in library) for all opponent cards.
        Uses exact arithmetic if decklist known: process-of-elimination.
        """
        if not self.known_decklist:
            return
        
        # Total cards unaccounted for (not seen in open zones, not confirmed played)
        total_unaccounted = sum(
            card.copies_remaining for card in self.card_info.values()
        )
        
        if total_unaccounted == 0:
            return
        
        # Distribute unaccounted cards between hand and library
        # P(card in hand) = (copies_remaining / total_unaccounted) * (hand_size / library_size)
        remaining_in_deck = self.cards_in_library_count
        remaining_in_hand = self.cards_in_hand_count
        
        self.predicted_hand = {}
        
        for card_name, card_info in self.card_info.items():
            if card_info.copies_remaining <= 0:
                self.predicted_hand[card_name] = 0.0
                continue
            
            remaining_copies = card_info.copies_remaining
            
            # Probability this card is in hand vs deck
            if remaining_in_deck + remaining_in_hand > 0:
                # Uniform distribution: each remaining card equally likely in hand/deck
                # Hypergeometric: copies_remaining cards, draw remaining_in_hand
                prob_in_hand = (remaining_copies * remaining_in_hand) / (
                    remaining_in_deck + remaining_in_hand
                )
            else:
                prob_in_hand = 0.0
            
            self.predicted_hand[card_name] = prob_in_hand

    def observe_behavior(self, action: Action, game_state: GameState) -> None:
        """Update model based on play patterns (mana usage, attack lines).
        
        Infer decision-making style and adjust archetype confidence.
        """
        self.actions_taken.append(action)
        
        # Analyze action patterns to infer playstyle
        action_type = str(action.action_type).lower() if hasattr(action, 'action_type') else ""
        
        # Count action types
        spell_casts = sum(1 for a in self.actions_taken if "cast" in str(a).lower())
        attacks = sum(1 for a in self.actions_taken if "attack" in str(a).lower())
        passes = sum(1 for a in self.actions_taken if "pass" in str(a).lower())
        
        # Update archetype probabilities based on playstyle
        if spell_casts > attacks:
            # More spells than attacks suggests control or combo
            if self.archetype_probabilities:
                for arch in ["control", "combo"]:
                    if arch in self.archetype_probabilities:
                        self.archetype_probabilities[arch] *= 1.1  # Boost confidence
        elif attacks > spell_casts:
            # More attacks suggests aggro or midrange
            if self.archetype_probabilities:
                for arch in ["aggro", "midrange"]:
                    if arch in self.archetype_probabilities:
                        self.archetype_probabilities[arch] *= 1.1
        
        # Normalize probabilities
        total = sum(self.archetype_probabilities.values()) or 1.0
        self.archetype_probabilities = {
            k: v / total for k, v in self.archetype_probabilities.items()
        }

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
                cards_remaining_in_deck=self.cards_in_library_count,
                cards_in_hand_count=self.cards_in_hand_count,
            )
        
        top = max(self.archetype_probabilities, key=self.archetype_probabilities.get)
        return ThreatAssessment(
            archetype=top,
            confidence=self.archetype_probabilities[top],
            predicted_hand=dict(self.predicted_hand),
            probability_has_counterspell=self._prob_has_counter(),
            probability_has_removal=self._prob_has("removal", base=0.2),
            probability_has_boardwipe=self._prob_has("boardwipe", base=0.1),
            cards_remaining_in_deck=self.cards_in_library_count,
            cards_in_hand_count=self.cards_in_hand_count,
        )

    async def _update_hand_predictions(self) -> None:
        """
        Predict likely hand contents from top archetype.
        If decklist known, use exact probabilities; else use archetype staples.
        """
        if self.known_decklist:
            # Probabilities already computed via process-of-elimination
            return
        
        # Fallback: archetype-based predictions (no decklist info)
        if not self.archetype_probabilities:
            return
        
        top = max(self.archetype_probabilities, key=self.archetype_probabilities.get)
        staples = await self.kg.get_archetype_signature_cards(top)
        unseen = [s["card"] for s in staples if s["card"] not in self.cards_seen_in_open_zones]
        
        # Uniform distribution over unseen staples
        if unseen:
            prob_each = 0.15 / len(unseen)
            self.predicted_hand = {card: prob_each for card in unseen[:10]}

    def _prob_has_counter(self) -> float:
        """
        Estimate P(opponent has counterspell in hand).
        If decklist known, exact count from process-of-elimination.
        """
        if not self.known_decklist:
            # Fallback: archetype-based heuristic
            if not self.archetype_probabilities:
                return 0.15
            top = max(self.archetype_probabilities, key=self.archetype_probabilities.get)
            if "control" in top.lower():
                return 0.6
            if "midrange" in top.lower():
                return 0.2
            return 0.1
        
        # Count counterspells in predicted hand
        counter_keywords = ["counter", "counterspell", "negate", "cancel"]
        total_prob = 0.0
        
        for card_name, prob_in_hand in self.predicted_hand.items():
            # Check if card name contains counter keyword
            if any(kw in card_name.lower() for kw in counter_keywords):
                total_prob += prob_in_hand
        
        return min(total_prob, 1.0)

    def _prob_has(self, card_type: str, base: float) -> float:
        """
        Estimate P(opponent has card of type) in hand.
        """
        if not self.known_decklist:
            return base
        
        # Count cards matching type in predicted hand
        total_prob = 0.0
        for card_name, prob_in_hand in self.predicted_hand.items():
            if card_type.lower() in card_name.lower():
                total_prob += prob_in_hand
        
        return min(total_prob, 1.0)

    def get_library_composition_remaining(self) -> dict[str, int]:
        """
        Get exact composition of opponent's remaining library.
        Only works if decklist is known. Uses process-of-elimination.
        
        Returns:
            {card_name: count_remaining_in_library}
        """
        if not self.known_decklist:
            return {}
        
        composition = {}
        for card_name, card_info in self.card_info.items():
            if card_info.copies_remaining > 0:
                composition[card_name] = card_info.copies_remaining
        
        return composition

    def get_draw_probabilities(self, num_cards: int = 1) -> dict[str, float]:
        """
        Compute probability of drawing each card in next N draws.
        Uses hypergeometric distribution.
        
        Args:
            num_cards: Number of cards about to be drawn
        
        Returns:
            {card_name: P(draw at least 1 copy)}
        """
        if not self.known_decklist or self.cards_in_library_count == 0:
            return {}
        
        remaining_in_deck = self.cards_in_library_count
        draw_prob: dict[str, float] = {}
        
        for card_name, card_info in self.card_info.items():
            if card_info.copies_remaining <= 0:
                draw_prob[card_name] = 0.0
                continue
            
            # P(at least 1 in next N draws) = 1 - P(0 in next N draws)
            # = 1 - C(deck-copies, N) / C(deck, N)
            # Approximation: 1 - (1 - copies/deck)^N for small num_cards
            
            prob_none = (
                (remaining_in_deck - card_info.copies_remaining) ** num_cards
                / (remaining_in_deck ** num_cards)
            ) if remaining_in_deck > 0 else 0.0
            
            draw_prob[card_name] = 1.0 - prob_none
        
        return draw_prob

    def predict_play(self, game_state: GameState) -> dict[str, Any]:
        """Predict opponent's next action based on learned patterns.
        
        Returns probability distribution over likely plays.
        """
        if not self.archetype_probabilities:
            return {"most_likely": "unknown", "confidence": 0.0}
        
        # Get most likely archetype
        likely_archetype = max(
            self.archetype_probabilities.items(),
            key=lambda x: x[1]
        )
        
        # Get the average confidence across known archetypes
        avg_confidence = (
            sum(self.archetype_probabilities.values()) / 
            max(len(self.archetype_probabilities), 1)
        )
        
        # Make predictions based on archetype and game state
        prediction = {
            "most_likely_archetype": likely_archetype[0],
            "archetype_confidence": likely_archetype[1],
            "average_confidence": avg_confidence,
            "likely_plays": self._infer_likely_plays(likely_archetype[0], game_state),
            "actions_observed": len(self.actions_taken),
        }
        
        return prediction
    
    def _infer_likely_plays(self, archetype: str, game_state: GameState) -> list[str]:
        """Generate list of likely plays based on archetype and game state."""
        likely_plays = []
        
        if archetype == "aggro":
            likely_plays = ["attack", "cast creature", "burn spell"]
        elif archetype == "control":
            likely_plays = ["cast instant", "counterspell", "draw cards"]
        elif archetype == "combo":
            likely_plays = ["tutor spell", "cast combo piece", "protection spell"]
        elif archetype == "midrange":
            likely_plays = ["cast creature", "attack", "removal spell"]
        elif archetype == "ramp":
            likely_plays = ["cast land ramp", "cast creature", "attack"]
        
        return likely_plays

