"""Card scorer for the deck-builder (G2).

Mixes four signals:

* **synergy**   (alpha)  — KG ``SYNERGIZES_WITH`` + learned evidence for
                           each card pair already in the deck.
* **archetype** (beta)   — simple rule-based type/role diversity bonus.
* **combo**     (gamma)  — KG ``detect_near_combos``: how many near-complete
                           combos does adding this card finish?
* **world_model** (delta) — predicted card value from the JEPA world model.

All KG calls are best-effort: if Neo4j is offline every KG sub-score
returns 0.0 and the scorer falls back to the archetype heuristic only.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Default mixing weights.
_ALPHA = 1.0   # synergy
_BETA  = 0.5   # archetype / type-diversity
_GAMMA = 2.0   # combo completion
_DELTA = 0.3   # world-model value


@dataclass
class CardScorer:
    """Score candidate cards for inclusion in a Commander deck.

    Args:
        kg:          MTGKnowledgeGraph instance (async); ``None`` disables KG
                     scoring.
        world_model: WorldModel instance; ``None`` disables WM scoring.
        alpha:       Synergy weight.
        beta:        Archetype / type-diversity weight.
        gamma:       Combo-completion weight.
        delta:       World-model value weight.
    """

    kg: Any = None
    world_model: Any = None
    alpha: float = _ALPHA
    beta:  float = _BETA
    gamma: float = _GAMMA
    delta: float = _DELTA

    # Per-brew caches — reset between brews with reset_caches().
    _synergy_cache: dict[tuple[str, str], float] = field(
        default_factory=dict, init=False, repr=False
    )
    _combo_cache: dict[frozenset[str], list[dict]] = field(
        default_factory=dict, init=False, repr=False
    )

    def reset_caches(self) -> None:
        self._synergy_cache.clear()
        self._combo_cache.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def score(
        self,
        candidate: dict[str, Any],
        current_deck_names: list[str],
    ) -> float:
        """Return a scalar score for adding *candidate* to *current_deck_names*."""
        name = candidate.get("name", "")
        synergy  = await self._synergy_score(name, current_deck_names)
        combo    = await self._combo_score(name, current_deck_names)
        archtype = self._archetype_score(candidate)
        wm_val   = self._world_model_score(name)
        return (
            self.alpha * synergy
            + self.beta  * archtype
            + self.gamma * combo
            + self.delta * wm_val
        )

    async def score_batch(
        self,
        candidates: list[dict[str, Any]],
        current_deck_names: list[str],
    ) -> list[tuple[dict, float]]:
        """Score all candidates concurrently.

        Returns a list of ``(card_dict, score)`` in the same order as
        *candidates*.
        """
        scores = await asyncio.gather(
            *(self.score(c, current_deck_names) for c in candidates)
        )
        return list(zip(candidates, scores))

    # ------------------------------------------------------------------
    # Sub-scores
    # ------------------------------------------------------------------

    async def _synergy_score(
        self, name: str, deck_names: list[str]
    ) -> float:
        if self.kg is None or not deck_names:
            return 0.0
        total = 0.0
        for deck_card in deck_names:
            key = (deck_card, name)
            if key in self._synergy_cache:
                total += self._synergy_cache[key]
                continue
            try:
                synergies = await self.kg.get_synergies_for(deck_card)
                hit = next(
                    (float(s.get("strength", 0)) for s in synergies
                     if s.get("card") == name),
                    0.0,
                )
            except Exception:
                hit = 0.0
            self._synergy_cache[key] = hit
            total += hit
        return total

    async def _combo_score(
        self, name: str, deck_names: list[str]
    ) -> float:
        if self.kg is None or not deck_names:
            return 0.0
        # Use a frozenset of up to the first 40 cards as cache key.
        key = frozenset(deck_names[:40])
        if key not in self._combo_cache:
            try:
                self._combo_cache[key] = await self.kg.detect_near_combos(
                    list(key)
                )
            except Exception:
                self._combo_cache[key] = []
        near = self._combo_cache[key]
        return float(sum(1 for c in near if c.get("missingPiece") == name))

    def _archetype_score(self, card: dict[str, Any]) -> float:
        """Lightweight rule-based type/role diversity bonus."""
        type_line = (card.get("type_line") or "").lower()
        oracle = (card.get("oracle_text") or "").lower()

        # Basic lands add no archetype value (the agent pads them separately).
        if "basic" in type_line and "land" in type_line:
            return 0.0
        # Non-basic lands get a mild bonus.
        if "land" in type_line:
            return 0.3
        # Mana ramp / rocks get a moderate bonus — always useful.
        if "{t}: add" in oracle or "add {" in oracle:
            return 0.8
        if "search your library for a" in oracle and "land" in oracle:
            return 0.7
        # Tutors get a mild bonus.
        if "search your library" in oracle:
            return 0.6
        # Counterspells and removal.
        if "counter target" in oracle or "destroy target" in oracle or "exile target" in oracle:
            return 0.5
        return 1.0

    def _world_model_score(self, name: str) -> float:
        if self.world_model is None:
            return 0.0
        try:
            return float(self.world_model.estimate_card_value(name))
        except Exception:
            return 0.0
