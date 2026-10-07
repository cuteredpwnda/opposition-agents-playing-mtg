"""KG-aware heuristic agent.

Same priority skeleton as :class:`HeuristicAgent` but, when more than one
``CAST_SPELL`` action is available, prefers the card that:

  1. completes a combo whose other pieces are already in play / in hand
     (uses ``MTGKnowledgeGraph.detect_near_combos``), then
  2. has the strongest synergy with cards already on our battlefield
     (uses ``get_synergies_for``), then
  3. falls back to the parent's deterministic tiebreak.

Stays deterministic given a seed.  All KG calls are best-effort: if Neo4j
is unreachable the agent silently degrades to the plain heuristic.

The KG handle is cached at construction time, but a per-decision LRU
cache keyed on ``(card_name, hand_or_battlefield_set)`` keeps Cypher
traffic to one query per turn-and-card pair.
"""
from __future__ import annotations

import logging
from typing import Any

from src.agents.heuristic_agent import HeuristicAgent
from src.engine_legacy.game_state import Action, ActionType, GameState, Zone

logger = logging.getLogger(__name__)


class KGHeuristicAgent(HeuristicAgent):
    """Heuristic agent that uses the knowledge graph to break CAST ties."""

    def __init__(
        self,
        player_id: str,
        name: str = "",
        seed: int | None = None,
        prefer_aggressive: bool = True,
        kg: Any = None,
    ) -> None:
        super().__init__(
            player_id=player_id,
            name=name or f"KGHeuristic({player_id})",
            seed=seed,
            prefer_aggressive=prefer_aggressive,
        )
        self._kg = kg
        self._kg_disabled = False
        # combo_score, synergy_score caches keyed on card name (per-game).
        self._combo_cache: dict[str, float] = {}
        self._synergy_cache: dict[tuple[str, frozenset[str]], float] = {}

    # ------------------------------------------------------------------ public
    async def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        # Re-use the parent's filtering + priority logic to find the top
        # candidates, but if the top candidates are all CAST_SPELL we
        # rerank them by combo / synergy score before falling through.
        casts = [a for a in legal_actions if a.action_type == ActionType.CAST_SPELL]
        if len(casts) <= 1 or self._kg_disabled or self._kg is None:
            action = await super().decide_action(game_state, legal_actions)
            # Tag the parent's reasoning trace as kg_heuristic-with-fallback.
            if isinstance(self.last_reasoning, dict):
                self.last_reasoning["agent_kind"] = "kg_heuristic"
                self.last_reasoning.setdefault("beliefs", {})["kg_active"] = False
                if self._kg_disabled:
                    self.last_reasoning["beliefs"]["kg_disabled_reason"] = "previous_query_failed"
                elif self._kg is None:
                    self.last_reasoning["beliefs"]["kg_disabled_reason"] = "no_kg_handle"
                elif len(casts) <= 1:
                    self.last_reasoning["beliefs"]["kg_disabled_reason"] = "fewer_than_2_casts"
            return action

        scored = await self._score_casts(game_state, casts)
        if not scored:
            action = await super().decide_action(game_state, legal_actions)
            if isinstance(self.last_reasoning, dict):
                self.last_reasoning["agent_kind"] = "kg_heuristic"
                self.last_reasoning.setdefault("beliefs", {})["kg_active"] = False
                self.last_reasoning["beliefs"]["kg_disabled_reason"] = "scoring_returned_empty"
            return action

        # Replace the cast subset with a single "best" cast in the legal
        # list and let the parent's priority pick handle the rest (so
        # PLAY_LAND still beats CAST_SPELL, etc.).
        best_cast, best_score = max(scored, key=lambda kv: kv[1])
        pruned = [a for a in legal_actions if a.action_type != ActionType.CAST_SPELL]
        pruned.append(best_cast)
        action = await super().decide_action(game_state, pruned)

        # Augment the parent's reasoning trace with KG details.
        if isinstance(self.last_reasoning, dict):
            top = sorted(scored, key=lambda kv: kv[1], reverse=True)[:5]
            top_payload = []
            for act, sc in top:
                name = self._card_name_for_action(game_state, act) or act.card_instance_id
                top_payload.append({
                    "action": f"CAST_SPELL({name})",
                    "score": float(sc),
                    "reason": ("near-combo closer" if sc >= 10.0
                               else "synergy with battlefield" if sc > 0.0
                               else "no KG signal"),
                })
            self.last_reasoning["agent_kind"] = "kg_heuristic"
            self.last_reasoning["top_candidates"] = top_payload
            self.last_reasoning["scores"] = [float(s) for _, s in scored]
            beliefs = self.last_reasoning.setdefault("beliefs", {})
            beliefs["kg_active"] = True
            beliefs["best_cast_score"] = float(best_score)
            beliefs["cast_options"] = len(casts)
            beliefs["kg_chose_cast"] = action.card_instance_id == best_cast.card_instance_id
            if action.card_instance_id == best_cast.card_instance_id:
                self.last_reasoning["rationale"] = (
                    f"KG re-rank: cast {self._card_name_for_action(game_state, best_cast)} "
                    f"(score={best_score:.2f}) selected from {len(casts)} options"
                )
        return action

    # ----------------------------------------------------------------- scoring
    async def _score_casts(
        self, game_state: GameState, casts: list[Action]
    ) -> list[tuple[Action, float]]:
        my_battlefield = self._card_names_in_zone(game_state, Zone.BATTLEFIELD)
        my_hand = self._card_names_in_zone(game_state, Zone.HAND)
        available = sorted(set(my_battlefield) | set(my_hand))

        # One-shot near-combo lookup: any card that closes a near-combo
        # gets a hard +10 score.
        near_combo_pieces: set[str] = set()
        try:
            near = await self._kg.detect_near_combos(available)
            for entry in near or []:
                missing = entry.get("missingPiece")
                if missing:
                    near_combo_pieces.add(str(missing))
        except Exception as e:  # network / driver issues — don't crash the game
            logger.debug("KG near-combo lookup failed (%s); disabling KG bias", e)
            self._kg_disabled = True
            return []

        battlefield_key = frozenset(my_battlefield)
        scored: list[tuple[Action, float]] = []
        for action in casts:
            card_name = self._card_name_for_action(game_state, action)
            if not card_name:
                continue
            score = 0.0
            if card_name in near_combo_pieces:
                score += 10.0
            score += await self._synergy_score(card_name, battlefield_key)
            scored.append((action, score))
        return scored

    async def _synergy_score(
        self, card_name: str, battlefield: frozenset[str]
    ) -> float:
        if self._kg_disabled or not battlefield:
            return 0.0
        key = (card_name, battlefield)
        if key in self._synergy_cache:
            return self._synergy_cache[key]
        try:
            synergies = await self._kg.get_synergies_for(card_name)
        except Exception as e:
            logger.debug("KG synergy lookup failed (%s); disabling KG bias", e)
            self._kg_disabled = True
            return 0.0
        score = 0.0
        for s in synergies or []:
            partner = s.get("card")
            strength = float(s.get("strength") or 0.0)
            if partner in battlefield:
                score += strength
        self._synergy_cache[key] = score
        return score

    # --------------------------------------------------------------- utilities
    def _card_names_in_zone(self, gs: GameState, zone: Zone) -> list[str]:
        out: list[str] = []
        for card in gs.cards:
            if card.zone != zone:
                continue
            if getattr(card, "controller_id", None) != self.player_id:
                continue
            name = getattr(card, "name", None) or (card.card_data or {}).get("name")
            if name:
                out.append(name)
        return out

    def _card_name_for_action(self, gs: GameState, action: Action) -> str | None:
        if not action.card_instance_id:
            return None
        for c in gs.cards:
            if c.instance_id == action.card_instance_id:
                return getattr(c, "name", None) or (c.card_data or {}).get("name")
        return None
