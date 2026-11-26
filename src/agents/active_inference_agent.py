"""Agent that uses active inference + opponent model for decision making."""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

from src.agents.base_agent import MTGAgent
from src.agents.random_agent import RandomAgent
from src.agents.active_inference import ActiveInferenceModule
from src.agents.opponent_model import OpponentModel
from src.knowledge.knowledge_graph import MTGKnowledgeGraph
from src.engine_legacy.game_state import GameState, Action, ActionType

logger = logging.getLogger(__name__)


class ActiveInferenceAgent(MTGAgent):
    """MTG Agent that chooses actions by minimizing expected free energy."""

    def __init__(
        self,
        player_id: str,
        kg: Optional[MTGKnowledgeGraph] = None,
        known_opponent_decklist: Optional[Dict[str, int]] = None,
        name: str = "ActiveInferenceAgent",
    ):
        super().__init__(player_id=player_id, name=name)

        self.kg = kg or self._try_init_kg()
        self.ai_module = ActiveInferenceModule(self.kg) if self.kg is not None else None
        self.opponent_model = OpponentModel(
            opponent_id="opponent",
            kg=self.kg,
            known_decklist=known_opponent_decklist,
        ) if self.kg is not None else None

        self.fallback = RandomAgent(player_id=player_id, name=f"{name}_fallback")

        # Keep track of observed opponent IDs to initialize beliefs
        self.opponent_ids: List[str] = []

    def _try_init_kg(self) -> Optional[MTGKnowledgeGraph]:
        try:
            return MTGKnowledgeGraph()
        except Exception as e:
            logger.warning("ActiveInferenceAgent: Could not connect to KG: %s", e)
            return None

    async def decide_action(
        self, game_state: GameState, legal_actions: List[Action]
    ) -> Action:
        if not legal_actions:
            return Action(action_type=ActionType.PASS_PRIORITY, player_id=self.player_id)

        if len(legal_actions) == 1:
            return legal_actions[0]

        # Basic heuristic fallback scores when no AI module
        if self.ai_module is None:
            return await self.fallback.decide_action(game_state, legal_actions)

        # Ensure beliefs are initialized for opponents
        for player in game_state.players:
            if player.player_id != self.player_id and player.player_id not in self.opponent_ids:
                self.opponent_ids.append(player.player_id)
                await self.ai_module.initialize_beliefs(player.player_id)

        # Rank actions by expected free energy (low is good)
        ranked = self.ai_module.rank_actions(legal_actions, game_state)

        # Use opponent model to influence action selection in Commander/multiplayer
        if self.opponent_model is not None and ranked:
            try:
                threat = await self.opponent_model.get_threat_assessment()
                if threat.probability_has_counterspell > 0.5:
                    # If opponent likely has counterspell, postpone non-urgent spells
                    ranked = [r for r in ranked if r[0].action_type != ActionType.CAST_SPELL or r[0].card_instance_id is None]
                    if not ranked:
                        ranked = self.ai_module.rank_actions(legal_actions, game_state)
            except Exception:
                pass

        # If we have multiple attacker choices, prioritize the highest-threat target
        declare_actions = [a for a in (r[0] for r in ranked) if a.action_type == ActionType.DECLARE_ATTACKERS]
        if declare_actions:
            scored = []
            for a in declare_actions:
                if not a.targets:
                    continue
                scored.append((a, self._score_attack_target(game_state, a.targets[0])))

            if scored:
                chosen_action = max(scored, key=lambda x: x[1])[0]
            else:
                chosen_action = ranked[0][0] if ranked else await self.fallback.decide_action(game_state, legal_actions)
        else:
            chosen_action = ranked[0][0] if ranked else await self.fallback.decide_action(game_state, legal_actions)

        # Attach meta debug info for analysis/policy tracing
        chosen_action.metadata["decision_mode"] = "active_inference"
        chosen_action.metadata["expected_free_energy"] = ranked[0][1] if ranked else None
        if chosen_action.action_type == ActionType.DECLARE_ATTACKERS and chosen_action.targets:
            chosen_action.metadata["target_score"] = self._score_attack_target(game_state, chosen_action.targets[0])

        logger.debug(
            "ActiveInferenceAgent chose %s with G=%s",
            chosen_action.action_type.value,
            ranked[0][1] if ranked else None,
        )

        # Reasoning trace ----------------------------------------------------
        try:
            from src.agents.reasoning import ReasoningTrace

            efe_scores = [float(g) for _, g in ranked][:32] if ranked else []
            top_payload = []
            for a, g in (ranked[:5] if ranked else []):
                top_payload.append({
                    "action": f"{a.action_type.value}({a.card_instance_id or ''})",
                    "score": float(g),  # expected free energy (lower = better)
                    "reason": "lowest expected free energy" if g == (ranked[0][1] if ranked else None) else "candidate",
                })
            try:
                chosen_idx = legal_actions.index(chosen_action)
            except ValueError:
                chosen_idx = -1
            beliefs = {
                "best_efe": float(ranked[0][1]) if ranked else None,
                "num_opponents_modelled": len(self.opponent_ids),
                "fallback_used": self.ai_module is None,
            }
            if self.opponent_model is not None:
                try:
                    threat = await self.opponent_model.get_threat_assessment()
                    beliefs["opp_p_counterspell"] = float(threat.probability_has_counterspell)
                except Exception:
                    pass
            self.set_reasoning(ReasoningTrace(
                agent_kind="active_inference",
                rationale=(
                    f"min-EFE over {len(legal_actions)} actions: "
                    f"chose {chosen_action.action_type.value} (G={ranked[0][1]:.3f})"
                    if ranked else "fallback heuristic (no AI module)"
                ),
                legal_action_count=len(legal_actions),
                chosen_index=chosen_idx,
                scores=efe_scores,
                top_candidates=top_payload,
                beliefs=beliefs,
            ))
        except Exception as e:  # never let reasoning logging break gameplay
            logger.debug("Failed to record ActiveInferenceAgent reasoning: %s", e)

        return chosen_action

    def _score_attack_target(self, game_state: GameState, target_id: str) -> float:
        target = next((p for p in game_state.players if p.player_id == target_id), None)
        if target is None:
            return 0.0

        base = max(0, 40 - target.life_total)
        commander_dmg = 0
        if hasattr(target, "commander_damage_received"):
            commander_dmg = sum(target.commander_damage_received.values())

        # Avoid tanking player with highest life when not needed
        score = base + commander_dmg * 2

        # Penalize knockout player if already about to lose to commander damage (e.g., to avoid gratitude alliances)
        if commander_dmg >= 21:
            score *= 0.8

        return score

    async def observe(self, game_state: GameState, action: Action) -> None:
        await super().observe(game_state, action)

        # Infer opponent model from observed action
        if self.opponent_model is not None and action.player_id != self.player_id:
            self.opponent_model.observe_behavior(action, game_state)
            self.opponent_model.observe_card_played(action.card_instance_id or "")
            self.opponent_model.observe_game_state(game_state)

        # Update active inference beliefs
        if self.ai_module is not None:
            observation = {
                "card_played": None,
                "blue_mana_open": 0,
            }
            if action.card_instance_id:
                card = next((c for c in game_state.cards if c.instance_id == action.card_instance_id), None)
                if card is not None:
                    observation["card_played"] = card.name

            # Track open blue mana on active player (if we can infer)
            active_player = game_state.active_player
            observation["blue_mana_open"] = active_player.mana_pool.get("U", 0)

            self.ai_module.update_beliefs(observation, game_state)

    def reset(self) -> None:
        super().reset()
        self.opponent_ids = []
        if self.ai_module is not None:
            self.ai_module.beliefs.clear()
        if self.opponent_model is not None:
            self.opponent_model.cards_seen_in_open_zones.clear()
            self.opponent_model.actions_taken.clear()
