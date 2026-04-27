"""
LLM-Fusion Agent — combines LLM strategic reasoning with world model planning.

The LLM proposes candidate strategies, the world model evaluates them via
dream rollouts, and the combined score determines the final action.

Architecture:
  1. LLM generates strategic intent ("hold removal for their combo piece")
  2. World model scores each legal action via dream search
  3. Knowledge Graph provides combo/synergy context
  4. Active Inference updates opponent beliefs
  5. Fusion layer combines all signals into a final decision

This is the strongest agent mode — slower than pure world model but more
robust on novel card interactions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.agents.base_agent import MTGAgent
from src.engine.game_state import Action, ActionType, GameState, Zone

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """Configuration for the LLM-Fusion agent."""

    # Weight for each signal in the fusion
    llm_weight: float = 0.3         # LLM strategic preference
    world_model_weight: float = 0.4  # Dream rollout expected value
    kg_weight: float = 0.15          # Knowledge graph combo/synergy bonus
    heuristic_weight: float = 0.15   # Fast board-evaluation fallback

    # Dream search config
    dream_rollouts: int = 8
    dream_depth: int = 10
    dream_temperature: float = 1.15

    # LLM config
    llm_provider: str = "ollama"   # "ollama" | "openai" | "anthropic"
    llm_model: str = "gemma4:e2b"
    llm_timeout: float = 15.0

    # When to skip LLM (for speed)
    skip_llm_if_obvious: bool = True  # Skip LLM if one action dominates by > threshold
    obvious_threshold: float = 0.8


class LLMFusionAgent(MTGAgent):
    """Agent that fuses LLM reasoning, world model planning, and KG context.

    Decision pipeline:
    1. Fast heuristic pass — score all actions cheaply
    2. KG context lookup — combo/synergy bonuses
    3. World model dream search — evaluate top candidates
    4. LLM strategic reasoning — for non-obvious decisions
    5. Weighted fusion — combine all signals
    """

    def __init__(
        self,
        player_id: str,
        name: str = "Fusion Agent",
        config: FusionConfig | None = None,
        world_model: Any = None,
        tokenizer: Any = None,
        knowledge_graph: Any = None,
        opponent_model: Any = None,
    ):
        super().__init__(player_id, name)
        self.config = config or FusionConfig()
        self.world_model = world_model
        self.tokenizer = tokenizer
        self.kg = knowledge_graph
        self.opponent_model = opponent_model

        # Lazy-init sub-components
        self._llm_agent = None
        self._random_fallback = None
        self._dynamics_hidden = None  # RNN hidden state for world model

    def _ensure_llm(self):
        """Lazy-init the LLM sub-agent."""
        if self._llm_agent is not None:
            return
        try:
            from src.agents.llm_agent import OllamaAgent
            self._llm_agent = OllamaAgent(
                player_id=self.player_id,
                name=f"{self.name}_llm",
                model=self.config.llm_model,
            )
        except Exception as e:
            logger.warning("LLM agent unavailable: %s", e)

    def _ensure_fallback(self):
        if self._random_fallback is None:
            from src.agents.random_agent import RandomAgent
            self._random_fallback = RandomAgent(
                player_id=self.player_id, name=f"{self.name}_fallback"
            )

    async def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        """Fuse all decision signals to choose the best action."""
        if not legal_actions:
            return Action(action_type=ActionType.PASS_PRIORITY, player_id=self.player_id)

        if len(legal_actions) == 1:
            return legal_actions[0]

        # --- Signal 1: Fast heuristic scores ---
        heuristic_scores = self._heuristic_scores(game_state, legal_actions)

        # --- Signal 2: Knowledge Graph bonuses ---
        kg_scores = await self._kg_scores(game_state, legal_actions)

        # --- Signal 3: World model dream search ---
        wm_scores = self._world_model_scores(game_state, legal_actions)

        # --- Signal 4: LLM strategic preference (skip if obvious) ---
        llm_scores = [0.0] * len(legal_actions)
        if not self._is_obvious(heuristic_scores, wm_scores):
            llm_scores = await self._llm_scores(game_state, legal_actions)

        # --- Weighted fusion ---
        cfg = self.config
        final_scores = []
        for i in range(len(legal_actions)):
            score = (
                cfg.heuristic_weight * heuristic_scores[i]
                + cfg.kg_weight * kg_scores[i]
                + cfg.world_model_weight * wm_scores[i]
                + cfg.llm_weight * llm_scores[i]
            )
            final_scores.append(score)

        best_idx = max(range(len(final_scores)), key=lambda i: final_scores[i])

        logger.debug(
            "Fusion decision: action=%s scores=[h=%.2f, kg=%.2f, wm=%.2f, llm=%.2f] → %.2f",
            legal_actions[best_idx].action_type.value,
            heuristic_scores[best_idx],
            kg_scores[best_idx],
            wm_scores[best_idx],
            llm_scores[best_idx],
            final_scores[best_idx],
        )
        return legal_actions[best_idx]

    async def observe(self, game_state: GameState, action: Action) -> None:
        """Update world model hidden state and opponent model on any observed action."""
        await super().observe(game_state, action)

        # Update opponent model
        if self.opponent_model and action.player_id != self.player_id:
            self.opponent_model.observe_behavior(action, game_state)

        # Update world model LSTM hidden state
        if self.world_model is not None and self.tokenizer is not None:
            try:
                import torch
                features = self.tokenizer.encode_state(game_state, self.player_id)
                tensors = {k: torch.tensor(v).unsqueeze(0).float() for k, v in features.items()}
                z, _, _ = self.world_model.encode(tensors)
                action_enc = self.tokenizer.encode_action(action)
                action_t = torch.tensor(action_enc).unsqueeze(0).float()
                _, self._dynamics_hidden, _ = self.world_model.predict(
                    z, action_t, self._dynamics_hidden
                )
            except Exception:
                pass  # Non-critical: hidden state just won't be updated

    # ------------------------------------------------------------------
    # Signal generators
    # ------------------------------------------------------------------

    def _heuristic_scores(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> list[float]:
        """Fast board-evaluation heuristic per action."""
        scores = []
        for action in legal_actions:
            score = 0.0
            if action.action_type == ActionType.CAST_SPELL:
                card = self._find_card(game_state, action.card_instance_id)
                if card and card.is_creature():
                    try:
                        power = int(card.power or 0)
                        score = 0.3 + power * 0.1
                    except (ValueError, TypeError):
                        score = 0.3
                else:
                    score = 0.4  # Non-creature spells (removal, draw)
            elif action.action_type == ActionType.PLAY_LAND:
                score = 0.25
            elif action.action_type == ActionType.DECLARE_ATTACKERS:
                score = 0.5  # Attacking is usually good
            elif action.action_type == ActionType.ACTIVATE_ABILITY:
                score = 0.2
            elif action.action_type == ActionType.PASS_PRIORITY:
                score = 0.05
            scores.append(score)
        return self._normalize(scores)

    async def _kg_scores(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> list[float]:
        """Score actions by KG combo/synergy potential."""
        if self.kg is None:
            return [0.0] * len(legal_actions)

        scores = [0.0] * len(legal_actions)
        try:
            hand_names = [c.name for c in game_state.cards
                         if c.zone == Zone.HAND and c.controller_id == self.player_id]
            bf_names = [c.name for c in game_state.cards
                       if c.zone == Zone.BATTLEFIELD and c.controller_id == self.player_id]
            available = list(set(hand_names + bf_names))

            combos = await self.kg.detect_available_combos(available)
            near_combos = await self.kg.detect_near_combos(available)
            combo_pieces = set()
            for combo in combos + near_combos:
                for piece in combo.get("pieces", []):
                    combo_pieces.add(piece)

            for i, action in enumerate(legal_actions):
                card = self._find_card(game_state, action.card_instance_id)
                if card and card.name in combo_pieces:
                    scores[i] = 0.8  # Part of a combo
                if card:
                    synergies = await self.kg.get_synergies_for(card.name)
                    if synergies:
                        bf_set = set(bf_names)
                        synergy_count = sum(1 for s in synergies if s.get("card") in bf_set)
                        scores[i] = max(scores[i], min(synergy_count * 0.2, 1.0))
        except Exception as e:
            logger.debug("KG scoring failed: %s", e)

        return self._normalize(scores)

    def _world_model_scores(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> list[float]:
        """Score actions via world model dream rollouts."""
        if self.world_model is None or self.tokenizer is None:
            return [0.0] * len(legal_actions)

        try:
            import torch

            features = self.tokenizer.encode_state(game_state, self.player_id)
            tensors = {k: torch.tensor(v).unsqueeze(0).float() for k, v in features.items()}
            z, _, _ = self.world_model.encode(tensors)

            action_encodings = []
            for action in legal_actions:
                enc = self.tokenizer.encode_action(action)
                action_encodings.append(enc)

            import numpy as np
            max_actions = max(len(action_encodings), 1)
            action_dim = len(action_encodings[0]) if action_encodings else 136
            padded = np.zeros((max_actions, action_dim), dtype=np.float32)
            mask = np.zeros(max_actions, dtype=np.float32)
            for i, enc in enumerate(action_encodings):
                padded[i] = enc
                mask[i] = 1.0

            action_t = torch.tensor(padded).unsqueeze(0)
            mask_t = torch.tensor(mask).unsqueeze(0)

            h = self._dynamics_hidden
            if h is None:
                h_dim = self.world_model.config.dynamics.hidden_dim
                h = (torch.zeros(1, 1, h_dim), torch.zeros(1, 1, h_dim))

            # Dream search: for each action, simulate rollouts
            scores = []
            for i, action in enumerate(legal_actions):
                action_enc = torch.tensor(action_encodings[i]).unsqueeze(0)
                total_reward = 0.0
                for _ in range(self.config.dream_rollouts):
                    z_sim = z.clone()
                    h_sim = (h[0].clone(), h[1].clone())
                    rollout_reward = 0.0
                    # First step: use the candidate action
                    z_sim, h_sim, done_p = self.world_model.predict(
                        z_sim, action_enc, h_sim,
                        temperature=self.config.dream_temperature,
                    )
                    for step in range(self.config.dream_depth - 1):
                        if done_p.item() > 0.5:
                            break
                        # Controller picks subsequent actions
                        act_idx, _ = self.world_model.act(
                            z_sim, h_sim[0][:, -1, :] if h_sim[0].dim() == 3 else h_sim[0],
                            action_t.squeeze(0), mask_t.squeeze(0),
                        )
                        next_action = action_t[0, min(act_idx, max_actions - 1)].unsqueeze(0)
                        z_sim, h_sim, done_p = self.world_model.predict(
                            z_sim, next_action, h_sim,
                            temperature=self.config.dream_temperature,
                        )
                        rollout_reward += self.config.dream_temperature ** (-step)
                    total_reward += rollout_reward
                scores.append(total_reward / max(self.config.dream_rollouts, 1))

            return self._normalize(scores)

        except Exception as e:
            logger.debug("World model scoring failed: %s", e)
            return [0.0] * len(legal_actions)

    async def _llm_scores(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> list[float]:
        """Get LLM preference scores for each action."""
        self._ensure_llm()
        if self._llm_agent is None:
            return [0.0] * len(legal_actions)

        try:
            chosen = await self._llm_agent.decide_action(game_state, legal_actions)
            scores = [0.0] * len(legal_actions)
            for i, action in enumerate(legal_actions):
                if action == chosen:
                    scores[i] = 1.0
                elif action.action_type == chosen.action_type:
                    scores[i] = 0.3  # Same action type gets partial credit
            return scores
        except Exception as e:
            logger.debug("LLM scoring failed: %s", e)
            return [0.0] * len(legal_actions)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_obvious(
        self, heuristic: list[float], wm: list[float]
    ) -> bool:
        """Check if the decision is obvious enough to skip LLM."""
        if not self.config.skip_llm_if_obvious:
            return False
        combined = [h + w for h, w in zip(heuristic, wm)]
        if not combined:
            return True
        best = max(combined)
        second = sorted(combined, reverse=True)[1] if len(combined) > 1 else 0.0
        return (best - second) > self.config.obvious_threshold

    def _find_card(self, game_state: GameState, card_instance_id: str | None):
        if not card_instance_id:
            return None
        return next((c for c in game_state.cards if c.instance_id == card_instance_id), None)

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        """Min-max normalize scores to [0, 1]."""
        if not scores:
            return scores
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-8:
            return [0.5] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]
