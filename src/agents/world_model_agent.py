"""
WorldModelAgent — MTGAgent subclass that uses the V+M+C world model.

Plays MTG by:
1. Encoding the real game state into latent space (V)
2. Optionally running dream search to evaluate actions (dream MCTS)
3. Selecting the best action via the Controller (C) or dream search
4. Updating beliefs via the Dynamics Model (M) on opponent actions

Supports two modes:
- Direct policy: Fast — just run C(z, h) → action. Good for real-time play.
- Dream search: Slower — simulate multiple futures per action. Stronger play.
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    import torch
    import numpy as np
except ImportError:
    raise ImportError("WorldModelAgent requires PyTorch and NumPy.")

from src.agents.base_agent import MTGAgent
from src.engine.game_state import Action, GameState, Zone
from src.world_model.card_embeddings import CardEmbeddingModel
from src.world_model.game_tokenizer import GameTokenizer
from src.world_model.world_model import WorldModel
from src.world_model.kg_encoder import KGContextEncoder

logger = logging.getLogger(__name__)


class WorldModelAgent(MTGAgent):
    """An agent that uses a trained world model to play MTG.

    Usage:
        # Load trained world model
        world_model = WorldModel.load("checkpoints/world_model_final.pt")
        tokenizer = GameTokenizer(card_embeddings=card_embed_model)

        agent = WorldModelAgent(
            player_id="wm_agent",
            world_model=world_model,
            tokenizer=tokenizer,
            mode="dream_search",
        )

        # In game loop:
        action = await agent.decide_action(game_state, legal_actions)
    """

    def __init__(
        self,
        player_id: str,
        world_model: WorldModel,
        tokenizer: GameTokenizer,
        kg_encoder: KGContextEncoder | None = None,
        name: str = "WorldModelAgent",
        mode: str = "direct",           # "direct" or "dream_search"
        dream_rollouts: int = 8,
        dream_depth: int = 10,
        deterministic: bool = False,
        device: str = "cpu",
    ):
        super().__init__(player_id=player_id, name=name)
        self.world_model = world_model.to(device)
        self.tokenizer = tokenizer
        self.kg_encoder = kg_encoder
        self.mode = mode
        self.dream_rollouts = dream_rollouts
        self.dream_depth = dream_depth
        self.deterministic = deterministic
        self.device = device

        # Persistent hidden state (updated each step via dynamics model)
        self._hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._last_z: Optional[torch.Tensor] = None

    async def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        """Choose an action using the world model.

        1. Encode current game state → z
        2. Encode each legal action → action embeddings
        3. Score actions via controller (direct) or dream search
        4. Return the highest-scoring legal action
        """
        if not legal_actions:
            raise ValueError("No legal actions available")

        # Drop CONCEDE from the candidate set: the rules engine always
        # offers it for debug parity, and an untrained or stochastic
        # controller would otherwise forfeit games at random.  Keeping
        # PASS_PRIORITY ensures the agent can still cleanly end its turn.
        from src.engine.game_state import ActionType as _AT
        non_concede = [a for a in legal_actions if a.action_type != _AT.CONCEDE]
        if non_concede:
            legal_actions = non_concede

        with torch.no_grad():
            # Step 1: Encode game state
            z, h = self._encode_state(game_state)

            # Step 2: Encode legal actions
            action_encodings, action_mask = self._encode_actions(legal_actions)

            # Step 3: Select action
            if self.mode == "dream_search" and len(legal_actions) > 1:
                action_idx = self._dream_search(z, h, action_encodings, action_mask)
            else:
                action_idx = self._direct_policy(z, h, action_encodings, action_mask)

        chosen_action = legal_actions[action_idx]
        logger.debug(
            "WorldModelAgent chose: %s %s",
            chosen_action.action_type.value,
            chosen_action.card_instance_id or "",
        )
        # Reasoning trace ----------------------------------------------------
        try:
            from src.agents.reasoning import ReasoningTrace
            beliefs: dict = {
                "mode": self.mode,
                "z_norm": float(z.norm().item()) if z is not None else None,
                "deterministic": self.deterministic,
            }
            scores_list: list[float] = []
            top_payload: list[dict] = []
            if getattr(self, "_last_policy_info", None) and isinstance(self._last_policy_info, dict):
                logits = self._last_policy_info.get("logits")
                probs = self._last_policy_info.get("probs")
                if probs is not None:
                    try:
                        scores_list = [float(p) for p in probs.flatten().tolist()][: len(legal_actions)]
                    except Exception:
                        scores_list = []
                if logits is not None and not scores_list:
                    try:
                        scores_list = [float(x) for x in logits.flatten().tolist()][: len(legal_actions)]
                    except Exception:
                        scores_list = []
                value = self._last_policy_info.get("value")
                if value is not None:
                    try:
                        beliefs["value_estimate"] = float(value.item() if hasattr(value, "item") else value)
                    except Exception:
                        pass
            elif getattr(self, "_last_dream_info", None) and isinstance(self._last_dream_info, dict):
                raw = self._last_dream_info.get("scores")
                try:
                    scores_list = [float(x) for x in raw.flatten().tolist()][: len(legal_actions)]
                except Exception:
                    scores_list = []
                beliefs["dream_rollouts"] = self.dream_rollouts
                beliefs["dream_depth"] = self.dream_depth

            if scores_list:
                ranked = sorted(enumerate(scores_list), key=lambda kv: kv[1], reverse=True)[:5]
                for idx, sc in ranked:
                    if idx >= len(legal_actions):
                        continue
                    a = legal_actions[idx]
                    top_payload.append({
                        "action": f"{a.action_type.value}({a.card_instance_id or ''})",
                        "score": sc,
                    })
            self.set_reasoning(ReasoningTrace(
                agent_kind="world_model",
                rationale=f"{self.mode} policy over {len(legal_actions)} legal actions",
                legal_action_count=len(legal_actions),
                chosen_index=action_idx,
                scores=scores_list,
                top_candidates=top_payload,
                beliefs=beliefs,
            ))
        except Exception as e:  # never let reasoning logging break gameplay
            logger.debug("Failed to record WorldModelAgent reasoning: %s", e)
        return chosen_action

    async def observe(self, game_state: GameState, action: Action) -> None:
        """Update the dynamics model hidden state after observing an action.

        This keeps the RNN's memory current even when the opponent acts.
        """
        await super().observe(game_state, action)

        if self._last_z is not None:
            with torch.no_grad():
                action_enc = torch.from_numpy(
                    self.tokenizer.encode_action(action)
                ).float().unsqueeze(0).to(self.device)

                _, self._hidden, _ = self.world_model.predict(
                    self._last_z, action_enc, self._hidden
                )

    def reset(self) -> None:
        """Reset for a new game."""
        super().reset()
        self._hidden = None
        self._last_z = None

    # -- Private helpers ----------------------------------------------------

    def _encode_state(self, game_state: GameState) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode game state into latent space and get hidden state vector."""
        # Tokenize
        player_idx = self._get_player_index(game_state)
        features = self.tokenizer.encode_state(game_state, player_idx)

        # KG context embedding (if available)
        kg_embedding = None
        if self.kg_encoder is not None:
            visible_ids = self._get_visible_card_names(game_state, player_idx)
            kg_embedding = self.kg_encoder(visible_ids, device=self.device)

        # Convert to batched tensors
        feature_tensors = {
            k: torch.from_numpy(v).float().unsqueeze(0).to(self.device)
            for k, v in features.items()
        }

        # Encode with optional KG context
        z, _, _ = self.world_model.encode(feature_tensors, kg_embedding=kg_embedding)
        self._last_z = z

        # Initialize hidden state if needed
        if self._hidden is None:
            self._hidden = self.world_model.dynamics.initial_hidden(batch_size=1)
            if self._hidden is not None:
                self._hidden = (
                    self._hidden[0].to(self.device),
                    self._hidden[1].to(self.device),
                )

        # Get hidden vector for controller
        h_vec = self.world_model.dynamics.get_hidden_state_vector(self._hidden)
        if h_vec.device != z.device:
            h_vec = h_vec.to(self.device)

        return z, h_vec

    def _encode_actions(
        self, legal_actions: list[Action]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode legal actions into tensor form."""
        encodings = []
        for action in legal_actions:
            enc = self.tokenizer.encode_action(action)
            encodings.append(torch.from_numpy(enc).float())

        # Pad to fixed size
        max_actions = max(len(encodings), 1)
        action_dim = encodings[0].shape[0]
        padded = torch.zeros(1, max_actions, action_dim, device=self.device)
        mask = torch.zeros(1, max_actions, device=self.device)

        for i, enc in enumerate(encodings):
            padded[0, i] = enc.to(self.device)
            mask[0, i] = 1.0

        return padded, mask

    def _direct_policy(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        action_encodings: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> int:
        """Select action directly via the controller."""
        action_idx, info = self.world_model.controller.select_action(
            z, h, action_encodings, action_mask, deterministic=self.deterministic
        )
        # Stash for reasoning trace.
        self._last_policy_info = info
        self._last_dream_info = None
        return action_idx

    def _dream_search(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        action_encodings: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> int:
        """Select action via dream search (simulated rollouts)."""
        best_idx, scores = self.world_model.dream_search(
            z, self._hidden, action_encodings, action_mask,
            num_rollouts=self.dream_rollouts,
            rollout_depth=self.dream_depth,
        )
        self._last_dream_info = {"scores": scores}
        self._last_policy_info = None
        return best_idx

    def _get_player_index(self, game_state: GameState) -> int:
        """Find which player index we are in the game state."""
        for i, player in enumerate(game_state.players):
            if player.name == self.player_id:
                return i
        return 0  # Default to player 0

    def _get_visible_card_names(self, game_state: GameState, player_index: int) -> list[list[str]]:
        """Collect all visible card names for KG context encoder."""
        player_id = game_state.players[player_index].player_id
        names = []

        # Own hand + battlefield
        own_hand = [c.name for c in game_state.cards if c.owner_id == player_id and c.zone == Zone.HAND]
        own_bf = [c.name for c in game_state.cards if c.owner_id == player_id and c.zone == Zone.BATTLEFIELD]

        # Opponent battlefield for all other players
        opp_bf = [c.name for c in game_state.cards if c.owner_id != player_id and c.zone == Zone.BATTLEFIELD]

        names.append(own_hand[: self.tokenizer.config.max_hand_size])
        names.append(own_bf[: self.tokenizer.config.max_battlefield_size])
        names.append(opp_bf[: self.tokenizer.config.max_battlefield_size])

        return names

