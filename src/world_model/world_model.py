"""
World Model Orchestrator — Combines V + M + C.

Provides high-level API for:
- Encoding real game states into latent space
- One-step and multi-step predictions ("dreaming")
- Action selection from real or dreamed states
- MCTS-style planning in latent space ("dream search")

This is the main entry point for using the world model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError("World model requires PyTorch. Install with: pip install torch")

from .controller import Controller, ControllerConfig
from .dynamics_model import DynamicsModel, DynamicsModelConfig
from .state_encoder import StateEncoder, StateEncoderConfig


@dataclass
class WorldModelConfig:
    """Top-level configuration combining V + M + C."""

    encoder: StateEncoderConfig = None      # type: ignore[assignment]
    dynamics: DynamicsModelConfig = None     # type: ignore[assignment]
    controller: ControllerConfig = None      # type: ignore[assignment]
    dream_steps: int = 50       # Max steps per dream rollout
    dream_temperature: float = 1.15  # τ > 1 for harder dreams
    num_dream_rollouts: int = 8  # Parallel rollouts for dream search
    discount: float = 0.99       # Reward discount factor

    def __post_init__(self):
        if self.encoder is None:
            self.encoder = StateEncoderConfig()
        if self.dynamics is None:
            self.dynamics = DynamicsModelConfig()
        if self.controller is None:
            self.controller = ControllerConfig()


class WorldModel(nn.Module):
    """Full V + M + C world model for MTG.

    Usage:
        wm = WorldModel(config)

        # Encode a real game state
        z, mu, logvar = wm.encode(tokenized_features)

        # Predict next state after an action
        z_next, h_next, done = wm.predict(z, action, h)

        # Select an action
        action_idx, log_prob = wm.act(z, h, legal_actions, legal_mask)

        # Dream: simulate a full game in latent space
        trajectory = wm.dream(z_start, h_start, action_fn)

        # Dream search: evaluate actions via multiple rollouts
        best_action = wm.dream_search(z, h, legal_actions)

        # Save / load
        wm.save("checkpoints/world_model.pt")
        wm = WorldModel.load("checkpoints/world_model.pt")
    """

    def __init__(self, config: WorldModelConfig | None = None):
        super().__init__()
        self.config = config or WorldModelConfig()

        self.encoder = StateEncoder(self.config.encoder)
        self.dynamics = DynamicsModel(self.config.dynamics)
        self.controller = Controller(self.config.controller)

    # -- Encoding -----------------------------------------------------------

    def encode(
        self, features: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode tokenized game state into latent space.

        Args:
            features: Dict of batched tensors from GameTokenizer

        Returns:
            z, mu, logvar from the state encoder (V)
        """
        return self.encoder.encode(features)

    # -- Prediction ---------------------------------------------------------

    def predict(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Predict next latent state given current state and action.

        Args:
            z: Current latent state
            action: Encoded action
            hidden: RNN hidden state
            temperature: Override default temperature

        Returns:
            z_next, hidden_next, done_probability
        """
        temp = temperature or self.config.dream_temperature
        return self.dynamics.step(z, action, hidden, temperature=temp)

    # -- Action Selection ---------------------------------------------------

    def act(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        legal_action_encodings: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor]:
        """Select an action using the controller.

        Args:
            z: Current latent state
            h: Dynamics model hidden state
            legal_action_encodings: Encoded legal actions
            legal_action_mask: Mask for valid actions
            deterministic: Greedy selection

        Returns:
            action_idx, log_probability
        """
        return self.controller.select_action(
            z, h, legal_action_encodings, legal_action_mask, deterministic
        )

    # -- Dreaming -----------------------------------------------------------

    def dream(
        self,
        z_start: torch.Tensor,
        hidden_start: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        legal_actions_fn=None,
        temperature: float | None = None,
    ) -> list[dict]:
        """Roll out a full game trajectory in latent space ("dreaming").

        The agent plays inside its own world model, never touching the
        real game engine. This enables cheap generation of training data.

        Args:
            z_start: Starting latent state
            hidden_start: Starting RNN hidden state
            legal_actions_fn: Callable(z, h) → (action_encodings, mask).
                             In dream mode, we approximate legal actions.
            temperature: Dream temperature (higher = harder)

        Returns:
            List of step dicts with z, action, reward, done at each step.
        """
        temp = temperature or self.config.dream_temperature
        z = z_start
        h = hidden_start or self.dynamics.initial_hidden(z.size(0))
        trajectory = []

        for step in range(self.config.dream_steps):
            # Get hidden vector for controller
            h_vec = self.dynamics.get_hidden_state_vector(h)

            # Get available actions (in dream mode, this is approximate)
            if legal_actions_fn is not None:
                action_encs, action_mask = legal_actions_fn(z, h_vec)
            else:
                # Fallback: use controller's raw output as action
                action_vec = self.controller(z, h_vec)
                # Package as single "action"
                action_encs = action_vec.unsqueeze(1)
                action_mask = torch.ones(z.size(0), 1, device=z.device)

            # Controller selects action
            action_idx, log_prob = self.controller.select_action(
                z, h_vec, action_encs, action_mask
            )

            # Use the selected action encoding
            batch_idx = torch.arange(z.size(0), device=z.device)
            action = action_encs[batch_idx, action_idx]

            # Dynamics model predicts next state
            z_next, h_next, done_prob = self.dynamics.step(z, action, h, temperature=temp)

            # Predict reward
            h_flat = self.dynamics.get_hidden_state_vector(h)
            reward_pred = self.dynamics.reward_head(h_flat).squeeze(-1)

            trajectory.append({
                "z": z.detach(),
                "action_idx": action_idx,
                "action": action.detach(),
                "log_prob": log_prob,  # keep gradient path for policy gradient
                "z_next": z_next.detach(),
                "reward": reward_pred.detach(),
                "done_prob": done_prob.detach(),
            })

            z = z_next
            h = h_next

            # Stop if game predicted as over
            if done_prob.mean().item() > 0.5:
                break

        return trajectory

    def dream_search(
        self,
        z: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]],
        legal_action_encodings: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
        num_rollouts: int | None = None,
        rollout_depth: int = 10,
    ) -> tuple[int, torch.Tensor]:
        """MCTS-style search in latent space.

        For each legal action, simulate multiple rollouts using the dynamics
        model and score based on predicted cumulative reward.

        Args:
            z: Current latent state (batch=1)
            hidden: RNN hidden state
            legal_action_encodings: (1, num_actions, action_dim)
            legal_action_mask: (1, num_actions)
            num_rollouts: Rollouts per action to average
            rollout_depth: Steps to simulate per rollout

        Returns:
            best_action_idx, raw action scores
        """
        n_rollouts = num_rollouts or self.config.num_dream_rollouts
        num_actions = legal_action_encodings.size(1)
        action_scores = torch.zeros(num_actions, device=z.device)

        if legal_action_mask is not None:
            valid_actions = legal_action_mask[0].nonzero(as_tuple=True)[0]
        else:
            valid_actions = torch.arange(num_actions, device=z.device)

        for action_idx in valid_actions:
            action = legal_action_encodings[0, action_idx].unsqueeze(0)
            total_reward = 0.0

            for _ in range(n_rollouts):
                z_sim = z.clone()
                h_sim = (hidden[0].clone(), hidden[1].clone()) if hidden else self.dynamics.initial_hidden(1)

                # Apply the candidate action
                z_sim, h_sim, done = self.dynamics.step(z_sim, action, h_sim, temperature=self.config.dream_temperature)
                rollout_reward = 0.0
                discount = 1.0

                # Continue rollout with controller policy
                for depth in range(rollout_depth):
                    if done.item() > 0.5:
                        break

                    h_vec = self.dynamics.get_hidden_state_vector(h_sim)
                    # Use forward output as action (simplified — no legal action check in dreams)
                    a_vec = self.controller(z_sim, h_vec)

                    z_sim, h_sim, done = self.dynamics.step(z_sim, a_vec, h_sim, temperature=self.config.dream_temperature)
                    r = self.dynamics.reward_head(self.dynamics.get_hidden_state_vector(h_sim))
                    rollout_reward += discount * r.item()
                    discount *= self.config.discount

                total_reward += rollout_reward

            action_scores[action_idx] = total_reward / n_rollouts

        best_idx = action_scores.argmax().item()
        return best_idx, action_scores

    # -- Persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        """Save full model state (V + M + C + config)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> WorldModel:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device)
