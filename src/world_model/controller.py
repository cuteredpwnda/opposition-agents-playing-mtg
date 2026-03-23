"""
C Model — Controller (Policy Network).

Maps the current latent state z and dynamics model hidden state h to
an action distribution. Deliberately kept simple — most of the model's
complexity lives in V and M.

Reference: Ha & Schmidhuber (2018), Section "Controller (C) Model"

The controller can be trained with:
- CMA-ES: Evolution strategy (original paper approach, ~1K params)
- Policy gradient: REINFORCE / PPO on dream rollouts
- Distillation: Initialize from LLM agent decisions, then fine-tune
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError(
        "Controller requires PyTorch. "
        "Install with: pip install torch"
    )

import numpy as np


@dataclass
class ControllerConfig:
    """Configuration for the controller (C model)."""

    latent_dim: int = 256          # Dimension of z from state encoder
    hidden_state_dim: int = 512    # Dimension of h from dynamics model
    action_dim: int = 136          # Dimension of output action vector
    max_legal_actions: int = 50    # Max number of legal actions to score
    use_hidden_layer: bool = False # Add a hidden layer (more expressive but more params)
    hidden_layer_dim: int = 64     # Size of optional hidden layer


class Controller(nn.Module):
    """C Model — Compact policy network.

    Linear version (original World Models):
        a_t = W_c [z_t; h_t] + b_c
        ~1K parameters, trainable with CMA-ES

    MLP version (for harder tasks):
        a_t = MLP([z_t; h_t])
        ~5-10K parameters, trainable with policy gradient

    Usage:
        controller = Controller(config)

        # Get raw action logits
        action_logits = controller(z, h)

        # Score specific legal actions
        scores = controller.score_actions(z, h, encoded_legal_actions)
        best_idx = scores.argmax()
    """

    def __init__(self, config: ControllerConfig | None = None):
        super().__init__()
        self.config = config or ControllerConfig()
        c = self.config

        input_dim = c.latent_dim + c.hidden_state_dim

        if c.use_hidden_layer:
            self.policy = nn.Sequential(
                nn.Linear(input_dim, c.hidden_layer_dim),
                nn.Tanh(),
                nn.Linear(c.hidden_layer_dim, c.action_dim),
            )
        else:
            # Linear controller — minimal params, original World Models approach
            self.policy = nn.Linear(input_dim, c.action_dim)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute raw action preference vector.

        Args:
            z: (batch, latent_dim) — current latent state from V
            h: (batch, hidden_state_dim) — hidden state from M

        Returns:
            action_logits: (batch, action_dim) — raw action scores
        """
        x = torch.cat([z, h], dim=-1)
        return self.policy(x)

    def score_actions(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        legal_action_encodings: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score a set of legal actions using dot-product attention.

        Instead of outputting a fixed-size action vector, compute compatibility
        scores between the controller's preference vector and each legal action's
        encoding. This handles MTG's variable action space.

        Args:
            z: (batch, latent_dim)
            h: (batch, hidden_state_dim)
            legal_action_encodings: (batch, num_actions, action_dim) — encoded legal actions
            legal_action_mask: (batch, num_actions) — 1 for valid, 0 for padding

        Returns:
            scores: (batch, num_actions) — log-probabilities over legal actions
        """
        # Get controller's action preference vector
        preference = self.forward(z, h)  # (B, action_dim)

        # Dot product with each legal action encoding
        scores = torch.einsum("bd,bnd->bn", preference, legal_action_encodings)

        # Mask out illegal actions
        if legal_action_mask is not None:
            scores = scores.masked_fill(legal_action_mask == 0, float("-inf"))

        # Return log-probabilities
        return F.log_softmax(scores, dim=-1)

    def select_action(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        legal_action_encodings: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor]:
        """Select an action index from legal actions.

        Args:
            z, h: Latent state and hidden state
            legal_action_encodings: Encoded legal actions
            legal_action_mask: Mask for valid actions
            deterministic: If True, always choose highest-scoring action

        Returns:
            action_idx: Index of chosen action
            log_prob: Log-probability of chosen action (for policy gradient)
        """
        log_probs = self.score_actions(z, h, legal_action_encodings, legal_action_mask)

        if deterministic:
            action_idx = log_probs.argmax(dim=-1).item()
        else:
            probs = torch.exp(log_probs)
            action_idx = torch.multinomial(probs, num_samples=1).item()

        return action_idx, log_probs[:, action_idx]

    def num_parameters(self) -> int:
        """Count total trainable parameters (should be small!)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -- CMA-ES interface ---------------------------------------------------

    def get_flat_params(self) -> np.ndarray:
        """Extract all parameters as a flat numpy array (for CMA-ES)."""
        params = []
        for p in self.parameters():
            params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_flat_params(self, flat_params: np.ndarray) -> None:
        """Set parameters from a flat numpy array (for CMA-ES)."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = torch.from_numpy(
                flat_params[offset : offset + numel].reshape(p.shape)
            ).float().to(p.device)
            offset += numel
