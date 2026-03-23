"""
M Model — Dynamics Model (MDN-RNN / Transformer).

Predicts the probability distribution of the next latent state:
    P(z_{t+1}, done_{t+1} | z_t, a_t, h_t)

Uses a Mixture Density Network (MDN) output layer to capture multi-modal
future states — critical for MTG where opponent responses create branching
futures (e.g., "will they counter my spell?").

The temperature parameter τ controls uncertainty during dream generation:
- τ > 1.0: More stochastic (harder dreams, prevents policy exploitation)
- τ < 1.0: More deterministic (easier dreams, but policy may overfit)

Reference: Ha & Schmidhuber (2018), Section "MDN-RNN (M) Model"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal, Normal
except ImportError:
    raise ImportError(
        "Dynamics model requires PyTorch. "
        "Install with: pip install torch"
    )

import math


@dataclass
class DynamicsModelConfig:
    """Configuration for the dynamics model (M model)."""

    latent_dim: int = 256        # Dimension of latent state z
    action_dim: int = 136        # Dimension of encoded action (8 types + 128 card embed)
    hidden_dim: int = 512        # RNN hidden state dimension
    num_layers: int = 2          # Number of LSTM layers
    num_gaussians: int = 5       # Number of Gaussian components in MDN
    temperature: float = 1.0     # Sampling temperature τ
    use_transformer: bool = False  # Use Transformer instead of LSTM
    transformer_heads: int = 8
    transformer_layers: int = 4
    max_sequence_length: int = 200  # Max turns * decisions per turn
    dropout: float = 0.1


class MDNHead(nn.Module):
    """Mixture Density Network output head.

    Outputs parameters of a Gaussian mixture model:
    - π (mixing coefficients): which Gaussian to sample from
    - μ (means): center of each Gaussian
    - σ (standard deviations): spread of each Gaussian

    For MTG, each Gaussian mode can represent a different game outcome
    (e.g., spell resolves vs. gets countered).
    """

    def __init__(self, input_dim: int, output_dim: int, num_gaussians: int):
        super().__init__()
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians

        # π: mixing coefficients (which Gaussian to choose)
        self.pi_layer = nn.Linear(input_dim, num_gaussians)
        # μ: means of each Gaussian
        self.mu_layer = nn.Linear(input_dim, num_gaussians * output_dim)
        # σ: log-std of each Gaussian (log for numerical stability)
        self.logsigma_layer = nn.Linear(input_dim, num_gaussians * output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim) — hidden state from RNN/Transformer

        Returns:
            pi: (batch, num_gaussians) — log mixing coefficients
            mu: (batch, num_gaussians, output_dim) — means
            sigma: (batch, num_gaussians, output_dim) — std devs
        """
        pi = self.pi_layer(x)  # (B, K)
        mu = self.mu_layer(x).view(-1, self.num_gaussians, self.output_dim)  # (B, K, D)
        logsigma = self.logsigma_layer(x).view(-1, self.num_gaussians, self.output_dim)
        # Clamp log-sigma to prevent numerical issues
        logsigma = torch.clamp(logsigma, min=-7.0, max=2.0)
        sigma = torch.exp(logsigma)  # (B, K, D)

        return pi, mu, sigma

    def sample(
        self, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample from the Gaussian mixture.

        Args:
            pi: (batch, K) — log mixing coefficients
            mu: (batch, K, D) — means
            sigma: (batch, K, D) — std devs
            temperature: Sampling temperature (>1 = more random)

        Returns:
            z: (batch, D) — sampled latent vector
        """
        # Apply temperature to mixing coefficients
        pi_scaled = pi / temperature
        pi_probs = F.softmax(pi_scaled, dim=-1)

        # Select which Gaussian to sample from
        k = Categorical(pi_probs).sample()  # (B,)

        # Gather the selected Gaussian's parameters
        batch_indices = torch.arange(mu.size(0), device=mu.device)
        selected_mu = mu[batch_indices, k]  # (B, D)
        selected_sigma = sigma[batch_indices, k] * temperature  # (B, D)

        # Sample: z = μ + σ * ε
        eps = torch.randn_like(selected_mu)
        z = selected_mu + selected_sigma * eps

        return z

    def log_prob(
        self, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log-probability of a target under the mixture.

        Args:
            pi: (batch, K) — log mixing coefficients
            mu: (batch, K, D) — means
            sigma: (batch, K, D) — std devs
            target: (batch, D) — observed next latent state

        Returns:
            log_prob: (batch,) — log probability
        """
        K = self.num_gaussians
        target_expanded = target.unsqueeze(1).expand_as(mu)  # (B, K, D)

        # Log probability under each Gaussian
        var = sigma.pow(2)
        log_normal = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(sigma)
            - 0.5 * (target_expanded - mu).pow(2) / var
        )
        log_normal = log_normal.sum(dim=-1)  # (B, K) — sum over dimensions

        # Log mixing coefficients
        log_pi = F.log_softmax(pi, dim=-1)  # (B, K)

        # Log-sum-exp over components
        log_prob = torch.logsumexp(log_pi + log_normal, dim=-1)  # (B,)

        return log_prob


class DynamicsModel(nn.Module):
    """M Model — Predicts next latent state distribution.

    Architecture:
        Input: [z_t, a_t] concatenated
        → LSTM (or Transformer) → hidden state h_t
        → MDN head → P(z_{t+1}) as Gaussian mixture
        → Done head → P(game_over)

    Usage:
        model = DynamicsModel(config)

        # Step-by-step (for dream rollouts):
        h = model.initial_hidden(batch_size=1)
        z_next, h_next, done_prob = model.step(z, action, h, temperature=1.0)

        # Sequence training:
        loss = model.sequence_loss(z_sequence, action_sequence, done_sequence)
    """

    def __init__(self, config: DynamicsModelConfig | None = None):
        super().__init__()
        self.config = config or DynamicsModelConfig()
        c = self.config

        input_dim = c.latent_dim + c.action_dim

        if c.use_transformer:
            self._build_transformer(input_dim, c)
        else:
            self._build_lstm(input_dim, c)

        # MDN output head for predicting next z
        self.mdn = MDNHead(c.hidden_dim, c.latent_dim, c.num_gaussians)

        # Game-over prediction head
        self.done_head = nn.Sequential(
            nn.Linear(c.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Reward prediction head (optional — for dream training)
        self.reward_head = nn.Sequential(
            nn.Linear(c.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _build_lstm(self, input_dim: int, c: DynamicsModelConfig) -> None:
        """Build LSTM-based sequence model."""
        self.model_type = "lstm"
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=c.hidden_dim,
            num_layers=c.num_layers,
            batch_first=True,
            dropout=c.dropout if c.num_layers > 1 else 0.0,
        )

    def _build_transformer(self, input_dim: int, c: DynamicsModelConfig) -> None:
        """Build Transformer-based sequence model."""
        self.model_type = "transformer"
        self.input_proj = nn.Linear(input_dim, c.hidden_dim)
        self.pos_encoding = nn.Embedding(c.max_sequence_length, c.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.hidden_dim,
            nhead=c.transformer_heads,
            dim_feedforward=c.hidden_dim * 4,
            dropout=c.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=c.transformer_layers)
        self.output_proj = nn.Linear(c.hidden_dim, c.hidden_dim)

    # -- Public API ---------------------------------------------------------

    def initial_hidden(self, batch_size: int = 1) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Get initial hidden state for LSTM rollouts.

        Returns None for Transformer (uses causal attention instead).
        """
        if self.model_type == "lstm":
            c = self.config
            device = next(self.parameters()).device
            h0 = torch.zeros(c.num_layers, batch_size, c.hidden_dim, device=device)
            c0 = torch.zeros(c.num_layers, batch_size, c.hidden_dim, device=device)
            return (h0, c0)
        return None

    def step(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Single-step prediction (for dream rollouts).

        Args:
            z: (batch, latent_dim) — current latent state
            action: (batch, action_dim) — encoded action taken
            hidden: LSTM hidden state (h, c) or None for Transformer
            temperature: Sampling temperature τ

        Returns:
            z_next: (batch, latent_dim) — sampled next latent state
            hidden_next: updated hidden state
            done_prob: (batch,) — probability that game is over
        """
        # Concatenate z and action
        x = torch.cat([z, action], dim=-1)  # (B, latent_dim + action_dim)

        if self.model_type == "lstm":
            x = x.unsqueeze(1)  # (B, 1, input_dim) — single time step
            output, hidden_next = self.lstm(x, hidden)
            h = output.squeeze(1)  # (B, hidden_dim)
        else:
            # For transformer, we'd need to maintain a context window
            # Simplified: project through transformer with single token
            x = self.input_proj(x).unsqueeze(1)  # (B, 1, hidden_dim)
            h = self.transformer(x).squeeze(1)
            h = self.output_proj(h)
            hidden_next = None

        # Predict next z from MDN
        pi, mu, sigma = self.mdn(h)
        z_next = self.mdn.sample(pi, mu, sigma, temperature=temperature)

        # Predict game-over probability
        done_logit = self.done_head(h).squeeze(-1)
        done_prob = torch.sigmoid(done_logit)

        return z_next, hidden_next, done_prob

    def sequence_loss(
        self,
        z_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        done_sequence: torch.Tensor,
        reward_sequence: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute loss over a full game trajectory.

        Args:
            z_sequence: (batch, seq_len, latent_dim) — encoded states
            action_sequence: (batch, seq_len, action_dim) — encoded actions
            done_sequence: (batch, seq_len) — game-over flags (0 or 1)
            reward_sequence: (batch, seq_len) — optional rewards

        Returns:
            Dict with "total", "z_pred", "done", "reward" losses.
        """
        B, T, _ = z_sequence.shape

        # Input: [z_t, a_t] for t=0..T-2; Target: z_{t+1} for t=1..T-1
        z_input = z_sequence[:, :-1]       # (B, T-1, D)
        a_input = action_sequence[:, :-1]  # (B, T-1, A)
        z_target = z_sequence[:, 1:]       # (B, T-1, D)
        done_target = done_sequence[:, 1:]  # (B, T-1)

        # Concatenate inputs
        x = torch.cat([z_input, a_input], dim=-1)  # (B, T-1, D+A)

        if self.model_type == "lstm":
            output, _ = self.lstm(x)  # (B, T-1, hidden_dim)
        else:
            x = self.input_proj(x)
            positions = torch.arange(T - 1, device=x.device).unsqueeze(0).expand(B, -1)
            x = x + self.pos_encoding(positions)
            # Causal mask
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T - 1, device=x.device)
            output = self.transformer(x, mask=causal_mask)
            output = self.output_proj(output)

        # Flatten for loss computation
        h_flat = output.reshape(B * (T - 1), -1)
        z_target_flat = z_target.reshape(B * (T - 1), -1)
        done_target_flat = done_target.reshape(B * (T - 1))

        # MDN loss: negative log-likelihood of actual z_{t+1}
        pi, mu, sigma = self.mdn(h_flat)
        z_log_prob = self.mdn.log_prob(pi, mu, sigma, z_target_flat)
        z_loss = -z_log_prob.mean()

        # Done prediction loss: binary cross-entropy
        done_logits = self.done_head(h_flat).squeeze(-1)
        done_loss = F.binary_cross_entropy_with_logits(done_logits, done_target_flat)

        # Reward prediction loss (optional)
        reward_loss = torch.tensor(0.0, device=z_sequence.device)
        if reward_sequence is not None:
            reward_target = reward_sequence[:, 1:].reshape(B * (T - 1))
            reward_pred = self.reward_head(h_flat).squeeze(-1)
            reward_loss = F.mse_loss(reward_pred, reward_target)

        total = z_loss + done_loss + reward_loss

        return {
            "total": total,
            "z_pred": z_loss,
            "done": done_loss,
            "reward": reward_loss,
        }

    def get_hidden_state_vector(
        self, hidden: Optional[tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Extract a flat hidden state vector for the Controller.

        For LSTM: concatenates the output hidden state h from all layers.
        For Transformer: returns the last output vector.

        Returns:
            h_vector: (batch, hidden_dim) or (batch, num_layers * hidden_dim)
        """
        if self.model_type == "lstm" and hidden is not None:
            h, c = hidden
            # Use the last layer's hidden state
            return h[-1]  # (B, hidden_dim)
        # For transformer, the caller should use the output directly
        return torch.zeros(1, self.config.hidden_dim)
