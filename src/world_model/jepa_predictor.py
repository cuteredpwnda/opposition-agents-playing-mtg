"""
JEPA Predictor — predicts the next latent embedding from z_t and action a_t.

In a Joint Embedding Predictive Architecture (JEPA), the predictor is the
"imagination engine": given the current encoded state z_t and the action
taken a_t, it predicts what embedding z_{t+1} the encoder would produce
for the resulting next state.

Key difference from DIAMOND / MDN-RNN:
  - Prediction target is z_{t+1} in embedding space — NOT pixels or raw features.
  - No decoder. No reconstruction. No pixel loss.
  - Two loss terms only (LeWM-style):
      prediction_loss = MSE( ẑ_{t+1},  sg(z_{t+1}) )
      kl_loss         = KL ( q(z) || N(0, I) )
      total           = prediction_loss + β * kl_loss
    where sg = stop-gradient, β is the single tunable hyperparameter.

This simplicity (vs. 6-term losses in previous JEPA variants) comes from the
Gaussian regularizer: it prevents representation collapse without needing
exponential moving average target networks or VQ-VAE discrete tokens.

The predictor architecture is a small Transformer (pre-norm, 2 layers) that
processes a single token = concat(z_t, a_t). An MLP fallback is available.

Reference: Maes et al. (2026) "LeWorldModel: Stable End-to-End JEPA from Pixels"
           arXiv:2603.19312v1
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError("JEPAPredictor requires PyTorch.")


@dataclass
class JEPAPredictorConfig:
    """Configuration for the JEPA predictor.

    Attributes:
        latent_dim:      Dimension of z — must match StateEncoderConfig.latent_dim.
        action_dim:      Encoded action dimension — must match DynamicsModelConfig.action_dim.
        hidden_dim:      Internal width of the predictor.
        num_layers:      Number of Transformer encoder layers (use_transformer=True).
        num_heads:       Attention heads (use_transformer=True).
        use_transformer: Transformer (True) or MLP (False).
        dropout:         Dropout probability.
        jepa_beta:       Weight of the Gaussian KL regularizer.
                         This is the *one* hyperparameter in LeWM.
                         Typical range: 0.1 – 2.0. Start with 1.0.
    """

    latent_dim: int = 256
    action_dim: int = 136
    hidden_dim: int = 512
    num_layers: int = 2
    num_heads: int = 4
    use_transformer: bool = True
    dropout: float = 0.1
    jepa_beta: float = 1.0


class JEPAPredictor(nn.Module):
    """Predicts the next latent embedding ẑ_{t+1} from (z_t, a_t).

    Training is driven by :meth:`jepa_loss`, which computes the prediction
    MSE against the stop-gradient target z_{t+1} plus a Gaussian KL
    regularizer on the encoder's (mu, logvar).

    The predictor is used in *two* contexts:

    1. **Training** — supervised by real game transitions:
          z_t, a_t → ẑ_{t+1}
          loss = jepa_loss(ẑ_{t+1}, z_{t+1}, mu_t, logvar_t)

    2. **Dream search** (rollout) — unrolling imagined futures without a
       rules engine, used as the M model's fast path in WorldModelAgent.

    Usage::

        predictor = JEPAPredictor(JEPAPredictorConfig(latent_dim=256))
        z_hat_next = predictor(z_t, a_t)
        losses = predictor.jepa_loss(z_hat_next, z_next, mu, logvar)
        losses["total"].backward()
    """

    def __init__(self, config: JEPAPredictorConfig | None = None):
        super().__init__()
        self.config = config or JEPAPredictorConfig()
        c = self.config

        input_dim = c.latent_dim + c.action_dim

        if c.use_transformer:
            self.input_proj = nn.Linear(input_dim, c.hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=c.hidden_dim,
                nhead=c.num_heads,
                dim_feedforward=c.hidden_dim * 2,
                dropout=c.dropout,
                batch_first=True,
                norm_first=True,   # Pre-norm: more stable than post-norm
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=c.num_layers
            )
            self.output_proj = nn.Linear(c.hidden_dim, c.latent_dim)
        else:
            # Lightweight MLP fallback — good for quick experiments
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, c.hidden_dim),
                nn.GELU(),
                nn.Dropout(c.dropout),
                nn.Linear(c.hidden_dim, c.hidden_dim),
                nn.GELU(),
                nn.Dropout(c.dropout),
                nn.Linear(c.hidden_dim, c.latent_dim),
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """Predict the embedding of the next game state.

        Args:
            z_t: (batch, latent_dim) — current latent state.
            a_t: (batch, action_dim) — encoded action taken at time t.

        Returns:
            z_hat_next: (batch, latent_dim) — predicted next-state embedding.
        """
        x = torch.cat([z_t, a_t], dim=-1)  # (B, latent_dim + action_dim)

        if self.config.use_transformer:
            x = self.input_proj(x).unsqueeze(1)   # (B, 1, hidden_dim)
            x = self.transformer(x)                # (B, 1, hidden_dim)
            return self.output_proj(x.squeeze(1))  # (B, latent_dim)
        else:
            return self.mlp(x)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def jepa_loss(
        self,
        z_hat_next: torch.Tensor,
        z_next: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the LeWM-style JEPA training loss.

        Two terms, one hyperparameter (β = ``self.config.jepa_beta``):

        .. code-block:: text

            prediction_loss = MSE( ẑ_{t+1}, sg(z_{t+1}) )
            kl_loss         = -0.5 * mean(1 + logvar - mu² - exp(logvar))
            total           = prediction_loss + β * kl_loss

        The stop-gradient on z_{t+1} is applied **inside this method** — the
        caller does not need to call ``.detach()`` on the target.

        Args:
            z_hat_next: (B, latent_dim) — predictor output.
            z_next:     (B, latent_dim) — encoder output for actual next state.
            mu:         (B, latent_dim) — encoder mean for current state.
            logvar:     (B, latent_dim) — encoder log-variance for current state.

        Returns:
            dict with keys "total", "prediction", "kl".
        """
        # Stop-gradient on the target: predictor is trained to chase the encoder,
        # but the encoder is NOT pulled toward the predictor's output.
        target = z_next.detach()

        prediction_loss = F.mse_loss(z_hat_next, target)

        # Gaussian KL divergence: KL( N(mu, exp(logvar)) || N(0, I) )
        kl_loss = -0.5 * torch.mean(
            1.0 + logvar - mu.pow(2) - logvar.exp()
        )

        total = prediction_loss + self.config.jepa_beta * kl_loss

        return {
            "total": total,
            "prediction": prediction_loss,
            "kl": kl_loss,
        }

    # ------------------------------------------------------------------
    # Convenience: surprise score
    # ------------------------------------------------------------------

    def surprise_score(
        self,
        z_hat_next: torch.Tensor,
        z_next_actual: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample surprise score = MSE(ẑ_{t+1}, z_{t+1}).

        High surprise → the model did not expect this transition.  Use this
        as a signal to fall back on KG-based reasoning or flag the state for
        human review.

        Args:
            z_hat_next:    (B, latent_dim) — predictor output.
            z_next_actual: (B, latent_dim) — actual next-state encoding.

        Returns:
            surprise: (B,) — per-sample MSE.
        """
        return F.mse_loss(z_hat_next, z_next_actual.detach(), reduction="none").mean(dim=-1)
