"""
V Model — State Encoder (VAE).

Encodes a structured MTG GameState into a compact latent vector z ∈ ℝ^d.
Uses a Variational Autoencoder architecture:
- Encoder: game features → μ, σ → z (reparameterization trick)
- Decoder: z → reconstructed game features

Unlike the original World Models VAE on pixels, this operates on structured
game state features produced by GameTokenizer.

Reference: Ha & Schmidhuber (2018), Section "VAE (V) Model"
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError(
        "State encoder requires PyTorch. "
        "Install with: pip install torch"
    )


@dataclass
class StateEncoderConfig:
    """Configuration for the state encoder (V model)."""

    input_dim: int = 2048        # Total flattened input feature dim
    hidden_dim: int = 512        # Hidden layer size
    latent_dim: int = 256        # Latent vector z dimension
    card_embed_dim: int = 128    # Card embedding dimension
    max_hand_size: int = 10
    max_battlefield_size: int = 20
    max_graveyard_size: int = 20
    dropout: float = 0.1
    kl_weight: float = 0.001     # KL divergence weight (β-VAE)


class SetEncoder(nn.Module):
    """Encodes a variable-length set of card embeddings into a fixed-size vector.

    Uses a permutation-invariant pooling approach:
    mean(MLP(card_embedding * mask)) for each card in the set.

    This handles variable numbers of cards in hand/battlefield/graveyard.
    """

    def __init__(self, card_dim: int, output_dim: int):
        super().__init__()
        self.card_dim = card_dim
        self.mlp = nn.Sequential(
            nn.Linear(card_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, cards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cards: (batch, max_cards, card_dim)
            mask: (batch, max_cards) — 1 where card exists, 0 for padding

        Returns:
            (batch, output_dim) — pooled set representation
        """
        # Apply MLP to each card
        encoded = self.mlp(cards)  # (B, max_cards, output_dim)

        # Masked mean pooling
        mask_expanded = mask.unsqueeze(-1)  # (B, max_cards, 1)
        encoded = encoded * mask_expanded
        count = mask_expanded.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = encoded.sum(dim=1) / count  # (B, output_dim)

        return pooled


class StateEncoder(nn.Module):
    """V Model — Variational Autoencoder for MTG game states.

    Architecture:
        1. Set encoders for variable-length card zones (hand, battlefield, etc.)
        2. MLP encoders for fixed-size features (player stats, phase, turn)
        3. Fusion layer combining all encodings
        4. VAE bottleneck: μ, σ → z via reparameterization
        5. Decoder: z → reconstructed features (for training)

    Usage:
        encoder = StateEncoder(config)
        z, mu, logvar = encoder.encode(tokenized_state)
        reconstructed = encoder.decode(z)
        loss = encoder.loss(tokenized_state, reconstructed, mu, logvar)
    """

    def __init__(self, config: StateEncoderConfig | None = None):
        super().__init__()
        self.config = config or StateEncoderConfig()
        c = self.config

        # Set encoders for variable-length zones
        self.hand_encoder = SetEncoder(c.card_embed_dim, c.hidden_dim // 4)
        self.battlefield_encoder = SetEncoder(c.card_embed_dim + 4, c.hidden_dim // 4)  # +4 for tapped/power/tough/sick
        self.opp_battlefield_encoder = SetEncoder(c.card_embed_dim, c.hidden_dim // 4)
        self.graveyard_encoder = SetEncoder(c.card_embed_dim, c.hidden_dim // 8)
        self.stack_encoder = SetEncoder(c.card_embed_dim + 2, c.hidden_dim // 8)

        # Fixed-size feature encoders
        player_feature_dim = 11  # life, mana(6), hand_count, lib_count, bf_count, land_plays
        self.player_encoder = nn.Sequential(
            nn.Linear(player_feature_dim * 2, c.hidden_dim // 4),  # Both players
            nn.ReLU(),
        )
        self.phase_encoder = nn.Sequential(
            nn.Linear(12 + 4, c.hidden_dim // 8),  # phase_onehot(12) + turn_features(4)
            nn.ReLU(),
        )

        # Compute fusion input dimension
        # hand + bf + opp_bf + gy + stack + player + phase
        fusion_dim = (
            c.hidden_dim // 4  # hand
            + c.hidden_dim // 4  # battlefield
            + c.hidden_dim // 4  # opp battlefield
            + c.hidden_dim // 8  # graveyard
            + c.hidden_dim // 8  # stack
            + c.hidden_dim // 4  # players
            + c.hidden_dim // 8  # phase/turn
        )

        # Fusion → latent
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, c.hidden_dim),
            nn.ReLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.ReLU(),
        )

        # VAE bottleneck
        self.fc_mu = nn.Linear(c.hidden_dim, c.latent_dim)
        self.fc_logvar = nn.Linear(c.hidden_dim, c.latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(c.latent_dim, c.hidden_dim),
            nn.ReLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.ReLU(),
            nn.Linear(c.hidden_dim, fusion_dim),  # Reconstruct fused features
        )

    def encode(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode tokenized game state features into latent space.

        Args:
            features: Dict of tensors from GameTokenizer.encode_state(),
                      converted to torch tensors and batched.

        Returns:
            z: (batch, latent_dim) — sampled latent vector
            mu: (batch, latent_dim) — mean of latent distribution
            logvar: (batch, latent_dim) — log-variance of latent distribution
        """
        # Encode each zone
        hand_enc = self.hand_encoder(features["hand_cards"], features["hand_mask"])

        # Battlefield: concatenate card embeddings with extra state (tapped, power, etc.)
        bf_cards = torch.cat([features["battlefield_cards"], features["battlefield_state"]], dim=-1)
        bf_enc = self.battlefield_encoder(bf_cards, features["battlefield_mask"])

        opp_bf_enc = self.opp_battlefield_encoder(
            features["opp_battlefield_cards"], features["opp_battlefield_mask"]
        )

        gy_enc = self.graveyard_encoder(features["graveyard_cards"], features["graveyard_mask"])

        # Stack: use first dim as mask (non-zero entries)
        stack = features["stack_features"]
        stack_mask = (stack.abs().sum(dim=-1) > 0).float()
        stack_enc = self.stack_encoder(stack, stack_mask)

        # Player features
        player_feats = torch.cat([features["player_features"], features["opponent_features"]], dim=-1)
        player_enc = self.player_encoder(player_feats)

        # Phase/turn
        phase_turn = torch.cat([features["phase_encoding"], features["turn_features"]], dim=-1)
        phase_enc = self.phase_encoder(phase_turn)

        # Fuse and return full hidden representation for loss targets
        hidden = self._fuse_features(
            hand_enc=hand_enc,
            bf_enc=bf_enc,
            opp_bf_enc=opp_bf_enc,
            gy_enc=gy_enc,
            stack_enc=stack_enc,
            player_enc=player_enc,
            phase_enc=phase_enc,
        )

        # VAE
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self._reparameterize(mu, logvar)

        return z, mu, logvar

    def _fuse_features(
        self,
        hand_enc: torch.Tensor,
        bf_enc: torch.Tensor,
        opp_bf_enc: torch.Tensor,
        gy_enc: torch.Tensor,
        stack_enc: torch.Tensor,
        player_enc: torch.Tensor,
        phase_enc: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse encoded components into the shared latent input space."""
        fused = torch.cat([hand_enc, bf_enc, opp_bf_enc, gy_enc, stack_enc, player_enc, phase_enc], dim=-1)
        return self.fusion(fused)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to fused feature space.

        Args:
            z: (batch, latent_dim)

        Returns:
            reconstructed: (batch, fusion_dim)
        """
        return self.decoder(z)

    def loss(
        self,
        features: dict[str, torch.Tensor],
        reconstructed: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute VAE loss = reconstruction + KL divergence.

        Args:
            features: Original input features
            reconstructed: Decoder output
            mu, logvar: Encoder outputs

        Returns:
            Dict with "total", "reconstruction", "kl" loss tensors.
        """
        # Reconstruction target: the fused representation prior to final fusion layer
        # (i.e., concat of per-zone encodings). The decoder output has size fusion_dim
        # and should reconstruct this vector.
        with torch.no_grad():
            hand_enc = self.hand_encoder(features["hand_cards"], features["hand_mask"])
            bf_cards = torch.cat([features["battlefield_cards"], features["battlefield_state"]], dim=-1)
            bf_enc = self.battlefield_encoder(bf_cards, features["battlefield_mask"])
            opp_bf_enc = self.opp_battlefield_encoder(features["opp_battlefield_cards"], features["opp_battlefield_mask"])
            gy_enc = self.graveyard_encoder(features["graveyard_cards"], features["graveyard_mask"])
            stack = features["stack_features"]
            stack_mask = (stack.abs().sum(dim=-1) > 0).float()
            stack_enc = self.stack_encoder(stack, stack_mask)
            player_feats = torch.cat([features["player_features"], features["opponent_features"]], dim=-1)
            player_enc = self.player_encoder(player_feats)
            phase_turn = torch.cat([features["phase_encoding"], features["turn_features"]], dim=-1)
            phase_enc = self.phase_encoder(phase_turn)

            target = torch.cat(
                [hand_enc, bf_enc, opp_bf_enc, gy_enc, stack_enc, player_enc, phase_enc],
                dim=-1,
            )

        recon_loss = F.mse_loss(reconstructed, target, reduction="mean")

        # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total = recon_loss + self.config.kl_weight * kl_loss

        return {
            "total": total,
            "reconstruction": recon_loss,
            "kl": kl_loss,
        }

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # During inference, just use the mean
