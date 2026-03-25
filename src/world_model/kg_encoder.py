"""
KG Context Encoder — strategic context from visible cards.

Encodes a set of visible card names (hand + battlefield) into a compact
semantic vector by aggregating KG-informed card embeddings via
multi-head self-attention.

The attention mechanism learns which cards carry the most strategic
weight given the current board — combo pieces, win conditions, mana
sources, disruption tools — without needing a live Neo4j connection.

Two modes:
  - text:       CardEmbeddingModel text embeddings (always available)
  - structural: CardGraphEmbedder GNN embeddings (requires PyG + trained checkpoint)

The output is concatenated with game-state features inside StateEncoder
to form the dual-input latent vector used by the JEPA predictor.

Reference: Maes et al. (2026) "LeWorldModel: Stable End-to-End JEPA from Pixels"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError("KGContextEncoder requires PyTorch.")

from .card_embeddings import CardEmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class KGContextEncoderConfig:
    """Configuration for the KG context encoder."""

    card_embed_dim: int = 128        # Input card embedding dim (must match CardEmbeddingModel)
    kg_embed_dim: int = 128          # Output context vector dim
    num_attention_heads: int = 4     # Heads for self-attention over card set
    max_visible_cards: int = 30      # max(hand_size + battlefield_size)
    dropout: float = 0.1


class KGContextEncoder(nn.Module):
    """Encodes visible cards into a strategic context vector.

    Input:  list of card name lists (one per batch item)
    Output: (batch, kg_embed_dim) attention-pooled semantic embedding

    The self-attention over the card set acts as a soft detector for
    combos, synergies, and archetypes present in the visible game state.
    When KG graph embeddings are used (CardGraphEmbedder), the attention
    attends over structurally richer features than text alone.

    Usage::

        card_model = CardEmbeddingModel()
        kg_enc = KGContextEncoder(card_model)

        # During game / training:
        cards = [["Lightning Bolt", "Mountain"], ["Tarmogoyf", "Thoughtseize"]]
        context = kg_enc(cards, device=device)  # (2, 128)
    """

    def __init__(
        self,
        card_embed_model: CardEmbeddingModel,
        config: KGContextEncoderConfig | None = None,
    ):
        super().__init__()
        self.card_embed_model = card_embed_model
        self.config = config or KGContextEncoderConfig()
        c = self.config

        # Project raw card embeddings to the internal attention dim.
        # This lets the input card_embed_dim differ from kg_embed_dim.
        self.card_proj = nn.Linear(c.card_embed_dim, c.kg_embed_dim)

        # Self-attention over the card set — learns which cards matter most
        # in the context of the full visible board.
        self.card_attention = nn.MultiheadAttention(
            embed_dim=c.kg_embed_dim,
            num_heads=c.num_attention_heads,
            dropout=c.dropout,
            batch_first=True,
        )

        # Final projection + nonlinearity before fusion
        self.output_proj = nn.Sequential(
            nn.Linear(c.kg_embed_dim, c.kg_embed_dim),
            nn.ReLU(),
            nn.Dropout(c.dropout),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        card_names: list[list[str]],
        device: Optional[torch.device] = None,
        *,
        precomputed_embeddings: Optional[torch.Tensor] = None,
        precomputed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode visible cards into a strategic context vector.

        Args:
            card_names: Batch of card name lists, e.g.
                        [["Lightning Bolt", "Mountain"], ["Tarmogoyf"]].
                        Ignored when *precomputed_embeddings* is provided.
            device:     Target device. Defaults to the module's parameter device.
            precomputed_embeddings: (B, N, card_embed_dim) — supply when
                        embeddings were already fetched outside this call (e.g.
                        from CardGraphEmbedder GNN output during training).
            precomputed_mask: (B, N) bool — True where the slot is padding.
                        Required when *precomputed_embeddings* is given.

        Returns:
            context: (batch, kg_embed_dim)
        """
        if device is None:
            device = next(self.parameters()).device

        if precomputed_embeddings is not None:
            raw = precomputed_embeddings.to(device)
            mask = (
                precomputed_mask.to(device)
                if precomputed_mask is not None
                else torch.zeros(raw.shape[:2], dtype=torch.bool, device=device)
            )
        else:
            raw, mask = self._embed_card_names(card_names, device)

        # raw: (B, N, card_embed_dim) → (B, N, kg_embed_dim)
        tokens = self.card_proj(raw)

        # Self-attention over the card token set.
        # key_padding_mask=True → the position is ignored (padding).
        attended, _ = self.card_attention(
            tokens, tokens, tokens,
            key_padding_mask=mask,
        )  # (B, N, kg_embed_dim)

        # Masked mean pooling — exclude padding positions
        valid = (~mask).float().unsqueeze(-1)          # (B, N, 1)
        count = valid.sum(dim=1).clamp(min=1.0)        # (B, 1)
        context = (attended * valid).sum(dim=1) / count  # (B, kg_embed_dim)

        return self.output_proj(context)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_card_names(
        self,
        card_names: list[list[str]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up CardEmbeddingModel vectors and build a padded batch.

        Args:
            card_names: List of card name lists, one per batch item.
            device:     Where to place the output tensors.

        Returns:
            embeddings: (B, max_visible_cards, card_embed_dim) — zero-padded
            mask:       (B, max_visible_cards) bool — True = padding slot
        """
        c = self.config
        B = len(card_names)
        embeddings = torch.zeros(B, c.max_visible_cards, c.card_embed_dim, device=device)
        # True means "ignore this position" in MultiheadAttention
        mask = torch.ones(B, c.max_visible_cards, dtype=torch.bool, device=device)

        for b, names in enumerate(card_names):
            for i, name in enumerate(names[: c.max_visible_cards]):
                emb = self.card_embed_model.get_embedding(name)
                if emb is not None:
                    embeddings[b, i] = torch.from_numpy(
                        np.asarray(emb, dtype=np.float32)
                    ).to(device)
                mask[b, i] = False  # This slot carries real data

        return embeddings, mask
