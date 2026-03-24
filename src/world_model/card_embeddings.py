"""
Card embedding model — maps card names to dense vectors.

Provides multiple strategies for computing card embeddings:
1. Text-based: Sentence transformer on oracle text (bootstrap, no game data needed)
2. Gameplay-based: Learned from game trajectories (like Word2Vec for cards)
3. Hybrid: Initialize from text, fine-tune during world model training

Card embeddings are the foundation of the world model — every game state
feature pipeline starts by looking up card embeddings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CardEmbeddingConfig:
    """Configuration for card embeddings."""

    embed_dim: int = 128
    text_model_name: str = "all-MiniLM-L6-v2"  # Sentence transformer model
    cache_dir: Path = Path("data/card_embeddings")
    use_text_bootstrap: bool = True
    fine_tune: bool = False


class CardEmbeddingModel:
    """Manages card name → ℝ^d embedding lookup.

    Usage:
        model = CardEmbeddingModel(config)
        model.build_from_scryfall(scryfall_bulk_data)
        embedding = model.get_embedding("Lightning Bolt")  # → np.ndarray(128,)
    """

    def __init__(self, config: CardEmbeddingConfig | None = None):
        self.config = config or CardEmbeddingConfig()
        self._embeddings: dict[str, np.ndarray] = {}
        self._text_encoder = None  # Lazy-loaded sentence transformer

    # -- Public API ---------------------------------------------------------

    def get_embedding(self, card_name: str) -> np.ndarray:
        """Get the embedding for a card by name.

        Returns a zero vector if the card is unknown and text bootstrap
        is not available.
        """
        if card_name in self._embeddings:
            return self._embeddings[card_name]

        # Try text-based embedding on the fly
        if self.config.use_text_bootstrap and self._text_encoder is not None:
            embed = self._compute_text_embedding(card_name, oracle_text="")
            self._embeddings[card_name] = embed
            return embed

        return np.zeros(self.config.embed_dim, dtype=np.float32)

    def get_all_embeddings(self) -> dict[str, np.ndarray]:
        """Return the full embedding dictionary."""
        return dict(self._embeddings)

    def build_from_scryfall(self, cards: list[dict[str, Any]]) -> None:
        """Build embeddings from a list of Scryfall card objects.

        Each card dict should have at minimum: "name", "oracle_text", "type_line".

        If a sentence transformer is available, uses text-based encoding.
        Otherwise, generates random embeddings as placeholders.

        Args:
            cards: List of Scryfall card JSON dicts.
        """
        logger.info(f"Building card embeddings for {len(cards)} cards...")

        if self.config.use_text_bootstrap:
            self._init_text_encoder()

        for card in cards:
            name = card.get("name", "")
            if not name or name in self._embeddings:
                continue

            oracle = card.get("oracle_text", "")
            type_line = card.get("type_line", "")
            mana_cost = card.get("mana_cost", "")

            if self._text_encoder is not None:
                text = f"{name}. {type_line}. {mana_cost}. {oracle}"
                self._embeddings[name] = self._compute_text_embedding(name, text)
            else:
                # Deterministic random embedding based on card name hash
                rng = np.random.RandomState(hash(name) % (2**31))
                self._embeddings[name] = rng.randn(self.config.embed_dim).astype(np.float32)

        logger.info(f"Built embeddings for {len(self._embeddings)} cards")

    def build_from_trajectories(self, trajectories: list[Any]) -> None:
        """Learn embeddings from gameplay data (Word2Vec-style).

        Cards that co-occur in hands/battlefields → similar embeddings.
        Cards that are played in similar game states → similar embeddings.

        Args:
            trajectories: List of game trajectory objects.
        """
        if self.config.use_text_bootstrap:
            self._init_text_encoder()

        # Collect card co-occurrence contexts from trajectory actions.
        contexts: dict[str, set[str]] = {}
        for traj in trajectories:
            for transition in getattr(traj, "transitions", []):
                card_name = getattr(transition, "card_name", None)
                if not card_name:
                    continue

                # Add as seen card
                contexts.setdefault(card_name, set())

                # Also include action type as weak context proxy
                action_type = getattr(transition, "action_type", "")
                if action_type:
                    contexts[card_name].add(action_type)

                # Include card names in metadata if present
                metadata_cards = transition.metadata.get("cards", []) if hasattr(transition, "metadata") else []
                for other in metadata_cards:
                    if other != card_name:
                        contexts[card_name].add(other)

        # Build seed embeddings for all cards in context
        for card_name in contexts.keys():
            if card_name not in self._embeddings:
                if self._text_encoder is not None:
                    self._embeddings[card_name] = self._compute_text_embedding(card_name, oracle_text="")
                else:
                    rng = np.random.RandomState(hash(card_name) % (2**31))
                    self._embeddings[card_name] = rng.randn(self.config.embed_dim).astype(np.float32)

        # A simple context smoothing pass
        for card_name, neighbors in contexts.items():
            base = self._embeddings.get(card_name, np.zeros(self.config.embed_dim, dtype=np.float32))
            neighbor_vecs = [self._embeddings.get(n) for n in neighbors if n in self._embeddings]
            if neighbor_vecs:
                neighbor_mean = np.mean(np.stack(neighbor_vecs), axis=0).astype(np.float32)
                self._embeddings[card_name] = 0.7 * base + 0.3 * neighbor_mean

        logger.info("Built gameplay-aware embeddings for %d cards from trajectories", len(self._embeddings))

    def save(self, path: Path | None = None) -> None:
        """Save embeddings to disk."""
        save_path = path or self.config.cache_dir / "embeddings.npz"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        names = list(self._embeddings.keys())
        vectors = np.stack([self._embeddings[n] for n in names])

        np.savez(save_path, names=names, vectors=vectors)
        logger.info(f"Saved {len(names)} embeddings to {save_path}")

    def load(self, path: Path | None = None) -> None:
        """Load embeddings from disk."""
        load_path = path or self.config.cache_dir / "embeddings.npz"
        if not load_path.exists():
            logger.warning(f"No embeddings found at {load_path}")
            return

        data = np.load(load_path, allow_pickle=True)
        names = data["names"]
        vectors = data["vectors"]

        self._embeddings = {
            str(name): vec.astype(np.float32)
            for name, vec in zip(names, vectors)
        }
        logger.info(f"Loaded {len(self._embeddings)} embeddings from {load_path}")

    # -- Private helpers ----------------------------------------------------

    def _init_text_encoder(self) -> None:
        """Lazy-load the sentence transformer model."""
        if self._text_encoder is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._text_encoder = SentenceTransformer(self.config.text_model_name)
            logger.info(f"Loaded text encoder: {self.config.text_model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Card embeddings will use random vectors. "
                "Install with: pip install sentence-transformers"
            )
            self._text_encoder = None

    def _compute_text_embedding(self, card_name: str, oracle_text: str) -> np.ndarray:
        """Compute a card embedding from its text using the sentence transformer."""
        if self._text_encoder is None:
            rng = np.random.RandomState(hash(card_name) % (2**31))
            return rng.randn(self.config.embed_dim).astype(np.float32)

        text = f"{card_name}. {oracle_text}" if oracle_text else card_name
        raw_embedding = self._text_encoder.encode(text, show_progress_bar=False)

        # Project to target dimension if needed
        if len(raw_embedding) != self.config.embed_dim:
            # Simple linear projection (in practice, use a learned projection)
            rng = np.random.RandomState(42)
            proj = rng.randn(len(raw_embedding), self.config.embed_dim).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True)
            return (raw_embedding @ proj).astype(np.float32)

        return raw_embedding.astype(np.float32)

    @property
    def num_cards(self) -> int:
        return len(self._embeddings)

    @property
    def embed_dim(self) -> int:
        return self.config.embed_dim
