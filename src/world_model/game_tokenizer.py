"""
Game tokenizer — converts GameState and Actions into numeric feature vectors.

This is the bridge between the symbolic game engine and the neural world model.
It handles variable-length inputs (different numbers of cards in hand, on board)
by using set-based pooling and fixed-size encodings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.engine.game_state import (
    Action,
    ActionType,
    CardInstance,
    GameState,
    Phase,
    Zone,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TokenizerConfig:
    """Configuration for the game tokenizer."""

    card_embed_dim: int = 128        # Dimension of card embedding vectors
    max_hand_size: int = 10          # Max cards in hand to encode
    max_battlefield_size: int = 20   # Max permanents on battlefield to encode
    max_graveyard_size: int = 20     # Max cards in graveyard to encode
    max_stack_size: int = 5          # Max items on stack to encode
    num_phases: int = 12             # Number of distinct phases
    num_action_types: int = 8        # Number of ActionType values
    max_mana_value: int = 15         # Clip mana pool values


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class GameTokenizer:
    """Converts GameState ↔ fixed-size feature vectors for the world model.

    Produces a dict of numpy arrays that can be fed into the State Encoder (V).
    """

    def __init__(
        self,
        config: TokenizerConfig | None = None,
        card_embeddings: dict[str, np.ndarray] | None = None,
    ):
        self.config = config or TokenizerConfig()
        self.card_embeddings = card_embeddings or {}

    # -- Public API ---------------------------------------------------------

    def encode_state(self, game_state: GameState, player_id: str) -> dict[str, np.ndarray]:
        """Encode a full GameState from the perspective of `player_id`.

        Returns a dict of feature arrays ready for the state encoder:
        - "player_features": (player_feature_dim,) — life, mana, flags
        - "opponent_features": (player_feature_dim,)
        - "hand_cards": (max_hand, card_embed_dim) — padded card embeddings
        - "hand_mask": (max_hand,) — 1 where card exists, 0 for padding
        - "battlefield_cards": (max_bf, card_embed_dim) — padded
        - "battlefield_mask": (max_bf,)
        - "battlefield_state": (max_bf, extra_features) — tapped, power, etc.
        - "graveyard_cards": (max_gy, card_embed_dim) — padded
        - "graveyard_mask": (max_gy,)
        - "stack_features": (max_stack, stack_item_dim) — padded
        - "phase_encoding": (num_phases,) — one-hot
        - "turn_features": (4,) — turn number, active player flag, etc.
        """
        opponent_id = self._get_opponent_id(game_state, player_id)

        return {
            "player_features": self._encode_player(game_state, player_id),
            "opponent_features": self._encode_player(game_state, opponent_id),
            "hand_cards": self._encode_zone_cards(game_state, player_id, Zone.HAND, self.config.max_hand_size),
            "hand_mask": self._encode_zone_mask(game_state, player_id, Zone.HAND, self.config.max_hand_size),
            "battlefield_cards": self._encode_zone_cards(game_state, player_id, Zone.BATTLEFIELD, self.config.max_battlefield_size),
            "battlefield_mask": self._encode_zone_mask(game_state, player_id, Zone.BATTLEFIELD, self.config.max_battlefield_size),
            "battlefield_state": self._encode_battlefield_extra(game_state, player_id),
            "opp_battlefield_cards": self._encode_zone_cards(game_state, opponent_id, Zone.BATTLEFIELD, self.config.max_battlefield_size),
            "opp_battlefield_mask": self._encode_zone_mask(game_state, opponent_id, Zone.BATTLEFIELD, self.config.max_battlefield_size),
            "graveyard_cards": self._encode_zone_cards(game_state, player_id, Zone.GRAVEYARD, self.config.max_graveyard_size),
            "graveyard_mask": self._encode_zone_mask(game_state, player_id, Zone.GRAVEYARD, self.config.max_graveyard_size),
            "stack_features": self._encode_stack(game_state),
            "phase_encoding": self._encode_phase(game_state.phase),
            "turn_features": self._encode_turn(game_state, player_id),
        }

    def encode_action(self, action: Action) -> np.ndarray:
        """Encode an Action into a fixed-size vector.

        Returns:
            (action_dim,) vector: [action_type_onehot | card_embedding | target_embedding]
        """
        # Action type one-hot
        action_types = list(ActionType)
        type_onehot = np.zeros(self.config.num_action_types, dtype=np.float32)
        try:
            idx = action_types.index(action.action_type)
            type_onehot[idx] = 1.0
        except ValueError:
            pass

        # Card embedding (if action involves a card)
        card_embed = np.zeros(self.config.card_embed_dim, dtype=np.float32)
        if action.card_instance_id and action.metadata.get("card_name"):
            card_name = action.metadata["card_name"]
            card_embed = self._get_card_embedding(card_name)

        return np.concatenate([type_onehot, card_embed])

    def compute_state_dim(self) -> int:
        """Calculate total flattened state feature dimension."""
        # This is approximate — actual dim depends on pooling strategy
        cfg = self.config
        player_dim = 12  # life + 6 mana + flags
        hand_dim = cfg.max_hand_size * cfg.card_embed_dim
        bf_dim = cfg.max_battlefield_size * (cfg.card_embed_dim + 4)  # +tapped/power/tough/sick
        gy_dim = cfg.max_graveyard_size * cfg.card_embed_dim
        stack_dim = cfg.max_stack_size * (cfg.card_embed_dim + 2)
        phase_dim = cfg.num_phases
        turn_dim = 4
        return (
            2 * player_dim + hand_dim + 2 * bf_dim + gy_dim
            + stack_dim + phase_dim + turn_dim
        )

    def compute_action_dim(self) -> int:
        """Dimension of an encoded action vector."""
        return self.config.num_action_types + self.config.card_embed_dim

    # -- Private helpers ----------------------------------------------------

    def _get_opponent_id(self, game_state: GameState, player_id: str) -> str:
        """Get the opponent's player_id."""
        for p in game_state.players:
            if p.player_id != player_id:
                return p.player_id
        return ""

    def _encode_player(self, game_state: GameState, player_id: str) -> np.ndarray:
        """Encode a player's vital stats into a feature vector."""
        player = next((p for p in game_state.players if p.player_id == player_id), None)
        if player is None:
            return np.zeros(12, dtype=np.float32)

        mana_order = ["W", "U", "B", "R", "G", "C"]
        mana_values = [
            min(player.mana_pool.get(c, 0), self.config.max_mana_value) / self.config.max_mana_value
            for c in mana_order
        ]

        hand_count = len(game_state.cards_in_zone(player_id, Zone.HAND))
        lib_count = len(game_state.cards_in_zone(player_id, Zone.LIBRARY))
        bf_count = len(game_state.cards_in_zone(player_id, Zone.BATTLEFIELD))

        return np.array([
            player.life_total / 40.0,  # Normalized life
            *mana_values,              # 6 mana colors
            hand_count / 10.0,         # Normalized hand size
            lib_count / 60.0,          # Normalized library size
            bf_count / 20.0,           # Normalized board size
            float(player.land_plays_remaining),
        ], dtype=np.float32)

    def _get_card_embedding(self, card_name: str) -> np.ndarray:
        """Look up or generate a card embedding."""
        if card_name in self.card_embeddings:
            return self.card_embeddings[card_name]
        # Return zero vector for unknown cards (will be trained)
        return np.zeros(self.config.card_embed_dim, dtype=np.float32)

    def _encode_zone_cards(
        self,
        game_state: GameState,
        player_id: str,
        zone: Zone,
        max_size: int,
    ) -> np.ndarray:
        """Encode cards in a zone as a padded matrix of embeddings."""
        cards = game_state.cards_in_zone(player_id, zone)[:max_size]
        result = np.zeros((max_size, self.config.card_embed_dim), dtype=np.float32)
        for i, card in enumerate(cards):
            result[i] = self._get_card_embedding(card.name)
        return result

    def _encode_zone_mask(
        self,
        game_state: GameState,
        player_id: str,
        zone: Zone,
        max_size: int,
    ) -> np.ndarray:
        """Create attention mask for cards in a zone (1=present, 0=padding)."""
        count = min(len(game_state.cards_in_zone(player_id, zone)), max_size)
        mask = np.zeros(max_size, dtype=np.float32)
        mask[:count] = 1.0
        return mask

    def _encode_battlefield_extra(
        self,
        game_state: GameState,
        player_id: str,
    ) -> np.ndarray:
        """Encode extra battlefield state per card (tapped, power, toughness, summoning sick)."""
        max_size = self.config.max_battlefield_size
        cards = game_state.cards_in_zone(player_id, Zone.BATTLEFIELD)[:max_size]
        result = np.zeros((max_size, 4), dtype=np.float32)
        for i, card in enumerate(cards):
            power = 0
            toughness = 0
            try:
                power = int(card.power) if card.power else 0
            except (ValueError, TypeError):
                pass
            try:
                toughness = int(card.toughness) if card.toughness else 0
            except (ValueError, TypeError):
                pass
            result[i] = [
                float(card.tapped),
                power / 15.0,
                toughness / 15.0,
                float(card.summoning_sick),
            ]
        return result

    def _encode_stack(self, game_state: GameState) -> np.ndarray:
        """Encode the stack (pending spells/abilities)."""
        max_size = self.config.max_stack_size
        item_dim = self.config.card_embed_dim + 2  # embed + is_spell + controller_idx
        result = np.zeros((max_size, item_dim), dtype=np.float32)
        for i, item in enumerate(game_state.stack[:max_size]):
            card_name = item.card_data.get("name", "")
            result[i, :self.config.card_embed_dim] = self._get_card_embedding(card_name)
            result[i, self.config.card_embed_dim] = float(item.is_spell)
            # Controller index (0 or 1)
            for j, p in enumerate(game_state.players):
                if p.player_id == item.controller_id:
                    result[i, self.config.card_embed_dim + 1] = float(j)
                    break
        return result

    def _encode_phase(self, phase: Phase) -> np.ndarray:
        """One-hot encode the current phase."""
        phases = list(Phase)
        onehot = np.zeros(self.config.num_phases, dtype=np.float32)
        try:
            idx = phases.index(phase)
            onehot[idx] = 1.0
        except ValueError:
            pass
        return onehot

    def _encode_turn(self, game_state: GameState, player_id: str) -> np.ndarray:
        """Encode turn-level context."""
        is_active = game_state.active_player.player_id == player_id
        is_priority = game_state.priority_player.player_id == player_id
        return np.array([
            game_state.turn_number / 30.0,  # Normalized turn number
            float(is_active),
            float(is_priority),
            len(game_state.stack) / 5.0,    # Normalized stack depth
        ], dtype=np.float32)
