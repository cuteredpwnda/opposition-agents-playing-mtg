"""
Neural reasoning module — GNN + Transformer + MLP fusion for board evaluation.

Combines graph neural networks (for KG reasoning), a transformer (for game
sequence encoding), and an MLP (for board-state numeric features) to produce
action-value estimates, position evaluation, and win probability.

Reference: Section 8.1-8.2 of PLAN.md.
Requires the [ml] optional dependency: pip install opposition-agents-mtg[ml]
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GATConv, global_mean_pool
except ImportError:
    raise ImportError(
        "Neural reasoner requires PyTorch + PyG. "
        "Install with: pip install opposition-agents-mtg[ml]"
    )

from src.engine.game_state import GameState, Phase, Zone


class NeuralReasoningModule(nn.Module):
    """Multi-modal neural architecture for MTG strategic reasoning.

    Inputs:
        1. KG subgraph (cards in play + neighbors) → GAT
        2. Game-action sequence → Transformer encoder
        3. Board-state numeric features → MLP

    Outputs:
        value     — position evaluation score
        policy    — action preference logits
        win_prob  — estimated win probability
    """

    def __init__(
        self,
        card_embed_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        board_features_dim: int = 512,
        policy_dim: int = 512,
    ):
        super().__init__()

        # Graph Attention Network for KG
        self.gat1 = GATConv(card_embed_dim, hidden_dim, heads=num_heads)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1)

        # Transformer for game sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Board state MLP
        self.board_encoder = nn.Sequential(
            nn.Linear(board_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Fusion + output heads
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.policy_head = nn.Linear(hidden_dim, policy_dim)
        self.win_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        graph_data,  # PyG Data batch
        game_sequence: torch.Tensor,  # (B, seq_len, hidden_dim)
        board_features: torch.Tensor,  # (B, board_features_dim)
    ) -> dict[str, torch.Tensor]:
        # Graph reasoning
        graph_embed = self.gat1(graph_data.x, graph_data.edge_index).relu()
        graph_embed = self.gat2(graph_embed, graph_data.edge_index)
        graph_embed = global_mean_pool(graph_embed, graph_data.batch)

        # Sequence reasoning
        seq_embed = self.sequence_encoder(game_sequence)
        seq_embed = seq_embed.mean(dim=1)

        # Board reasoning
        board_embed = self.board_encoder(board_features)

        # Fuse
        fused = self.fusion(
            torch.cat([graph_embed, seq_embed, board_embed], dim=-1)
        ).relu()

        return {
            "value": self.value_head(fused),
            "policy": self.policy_head(fused),
            "win_prob": torch.sigmoid(self.win_head(fused)),
        }


class BoardStateEncoder:
    """Converts a GameState into a fixed-size numeric tensor (Section 8.2)."""

    def encode(self, game_state: GameState, player_id: str) -> torch.Tensor:
        features: list[float] = []

        # Life totals (normalized to 40 for Commander)
        for pstate in game_state.players.values():
            features.append(pstate.life / 40.0)

        # Card counts per zone per player
        for pid in game_state.players:
            for zone_name in ["hand", "battlefield", "graveyard", "library"]:
                cards = game_state.cards_in_zone.get((pid, zone_name), [])
                divisor = {"hand": 10, "battlefield": 20, "graveyard": 40, "library": 99}
                features.append(len(cards) / divisor.get(zone_name, 10))

        # Creature stats per player
        for pid in game_state.players:
            bf = game_state.cards_in_zone.get((pid, "battlefield"), [])
            creatures = [c for c in bf if c.is_creature]
            total_power = sum(c.power or 0 for c in creatures)
            total_tough = sum(c.toughness or 0 for c in creatures)
            features.extend([
                total_power / 50.0,
                total_tough / 50.0,
                len(creatures) / 15.0,
            ])

        # Mana available (WUBRGC)
        me = game_state.players.get(player_id)
        if me:
            for color in "WUBRGC":
                features.append(me.mana_pool.get(color, 0) / 10.0)

        # Phase one-hot
        phases = list(Phase)
        phase_vec = [0.0] * len(phases)
        try:
            phase_vec[phases.index(game_state.phase)] = 1.0
        except ValueError:
            pass
        features.extend(phase_vec)

        # Stack depth, turn number
        features.append(len(game_state.stack) / 10.0)
        features.append(game_state.turn_number / 20.0)

        return torch.tensor(features, dtype=torch.float32)
