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
    _HAS_PYG = True
except ImportError:
    torch = None
    nn = None
    GATConv = None
    global_mean_pool = None
    _HAS_PYG = False

from src.engine.game_state import GameState, Phase, Zone

if not _HAS_PYG:
    class NeuralReasoningModule:
        """Fallback stub for environments without PyTorch/PyG."""

        def __init__(self, *args, **kwargs):
            logger = __import__('logging').getLogger(__name__)
            logger.warning('NeuralReasoningModule fallback active: PyTorch/PyG not installed')

        def __call__(self, *args, **kwargs):
            import random
            return {
                'value': 0.0,
                'policy': [random.random() for _ in range(10)],
                'win_prob': 0.5,
            }

    class BoardStateEncoder:
        """Fallback board state encoder."""

        def encode(self, game_state: GameState, player_id: str):
            try:
                import numpy as np
                return np.zeros(64, dtype=float)
            except ImportError:
                return [0.0] * 64

else:
    class NeuralReasoningModule(nn.Module):
        """Multi-modal neural architecture for MTG strategic reasoning."""

        def __init__(
            self,
            card_embed_dim: int = 128,
            hidden_dim: int = 256,
            num_heads: int = 8,
            board_features_dim: int = 512,
            policy_dim: int = 512,
        ):
            super().__init__()
            self.gat1 = GATConv(card_embed_dim, hidden_dim, heads=num_heads)
            self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, batch_first=True
            )
            self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
            self.board_encoder = nn.Sequential(
                nn.Linear(board_features_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
            self.value_head = nn.Linear(hidden_dim, 1)
            self.policy_head = nn.Linear(hidden_dim, policy_dim)
            self.win_head = nn.Linear(hidden_dim, 1)

        def forward(self, graph_data, game_sequence, board_features):
            graph_embed = self.gat1(graph_data.x, graph_data.edge_index).relu()
            graph_embed = self.gat2(graph_embed, graph_data.edge_index)
            graph_embed = global_mean_pool(graph_embed, graph_data.batch)
            seq_embed = self.sequence_encoder(game_sequence).mean(dim=1)
            board_embed = self.board_encoder(board_features)
            fused = self.fusion(torch.cat([graph_embed, seq_embed, board_embed], dim=-1)).relu()
            return {
                'value': self.value_head(fused),
                'policy': self.policy_head(fused),
                'win_prob': torch.sigmoid(self.win_head(fused)),
            }

    class BoardStateEncoder:
        """Converts a GameState into a fixed-size numeric tensor."""

        def encode(self, game_state, player_id):
            features = []
            for pstate in game_state.players:
                features.append(pstate.life_total / 40.0)
            for pstate in game_state.players:
                for zone_name in [Zone.HAND, Zone.BATTLEFIELD, Zone.GRAVEYARD, Zone.LIBRARY]:
                    features.append(len(game_state.cards_in_zone(pstate.player_id, zone_name)) / 60.0)
            for pstate in game_state.players:
                bf = game_state.cards_in_zone(pstate.player_id, Zone.BATTLEFIELD)
                creatures = [c for c in bf if c.is_creature()]
                total_power = sum(int(c.power or 0) for c in creatures)
                total_tough = sum(int(c.toughness or 0) for c in creatures)
                features.extend([total_power / 50.0, total_tough / 50.0, len(creatures) / 15.0])
            me = next((p for p in game_state.players if p.player_id == player_id), None)
            if me:
                for color in ['W', 'U', 'B', 'R', 'G', 'C']:
                    features.append(me.mana_pool.get(color, 0) / 10.0)
            phases = list(Phase)
            phase_vec = [0.0] * len(phases)
            try:
                phase_vec[phases.index(game_state.phase)] = 1.0
            except ValueError:
                pass
            features.extend(phase_vec)
            features.append(len(game_state.stack) / 10.0)
            features.append(game_state.turn_number / 20.0)
            return torch.tensor(features, dtype=torch.float32)
