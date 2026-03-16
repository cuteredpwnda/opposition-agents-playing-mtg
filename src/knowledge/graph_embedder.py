"""
GNN graph embedder — train on Neo4j graph, write embeddings back.

Reference: PyTorch Geometric (PyG) for GraphSAGE/GAT.
Embeddings are stored in Neo4j's native vector index for similarity search.
"""

from __future__ import annotations

# This module requires the [ml] optional dependency:
#   pip install opposition-agents-mtg[ml]

try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import SAGEConv
except ImportError:
    raise ImportError(
        "Graph embedder requires PyTorch Geometric. "
        "Install with: pip install opposition-agents-mtg[ml]"
    )


class CardGraphEmbedder(nn.Module):
    """Produces d-dimensional embeddings for each card node.

    Trained on the graph exported from Neo4j, capturing combo/synergy/archetype
    neighborhood structure via GraphSAGE message passing.

    After training, embeddings are written back to Neo4j:
        MATCH (c:Card {cardName: $name}) SET c.embedding = $vector
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 384,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
