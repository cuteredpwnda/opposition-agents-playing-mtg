"""
Train CardGraphEmbedder on the Neo4j card graph and write vectors back.

Pipeline:
  1. Export card nodes + edges from Neo4j into PyG Data format
  2. Train a 2-layer GraphSAGE via self-supervised link prediction
  3. Write resulting embeddings back to Neo4j (SET c.embedding = $vector)
  4. Optionally export embeddings to a local .pt cache for offline use

Self-supervised link prediction trains the GNN to predict whether an edge
exists between two nodes.  This forces card embeddings to encode
neighborhood structure: cards that share combos, synergies, or archetypes
end up close in embedding space — exactly the signal we need for the
KGContextEncoder in the JEPA world model.

Usage:
    python -m scripts.train_graph_embeddings            # full pipeline
    python -m scripts.train_graph_embeddings --export-only  # skip training
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch_geometric.data import Data
    from torch_geometric.transforms import RandomLinkSplit
except ImportError:
    raise ImportError(
        "Graph embedding training requires PyTorch + PyG. "
        "Install with: pip install -r requirements-ml.txt"
    )

from src.config import settings
from src.knowledge.graph_embedder import CardGraphEmbedder

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Neo4j graph export
# -----------------------------------------------------------------------

EXPORT_NODES_QUERY = """
MATCH (c:Card)
RETURN c.cardName AS name,
       c.manaValue      AS cmc,
       c.power           AS power,
       c.toughness        AS toughness,
       c.edhrecRank       AS edhrecRank,
       c.colors           AS colors,
       c.keywords         AS keywords,
       labels(c)          AS labels
ORDER BY c.cardName
"""

EXPORT_EDGES_QUERY = """
MATCH (a:Card)-[r]->(b:Card)
RETURN a.cardName AS src, b.cardName AS dst, type(r) AS rel
"""


async def export_graph_from_neo4j() -> tuple[Data, dict[str, int]]:
    """Pull card nodes and edges from Neo4j into a PyG Data object.

    Returns:
        data: PyG Data(x, edge_index) ready for GNN training.
        name_to_idx: Mapping from card name to node index.
    """
    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )

    # --- Nodes ---
    async with driver.session() as session:
        result = await session.run(EXPORT_NODES_QUERY)
        records = [dict(r) async for r in result]

    if not records:
        raise RuntimeError("No Card nodes in Neo4j. Run import_scryfall first.")

    name_to_idx: dict[str, int] = {}
    node_features_list: list[list[float]] = []

    # Colour set → 5-dim binary
    colour_order = ["W", "U", "B", "R", "G"]

    for i, rec in enumerate(records):
        name_to_idx[rec["name"]] = i
        cmc = float(rec["cmc"] or 0)
        power = _safe_float(rec["power"])
        toughness = _safe_float(rec["toughness"])
        edhrec = float(rec["edhrecRank"] or 100_000) / 100_000  # normalise

        colors = rec["colors"] or []
        color_vec = [1.0 if c in colors else 0.0 for c in colour_order]

        # Type indicators from Neo4j labels
        lbls = rec["labels"] or []
        is_creature = 1.0 if "Creature" in lbls else 0.0
        is_instant = 1.0 if "Instant" in lbls else 0.0
        is_sorcery = 1.0 if "Sorcery" in lbls else 0.0
        is_enchantment = 1.0 if "Enchantment" in lbls else 0.0
        is_artifact = 1.0 if "Artifact" in lbls else 0.0
        is_planeswalker = 1.0 if "Planeswalker" in lbls else 0.0
        is_land = 1.0 if "Land" in lbls else 0.0

        keywords = rec["keywords"] or []
        kw_count = float(len(keywords)) / 10.0  # normalise

        node_features_list.append([
            cmc / 16.0, power / 15.0, toughness / 15.0, edhrec,
            *color_vec,
            is_creature, is_instant, is_sorcery, is_enchantment,
            is_artifact, is_planeswalker, is_land,
            kw_count,
        ])

    x = torch.tensor(node_features_list, dtype=torch.float)
    logger.info("Exported %d card nodes (%d features each)", x.size(0), x.size(1))

    # --- Edges ---
    async with driver.session() as session:
        result = await session.run(EXPORT_EDGES_QUERY)
        edge_records = [dict(r) async for r in result]

    src_ids, dst_ids = [], []
    for er in edge_records:
        s, d = er["src"], er["dst"]
        if s in name_to_idx and d in name_to_idx:
            src_ids.append(name_to_idx[s])
            dst_ids.append(name_to_idx[d])

    if src_ids:
        edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
    else:
        # No edges — create a self-loop for every node so GNN doesn't fail
        logger.warning("No edges found. Using self-loops as fallback.")
        n = x.size(0)
        edge_index = torch.stack([torch.arange(n), torch.arange(n)])

    logger.info("Exported %d edges", edge_index.size(1))

    await driver.close()

    data = Data(x=x, edge_index=edge_index)
    return data, name_to_idx


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_graph_embedder(
    data: Data,
    out_channels: int = 128,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[CardGraphEmbedder, torch.Tensor]:
    """Train CardGraphEmbedder via self-supervised link prediction.

    Args:
        data: PyG Data with x, edge_index.
        out_channels: Embedding dimension (should match KGContextEncoder card_embed_dim).
        epochs: Training epochs.
        lr: Learning rate.
        device: Target device.

    Returns:
        model: Trained CardGraphEmbedder.
        embeddings: (N, out_channels) tensor of card embeddings.
    """
    in_channels = data.x.size(1)
    model = CardGraphEmbedder(
        in_channels=in_channels,
        hidden_channels=max(out_channels, 128),
        out_channels=out_channels,
    ).to(device)

    # 85% train / 5% val / 10% test edge split
    transform = RandomLinkSplit(
        num_val=0.05, num_test=0.1,
        is_undirected=False,
        add_negative_train_samples=True,
    )
    train_data, val_data, _test_data = transform(data)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    best_val_auc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        z = model(train_data.x, train_data.edge_index)

        # Link prediction: positive edges should have high dot-product
        pos_edge = train_data.edge_label_index[:, train_data.edge_label == 1]
        neg_edge = train_data.edge_label_index[:, train_data.edge_label == 0]

        pos_score = (z[pos_edge[0]] * z[pos_edge[1]]).sum(dim=-1)
        neg_score = (z[neg_edge[0]] * z[neg_edge[1]]).sum(dim=-1)

        labels = torch.cat([
            torch.ones(pos_score.size(0), device=device),
            torch.zeros(neg_score.size(0), device=device),
        ])
        scores = torch.cat([pos_score, neg_score])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        loss.backward()
        optimizer.step()

        # Validation
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                z_val = model(val_data.x, val_data.edge_index)
                vp = val_data.edge_label_index[:, val_data.edge_label == 1]
                vn = val_data.edge_label_index[:, val_data.edge_label == 0]
                vp_s = (z_val[vp[0]] * z_val[vp[1]]).sum(dim=-1)
                vn_s = (z_val[vn[0]] * z_val[vn[1]]).sum(dim=-1)
                # Simple AUC proxy: mean(pos > neg)
                auc_proxy = (vp_s.mean() - vn_s.mean()).item()

            logger.info(
                "Epoch %d: loss=%.4f val_auc_proxy=%.4f", epoch + 1, loss.item(), auc_proxy,
            )
            if auc_proxy > best_val_auc:
                best_val_auc = auc_proxy
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Produce final embeddings on full graph
    model.eval()
    full_data = data.to(device)
    with torch.no_grad():
        embeddings = model(full_data.x, full_data.edge_index).cpu()

    logger.info("Trained graph embeddings: %s", embeddings.shape)
    return model, embeddings


# -----------------------------------------------------------------------
# Write back to Neo4j
# -----------------------------------------------------------------------

async def write_embeddings_to_neo4j(
    embeddings: torch.Tensor,
    name_to_idx: dict[str, int],
) -> int:
    """Write card embeddings back to Neo4j as a vector property.

    Sets ``c.graphEmbedding`` on each Card node.

    Returns:
        Number of cards updated.
    """
    from neo4j import AsyncGraphDatabase

    idx_to_name = {v: k for k, v in name_to_idx.items()}
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )

    count = 0
    async with driver.session() as session:
        for idx in range(embeddings.size(0)):
            name = idx_to_name.get(idx)
            if name is None:
                continue
            vec = embeddings[idx].tolist()
            await session.run(
                "MATCH (c:Card {cardName: $name}) SET c.graphEmbedding = $vec",
                name=name, vec=vec,
            )
            count += 1

    await driver.close()
    logger.info("Wrote %d embeddings to Neo4j", count)
    return count


# -----------------------------------------------------------------------
# Local cache
# -----------------------------------------------------------------------

def save_embedding_cache(
    embeddings: torch.Tensor,
    name_to_idx: dict[str, int],
    path: str = "data/card_graph_embeddings.pt",
) -> None:
    """Save embeddings + name mapping to a local file for offline use."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "name_to_idx": name_to_idx}, path)
    logger.info("Saved embedding cache → %s", path)


def load_embedding_cache(
    path: str = "data/card_graph_embeddings.pt",
) -> tuple[torch.Tensor, dict[str, int]]:
    """Load cached graph embeddings."""
    data = torch.load(path, weights_only=False)
    return data["embeddings"], data["name_to_idx"]


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _safe_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

async def _async_main(args: argparse.Namespace) -> None:
    out_channels = args.embed_dim

    if args.export_only:
        logger.info("Export-only mode — loading cached embeddings")
        embeddings, name_to_idx = load_embedding_cache(args.cache_path)
    else:
        logger.info("Step 1/4: Exporting graph from Neo4j…")
        data, name_to_idx = await export_graph_from_neo4j()

        logger.info("Step 2/4: Training graph embedder…")
        _model, embeddings = train_graph_embedder(
            data, out_channels=out_channels, epochs=args.epochs, lr=args.lr,
        )

        logger.info("Step 3/4: Saving local cache…")
        save_embedding_cache(embeddings, name_to_idx, args.cache_path)

    if not args.skip_writeback:
        logger.info("Step 4/4: Writing embeddings back to Neo4j…")
        await write_embeddings_to_neo4j(embeddings, name_to_idx)

    logger.info("Done! %d card embeddings of dim %d", embeddings.size(0), embeddings.size(1))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="Train card graph embeddings")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cache-path", default="data/card_graph_embeddings.pt")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip training, just write cached embeddings to Neo4j")
    parser.add_argument("--skip-writeback", action="store_true",
                        help="Skip writing embeddings to Neo4j")
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
