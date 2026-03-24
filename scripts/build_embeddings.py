"""
Build GNN embeddings and write back to Neo4j vector index.

Pipeline:
1. Export card graph from Neo4j (nodes + edges)
2. Build PyG Data object
3. Train 2-layer GraphSAGE encoder
4. Write 384-dim embeddings back to each :Card node
5. Create Neo4j vector index for similarity search

Usage: python -m scripts.build_embeddings
"""

from __future__ import annotations

import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def export_graph():
    """Export card graph from Neo4j into node features + edge list."""
    from neo4j import AsyncGraphDatabase
    from src.config import settings

    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )

    try:
        async with driver.session() as session:
            # Get all card nodes with basic features
            result = await session.run("""
                MATCH (c:Card)
                RETURN c.cardName AS name,
                       c.manaValue AS cmc,
                       c.typeLine AS type_line,
                       c.oracleText AS oracle_text,
                       c.colors AS colors,
                       c.colorIdentity AS color_identity,
                       c.power AS power,
                       c.toughness AS toughness,
                       c.edhrecRank AS edhrec_rank,
                       id(c) AS neo4j_id
                ORDER BY neo4j_id
            """)
            cards = [dict(record) async for record in result]
            logger.info("Exported %d card nodes", len(cards))

            if not cards:
                return None, None, None

            # Build name → index mapping
            name_to_idx = {c["name"]: i for i, c in enumerate(cards)}

            # Get edges (SYNERGIZES_WITH, PART_OF_COMBO, COUNTERS, HAS_KEYWORD)
            result = await session.run("""
                MATCH (a:Card)-[r:SYNERGIZES_WITH|PART_OF_COMBO|COUNTERS|HAS_KEYWORD]-(b:Card)
                RETURN a.cardName AS source, b.cardName AS target, type(r) AS rel_type
            """)
            edges = [dict(record) async for record in result]
            logger.info("Exported %d edges", len(edges))

            return cards, name_to_idx, edges
    finally:
        await driver.close()


def build_features(cards):
    """Build simple numeric feature vectors for each card."""
    import numpy as np

    features = []
    all_colors = ["W", "U", "B", "R", "G"]

    for card in cards:
        f = []
        # CMC (normalized)
        cmc = float(card.get("cmc") or 0)
        f.append(min(cmc / 15.0, 1.0))

        # Color identity (5-dim binary)
        ci = card.get("color_identity") or []
        for c in all_colors:
            f.append(1.0 if c in ci else 0.0)

        # Type flags
        tl = (card.get("type_line") or "").lower()
        for t in ["creature", "instant", "sorcery", "enchantment", "artifact", "planeswalker", "land"]:
            f.append(1.0 if t in tl else 0.0)

        # Power/toughness (normalized)
        try:
            power = float(card.get("power") or 0)
        except (ValueError, TypeError):
            power = 0.0
        try:
            toughness = float(card.get("toughness") or 0)
        except (ValueError, TypeError):
            toughness = 0.0
        f.append(min(power / 15.0, 1.0))
        f.append(min(toughness / 15.0, 1.0))

        # EDHREC rank (normalized, lower = more popular)
        rank = card.get("edhrec_rank")
        f.append(1.0 - min((rank or 30000) / 30000.0, 1.0))

        features.append(f)

    return np.array(features, dtype=np.float32)


def train_graphsage(features, edge_index, num_nodes, hidden_dim=256, out_dim=384, epochs=100):
    """Train a 2-layer GraphSAGE model on the card graph."""
    import torch
    from src.knowledge.graph_embedder import CardGraphEmbedder

    in_dim = features.shape[1]
    model = CardGraphEmbedder(in_channels=in_dim, hidden_channels=hidden_dim, out_channels=out_dim)
    x = torch.tensor(features, dtype=torch.float32)
    ei = torch.tensor(edge_index, dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, ei)

        # Self-supervised: connected nodes should have similar embeddings
        # Sample positive pairs from edges
        if ei.shape[1] > 0:
            src_emb = out[ei[0]]
            dst_emb = out[ei[1]]
            pos_loss = -torch.nn.functional.cosine_similarity(src_emb, dst_emb).mean()

            # Negative sampling: random pairs should be dissimilar
            neg_src = torch.randint(0, num_nodes, (min(ei.shape[1], 1000),))
            neg_dst = torch.randint(0, num_nodes, (min(ei.shape[1], 1000),))
            neg_loss = torch.nn.functional.cosine_similarity(out[neg_src], out[neg_dst]).mean()

            loss = pos_loss + neg_loss
        else:
            # No edges: just regularize
            loss = out.norm(dim=1).mean() * 0.01

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, epochs, loss.item())

    model.eval()
    with torch.no_grad():
        embeddings = model(x, ei).numpy()

    return embeddings


async def write_embeddings(cards, embeddings):
    """Write embeddings back to Neo4j card nodes."""
    from neo4j import AsyncGraphDatabase
    from src.config import settings

    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )

    try:
        batch_size = 500
        written = 0
        for i in range(0, len(cards), batch_size):
            batch = []
            for j in range(i, min(i + batch_size, len(cards))):
                batch.append({
                    "name": cards[j]["name"],
                    "embedding": embeddings[j].tolist(),
                })

            async with driver.session() as session:
                await session.run("""
                    UNWIND $batch AS item
                    MATCH (c:Card {cardName: item.name})
                    SET c.embedding = item.embedding
                """, batch=batch)
            written += len(batch)
            if written % 5000 == 0:
                logger.info("Written %d/%d embeddings", written, len(cards))

        logger.info("All %d embeddings written to Neo4j", written)
    finally:
        await driver.close()


async def main() -> None:
    """Export graph from Neo4j → train GraphSAGE → write embeddings back."""
    import numpy as np
    from src.knowledge.n10s_setup import N10sSetup

    # 1. Create vector index
    setup = N10sSetup()
    try:
        await setup.create_vector_index(dimensions=384)
        logger.info("Vector index created (or already exists)")
    except Exception:
        logger.info("Vector index setup skipped")
    finally:
        await setup.close()

    # 2. Export graph
    cards, name_to_idx, edges = await export_graph()
    if cards is None:
        logger.warning("No cards in Neo4j — run import_scryfall.py first")
        return

    # 3. Build features & edge index
    features = build_features(cards)
    logger.info("Feature matrix: %s", features.shape)

    # Build edge_index from edge list
    edge_src, edge_dst = [], []
    for e in edges:
        src = name_to_idx.get(e["source"])
        dst = name_to_idx.get(e["target"])
        if src is not None and dst is not None:
            edge_src.append(src)
            edge_dst.append(dst)
            # Undirected
            edge_src.append(dst)
            edge_dst.append(src)

    if edge_src:
        edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    logger.info("Edge index: %s", edge_index.shape)

    # 4. Train GraphSAGE
    embeddings = train_graphsage(
        features, edge_index, num_nodes=len(cards),
        hidden_dim=256, out_dim=384, epochs=100,
    )
    logger.info("Embeddings shape: %s", embeddings.shape)

    # 5. Save locally
    np.savez(
        "data/card_embeddings/graph_embeddings.npz",
        names=[c["name"] for c in cards],
        embeddings=embeddings,
    )
    logger.info("Saved to data/card_embeddings/graph_embeddings.npz")

    # 6. Write to Neo4j
    await write_embeddings(cards, embeddings)


if __name__ == "__main__":
    import os
    os.makedirs("data/card_embeddings", exist_ok=True)
    asyncio.run(main())
