"""Walk through the KG cookbook (paper/reports/04) live.

Runs the headline queries from each cookbook section against
``mtg-neo4j`` and prints a short, human-readable summary.  Useful as a
post-build smoke test to confirm the graph is populated end-to-end.

    python -m examples.kg_cookbook_walkthrough
"""

from __future__ import annotations

import asyncio
from typing import Any

from neo4j import AsyncGraphDatabase

from src.config import settings


async def _run(driver, query: str, **params: Any) -> list[dict]:
    async with driver.session() as session:
        result = await session.run(query, **params)
        return [r.data() async for r in result]


def _hr(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


async def main() -> None:
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    try:
        # ---------------------------------------------------------------
        _hr("1. Inventory")
        labels = await _run(
            driver,
            "CALL db.labels() YIELD label "
            "CALL { WITH label MATCH (n) WHERE label IN labels(n) RETURN count(n) AS n } "
            "RETURN label, n ORDER BY n DESC LIMIT 12",
        )
        for r in labels:
            print(f"  {r['label']:<20} {r['n']:>10,}")
        rels = await _run(
            driver,
            "CALL db.relationshipTypes() YIELD relationshipType "
            "CALL { WITH relationshipType MATCH ()-[r]->() WHERE type(r) = relationshipType "
            "       RETURN count(r) AS n } "
            "RETURN relationshipType AS t, n ORDER BY n DESC",
        )
        print()
        for r in rels:
            print(f"  {r['t']:<25} {r['n']:>10,}")

        # ---------------------------------------------------------------
        _hr("2. Card lookup — Lightning Bolt")
        rows = await _run(
            driver,
            "MATCH (c:Card {cardName: 'Lightning Bolt'}) "
            "RETURN c.manaCostText AS cost, c.oracleText AS oracle, "
            "       [(c)-[:LEGAL_IN]->(f:Format) | f.name] AS formats",
        )
        if rows:
            r = rows[0]
            print(f"  cost   : {r['cost']}")
            print(f"  oracle : {(r['oracle'] or '')[:120]}...")
            print(f"  formats: {', '.join(sorted(r['formats'] or []))}")
        else:
            print("  (Lightning Bolt missing — run stage 2)")

        # ---------------------------------------------------------------
        _hr("3. Combos")
        n_combos = await _run(driver, "MATCH (c:Combo) RETURN count(c) AS n")
        print(f"  Combo nodes: {n_combos[0]['n']:,}")
        if n_combos[0]["n"] > 0:
            sample = await _run(
                driver,
                "MATCH (combo:Combo)<-[:PART_OF_COMBO]-(c:Card) "
                "WITH combo, collect(c.cardName) AS pieces "
                "RETURN combo.comboDescription AS desc, pieces "
                "LIMIT 3",
            )
            for r in sample:
                desc = (r["desc"] or "")[:60]
                print(f"  • {desc:<60} {r['pieces'][:3]}")

        # ---------------------------------------------------------------
        _hr("4. Synergies (dynamic — written by stage-4.5 enrichment)")
        n_syn = await _run(
            driver,
            "MATCH ()-[s:SYNERGIZES_WITH]-() RETURN count(s) AS n",
        )
        print(f"  SYNERGIZES_WITH edges: {n_syn[0]['n']:,}")
        if n_syn[0]["n"] > 0:
            top = await _run(
                driver,
                "MATCH (a:Card)-[s:SYNERGIZES_WITH]-(b:Card) "
                "WHERE s.strength IS NOT NULL AND a.cardName < b.cardName "
                "RETURN a.cardName AS a, b.cardName AS b, s.strength AS w "
                "ORDER BY s.strength DESC LIMIT 5",
            )
            for r in top:
                print(f"  • {r['a']:<25} ↔ {r['b']:<25} strength={r['w']:.3f}")

        # ---------------------------------------------------------------
        _hr("8. Embeddings")
        n_emb = await _run(
            driver,
            "MATCH (c:Card) WHERE c.embedding IS NOT NULL RETURN count(c) AS n",
        )
        print(f"  cards with embedding: {n_emb[0]['n']:,}")
        if n_emb[0]["n"] > 0:
            d = await _run(
                driver,
                "MATCH (c:Card) WHERE c.embedding IS NOT NULL "
                "RETURN size(c.embedding) AS d LIMIT 1",
            )
            print(f"  embedding dim: {d[0]['d']}")

            # Vector index test
            idx = await _run(
                driver,
                "SHOW INDEXES YIELD name WHERE name = 'cardEmbeddings' RETURN name",
            )
            if idx:
                sample = await _run(
                    driver,
                    "MATCH (c:Card) WHERE c.embedding IS NOT NULL "
                    "RETURN c.cardName AS name, c.embedding AS e LIMIT 1",
                )
                similar = await _run(
                    driver,
                    "CALL db.index.vector.queryNodes('cardEmbeddings', 5, $e) "
                    "YIELD node, score RETURN node.cardName AS name, score",
                    e=sample[0]["e"],
                )
                print(f"  nearest to {sample[0]['name']!r}:")
                for r in similar:
                    print(f"    {r['score']:.3f}  {r['name']}")
            else:
                print("  (vector index 'cardEmbeddings' not yet created)")

        # ---------------------------------------------------------------
        _hr("9. Quality")
        orphans = await _run(
            driver,
            "MATCH (c:Card) WHERE NOT (c)--() RETURN count(c) AS n",
        )
        total = await _run(driver, "MATCH (c:Card) RETURN count(c) AS n")
        pct = 100.0 * orphans[0]["n"] / max(total[0]["n"], 1)
        print(f"  orphans: {orphans[0]['n']:,} / {total[0]['n']:,}  ({pct:.1f}%)")

    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
