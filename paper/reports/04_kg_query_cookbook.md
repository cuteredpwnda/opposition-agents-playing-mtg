# Report 04 — KG Exploration Cookbook

A copy-pasteable Cypher cheat-sheet for the live Neo4j graph in `mtg-neo4j`.
Open the browser at <http://localhost:7474> (default `neo4j` / password
from `.env`) and paste any of the queries below.

The runtime queries used by the agent layer live in
[src/knowledge/knowledge_graph.py](../../src/knowledge/knowledge_graph.py)
and [src/knowledge/graph_rag.py](../../src/knowledge/graph_rag.py).

---

## 1. Sanity-check the imports

```cypher
// Total cards + label histogram
MATCH (c:Card)
RETURN count(c) AS total_cards;

CALL db.labels() YIELD label
CALL { WITH label
       MATCH (n) WHERE label IN labels(n)
       RETURN count(n) AS n }
RETURN label, n ORDER BY n DESC;

// Relationship-type histogram
CALL db.relationshipTypes() YIELD relationshipType
CALL { WITH relationshipType
       MATCH ()-[r]->() WHERE type(r) = relationshipType
       RETURN count(r) AS n }
RETURN relationshipType, n ORDER BY n DESC;

// How many cards have a graph embedding?
MATCH (c:Card) WHERE c.embedding IS NOT NULL
RETURN count(c) AS cards_with_embedding;
```

## 2. Card lookup

```cypher
// Single card with all properties
MATCH (c:Card {cardName: "Lightning Bolt"})
RETURN c;

// Search by partial text (uses Lucene full-text index)
CALL db.index.fulltext.queryNodes('cardSearch', 'sacrifice creature draw')
YIELD node, score
RETURN node.cardName, node.oracleText, score
ORDER BY score DESC LIMIT 10;

// All instants ≤ 2 mana legal in Modern
MATCH (c:Instant)-[:LEGAL_IN]->(:Format {name: 'modern'})
WHERE c.manaValue <= 2
RETURN c.cardName, c.manaCostText, c.oracleText
ORDER BY c.manaValue, c.cardName;
```

## 3. Combos

```cypher
// All combos that include a card
MATCH (c:Card {cardName: "Devoted Druid"})-[:PART_OF_COMBO]->(combo:Combo)
MATCH (combo)<-[:PART_OF_COMBO]-(piece:Card)
RETURN combo.comboDescription, collect(DISTINCT piece.cardName) AS pieces;

// Which combos can I assemble from a hypothetical card pool?
WITH ["Devoted Druid","Vizier of Remedies","Walking Ballista"] AS pool
MATCH (combo:Combo)
WITH combo, pool, [(combo)<-[:PART_OF_COMBO]-(c) | c.cardName] AS pieces
WHERE ALL(p IN pieces WHERE p IN pool)
RETURN combo.comboDescription, pieces;

// Near-combos — one piece missing
WITH ["Devoted Druid","Vizier of Remedies"] AS pool
MATCH (combo:Combo)
WITH combo, pool, [(combo)<-[:PART_OF_COMBO]-(c) | c.cardName] AS pieces
WITH combo, pieces, [p IN pieces WHERE NOT p IN pool] AS missing
WHERE size(missing) = 1
RETURN combo.comboDescription, pieces, missing[0] AS missing_piece;
```

## 4. Synergies (live, grows with self-play)

```cypher
// Top synergies discovered by KGEnrichment
MATCH (a:Card)-[s:SYNERGIZES_WITH]-(b:Card)
WHERE s.strength IS NOT NULL
RETURN a.cardName, b.cardName, s.strength
ORDER BY s.strength DESC LIMIT 25;

// Synergy partners of one card
MATCH (:Card {cardName: "Lightning Bolt"})-[s:SYNERGIZES_WITH]-(other:Card)
RETURN other.cardName, s.strength
ORDER BY s.strength DESC LIMIT 10;

// Cards whose self-play win-rate is dynamic (stage-4.5 stats)
MATCH (c:Card)
WHERE c.gamesPlayed >= 5
RETURN c.cardName, c.gamesPlayed, c.gamesWon, c.winRate
ORDER BY c.winRate DESC LIMIT 25;
```

## 5. Counter-play

```cypher
// What answers a given threat?
MATCH (answer:Card)-[:COUNTERS]->(threat:Card {cardName: "Sheoldred, the Apocalypse"})
OPTIONAL MATCH (answer)-[:LEGAL_IN]->(f:Format)
RETURN answer.cardName, answer.manaCostText, collect(DISTINCT f.name) AS formats;
```

## 6. Archetypes

```cypher
// Infer likely archetype from observed cards
WITH ["Goblin Guide","Lightning Bolt","Eidolon of the Great Revel"] AS seen
UNWIND seen AS cardName
MATCH (:Card {cardName: cardName})-[:BELONGS_TO_ARCHETYPE]->(a:Archetype)
WITH a, count(DISTINCT cardName) AS hits, size(seen) AS total
RETURN a.name, hits, toFloat(hits)/total AS confidence
ORDER BY confidence DESC LIMIT 5;

// Signature cards of an archetype
MATCH (:Archetype {name: 'Burn'})<-[:BELONGS_TO_ARCHETYPE]-(c:Card)
RETURN c.cardName, c.manaCostText
ORDER BY c.edhrecRank;
```

## 7. Graph traversal (APOC)

```cypher
// 2-hop strategic neighbourhood around a card
MATCH (start:Card {cardName: "Thoracle"})
CALL apoc.path.subgraphAll(start, {
    maxLevel: 2,
    relationshipFilter:
        'PART_OF_COMBO|SYNERGIZES_WITH|COUNTERS|ENABLES|BELONGS_TO_ARCHETYPE|HAS_KEYWORD'
}) YIELD nodes, relationships
RETURN [n IN nodes | labels(n)[0] + ': ' + coalesce(n.cardName, n.name, n.comboDescription, '')] AS entities,
       [r IN relationships | type(r)] AS edge_types;

// PageRank — most "important" cards in the graph
CALL gds.pageRank.stream('cardGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).cardName AS card, score
ORDER BY score DESC LIMIT 20;
```

## 8. Vector / semantic search

```cypher
// Find 10 cards semantically similar to a given embedding
// (replace $embedding with a 128-d float list from CardEmbeddingModel)
CALL db.index.vector.queryNodes('cardEmbeddings', 10, $embedding)
YIELD node, score
RETURN node.cardName, node.typeLine, score;

// Cards in the same Louvain community
MATCH (c:Card {cardName: "Birds of Paradise"})
WITH c.strategicCluster AS cluster
MATCH (m:Card {strategicCluster: cluster})
RETURN cluster, count(m) AS size, collect(m.cardName)[..15] AS sample;
```

## 9. Quality / debugging

```cypher
// Cards with no edges (orphans) — should be 0 after a clean import
MATCH (c:Card)
WHERE NOT (c)--()
RETURN count(c);

// Combos whose card list isn't fully in the graph
MATCH (combo:Combo)<-[:PART_OF_COMBO]-(c)
WITH combo, count(c) AS pieces
WHERE pieces < 2
RETURN combo.comboDescription, pieces;

// Show me the dynamic edges added by stage 4.5 in the last hour
MATCH ()-[r:SYNERGIZES_WITH]-()
WHERE r.strength IS NOT NULL
RETURN count(r) AS dynamic_synergies;
```

## 10. From the Python side

```python
import asyncio
from src.knowledge.knowledge_graph import MTGKnowledgeGraph

async def main():
    kg = MTGKnowledgeGraph()
    print(await kg.get_combos_containing("Devoted Druid"))
    print(await kg.detect_available_combos(["Devoted Druid","Vizier of Remedies","Walking Ballista"]))
    print(await kg.subgraph_context("Lightning Bolt", depth=2))
    await kg.close()

asyncio.run(main())
```

---

### Tip — running a quick smoke query from PowerShell

```powershell
docker exec -it mtg-neo4j cypher-shell -u neo4j -p $env:NEO4J_PASSWORD `
  "MATCH (c:Card) RETURN count(c) AS n;"
```

---

## 11. Property schema (verified 2026-04-29)

The actual property names differ from some older query examples.
Always use these names:

### Card node (`c:Card`)
| Property | Type | Notes |
|---|---|---|
| `cardName` | string | ⚠️ **not** `name` |
| `scryfallId` | string | UUID |
| `oracleText` | string | |
| `typeLine` | string | e.g. `"Legendary Creature — Goblin Warrior"` |
| `manaCostText` | string | e.g. `"{3}{R}{R}"` |
| `manaValue` | int | converted mana cost |
| `power` / `toughness` | string | `null` for non-creatures |
| `colorIdentity` | list[string] | e.g. `["R"]` |
| `colors` | list[string] | |
| `rarity` | string | `common` / `uncommon` / `rare` / `mythic` |
| `setCode` | string | |
| `edhrecRank` | int | lower = more popular |
| `embedding` | list[float] | 128-d GraphSAGE embedding (all 36,909 cards populated) |
| `gamesPlayed` | int | written by KGEnrichment stage (0 until enrichment runs) |
| `gamesWon` | int | written by KGEnrichment stage |
| `winRate` | float | written by KGEnrichment stage |

### Combo node (`n:Combo`)
| Property | Type |
|---|---|
| `comboId` | string |
| `comboName` | string |
| `comboDescription` | string |
| `result` | string |
| `cardCount` | int |
| `outcomeCategories` | list[string] |
| `outcomeMagnitudes` | list[string] |

### Relationship summary (as of 2026-04-29)
| Relationship | Count | Notes |
|---|---|---|
| `LEGAL_IN` | 338,727 | Card → Format |
| `PART_OF_COMBO` | 29,557 | Card → Combo |
| `PRODUCES` | 24,776 | Combo → Outcome |
| `HAS_KEYWORD` | 23,067 | Card → Keyword |
| `SYNERGIZES_WITH` | 0* | written by `scripts/run_kg_enrichment.py` |
| `SUPPORTED_BY` | 0* | links Card ↔ LearnedSynergyEvidence node |

*\* Not yet written — run `scripts/run_kg_enrichment.py` after self-play.*

### ⚠️ Common mistake — `c.name` returns null
```cypher
-- WRONG (returns null for all rows):
MATCH (c:Card)-[:PART_OF_COMBO]->(combo:Combo)
RETURN c.name, count(combo)

-- CORRECT:
MATCH (c:Card)-[:PART_OF_COMBO]->(combo:Combo)
WITH c, count(combo) AS n ORDER BY n DESC LIMIT 10
RETURN c.cardName, n
```

## 12. KG Extension Layer (self-play synergies)

Run enrichment once you have trajectories in `data/trajectories/`:

```powershell
# Dry run — see what would be written without touching Neo4j
.\.venv\Scripts\python.exe scripts\run_kg_enrichment.py --dry-run

# Write synergies with relaxed thresholds (good for small trajectory sets)
.\.venv\Scripts\python.exe scripts\run_kg_enrichment.py --min-co 3 --min-lift 1.2
```

Query learned synergies after enrichment:

```cypher
// All learned synergy pairs, sorted by lift
MATCH (a:Card)-[s:SYNERGIZES_WITH]->(b:Card)
RETURN a.cardName, b.cardName, s.lift, s.pairWinRate, s.coOccurrences
ORDER BY s.lift DESC LIMIT 25;

// LearnedSynergyEvidence nodes (provenance trail)
MATCH (ev:LearnedSynergyEvidence)
RETURN ev.runId, ev.source, ev.lift, ev.pairWinRate,
       ev.cardA, ev.cardB
ORDER BY ev.lift DESC LIMIT 20;

// Cards whose stats were updated
MATCH (c:Card)
WHERE c.gamesPlayed >= 3
RETURN c.cardName, c.gamesPlayed, c.gamesWon, c.winRate
ORDER BY c.winRate DESC LIMIT 25;

// Count all dynamic edges (should be > 0 after enrichment)
MATCH ()-[r:SYNERGIZES_WITH]->()
RETURN count(r) AS synergy_edges;
```
