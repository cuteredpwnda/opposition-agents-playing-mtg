# Report 03 — KG Build Status (Static vs. Dynamic)

**Date:** 2026-04-28

## Question

> Is the KG building dynamic already?

**Yes — partially.** The graph is constructed in two distinct phases that
run in different time scales:

## Phase 1 — Static bootstrap (offline, run once per data refresh)

| Module | What it does |
|---|---|
| [src/knowledge/n10s_setup.py](../../src/knowledge/n10s_setup.py) `N10sSetup.full_setup()` | Imports the OWL ontology TBox + ABox via `n10s.onto.import.fetch` / `n10s.rdf.import.fetch`, creates uniqueness constraints, full-text + vector indexes. |
| [src/knowledge/kg_builder.py](../../src/knowledge/kg_builder.py) `KGBuilder.import_scryfall_cards()` | Bulk-inserts 37 384 `:Card` nodes from `data/scryfall/oracle-cards.json` (UNWIND batches of 500). Sets sub-labels (`:Creature`, `:Instant`, …) from `type_line`. |
| `KGBuilder.import_combos()` | Inserts `:Combo` nodes + `[:PART_OF_COMBO]` edges from Commander Spellbook v2. |
| `KGBuilder.import_legalities()` | Creates `[:LEGAL_IN]` edges to `:Format` nodes. |
| [scripts/train_graph_embeddings.py](../../scripts/train_graph_embeddings.py) | Trains a 2-layer GraphSAGE on the exported edge set, writes 128-d `embedding` back to every `:Card`. |

These steps are wrapped in stages 2–3 of `scripts/train_pipeline.py` and
typically run once per Scryfall update.

## Phase 2 — Dynamic enrichment from gameplay (every pipeline run)

| Module | What it writes |
|---|---|
| [src/knowledge/kg_enrichment.py](../../src/knowledge/kg_enrichment.py) `KGEnrichment.enrich_from_trajectories()` | Mines synergies, combos, win-rate stats from a `TrajectoryStore`. |
| `MTGKnowledgeGraph.add_synergy()` | `MERGE (a)-[r:SYNERGIZES_WITH]-(b) ON MATCH SET r.strength = r.strength + weight` — additive edge weight reinforced over many games. |
| `MTGKnowledgeGraph.update_card_stats()` | `c.gamesPlayed`, `c.gamesWon`, `c.winRate` updated card-by-card. |

This **does** mutate Neo4j every time stage 4.5 runs, so the KG genuinely
grows with self-play. Concretely, after the first 5-game run the
enrichment report logged:

```
Enrichment complete: 40 synergies proposed (40 written),
                     0 combos proposed,
                     0 card stats updated
```

(card stats threshold is `min_games_for_stats=10` so we don't yet hit it
with only 5 games.)

## What is **not** dynamic yet

- Surprise-driven KG patching: `SurpriseDetector.analyze_trajectory()`
  identifies high-error transitions but currently only logs them; it does
  not yet propose new ontology edges. (Queued in
  `IMPLEMENTATION_PLAN.md`.)
- Archetype clustering: `[:BELONGS_TO_ARCHETYPE]` edges still come from
  the static ontology. There is a Louvain hook in
  [src/knowledge/graph_rag.py](../../src/knowledge/graph_rag.py)
  `run_community_detection()` but it is not wired into the training loop.
- LLM-extracted oracle-text effects: `Effect` nodes and
  `[:PRODUCES_EFFECT]` edges are populated from the static ontology only.

## Schema (current)

Node labels:
`Card`, `Creature`, `Instant`, `Sorcery`, `Enchantment`, `Artifact`,
`Planeswalker`, `Land`, `Battle`, `Combo`, `Archetype`, `Keyword`,
`Effect`, `Format`.

Relationship types:
`PART_OF_COMBO`, `SYNERGIZES_WITH`, `COUNTERS`, `ENABLES`, `HAS_KEYWORD`,
`BELONGS_TO_ARCHETYPE`, `LEGAL_IN`, `BANNED_IN`, `PRODUCES_EFFECT`,
`HAS_WIN_CONDITION`, `STRONG_AGAINST`, `WEAK_AGAINST`, `SCO`, `SPO`,
`DOMAIN`, `RANGE`.

Card properties used by queries:
`cardName`, `oracleText`, `manaCostText`, `manaValue`, `typeLine`,
`colors`, `colorIdentity`, `keywords`, `power`, `toughness`, `loyalty`,
`rarity`, `setCode`, `edhrecRank`, `embedding` (128-d vector),
`strategicCluster` (Louvain), `gamesPlayed`, `gamesWon`, `winRate`.
