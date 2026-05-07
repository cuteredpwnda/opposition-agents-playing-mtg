# Open-Source MTG Game Engine & GraphRAG Research

*Compiled: May 2026*

---

## Table of Contents

1. [Python MTG Game Engines](#1-python-mtg-game-engines)
2. [Java/JVM MTG Engines (Reference)](#2-javajvm-mtg-engines-reference)
3. [MTG AI/Agent Projects](#3-mtg-aiagent-projects)
4. [Card Data & API Libraries](#4-card-data--api-libraries)
5. [GraphRAG & Knowledge Graph Projects](#5-graphrag--knowledge-graph-projects)
6. [Component Reusability Matrix](#6-component-reusability-matrix)
7. [GraphRAG Integration Architecture for MTG](#7-graphrag-integration-architecture-for-mtg)
8. [Recommendations](#8-recommendations)
9. [Collective Intelligence Pattern (Cross-Source)](#9-collective-intelligence-pattern-cross-source)

---

## 1. Python MTG Game Engines

### 1.1 open-mtg (hlynurd/open-mtg) ⭐⭐⭐ — HIGHEST RELEVANCE

| Attribute | Detail |
|-----------|--------|
| **URL** | https://github.com/hlynurd/open-mtg |
| **Language** | Python (78.7%), Jupyter Notebook (21.3%) |
| **Stars** | 152 |
| **Forks** | 33 |
| **License** | MIT |
| **Last Update** | ~2019 (inactive, 7 years old) |
| **Contributors** | 2 |

**What it implements:**
- ✅ Game state management (zones, players, life totals)
- ✅ Turn structure / phase management (phases as enum)
- ✅ Main phase (land drops, sorcery-speed spells)
- ✅ Mana system (mana pool, paying costs, generic mana combinations)
- ✅ Combat system (declare attackers/blockers, damage assignment ordering per CR 509.2 / 510.1c)
- ✅ Legal action generation (`game.get_moves()` returns indexed list of legal actions)
- ✅ AI via Monte Carlo Tree Search (MCTS) and Minimax
- ⚠️ Limited card pool (8th Edition core set decks only)
- ❌ No stack implementation
- ❌ No triggered abilities
- ❌ No instant-speed interaction

**Key design patterns we can adopt:**
- Clean `game.get_moves()` → `game.make_move(move)` interface — perfect for RL agents
- Phases as an enum with `current_phase_index`
- Attacker/blocker combinations computed as indexed lists
- Deep-copyable game state for tree search
- Player initialization with Python decklists

**Reusable components:**
- Combat damage assignment logic (CR-compliant ordering and distribution)
- Phase/step enum structure
- MCTS agent implementation pattern
- Game loop architecture (move generation → selection → execution)

---

### 1.2 mtg-python-engine (wanqizhu/mtg-python-engine) ⭐⭐⭐ — HIGHEST RELEVANCE

| Attribute | Detail |
|-----------|--------|
| **URL** | https://github.com/wanqizhu/mtg-python-engine |
| **Language** | Python (99.9%) |
| **Stars** | 67 |
| **Forks** | 21 |
| **License** | MIT |
| **Last Update** | May 2025 (mostly developed 2017-2019) |
| **Contributors** | 2 |

**What it implements:**
- ✅ Game state management (zones via `gameobject.GameObject`)
- ✅ Turn structure / full phase management
- ✅ The stack (`play.Play()` objects — spells and abilities on the stack, LIFO resolution)
- ✅ Mana system (cost payment, mana abilities)
- ✅ Combat system (`game.handle_combat_phase()`)
- ✅ State-based actions (`game.check_state_based_actions()`)
- ✅ Triggered abilities (onEtB, onAttack, intervening-if clauses)
- ✅ Activated abilities (cost/effect model with `can_activate()`)
- ✅ Static abilities & continuous effects (layer system, effects with expiration)
- ✅ Targeting system with legality checks (fizzling on illegal targets)
- ✅ Token creation
- ✅ Card data parsing (MTGJSON/Cockatrice → Python card classes)
- ✅ Counter system (+1/+1 etc.)
- ⚠️ Small card pool (~56/256 M15 cards, partial cube)
- ⚠️ Console-based UI for interaction
- ⚠️ Game rewind via deepcopy when illegal actions attempted

**Key design patterns:**
- Cards parsed into individual Python classes inheriting from `card.Card`
- Abilities defined in text files, processed by `cards.setup_cards()`
- `permanent.Permanent` wraps cards on battlefield with effects tracking
- Effects system: `permanent.add_effect(effect_name, details, source, expiration, toggle_func)`
- Spell resolution via `play.Play.apply()` with target legality checking
- Comprehensive Rules section tracking (documents which CR rules are implemented)

**Reusable components:**
- Stack implementation architecture (everything is a `Play` object)
- Triggered ability system (trigger conditions, effects, intervening-if)
- Static/continuous effects with toggle and expiration
- Targeting system with criteria-based validation
- State-based action checking pattern
- Card data parsing pipeline (MTGJSON → Card classes)
- Effect layers approach

---

### 1.3 cardboard (Julian/cardboard) — LOW RELEVANCE

| Attribute | Detail |
|-----------|--------|
| **URL** | https://github.com/Julian/cardboard |
| **Language** | Python (99.8%) |
| **Stars** | 7 |
| **License** | MIT |
| **Last Update** | 2012 (15 years old, abandoned) |
| **Contributors** | 1 |

**Assessment:** Very early-stage Python MTG engine. Too old and incomplete to be useful. Its existence confirms the difficulty of building a full MTG engine in Python. Not recommended for reuse.

---

## 2. Java/JVM MTG Engines (Reference)

### 2.1 Forge (Card-Forge/forge) ⭐⭐⭐ — GOLD STANDARD REFERENCE

| Attribute | Detail |
|-----------|--------|
| **URL** | https://github.com/Card-Forge/forge |
| **Language** | Java (98.1%), Python (0.8%) |
| **Stars** | 2,200 |
| **Forks** | 872 |
| **License** | GPL-3.0 |
| **Last Update** | Active (commits within hours) |
| **Contributors** | 228 |

**What it implements:**
- ✅ **Complete MTG rules engine** — the most comprehensive open-source implementation
- ✅ All game formats (Standard, Commander, Draft, Sealed, Legacy, etc.)
- ✅ 20,000+ cards implemented
- ✅ Full AI system (`forge-ai` module) with multiple difficulty levels
- ✅ All game zones, phases, stack, priority, combat
- ✅ State-based actions, triggered/activated/static abilities
- ✅ Mana system with complex interactions
- ✅ Cross-platform (Desktop, Android, iOS)
- ✅ Adventure mode, Quest mode
- ✅ LDA (Latent Dirichlet Allocation) for deck archetypes (`forge-lda`)

**Why it matters for us:**
- **Architecture reference**: Forge's modular architecture (`forge-core`, `forge-game`, `forge-ai`, `forge-gui`) is the blueprint for how to structure an MTG engine
- **Rules reference**: When implementing complex rules interactions, Forge's source code is the most reliable open-source reference
- **AI patterns**: The `forge-ai` module shows how to build AI that understands card evaluation, threat assessment, combat math
- **Card scripting**: Forge's card definition format shows how to encoding card abilities declaratively

**Limitations for direct reuse:**
- Java (not Python) — cannot directly import
- GPL-3.0 license — viral copyleft, any derivative work must also be GPL-3.0
- Massive codebase — extracting components is non-trivial

---

### 2.2 Magarena (magarena/magarena) ⭐⭐ — AI REFERENCE

| Attribute | Detail |
|-----------|--------|
| **URL** | https://github.com/magarena/magarena |
| **Language** | Java (66.6%), Groovy (30.5%) |
| **Stars** | 438 |
| **Forks** | 101 |
| **License** | GPL-3.0 |
| **Last Update** | 2021 (3 years inactive) |
| **Contributors** | 18 |

**What it implements:**
- ✅ Single-player MTG against AI
- ✅ Monte Carlo Tree Search AI (MCTS) — implemented by Melvin Zhang
- ✅ MMAB and MTDF AI algorithms 
- ✅ Card scripting via Groovy DSL
- ✅ Scryfall image integration
- ✅ Substantial card pool

**Key value:**
- **MCTS for MTG**: This is the best open-source reference for applying MCTS specifically to Magic gameplay
- **Algorithm comparison**: Contains experimental results for MMAB-H vs MTDF-H
- **AI evaluation framework**: Has infrastructure for comparing AI algorithms
- GPL-3.0 licensed — same copyleft restrictions as Forge

---

### 2.3 Argentum Engine (wingedsheep/argentum-engine) ⭐⭐⭐ — MODERN REFERENCE

| Attribute | Detail |
|-----------|--------|
| **URL** | https://github.com/wingedsheep/argentum-engine |
| **Language** | Kotlin (87.2%), TypeScript (12.0%) |
| **Stars** | 3 (new project) |
| **License** | Not specified |
| **Last Update** | Active (commits within hours) |
| **Contributors** | 2 (developer + Claude AI) |

**What it implements:**
- ✅ Full turn structure (phases, steps, priority)
- ✅ Stack and spell resolution
- ✅ Combat (attackers, blockers, damage assignment)
- ✅ Triggered, activated, and static abilities
- ✅ Keywords (flying, trample, deathtouch, morph, cycling, delve, etc.)
- ✅ State-based actions
- ✅ Targeting and legality checks
- ✅ Rule 613 layer system for continuous effects
- ✅ Replacement effects
- ✅ LLM-powered AI opponent (via OpenRouter)
- ✅ Online multiplayer with WebSocket
- ✅ Booster draft with up to 8 players
- ✅ Web client (React/TypeScript)
- ✅ Card definitions as pure data via Kotlin DSL
- ✅ Scryfall data integration

**Why it's especially relevant:**
- **Very active** — commits hourly, actively being developed
- **Modern architecture** — immutable game state, pure functional API 
- **LLM AI integration** — already uses OpenRouter for AI opponents; AI receives masked game state and responds through standard game protocol
- **Fallback heuristics** — when LLM fails, falls back to rule-based play
- **Built with Claude** — co-developed with AI, meaning its architecture is already AI-friendly
- Architecture patterns (immutable state, Kotlin DSL for cards, modular engine) translate well to Python

---

## 3. MTG AI/Agent Projects

### 3.1 mtg-player (theRealMarkCastillo/mtg-player) ⭐⭐⭐ — DIRECTLY RELEVANT

| Attribute | Detail |
|-----------|--------|
| **URL** | https://github.com/theRealMarkCastillo/mtg-player |
| **Language** | Python (99.9%) |
| **Stars** | 2 |
| **License** | MIT |
| **Last Update** | November 2025 (~4 months ago) |
| **Contributors** | 1 |

**What it implements:**
- ✅ LLM-powered MTG Commander AI agent
- ✅ Custom rules engine (game state, move validation, turn structure, combat, stack)
- ✅ Agentic architecture: LLM + tools (`get_game_state()`, `get_legal_actions()`, `execute_action()`, `analyze_threats()`, `get_stack_state()`, `can_respond()`, `evaluate_position()`)
- ✅ Chain-of-Thought reasoning for strategic planning
- ✅ Instant-speed stack interaction (Phase 4 complete)
- ✅ Turn history & memory system
- ✅ Political combat intelligence (4-factor scoring for multiplayer)
- ✅ Opponent modeling tool
- ✅ Strategy recommendation tool
- ✅ "Can I Win" lethal analysis tool
- ✅ Heuristic (no-LLM) mode for testing
- ✅ Multi-provider LLM support (OpenAI, Anthropic, Ollama, LM Studio, OpenRouter)
- ✅ 152-card Commander staples database
- ✅ Archetype deck builders (ramp, control, midrange)
- ✅ 4-player Commander with 40 life and commander zone
- ✅ Comprehensive logging system
- ✅ Pydantic v2 data models
- ✅ 20 tests passing

**Key architecture (directly reusable):**
```
src/
├── core/
│   ├── game_state.py     # Game state representation
│   ├── player.py         # Player state
│   ├── card.py           # Card models
│   ├── rules_engine.py   # Core rules implementation
│   └── stack.py          # Stack implementation
├── agent/
│   ├── llm_agent.py      # LLM decision-making agent
│   └── prompts.py        # Prompt templates
├── tools/
│   ├── game_tools.py     # Game state/action/stack/response tools
│   └── evaluation_tools.py  # Position evaluation
├── utils/
│   └── logger.py         # Logging utilities
└── data/
    └── cards.py          # Card database + deck builders
```

**Reusable components for our project:**
- **Tool-based agent architecture** — LLM calls tools, tools interact with engine, engine validates
- **Prompt engineering patterns** for MTG strategic reasoning
- **Heuristic fallback** — rule-based AI when LLM unavailable
- **Political combat scoring** — 4-factor priority for Commander multiplayer
- **Opponent modeling** approach
- **Multi-provider LLM abstraction**
- **Stack implementation** for instant-speed interaction

**Limitations:**
- Small card pool (152 cards, manually defined)
- Simplified rules engine (not Comprehensive Rules compliant)
- No knowledge graph or ontology
- No reinforcement learning
- No Scryfall integration for card data

---

### 3.2 MTG-Judge (oglantz/MTG-Judge)

| Attribute | Detail |
|-----------|--------|
| **URL** | https://github.com/oglantz/MTG-Judge |
| **Language** | Python (100%) |
| **Stars** | 0 |
| **License** | Not specified |
| **Last Update** | Active (days ago) |
| **Contributors** | 3 |

**What it implements:**
- AI-powered MTG rules judge
- Vector store for rules/rulings lookup
- Multiple LLM model testing (Qwen 2.5 3B working)
- RAG over MTG Comprehensive Rules

**Relevance:** Could inform our Judge Agent design. Uses RAG over the Comprehensive Rules document — a pattern we'll need for rules arbitration.

---

### 3.3 chat-mtg (nick-monto/chat-mtg)

| Attribute | Detail |
|-----------|--------|
| **URL** | https://github.com/nick-monto/chat-mtg |
| **Language** | Python |
| **Stars** | 0 |
| **Last Update** | Active (days ago) |

**What it implements:** Crew AI agent for MTG judge functionality. RAG and tool use with agentic architecture. Very small/early project but shows the pattern of using multi-agent frameworks for MTG.

---

### 3.4 ai.kvasir (cahyaong/ai.kvasir)

| Attribute | Detail |
|-----------|--------|
| **URL** | https://github.com/cahyaong/ai.kvasir |
| **Language** | C# |
| **Stars** | 6 |
| **Last Update** | January 2025 |

**What it implements:** ML/AI training platform for MTG agents. Simulation framework with WPF visualization. C# — not directly reusable but confirms the approach of treating MTG as an ML training ground.

---

## 4. Card Data & API Libraries

### 4.1 Scrython (NandaScott/Scrython) ⭐⭐⭐ — USE THIS

| Attribute | Detail |
|-----------|--------|
| **URL** | https://github.com/NandaScott/Scrython |
| **Language** | Python (100%) |
| **Stars** | 158 |
| **Forks** | 24 |
| **License** | MIT |
| **Last Update** | Active (v2.0.2, Jan 2026) |
| **PyPI** | `pip install scrython` |

**What it provides:**
- ✅ Complete Scryfall API wrapper for Python
- ✅ Card search (fuzzy, exact, Scryfall syntax queries)
- ✅ Bulk data download (oracle cards, all cards, rulings) — no rate limits
- ✅ Built-in rate limiting (10 req/s, Scryfall compliant)
- ✅ Built-in caching with TTL
- ✅ Card properties: `mana_cost`, `cmc`, `type_line`, `oracle_text`, `colors`, `legalities`, `image_uris`, `prices`
- ✅ Convenience methods: `is_legal_in('commander')`, `has_color('R')`, `is_creature`, `is_instant`
- ✅ Set information
- ✅ Auto-pagination for search results
- ✅ Serialization (to_dict, to_json, from_dict)
- ✅ Collection queries (batch card lookup)
- ✅ Autocomplete

**Integration plan:**
```python
import scrython

# Bulk download all cards (no rate limits)
bulk = scrython.bulk_data.ByType(type='oracle_cards')
cards = bulk.download()

# Also download all rulings
rulings_bulk = scrython.bulk_data.ByType(type='rulings')
rulings = rulings_bulk.download()
```

This should be our **primary card data source**. Download bulk data → parse into our Card model → populate the knowledge graph.

---

### 4.2 Scryfall API (Direct)

The Scryfall REST API at `https://api.scryfall.com/` provides:
- `/cards/search` — Full-text search with Scryfall syntax
- `/cards/named` — Exact/fuzzy card lookup
- `/cards/collection` — Batch card lookup (up to 75 per request)
- `/bulk-data` — Full database dumps (oracle cards, all printings, rulings)
- `/sets` — Set information
- `/symbology` — Mana symbol information
- `/rulings` — Card-specific rulings

For our project, Scrython wraps this perfectly. Use Scrython for one-off queries; use bulk data downloads for initial knowledge graph population.

---

### 4.3 Commander Spellbook

| Attribute | Detail |
|-----------|--------|
| **URL** | https://commanderspellbook.com |
| **API** | https://backend.commanderspellbook.com/api/v2/ |

**What it provides:**
- 10,000+ curated MTG combos
- Cards involved, result (infinite mana, infinite damage, etc.)
- Prerequisites and steps
- Color identity filtering

**Integration:** Scrape or API-call to populate the `Combo` nodes in our knowledge graph. This is the single best source for combo data.

---

## 5. GraphRAG & Knowledge Graph Projects

### 5.1 Microsoft GraphRAG (microsoft/graphrag) ⭐⭐⭐ — PRIMARY GRAPHRAG TOOL

| Attribute | Detail |
|-----------|--------|
| **URL** | https://github.com/microsoft/graphrag |
| **Language** | Python (88.2%), Jupyter Notebook (11.8%) |
| **Stars** | 31,500 |
| **Forks** | 3,300 |
| **License** | MIT |
| **Last Update** | Active (v3.0.6, last week) |
| **Used By** | 470+ projects |

**What it does:**
1. **Indexing pipeline**: Takes unstructured text → extracts entities & relationships → builds knowledge graph → performs hierarchical clustering (Leiden algorithm) → generates community summaries
2. **Query modes**:
   - **Global Search**: Holistic reasoning over the entire corpus using community summaries
   - **Local Search**: Entity-specific reasoning by fanning out to neighbors
   - **DRIFT Search**: Combines entity-specific with community context
   - **Basic Search**: Standard vector similarity (baseline RAG fallback)
3. **Key features**:
   - Automatic entity extraction from text
   - Relationship extraction
   - Hierarchical community detection
   - Community summarization
   - Prompt tuning framework

**Why it matters for our project:**
GraphRAG was designed for exactly the kind of complex, relational reasoning that MTG strategy requires. Baseline RAG (vector search over card text) would miss the *connections* between cards — combos, synergies, counter-play patterns. GraphRAG builds a structured knowledge graph and reasons over it hierarchically.

---

## 6. Component Reusability Matrix

| Component | open-mtg | mtg-python-engine | mtg-player | Forge | Argentum | Scrython |
|-----------|---------|-------------------|------------|-------|----------|----------|
| **Game State** | ✅ Basic | ✅ Good | ✅ Good | ✅ Complete | ✅ Complete | — |
| **Turn/Phases** | ✅ Enum | ✅ Full | ✅ Full | ✅ Full | ✅ Full | — |
| **Stack** | ❌ | ✅ Play objects | ✅ Basic | ✅ Complete | ✅ Complete | — |
| **Mana System** | ✅ Basic | ✅ Good | ✅ Basic | ✅ Complete | ✅ Complete | — |
| **Combat** | ✅ CR-compliant | ✅ Good | ✅ Basic | ✅ Complete | ✅ Complete | — |
| **Rules Engine** | ⚠️ Partial | ✅ Substantial | ✅ Basic | ✅ Complete | ✅ Complete | — |
| **SBAs** | ❌ | ✅ | ⚠️ Partial | ✅ | ✅ | — |
| **Triggered Abilities** | ❌ | ✅ | ✅ ETB/dies | ✅ | ✅ | — |
| **Card Data** | ❌ | ⚠️ MTGJSON parser | ⚠️ 152 cards | ✅ 20k+ | ✅ Multi-set | ✅ All cards |
| **AI/Agents** | ✅ MCTS | ❌ | ✅ LLM Agent | ✅ Heuristic AI | ✅ LLM + Heuristic | — |
| **LLM Integration** | ❌ | ❌ | ✅ Multi-provider | ❌ | ✅ OpenRouter | — |
| **Python** | ✅ | ✅ | ✅ | ❌ Java | ❌ Kotlin | ✅ |
| **License** | MIT ✅ | MIT ✅ | MIT ✅ | GPL-3.0 ⚠️ | Unspecified | MIT ✅ |

---

## 7. GraphRAG Integration Architecture for MTG

### 7.1 Why GraphRAG for MTG?

Standard RAG (vector search) would work for simple card lookups ("what does Lightning Bolt do?"). But MTG strategic questions are inherently *relational*:

- "What beats a Devoted Druid + Vizier of Remedies combo?" → requires traversing COUNTERS edges
- "What are the best Elf tribal synergies in Commander?" → requires SYNERGIZES_WITH fanout from a tribe
- "How does the Grixis Death's Shadow matchup play out against UW Control?" → requires holistic archetype-level reasoning

GraphRAG excels at exactly these types of queries because it builds and reasons over a knowledge graph rather than flat text chunks.

### 7.2 Building the MTG Knowledge Graph with GraphRAG

**Step 1: Corpus Preparation**

Assemble text documents from multiple sources:
```
corpus/
├── cards/              # Oracle text for all Standard/Commander-legal cards
├── combos/             # Commander Spellbook combo descriptions
├── rulings/            # Scryfall rulings per card
├── archetypes/         # Archetype descriptions scraped from MTGGoldfish/EDHREC
├── comprehensive_rules/ # MTG Comprehensive Rules text
└── strategy_guides/    # Strategy articles, matchup guides
```

**Step 2: GraphRAG Indexing**

Run GraphRAG's indexing pipeline on this corpus:
```bash
graphrag init --root ./mtg-corpus
graphrag index --root ./mtg-corpus
```

This will:
1. **Extract entities**: Cards, Keywords, Mechanics, Archetypes, Players, Formats
2. **Extract relationships**: SYNERGIZES_WITH, COUNTERS, ENABLES, PART_OF_COMBO, HAS_KEYWORD, BELONGS_TO_ARCHETYPE
3. **Build communities**: Groups of tightly-connected cards (tribal clusters, combo packages, archetype cores)
4. **Generate summaries**: "This community represents the Grixis Death's Shadow archetype, centered around cards like Death's Shadow, Thoughtseize, and Street Wraith that leverage low life totals for advantage..."

**Step 3: Custom Entity/Relationship Types**

Tune GraphRAG prompts to extract MTG-specific entities:

```yaml
# graphrag entity extraction prompt tuning
entity_types:
  - Card
  - Combo
  - Archetype
  - Keyword
  - ManaColor
  - Format
  - WinCondition
  - CounterPlay

relationship_types:
  - SYNERGIZES_WITH
  - COUNTERS
  - ENABLES
  - PART_OF_COMBO
  - HAS_KEYWORD
  - BELONGS_TO_ARCHETYPE
  - ANSWERS
  - TRIGGERS
```

### 7.3 Query Integration with Agents

```python
# Agent uses GraphRAG for strategic reasoning
class NeuralReasoningModule:
    def __init__(self):
        self.graphrag = GraphRAGClient("./mtg-corpus")
    
    def find_counterplay(self, threat_card: str) -> list[str]:
        """Global search: what answers this threat across the metagame?"""
        result = self.graphrag.global_search(
            f"What cards and strategies effectively counter {threat_card}?"
        )
        return result.response
    
    def evaluate_synergies(self, cards: list[str]) -> dict:
        """Local search: how do these cards work together?"""
        result = self.graphrag.local_search(
            f"What synergies exist between {', '.join(cards)}?"
        )
        return result.response
    
    def identify_archetype(self, decklist: list[str]) -> str:
        """DRIFT search: what archetype does this deck represent?"""
        result = self.graphrag.drift_search(
            f"What MTG archetype is represented by a deck containing: {', '.join(decklist[:10])}?"
        )
        return result.response
    
    def find_combos(self, hand_and_board: list[str]) -> list[dict]:
        """Local search: are there any combos available?"""
        result = self.graphrag.local_search(
            f"What combos can be assembled from these cards: {', '.join(hand_and_board)}?"
        )
        return result.response
```

### 7.4 Hybrid Approach: GraphRAG + Custom Knowledge Graph

The optimal architecture combines:

1. **Custom KG (NetworkX/Neo4j)** — our `MTGKnowledgeGraph` class from PLAN.md, populated from structured data (Scryfall, Commander Spellbook). This is fast, deterministic, and queryable with graph algorithms.

2. **GraphRAG layer** — built on top of unstructured text (strategy articles, comprehensive rules, oracle text). This handles natural language queries and discovers relationships that aren't explicitly encoded.

3. **GNN module** — Graph Neural Networks trained on the custom KG to learn embeddings for cards, combos, and archetypes. These embeddings feed into the agent's strategic evaluation.

```
                    Agent Strategic Query
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
    ┌─────────────┐ ┌─────────────┐ ┌──────────────┐
    │  Custom KG  │ │  GraphRAG   │ │  GNN Module  │
    │ (NetworkX)  │ │ (Microsoft) │ │  (PyG/DGL)   │
    │             │ │             │ │              │
    │ Structured  │ │ Unstructured│ │ Learned      │
    │ queries:    │ │ queries:    │ │ embeddings:  │
    │ - Combos    │ │ - Strategy  │ │ - Card sims  │
    │ - Synergies │ │ - Matchups  │ │ - Board eval │
    │ - Counters  │ │ - Rulings   │ │ - Combo prob │
    └─────────────┘ └─────────────┘ └──────────────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    Fused Strategic Intel
                           │
                    Agent Decision Making
```

### 7.5 Alternative GraphRAG Implementations

| Project | Stars | Approach |
|---------|-------|----------|
| **microsoft/graphrag** | 31.5k | Full pipeline: extract → cluster → summarize → query |
| **nano-graphrag** | ~5k | Lightweight GraphRAG implementation |
| **LightRAG** | ~10k | Lightweight alternative focusing on speed |
| **fast-graphrag** | ~2k | Optimized for performance |

Microsoft's GraphRAG is the most mature and well-documented. Start there and optimize later if needed.

---

## 8. Recommendations

## 9. Collective Intelligence Pattern (Cross-Source)

Across the surveyed systems, the strongest transferable pattern for this
repository is a shared-memory architecture rather than a single best agent.

- Use a common symbolic substrate (KG/ontology) that all agents can query.
- Keep base facts immutable and append learned evidence as provenance-tagged
    extension objects.
- Train heterogeneous agents (heuristic, LLM, world-model, active-inference)
    against the same environment and aggregate their trajectory evidence.
- Re-inject aggregated evidence into subsequent training/inference cycles.

This pattern yields a collective-intelligence effect: each generation benefits
from strategic evidence discovered by prior generations, without sacrificing
auditability or reproducibility.

### 8.1 What to Build vs. What to Reuse

| Component | Recommendation |
|-----------|---------------|
| **Card Data Layer** | **USE Scrython** — `pip install scrython`, bulk download oracle cards + rulings. MIT license. No need to build this. |
| **Game State Model** | **REFERENCE mtg-python-engine + mtg-player**, but **build fresh** with Pydantic V2. Their data models inform ours but aren't modern enough. |
| **Stack** | **REFERENCE mtg-python-engine** (Play objects pattern) and **Argentum Engine** (immutable state), **build fresh** with our LangGraph integration. |
| **Turn/Phase Structure** | **ADAPT from mtg-player**: their `rules_engine.py` + `game_state.py` are a good starting point. MIT licensed. |
| **Combat System** | **REFERENCE open-mtg** for CR-compliant damage assignment ordering, **build fresh** with our priority system. |
| **Mana System** | **BUILD FRESH** — none of the Python implementations handle the full complexity well. Reference Forge's Java code for correctness. |
| **Rules Engine** | **BUILD FRESH**, using **Forge as the behavioral reference** for correctness. Our LLM-based approach fundamentally differs from hard-coded rules. |
| **Legal Action Generation** | **ADAPT open-mtg pattern** (`get_moves()` → index list). Extend to support priority and instant-speed. |
| **Agent Architecture** | **ADAPT from mtg-player**: LLM + tool-calling pattern is exactly what we need. Add our Active Inference layer on top. |
| **Knowledge Graph** | **BUILD with NetworkX** (custom KG) + **USE Microsoft GraphRAG** (unstructured reasoning) + **PyTorch Geometric** (GNN embeddings). |
| **Judge Agent** | **REFERENCE MTG-Judge** for RAG-over-rules pattern. Build our own with better prompt engineering. |
| **Scryfall Integration** | **USE Scrython** directly. |
| **Combo Database** | **SCRAPE Commander Spellbook API** → populate KG combo nodes. |

### 8.2 Priority Dependencies to Install

```bash
pip install scrython           # Scryfall API wrapper
pip install graphrag           # Microsoft GraphRAG
pip install networkx           # Custom knowledge graph
pip install langgraph          # Agent orchestration
pip install pydantic           # Data models
pip install torch torch-geometric  # GNN embeddings (later phase)
```

### 8.3 Repos to Star/Watch

1. **wingedsheep/argentum-engine** — most active, modern architecture, LLM AI integration
2. **Card-Forge/forge** — gold standard rules reference
3. **microsoft/graphrag** — our GraphRAG foundation
4. **theRealMarkCastillo/mtg-player** — closest Python predecessor to our project
5. **hlynurd/open-mtg** — MCTS AI reference for game tree search
6. **NandaScott/Scrython** — our card data library

### 8.4 License Compatibility

Our project (presumably MIT or Apache-2.0):
- ✅ **MIT**: open-mtg, mtg-python-engine, mtg-player, Scrython, GraphRAG — all compatible
- ⚠️ **GPL-3.0**: Forge, Magarena — reference only, do NOT copy code directly (GPL is viral)
- ❓ **Unspecified**: Argentum Engine — reference only until license is clarified
