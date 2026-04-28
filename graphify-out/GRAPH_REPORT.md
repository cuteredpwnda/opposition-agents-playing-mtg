# Graph Report - src  (2026-04-28)

## Corpus Check
- 110 files · ~82,101 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1739 nodes · 6531 edges · 32 communities detected
- Extraction: 33% EXTRACTED · 67% INFERRED · 0% AMBIGUOUS · INFERRED: 4354 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]

## God Nodes (most connected - your core abstractions)
1. `GameState` - 432 edges
2. `Zone` - 320 edges
3. `MTGKnowledgeGraph` - 235 edges
4. `CardInstance` - 211 edges
5. `Action` - 167 edges
6. `ActionType` - 107 edges
7. `TrajectoryStore` - 101 edges
8. `RandomAgent` - 98 edges
9. `MTGAgent` - 95 edges
10. `RulesEngine` - 91 edges

## Surprising Connections (you probably didn't know these)
- `Combo detector — uses KG to detect available and near-miss combos.  Reference:` --uses--> `MTGKnowledgeGraph`  [INFERRED]
  src\agents\combo_detector.py → src\knowledge\knowledge_graph.py
- `A detected combo or near-combo.` --uses--> `MTGKnowledgeGraph`  [INFERRED]
  src\agents\combo_detector.py → src\knowledge\knowledge_graph.py
- `Uses the knowledge graph to detect combos from available cards.` --uses--> `MTGKnowledgeGraph`  [INFERRED]
  src\agents\combo_detector.py → src\knowledge\knowledge_graph.py
- `Find combos where ALL pieces are in hand + battlefield.` --uses--> `MTGKnowledgeGraph`  [INFERRED]
  src\agents\combo_detector.py → src\knowledge\knowledge_graph.py
- `Find combos missing exactly one piece.` --uses--> `MTGKnowledgeGraph`  [INFERRED]
  src\agents\combo_detector.py → src\knowledge\knowledge_graph.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.0
Nodes (312): Activated abilities system — handles permanent abilities controlled by players., Get all activated abilities on a card that a player can currently use., Check if player can pay an ability's cost.          Cost includes:     - Mana, Extract individual mana symbols from cost text.          E.g., "{2}{U}{B}" → [, Extract activated abilities from a card's oracle text.          This finds abi, Resolve an activated ability's effect.          Effects supported:     - Add, Extract just the cost part from '{cost}: effect' pattern.          E.g., "{T}:, Check if ability is a mana ability (can be used anytime).      Mana abilities (+304 more)

### Community 1 - "Community 1"
Cohesion: 0.0
Nodes (186): ActiveInferenceAgent, CardEmbeddingConfig, CardEmbeddingModel, Card embedding model — maps card names to dense vectors.  Provides multiple st, Learn embeddings from gameplay data (Word2Vec-style).          Cards that co-o, Save embeddings to disk., Load embeddings from disk., Lazy-load the sentence transformer model. (+178 more)

### Community 2 - "Community 2"
Cohesion: 0.0
Nodes (141): can_pay_ability_cost(), extract_cost(), extract_mana_cost(), get_legal_activated_abilities(), is_mana_ability(), parse_abilities(), resolve_ability(), _apply_lifelink() (+133 more)

### Community 3 - "Community 3"
Cohesion: 0.0
Nodes (73): Agent that uses active inference + opponent model for decision making., MTG Agent that chooses actions by minimizing expected free energy., list_archetypes(), AgentMemory, MTGAgent, Abstract agent interface for MTG gameplay.  Every agent must implement `decide, Per-game memory for an agent: observations, key moments, plans., Observe an action taken (by any player). Override for belief updates. (+65 more)

### Community 4 - "Community 4"
Cohesion: 0.0
Nodes (75): LogLevel, Comprehensive game logging for MTG agents.  Logs all game events with timestam, Color, Mana colors — mirrors mtg:Color in the ontology., EnrichmentConfig, EnrichmentReport, Knowledge Graph auto-enrichment from self-play trajectories.  Analyzes game tr, Thresholds for KG enrichment proposals. (+67 more)

### Community 5 - "Community 5"
Cohesion: 0.0
Nodes (77): CardDatabase, create_minimal_card_data(), Local card database — zero API calls during gameplay.  All card data is resolv, Game-session-local card database.          Holds pre-loaded card data for all, Register a card in the database., Retrieve card data by name., Check if card is in database., Resolve a raw deck into full card data.                  Input: List of dicts (+69 more)

### Community 6 - "Community 6"
Cohesion: 0.0
Nodes (23): ActiveInferenceModule, Write discovered synergies to the knowledge graph., Update per-card win rate statistics in the KG., Find cards that synergize with a given card., Find cards that counter/answer a given card., Given cards observed, rank likely archetypes., Get the signature/staple cards for an archetype., APOC subgraph expansion — get strategic context around a card.          Return (+15 more)

### Community 7 - "Community 7"
Cohesion: 0.0
Nodes (32): MTGGraphRAG, GraphRAG-style retrieval implemented natively on Neo4j.  Instead of running a, Neo4j native vector index search — find similar cards., Lucene full-text search over card names and oracle text., Get the strategic cluster a card belongs to., Archetype matchup analysis using graph traversal., Detect available combos from hand + battlefield., Run Louvain community detection and write clusters to nodes.          Returns (+24 more)

### Community 8 - "Community 8"
Cohesion: 0.0
Nodes (25): Log an event.                  Args:             level: Log severity level, Pretty-print log entry., GameResult, GameSimulator, GameRecord, Tournament System & Strategy Analytics.  Runs comprehensive tournaments betwee, Run round-robin tournament (each strategy plays each other)., Play best-of series between two strategies.                  Args: (+17 more)

### Community 9 - "Community 9"
Cohesion: 0.0
Nodes (34): BracketReport, can_play_together(), check_legality(), classify_bracket(), _color_identity(), _flat(), _has_partner(), _is_basic_land() (+26 more)

### Community 10 - "Community 10"
Cohesion: 0.0
Nodes (32): active_dungeon(), add_command_zone_object(), become_monarch(), command_zone_objects_for(), create_emblem(), emblems_for(), Command-zone operational helpers.  Covers the non-card command-zone state that, Advance ``controller_id`` one room into a dungeon (CR 309.4).      If the play (+24 more)

### Community 11 - "Community 11"
Cohesion: 0.0
Nodes (11): Adapter for a lightweight Schmidhuber-style world model path., Simple wrapper implementing same interface used by WorldModelAgent., SchmidhuberWorldModelAdapter, Adapter layer to integrate galilai-group/stable-worldmodel (LeWM) into this repo, Adapter wrapper around stable_worldmodel API., Train stable-worldmodel on HDF5 trajectories., StableWorldModelAdapter, Training pipeline for Schmidhuber-style world model (forward dynamics). (+3 more)

### Community 12 - "Community 12"
Cohesion: 0.0
Nodes (14): load(), load_default(), LocalRulingsCache, Local cache for Scryfall rulings + errata.  Reads `data/scryfall/rulings.json`, In-memory rulings cache keyed by oracle_id, with name index., Return all rulings for a card (case-insensitive name match)., _extract_card_names(), JudgeAgent (+6 more)

### Community 13 - "Community 13"
Cohesion: 0.0
Nodes (10): apply_replacement_effects(), Replacement effects — "instead" effects that modify game events before they happ, Representation of a replacement effect., Registry of active replacement effects on the board., Register a replacement effect., Unregister a replacement effect., Get all replacement effects for an event type., Apply replacement effects that modify an event.          For example:     - " (+2 more)

### Community 14 - "Community 14"
Cohesion: 0.0
Nodes (7): Async Scryfall API client with rate limiting.  For most use cases, prefer the, Async client for the Scryfall REST API with rate limiting., GET /cards/named?exact={name}, Fetch rulings for a card by resolving its Scryfall ID first., GET /cards/search?q={query} using Scryfall search syntax., Get download URI for a Scryfall bulk data type., ScryfallClient

### Community 15 - "Community 15"
Cohesion: 0.0
Nodes (12): A static ability that continuously modifies game state (not activated or trigger, StaticAbility, _ability_affects_card(), get_effective_power_toughness(), get_static_abilities_affecting(), has_keyword(), _parse_int(), _parse_scope() (+4 more)

### Community 16 - "Community 16"
Cohesion: 0.0
Nodes (7): MDNHead, Sample from the Gaussian mixture.          Args:             pi: (batch, K) —, Compute log-probability of a target under the mixture.          Args:, Build LSTM-based sequence model., Build Transformer-based sequence model., Mixture Density Network output head.      Outputs parameters of a Gaussian mix, Args:             x: (batch, input_dim) — hidden state from RNN/Transformer

### Community 17 - "Community 17"
Cohesion: 0.0
Nodes (12): _card(), _fill_to_60(), get_archetype(), izzet_burn(), _land(), mono_black_midrange(), mono_blue_control(), mono_green_ramp() (+4 more)

### Community 18 - "Community 18"
Cohesion: 0.0
Nodes (10): Return True to KEEP the current opening hand, False to mulligan.          Defa, HandStats, Strategy-aware mulligan policy.  Pure functions used by ``MTGAgent.decide_mull, Choose ``n`` cards from ``hand`` to put on the bottom of the library.      Str, Cheap summary of an opening hand used by every heuristic., Return True if the agent should keep this opening hand.      Always keeps once, select_bottom_cards(), should_keep() (+2 more)

### Community 19 - "Community 19"
Cohesion: 0.0
Nodes (3): CardCache, Local card data cache — SQLite-backed to avoid repeated Scryfall calls., SQLite cache of Scryfall card data for fast local lookup.

### Community 20 - "Community 20"
Cohesion: 0.0
Nodes (7): ComboDetector, ComboResult, Combo detector — uses KG to detect available and near-miss combos.  Reference:, A detected combo or near-combo., Uses the knowledge graph to detect combos from available cards., Find combos where ALL pieces are in hand + battlefield., Find combos missing exactly one piece.

### Community 21 - "Community 21"
Cohesion: 0.0
Nodes (7): get_llm_spec(), list_llms(), LLMSpec, Catalogue of LLM models used by ``OllamaAgent`` and ablation sweeps.  Every en, Return all registered LLMs that fit ``max_vram_gb`` (None = all)., Hardware-aware description of an Ollama-served model., Look up an LLM spec by its canonical ``name``.

### Community 22 - "Community 22"
Cohesion: 0.0
Nodes (5): BaseSettings, Environment config — loaded from .env or environment variables., Minimal fallback BaseSettings for environments without pydantic., Application settings resolved from environment / .env file., Settings

### Community 23 - "Community 23"
Cohesion: 0.0
Nodes (3): CardGraphEmbedder, GNN graph embedder — train on Neo4j graph, write embeddings back.  Reference:, Produces d-dimensional embeddings for each card node.      Trained on the grap

### Community 24 - "Community 24"
Cohesion: 0.0
Nodes (3): Encodes a variable-length set of card embeddings into a fixed-size vector., Args:             cards: (batch, max_cards, card_dim)             mask: (batch, SetEncoder

### Community 25 - "Community 25"
Cohesion: 0.0
Nodes (3): Encode tokenized game state features into latent space.          Args:, Fuse encoded components into the shared latent input space.          When *kg_, Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I).

### Community 26 - "Community 26"
Cohesion: 0.0
Nodes (3): build_rules_vectorstore(), Comprehensive Rules vectorstore — RAG over the full MTG rules document.  The C, Build a FAISS vectorstore over the MTG Comprehensive Rules.      Returns a FAI

### Community 27 - "Community 27"
Cohesion: 0.0
Nodes (1): True if the model is within the <= 7 B effective-size budget.

### Community 28 - "Community 28"
Cohesion: 0.0
Nodes (1): Whether this card instance is a token (CR 111.1).          Tokens must be flag

### Community 29 - "Community 29"
Cohesion: 0.0
Nodes (1): Extract keywords from oracle text and keywords field.

### Community 30 - "Community 30"
Cohesion: 0.0
Nodes (1): Convert a Scryfall card JSON to Neo4j node properties.

### Community 31 - "Community 31"
Cohesion: 0.0
Nodes (1): Convert a relative or absolute path to a file:// URI for Neo4j import.

## Knowledge Gaps
- **189 isolated node(s):** `Environment config — loaded from .env or environment variables.`, `Minimal fallback BaseSettings for environments without pydantic.`, `Application settings resolved from environment / .env file.`, `Catalogue of LLM models used by ``OllamaAgent`` and ablation sweeps.  Every en`, `Hardware-aware description of an Ollama-served model.` (+184 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 27`** (1 nodes): `True if the model is within the <= 7 B effective-size budget.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Whether this card instance is a token (CR 111.1).          Tokens must be flag`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Extract keywords from oracle text and keywords field.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Convert a Scryfall card JSON to Neo4j node properties.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Convert a relative or absolute path to a file:// URI for Neo4j import.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.