# Graph Report - .  (2026-04-29)

## Corpus Check
- Large corpus: 382 files · ~733,158 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder, or use --no-semantic to run AST-only.

## Summary
- 3573 nodes · 14627 edges · 52 communities detected
- Extraction: 31% EXTRACTED · 69% INFERRED · 0% AMBIGUOUS · INFERRED: 10050 edges (avg confidence: 0.58)
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
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]

## God Nodes (most connected - your core abstractions)
1. `GameState` - 799 edges
2. `CardInstance` - 660 edges
3. `Zone` - 647 edges
4. `PlayerState` - 433 edges
5. `Strategy` - 325 edges
6. `MTGKnowledgeGraph` - 323 edges
7. `AgentGamePlayer` - 287 edges
8. `Action` - 240 edges
9. `GameSimulator` - 236 edges
10. `GameResult` - 199 edges

## Surprising Connections (you probably didn't know these)
- `Ablation sweep over world-model sizes and LLM sizes.  Modes ----- - ``--swee` --uses--> `WorldModel`  [INFERRED]
  scripts\ablation_sweep.py → src\world_model\world_model.py
- `One row of the ablation grid.` --uses--> `WorldModel`  [INFERRED]
  scripts\ablation_sweep.py → src\world_model\world_model.py
- `Stub: probe Ollama availability and record metadata.      The full version ins` --uses--> `WorldModel`  [INFERRED]
  scripts\ablation_sweep.py → src\world_model\world_model.py
- `# NOTE: Real round-robin via BenchmarkSuite goes here once the` --uses--> `WorldModel`  [INFERRED]
  scripts\ablation_sweep.py → src\world_model\world_model.py
- `Run n10s SHACL validation against the knowledge graph.  Usage: python -m scrip` --uses--> `N10sSetup`  [INFERRED]
  scripts\validate_kg.py → src\knowledge\n10s_setup.py

## Hyperedges (group relationships)
- **All Agent Implementations** —  [EXTRACTED]
- **World Model Components** —  [EXTRACTED]
- **Core Engine Modules** —  [EXTRACTED]
- **External Data Sources** —  [EXTRACTED]
- **Krenko Mob Boss Deck Family** —  [EXTRACTED]
- **Commander Bracket 4 Gate Lists** —  [EXTRACTED]
- **EDH Deck Corpus** —  [EXTRACTED]
- **Modern Deck Corpus** —  [EXTRACTED]
- **Shared Black Removal Package** —  [INFERRED]
- **Standard Deck Gauntlet** —  [INFERRED]
- **World Model Research Lineage** —  [INFERRED]
- **V+M+C World Model Framework** —  [EXTRACTED]
- **Knowledge Graph Build Pipeline (Stages 2-3)** —  [EXTRACTED]
- **Sphinx API Documentation Suite** —  [EXTRACTED]
- **** —  [EXTRACTED]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.0
Nodes (706): can_pay_ability_cost(), extract_cost(), extract_mana_cost(), get_legal_activated_abilities(), is_mana_ability(), parse_abilities(), Activated abilities system — handles permanent abilities controlled by players., Get all activated abilities on a card that a player can currently use. (+698 more)

### Community 1 - "Community 1"
Cohesion: 0.0
Nodes (405): AgentStats, Play one game where model_first is seated as player1 (goes first)., evaluate_world_model(), Stub: build the model, count parameters, do a smoke forward pass.      A full, ActiveInferenceAgent, Agent that uses active inference + opponent model for decision making., MTG Agent that chooses actions by minimizing expected free energy., MTGAgent (+397 more)

### Community 2 - "Community 2"
Cohesion: 0.0
Nodes (288): Decide whether to hold mana open for reactive plays.          Returns recommen, Bayesian update of beliefs given a new game observation.          Observations, categorise(), main(), Return frequency counts per category., render_report(), Retrieve card data by name., Resolve a raw deck into full card data.                  Input: List of dicts (+280 more)

### Community 3 - "Community 3"
Cohesion: 0.0
Nodes (220): Strategy, play_simple_game(), Simple game test without LLM to check hand logging., Play a game without LLM to test hand logging., AgentGamePlayer, Log an event.                  Args:             level: Log severity level, GameResult, GameSimulator (+212 more)

### Community 4 - "Community 4"
Cohesion: 0.0
Nodes (169): ActiveInferenceModule, OpponentBelief, Active inference module — Bayesian belief tracking + expected free energy.  Im, Extract mana colors from a card's mana cost string., How much will this action reduce uncertainty?          High for: Thoughtseize,, How much does this action advance the win condition?          High for: combo, Probabilistic belief state about one opponent., Maintains probabilistic beliefs about hidden game information     and selects a (+161 more)

### Community 5 - "Community 5"
Cohesion: 0.0
Nodes (128): Learn embeddings from gameplay data (Word2Vec-style).          Cards that co-o, Color, Mana colors — mirrors mtg:Color in the ontology., HeuristicAgent, JsonlActionTrace, _summarize_action(), _summarize_state(), EnrichmentConfig (+120 more)

### Community 6 - "Community 6"
Cohesion: 0.0
Nodes (176): src/agents/active_inference_agent.py — ActiveInferenceAgent, Active Inference Framework, ActiveInferenceAgent, HeuristicAgent, HierarchicalAgent, KGHeuristicAgent, LLMAgent, LLMFusionAgent (+168 more)

### Community 7 - "Community 7"
Cohesion: 0.0
Nodes (93): fetch_one(), main(), to_text(), Check whether every card in a decklist is present in the offline DB.  Usage:, main(), report_one(), BracketReport, can_play_together() (+85 more)

### Community 8 - "Community 8"
Cohesion: 0.0
Nodes (94): Aggro Archetype, Control Archetype, Convoke / Token Swarm Archetype, Tempo / Bounce-Loop Archetype, Toxic / Poison Archetype, Archetype: Aggro, Archetype: Artifact Combo, Archetype: Big Mana / Ramp (+86 more)

### Community 9 - "Community 9"
Cohesion: 0.0
Nodes (71): _agent_stats_from(), _build_simple_deck(), GameRecord, main(), play_one_game(), _print_game_diagnostics(), _print_summary(), resolve_model_name() (+63 more)

### Community 10 - "Community 10"
Cohesion: 0.0
Nodes (50): active_dungeon(), add_command_zone_object(), become_monarch(), command_zone_objects_for(), create_emblem(), emblems_for(), Command-zone operational helpers.  Covers the non-card command-zone state that, Advance ``controller_id`` one room into a dungeon (CR 309.4).      If the play (+42 more)

### Community 11 - "Community 11"
Cohesion: 0.0
Nodes (38): Return True to KEEP the current opening hand, False to mulligan.          Defa, _describe_hand(), _parse_action_index(), HandStats, Strategy-aware mulligan policy.  Pure functions used by ``MTGAgent.decide_mull, Choose ``n`` cards from ``hand`` to put on the bottom of the library.      Str, Cheap summary of an opening hand used by every heuristic., Return True if the agent should keep this opening hand.      Always keeps once (+30 more)

### Community 12 - "Community 12"
Cohesion: 0.0
Nodes (39): create_game_with_full_support(), test_card_without_triggers(), test_check_attack_triggers(), test_check_cast_triggers(), test_check_death_triggers(), test_multiple_damage_values(), test_multiple_trigger_types(), test_parse_attack_triggers() (+31 more)

### Community 13 - "Community 13"
Cohesion: 0.0
Nodes (20): classify_feature(), classify_features(), combo_display_name(), display_name(), Outcome, Categorize Spellbook ``feature`` strings into structured outcome buckets.  Spe, Classify a sequence of features, deduping on (category, magnitude)., Build a short, human-readable combo name from its component cards.      Exampl (+12 more)

### Community 14 - "Community 14"
Cohesion: 0.0
Nodes (18): apply_damage_to_player(), apply_lifegain(), apply_replacements(), install_replacements_for(), _parse_damage_prevention(), _parse_death_to_exile(), _parse_lifegain_double(), _registry() (+10 more)

### Community 15 - "Community 15"
Cohesion: 0.0
Nodes (15): _card(), _fill_to_60(), get_archetype(), izzet_burn(), _land(), list_archetypes(), mono_black_midrange(), mono_blue_control() (+7 more)

### Community 16 - "Community 16"
Cohesion: 0.0
Nodes (14): main(), pull_checkpoints(), Push and run the training pipeline on a remote machine.  Syncs the project to, Pull trained model checkpoints from the remote machine., Run a command and stream output., Rsync the project to the remote machine., Execute a command on the remote machine via SSH., Start the Docker-based training stack on the remote machine. (+6 more)

### Community 17 - "Community 17"
Cohesion: 0.0
Nodes (3): CardCache, Local card data cache — SQLite-backed to avoid repeated Scryfall calls., SQLite cache of Scryfall card data for fast local lookup.

### Community 18 - "Community 18"
Cohesion: 0.0
Nodes (6): MDNHead, Compute log-probability of a target under the mixture.          Args:, Build LSTM-based sequence model., Build Transformer-based sequence model., Mixture Density Network output head.      Outputs parameters of a Gaussian mix, Args:             x: (batch, input_dim) — hidden state from RNN/Transformer

### Community 19 - "Community 19"
Cohesion: 0.0
Nodes (12): Midrange Archetype, Cut Down, Duress, Go for the Throat, Preacher of the Schism, Sheoldred, the Apocalypse, Mono-Black, Black-Green (Golgari) (+4 more)

### Community 20 - "Community 20"
Cohesion: 0.0
Nodes (9): fetch_rules(), _filename_from_url(), _find_rules_url(), _is_allowed(), main(), Make ``data/rules/latest.txt`` point at the most recent download.      Uses a, Extract the most recent ``MagicCompRules*.txt`` href from the rules page., Fetch the latest CR text file. Returns the downloaded path. (+1 more)

### Community 21 - "Community 21"
Cohesion: 0.0
Nodes (9): fetch_all_combos(), fetch_combos_merged(), fetch_edhrec_combos(), _polite_get(), Combo data fetchers.  Primary source: Commander Spellbook (https://commandersp, Fetch the global EDHREC combos page.      EDHREC exposes a JSON mirror of its, Convenience: fetch both sources and write a merged file.      Spellbook entrie, GET with full-jitter exponential backoff and Retry-After handling.      Return (+1 more)

### Community 22 - "Community 22"
Cohesion: 0.0
Nodes (4): AgentMemory, Abstract agent interface for MTG gameplay.  Every agent must implement `decide, Reset agent state for a new game., Per-game memory for an agent: observations, key moments, plans.

### Community 23 - "Community 23"
Cohesion: 0.0
Nodes (8): ward_cost(), _card(), Ward cost parsing (CR 702.21)., test_ward_absent_no_false_positive(), test_ward_bare(), test_ward_coloured_hybrid(), test_ward_numeric(), test_ward_with_other_text()

### Community 24 - "Community 24"
Cohesion: 0.0
Nodes (5): BaseSettings, Environment config — loaded from .env or environment variables., Minimal fallback BaseSettings for environments without pydantic., Application settings resolved from environment / .env file., Settings

### Community 25 - "Community 25"
Cohesion: 0.0
Nodes (6): _echo_trigger(), _make_echo_creature(), _state_with_creature(), test_pays_echo_for_card_with_activated_ability_text(), test_pays_echo_for_high_power_creature(), test_skips_echo_for_vanilla_low_power()

### Community 26 - "Community 26"
Cohesion: 0.0
Nodes (3): Encodes a variable-length set of card embeddings into a fixed-size vector., Args:             cards: (batch, max_cards, card_dim)             mask: (batch, SetEncoder

### Community 27 - "Community 27"
Cohesion: 0.0
Nodes (3): create_minimal_card_data(), Local card database — zero API calls during gameplay.  All card data is resolv, Create minimal card data dict.          Used for test decks and simple card cr

### Community 28 - "Community 28"
Cohesion: 0.0
Nodes (3): build_rules_vectorstore(), Comprehensive Rules vectorstore — RAG over the full MTG rules document.  The C, Build a FAISS vectorstore over the MTG Comprehensive Rules.      Returns a FAI

### Community 29 - "Community 29"
Cohesion: 0.0
Nodes (4): Ramp / Big-Mana Archetype, Atraxa, Grand Unifier, Five-Color (Domain), Standard Domain Ramp

### Community 30 - "Community 30"
Cohesion: 0.0
Nodes (4): Forge (Card-Forge/forge) — Java MTG Engine Reference, mtg-python-engine (wanqizhu) — External Python Engine, open-mtg (hlynurd/open-mtg) — External Python Engine, RESEARCH.md — MTG Engine & GraphRAG Survey

### Community 31 - "Community 31"
Cohesion: 0.0
Nodes (0): 

### Community 32 - "Community 32"
Cohesion: 0.0
Nodes (1): Thin CLI wrapper that delegates to :mod:`src.training.neural_reasoner_trainer`.

### Community 33 - "Community 33"
Cohesion: 0.0
Nodes (0): 

### Community 34 - "Community 34"
Cohesion: 0.0
Nodes (0): 

### Community 35 - "Community 35"
Cohesion: 0.0
Nodes (0): 

### Community 36 - "Community 36"
Cohesion: 0.0
Nodes (0): 

### Community 37 - "Community 37"
Cohesion: 0.0
Nodes (0): 

### Community 38 - "Community 38"
Cohesion: 0.0
Nodes (0): 

### Community 39 - "Community 39"
Cohesion: 0.0
Nodes (0): 

### Community 40 - "Community 40"
Cohesion: 0.0
Nodes (0): 

### Community 41 - "Community 41"
Cohesion: 0.0
Nodes (0): 

### Community 42 - "Community 42"
Cohesion: 0.0
Nodes (1): True if the model is within the <= 7 B effective-size budget.

### Community 43 - "Community 43"
Cohesion: 0.0
Nodes (1): Whether this card instance is a token (CR 111.1).          Tokens must be flag

### Community 44 - "Community 44"
Cohesion: 0.0
Nodes (1): Extract keywords from oracle text and keywords field.

### Community 45 - "Community 45"
Cohesion: 0.0
Nodes (1): Convert a Scryfall card JSON to Neo4j node properties.

### Community 46 - "Community 46"
Cohesion: 0.0
Nodes (1): Convert a relative or absolute path to a file:// URI for Neo4j import.

### Community 47 - "Community 47"
Cohesion: 0.0
Nodes (0): 

### Community 48 - "Community 48"
Cohesion: 0.0
Nodes (0): 

### Community 49 - "Community 49"
Cohesion: 0.0
Nodes (0): 

### Community 50 - "Community 50"
Cohesion: 0.0
Nodes (0): 

### Community 51 - "Community 51"
Cohesion: 0.0
Nodes (1): src/engine/continuous_effects.py

## Knowledge Gaps
- **234 isolated node(s):** `Simple MTG game runner for testing.  Usage:   python main.py`, `Walk through the KG cookbook (paper/reports/04) live.  Runs the headline queri`, `Return frequency counts per category.`, `Look up the download URL + size for a bulk-data type.`, `Stream a large file to disk with a simple progress indicator.` (+229 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 31`** (2 nodes): `doctools.js`, `_ready()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (2 nodes): `train_neural_reasoner.py`, `Thin CLI wrapper that delegates to :mod:`src.training.neural_reasoner_trainer`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `conf.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `searchindex.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `documentation_options.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `language_data.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `agents_tech_report.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `full_project_overview.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `kg_jepa_world_model.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `kg_jepa_world_model_old.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `repo_for_girlfriend.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `True if the model is within the <= 7 B effective-size budget.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Whether this card instance is a token (CR 111.1).          Tokens must be flag`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Extract keywords from oracle text and keywords field.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Convert a Scryfall card JSON to Neo4j node properties.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Convert a relative or absolute path to a file:// URI for Neo4j import.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `src/engine/continuous_effects.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.