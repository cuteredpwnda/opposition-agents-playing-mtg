# Graph Report - .  (2026-04-23)

## Corpus Check
- 170 files · ~142,407 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2253 nodes · 10137 edges · 36 communities detected
- Extraction: 26% EXTRACTED · 74% INFERRED · 0% AMBIGUOUS · INFERRED: 7516 edges (avg confidence: 0.57)
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

## God Nodes (most connected - your core abstractions)
1. `GameState` - 635 edges
2. `Zone` - 514 edges
3. `CardInstance` - 502 edges
4. `PlayerState` - 338 edges
5. `Strategy` - 301 edges
6. `MTGKnowledgeGraph` - 270 edges
7. `AgentGamePlayer` - 261 edges
8. `GameSimulator` - 210 edges
9. `GameResult` - 194 edges
10. `Action` - 185 edges

## Surprising Connections (you probably didn't know these)
- `Run n10s SHACL validation against the knowledge graph.  Usage: python -m scrip` --uses--> `N10sSetup`  [INFERRED]
  scripts\validate_kg.py → src\knowledge\n10s_setup.py
- `Extract likely card names from a situation description.          Heuristic: ca` --uses--> `GameState`  [INFERRED]
  src\judge\judge_agent.py → src\engine\game_state.py
- `test_create_mock_deck_size()` --calls--> `create_mock_deck()`  [INFERRED]
  tests\test_benchmark.py → src\training\deck_utils.py
- `test_stable_worldmodel_adapter_importable()` --calls--> `StableWorldModelAdapter`  [INFERRED]
  tests\test_world_model_pytest.py → src\world_model\stable_worldmodel_adapter.py
- `create_player_with_deck()` --calls--> `PlayerState`  [INFERRED]
  examples\load_real_decklists.py → src\engine\game_state.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.0
Nodes (283): ActiveInferenceAgent, Benchmark comparator for agent strategy evaluation., Compare multiple agent classes in a round-robin set of matches., Run the benchmark and return a results summary., Train a 2-layer GraphSAGE model on the card graph., train_graphsage(), CardEmbeddingConfig, CardEmbeddingModel (+275 more)

### Community 1 - "Community 1"
Cohesion: 0.0
Nodes (213): Strategy, AgentGamePlayer, GameResult, GameSimulator, OllamaConnector, build_izzet_control_deck(), build_mono_green_aggro_deck(), build_ur_aggro_deck() (+205 more)

### Community 2 - "Community 2"
Cohesion: 0.0
Nodes (156): ActiveInferenceModule, Agent that uses active inference + opponent model for decision making., OpponentBelief, Active inference module — Bayesian belief tracking + expected free energy.  Im, Decide whether to hold mana open for reactive plays.          Returns recommen, Extract mana colors from a card's mana cost string., How much will this action reduce uncertainty?          High for: Thoughtseize,, How much does this action advance the win condition?          High for: combo (+148 more)

### Community 3 - "Community 3"
Cohesion: 0.0
Nodes (141): BenchmarkSuite, main(), run_default_benchmark(), CardCache, Local card data cache — SQLite-backed to avoid repeated Scryfall calls., SQLite cache of Scryfall card data for fast local lookup., Register a card in the database., Retrieve card data by name. (+133 more)

### Community 4 - "Community 4"
Cohesion: 0.0
Nodes (143): extract_cost(), Activated abilities system — handles permanent abilities controlled by players., Get all activated abilities on a card that a player can currently use., Check if player can pay an ability's cost.          Cost includes:     - Mana, Extract individual mana symbols from cost text.          E.g., "{2}{U}{B}" → [, Resolve an activated ability's effect.          Effects supported:     - Add, Extract activated abilities from a card's oracle text.          This finds abi, Extract just the cost part from '{cost}: effect' pattern.          E.g., "{T}: (+135 more)

### Community 5 - "Community 5"
Cohesion: 0.0
Nodes (85): Evaluate the current board position using KG queries.                  Returns, build_features(), export_graph(), main(), Build GNN embeddings and write back to Neo4j vector index.  Pipeline: 1. Expo, Write embeddings back to Neo4j card nodes., Export graph from Neo4j → train GraphSAGE → write embeddings back., Export card graph from Neo4j into node features + edge list. (+77 more)

### Community 6 - "Community 6"
Cohesion: 0.0
Nodes (120): MTG Agent that chooses actions by minimizing expected free energy., AgentMemory, MTGAgent, Abstract agent interface for MTG gameplay.  Every agent must implement `decide, Per-game memory for an agent: observations, key moments, plans., Base class for all MTG-playing agents.      Subclasses must implement `decide_, Choose an action from the legal actions given the game state., Observe an action taken (by any player). Override for belief updates. (+112 more)

### Community 7 - "Community 7"
Cohesion: 0.0
Nodes (84): Enum, GameCoordinator, GamePhaseAction, Game Execution Bridge for Agent Strategic Play.  This module connects agents t, Decide what to play during main phase.                  Uses strategist to eva, Decide which creatures to attack with.                  Args:             gam, Decide if/how to block an attacker.                  Args:             attack, Decide if agent wants to respond to opposing spell.                  Args: (+76 more)

### Community 8 - "Community 8"
Cohesion: 0.0
Nodes (98): LangGraph state machine for MTG game flow.  Encodes the full MTG turn structur, Untap all permanents controlled by the active player., Upkeep step — trigger upkeep abilities and check SBAs., Active player draws a card (skip on turn 1 in 2-player)., Discard to hand size, remove damage, advance turn., Set up priority passing — APNAP order., Resolve the top item on the stack., Placeholder — the actual decision is made by the game_runner     which calls th (+90 more)

### Community 9 - "Community 9"
Cohesion: 0.0
Nodes (59): AgentStrategist, PlayEvaluation, PlayRecommendation, Agent Strategies and Decision-Making Using Neo4j Knowledge Graph.  This module, Determine if we should block an attacker.                  Args:, Evaluate if we should play a card from hand.                  Args:, Get the next recommended action for this player.                  Evaluates:, Get a text description of this agent's strategy. (+51 more)

### Community 10 - "Community 10"
Cohesion: 0.0
Nodes (45): can_pay_ability_cost(), extract_mana_cost(), get_legal_activated_abilities(), is_mana_ability(), parse_abilities(), Mark a specific card instance as known to a player., Check if a player knows a given card instance., _ability_affects_card() (+37 more)

### Community 11 - "Community 11"
Cohesion: 0.0
Nodes (31): create_game_with_full_support(), test_card_without_triggers(), test_check_attack_triggers(), test_check_cast_triggers(), test_check_death_triggers(), test_multiple_damage_values(), test_multiple_trigger_types(), test_parse_attack_triggers() (+23 more)

### Community 12 - "Community 12"
Cohesion: 0.0
Nodes (14): main(), pull_checkpoints(), Push and run the training pipeline on a remote machine.  Syncs the project to, Pull trained model checkpoints from the remote machine., Run a command and stream output., Rsync the project to the remote machine., Execute a command on the remote machine via SSH., Start the Docker-based training stack on the remote machine. (+6 more)

### Community 13 - "Community 13"
Cohesion: 0.0
Nodes (7): MDNHead, Compute log-probability of a target under the mixture.          Args:, Build LSTM-based sequence model., Build Transformer-based sequence model., Compute loss over a full game trajectory.          Args:             z_sequen, Mixture Density Network output head.      Outputs parameters of a Gaussian mix, Args:             x: (batch, input_dim) — hidden state from RNN/Transformer

### Community 14 - "Community 14"
Cohesion: 0.0
Nodes (8): apply_continuous_effects(), ContinuousEffectRegistry, Continuous effects — static effects modifying game rules (layer system).  Refe, Registry of active continuous effects on the board., Register a continuous effect., Remove all effects from a source card., Get all effects for a specific layer., Apply all continuous effects using the layer system.          Layers (in order

### Community 15 - "Community 15"
Cohesion: 0.0
Nodes (8): apply_replacement_effects(), Replacement effects — "instead" effects that modify game events before they happ, Registry of active replacement effects on the board., Register a replacement effect., Unregister a replacement effect., Get all replacement effects for an event type., Apply replacement effects that modify an event.          For example:     - ", ReplacementEffectRegistry

### Community 16 - "Community 16"
Cohesion: 0.0
Nodes (12): advance_priority(), all_players_passed(), get_priority_order(), priority_action_result(), Priority passing logic — APNAP order for multiplayer.  In MTG, after any game, Return player IDs in APNAP order starting from active player., Ensure priority player is valid when players are removed mid-game., Move priority to the next player in APNAP order. (+4 more)

### Community 17 - "Community 17"
Cohesion: 0.0
Nodes (6): Get text summary of player's hand.                  Args:             game: C, Get text summary of board state.                  Args:             game: Cur, Use LLM to decide what to play in main phase.                  Args:, Use LLM to decide combat strategy.                  Args:             game: C, Parse LLM response into structured decision.                  Args:, Generate text from Ollama.                  Args:             prompt: Input p

### Community 18 - "Community 18"
Cohesion: 0.0
Nodes (11): apply_deathtouch(), apply_lifelink(), can_block_with(), has_keyword(), menace_can_be_blocked(), Keyword ability implementations — Flying, Haste, Lifelink, Menace, etc.  This, Check if a card has a specific keyword., Check if defender can block attacker (considering evasion keywords). (+3 more)

### Community 19 - "Community 19"
Cohesion: 0.0
Nodes (3): Encodes a variable-length set of card embeddings into a fixed-size vector., Args:             cards: (batch, max_cards, card_dim)             mask: (batch, SetEncoder

### Community 20 - "Community 20"
Cohesion: 0.0
Nodes (1): _safe_int_score()

### Community 21 - "Community 21"
Cohesion: 0.0
Nodes (3): create_minimal_card_data(), Local card database — zero API calls during gameplay.  All card data is resolv, Create minimal card data dict.          Used for test decks and simple card cr

### Community 22 - "Community 22"
Cohesion: 0.0
Nodes (3): build_rules_vectorstore(), Comprehensive Rules vectorstore — RAG over the full MTG rules document.  Refer, Build a FAISS vectorstore over the MTG Comprehensive Rules.      Returns a FAI

### Community 23 - "Community 23"
Cohesion: 0.0
Nodes (1): TestPhases

### Community 24 - "Community 24"
Cohesion: 0.0
Nodes (1): Reward functions for RL training.  Combines terminal (win/loss) with intermedi

### Community 25 - "Community 25"
Cohesion: 0.0
Nodes (0): 

### Community 26 - "Community 26"
Cohesion: 0.0
Nodes (0): 

### Community 27 - "Community 27"
Cohesion: 0.0
Nodes (0): 

### Community 28 - "Community 28"
Cohesion: 0.0
Nodes (1): Check if this card is a token (lacks a card set, o/w has token=true).

### Community 29 - "Community 29"
Cohesion: 0.0
Nodes (1): Extract keywords from oracle text and keywords field.

### Community 30 - "Community 30"
Cohesion: 0.0
Nodes (1): Convert a Scryfall card JSON to Neo4j node properties.

### Community 31 - "Community 31"
Cohesion: 0.0
Nodes (1): Convert a relative or absolute path to a file:// URI for Neo4j import.

### Community 32 - "Community 32"
Cohesion: 0.0
Nodes (0): 

### Community 33 - "Community 33"
Cohesion: 0.0
Nodes (0): 

### Community 34 - "Community 34"
Cohesion: 0.0
Nodes (0): 

### Community 35 - "Community 35"
Cohesion: 0.0
Nodes (0): 

## Knowledge Gaps
- **172 isolated node(s):** `Simple MTG game runner for testing.  Usage:   python main.py`, `Push and run the training pipeline on a remote machine.  Syncs the project to`, `Run a command and stream output.`, `Rsync the project to the remote machine.`, `Execute a command on the remote machine via SSH.` (+167 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 24`** (2 nodes): `Reward functions for RL training.  Combines terminal (win/loss) with intermedi`, `rewards.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `full_project_overview.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `kg_jepa_world_model.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `kg_jepa_world_model_old.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Check if this card is a token (lacks a card set, o/w has token=true).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Extract keywords from oracle text and keywords field.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Convert a Scryfall card JSON to Neo4j node properties.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Convert a relative or absolute path to a file:// URI for Neo4j import.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.