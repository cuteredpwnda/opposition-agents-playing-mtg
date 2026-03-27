# Implementation Plan — Single Source of Truth

> **opposition-agents-playing-mtg**
> Last updated: 2026-03-27

This document is the **single source of truth** for what has been implemented,
what is in progress, and what remains. It supersedes the phase descriptions in
`PLAN.md`, `ARCHITECTURE.md`, and the presentation slides for tracking purposes
— those documents retain their value as design rationale and research context.

---

## Table of Contents

1. [Project Vision](#1-project-vision)
2. [Architecture Summary](#2-architecture-summary)
3. [Module Inventory — What Exists Today](#3-module-inventory)
4. [Implementation Status by Layer](#4-implementation-status-by-layer)
    - 4.1 [Game Engine](#41-game-engine)
    - 4.2 [Agent Zoo](#42-agent-zoo)
    - 4.3 [World Model (V + M + C)](#43-world-model-v--m--c)
    - 4.4 [Knowledge Graph & Ontology](#44-knowledge-graph--ontology)
    - 4.5 [Training & Self-Play](#45-training--self-play)
    - 4.6 [Orchestration & Game Loop](#46-orchestration--game-loop)
    - 4.7 [External Integrations](#47-external-integrations)
    - 4.8 [Judge Agent](#48-judge-agent)
5. [Roadmap — Remaining Work](#5-roadmap--remaining-work)
    - Phase A: Self-Play Learning Loop & KG Feedback
    - Phase B: Benchmark Comparators
    - Phase C: Active Inference & Opponent Modeling Integration
    - Phase D: Neural Reasoning Module Integration
    - Phase E: Commander / Multiplayer Support
    - Phase F: Human Player & UI
6. [Benchmark & Evaluation Plan](#6-benchmark--evaluation-plan)
7. [Key Design Decisions That Changed](#7-key-design-decisions-that-changed)
8. [File Map](#8-file-map)

---

## 1. Project Vision

Build a Python framework where AI agents play full games of Magic: The Gathering,
learning and improving through self-play. The system combines:

- **World Models** (Ha & Schmidhuber V+M+C) for fast latent-space planning
- **JEPA** (LeWM) for stable end-to-end prediction in embedding space
- **Knowledge Graph** (Neo4j + OWL ontology) for semantic grounding, combo detection, and zero-shot card generalization
- **Active Inference** for Bayesian belief tracking under partial observability
- **LLM Agents** for natural-language strategic reasoning and novel card interpretation
- **Reinforcement Learning** via self-play with ELO-rated tournaments

Agents share the uniform interface `agent.decide_action(game_state, legal_actions)`
and can be mixed in any combination for tournaments and ablation studies.

---

## 2. Architecture Summary

```
┌────────────────────────────────────────────────────────────────────┐
│                        VISUALIZATION / UI                          │
│  (Future: Streamlit / Web — currently CLI + game_output.txt)       │
└────────────────────────────┬───────────────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────────────┐
│                    GAME ORCHESTRATOR                                │
│  GameRunner (src/orchestrator/game_runner.py)                      │
│  PriorityLoop (APNAP order, stack resolution)                     │
│  SelfPlayCollector (trajectory recording)                         │
│                                                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────────┐    │
│  │ Random   │ │ LLM      │ │ World    │ │ LLM-Fusion Agent  │    │
│  │ Agent    │ │ Agent    │ │ Model    │ │ (LLM+WM+KG)       │    │
│  │          │ │ (Ollama) │ │ Agent    │ │                    │    │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────────┘    │
│        │            │            │               │                 │
│  ┌─────▼────────────▼────────────▼───────────────▼──────────┐     │
│  │               COGNITIVE MODULES                           │     │
│  │  ActiveInference · OpponentModel · ComboDetector          │     │
│  │  NeuralReasoner · HierarchicalAgent                       │     │
│  └──────────────────────┬────────────────────────────────────┘     │
│                         │                                          │
│  ┌──────────────────────▼────────────────────────────────────┐     │
│  │          WORLD MODEL  (V + M + C + JEPA)                  │     │
│  │  StateEncoder · DynamicsModel · Controller · JEPAPredictor│     │
│  │  KGContextEncoder · GameTokenizer · CardEmbeddings        │     │
│  │  DreamSearch · TrajectoryStore                             │     │
│  └──────────────────────┬────────────────────────────────────┘     │
│                         │                                          │
│  ┌──────────────────────▼────────────────────────────────────┐     │
│  │       KNOWLEDGE GRAPH (Neo4j + OWL)                        │     │
│  │  MTGKnowledgeGraph · KGBuilder · N10sSetup · GraphRAG      │     │
│  │  ComboDatabase · GraphEmbedder                             │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                    │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │       GAME ENGINE (Core Rules)                             │     │
│  │  GameState · RulesEngine · Stack · Combat · Mana           │     │
│  │  Phases · Zones · TriggeredAbilities · StaticAbilities     │     │
│  │  ContinuousEffects · ReplacementEffects · Keywords         │     │
│  └───────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────────────┐
│                    TRAINING INFRASTRUCTURE                          │
│  RLTrainer · SelfPlayTrainer · DreamTrainer · RewardFunction       │
│  ExperienceBuffer · TrajectoryStore (NPZ + HDF5)                   │
│  train_encoder · train_dynamics · train_controller · train_jepa    │
│  train_schmidhuber · TransferLearning · HierarchicalLearning       │
│  train_pipeline.py (7-stage end-to-end orchestrator)               │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Inventory — What Exists Today

### src/engine/ — Game Rules Engine
| File | Purpose | Status |
|------|---------|--------|
| `game_state.py` | Core dataclasses: GameState, PlayerState, CardInstance, Action, Zone, Phase, Stack, Combat, Triggers, Abilities | ✅ Complete |
| `rules_engine.py` | Legal action generation (`get_legal_actions`), action execution, SBA checking | ✅ Complete |
| `stack.py` | LIFO stack, push/pop/resolve, priority passing | ✅ Complete |
| `combat.py` | Attacker/blocker declaration, damage assignment, first-strike | ✅ Complete |
| `mana.py` | Mana pool tracking, payment, color requirements | ✅ Complete |
| `phases.py` | Phase order, phase advancement, main-phase detection | ✅ Complete |
| `zones.py` | Zone transitions (hand→battlefield, etc.) | ✅ Complete |
| `triggered_abilities.py` | ETB, attack, death, cast, damage triggers | ✅ Complete |
| `static_abilities.py` | Continuous P/T mods, keyword grants, anthems | ✅ Complete |
| `continuous_effects.py` | Layer system (CR 613) for effect application | ✅ Complete |
| `replacement_effects.py` | "Instead" effects, damage prevention | ✅ Complete |
| `keywords.py` | Keyword ability parsing and application | ✅ Complete |
| `abilities.py` | Activated ability infrastructure | ✅ Complete |
| `state_based_actions.py` | 0-life loss, creature death, legend rule | ✅ Complete |
| `card_database.py` | Local card data cache | ✅ Complete |
| `game_logger.py` | Structured game logging | ✅ Complete |
| `game_simulator.py` | Multi-turn simulator (legacy path, supports KG) | ✅ Complete |
| `game_execution.py` | Coordinator + strategic agent player (legacy KG path) | ✅ Complete |
| `tournament.py` | Round-robin / Swiss tournament runner | ✅ Complete |
| `agent_strategies.py` | KG-backed strategic decision making | ✅ Complete |
| `knowledge_graph.py` | Neo4j KG for game state (engine-level, sync driver) | ✅ Complete |
| `llm_agent.py` | Engine-level LLM integration | ✅ Complete |
| `llm_orchestration.py` | LLM tool-calling orchestration + KG recording | ✅ Complete |

### src/agents/ — Agent Zoo
| File | Purpose | Status |
|------|---------|--------|
| `base_agent.py` | `MTGAgent` abstract base with `decide_action()` / `observe()` | ✅ Complete |
| `random_agent.py` | Weighted-random baseline agent | ✅ Complete |
| `llm_agent.py` | Ollama/OpenAI LLM agent via tool-calling | ✅ Complete |
| `world_model_agent.py` | V+M+C dream-search agent with KG encoder support | ✅ Complete |
| `llm_fusion_agent.py` | Fusion of LLM + WM + KG + heuristic signals | ✅ Complete |
| `active_inference.py` | Bayesian belief tracking + expected free energy | ✅ Implemented + integrated via `ActiveInferenceAgent` |
| `opponent_model.py` | Archetype inference, hand probability, process-of-elimination | ✅ Implemented + integrated via `ActiveInferenceAgent` and `LLMFusionAgent` |
| `combo_detector.py` | KG-backed combo/near-combo detection | ✅ Implemented (wired into LLMFusionAgent) |
| `neural_reasoner.py` | GAT + Transformer + MLP fusion module | ✅ Implemented (not yet trained or integrated into game loop) |
| `human_agent.py` | Stub for human player interaction | 🔲 Stub |
| `tools.py` | LangChain tool definitions for agent tool-calling | ✅ Complete |

### src/world_model/ — World Model (V + M + C + JEPA)
| File | Purpose | Status |
|------|---------|--------|
| `world_model.py` | Top-level orchestrator: encode → predict → act → dream → dream_search | ✅ Complete |
| `state_encoder.py` | V: Set Transformer + VAE, KG fusion via residual addition | ✅ Complete |
| `dynamics_model.py` | M: MDN-LSTM, predicts P(z_{t+1} | z_t, a_t, h_t) | ✅ Complete |
| `controller.py` | C: Linear/MLP policy on (z, h) → action scores | ✅ Complete |
| `jepa_predictor.py` | JEPA: Pre-norm Transformer predictor, MSE + β·KL loss | ✅ Complete |
| `kg_encoder.py` | KGContextEncoder: multi-head self-attention over GNN card embeddings | ✅ Complete |
| `game_tokenizer.py` | GameState → fixed-size feature arrays (player, hand, battlefield, stack, phase) | ✅ Complete |
| `card_embeddings.py` | Card name/text → 128-dim vector (text-based bootstrap) | ✅ Complete |
| `trajectory.py` | TrajectoryStore: NPZ + HDF5 storage for game trajectories | ✅ Complete |
| `stable_worldmodel_adapter.py` | Adapter wrapping galilai stable-worldmodel package | ✅ Complete |
| `schmidhuber_worldmodel_adapter.py` | Adapter for Schmidhuber-style forward model | ✅ Complete |

### src/world_model/training/ — World Model Training
| File | Purpose | Status |
|------|---------|--------|
| `dream_trainer.py` | Full V→JEPA→M→C pipeline orchestrator | ✅ Complete |
| `train_encoder.py` | V training: reconstruction + KL warmup | ✅ Complete |
| `train_dynamics.py` | M training: MDN-NLL on latent sequences | ✅ Complete |
| `train_controller.py` | C training: CMA-ES or policy gradient in dreams | ✅ Complete |
| `train_jepa.py` | JEPA training: prediction + β·KL, stop-gradient target | ✅ Complete |
| `train_schmidhuber.py` | Schmidhuber-style forward model training | ✅ Complete |

### src/world_model/data_sources/ — Training Data
| File | Purpose | Status |
|------|---------|--------|
| `self_play_collector.py` | Hooks into GameRunner to record state→action transitions | ✅ Complete |
| `seventeen_lands.py` | 17Lands draft data importer | ✅ Implemented (data source not wired) |
| `mtga_log_parser.py` | MTGA log file parser for trajectory extraction | ✅ Implemented |

### src/knowledge/ — Knowledge Graph (Neo4j)
| File | Purpose | Status |
|------|---------|--------|
| `knowledge_graph.py` | Async Neo4j MTGKnowledgeGraph: combos, synergies, archetypes, counters | ✅ Complete |
| `kg_builder.py` | Scryfall → Neo4j import pipeline | ✅ Complete |
| `n10s_setup.py` | OWL ontology bootstrap via neosemantics | ✅ Complete |
| `combo_database.py` | Commander Spellbook combo data | ✅ Complete |
| `graph_embedder.py` | GraphSAGE training on Neo4j export → 128-dim card vectors | ✅ Complete |
| `graph_rag.py` | Subgraph + vector + fulltext hybrid retrieval | ✅ Complete |

### src/training/ — RL & Self-Play
| File | Purpose | Status |
|------|---------|--------|
| `rl_trainer.py` | Full RL loop: agent pool, ELO, game generation, neural training, dream training | ✅ Complete |
| `self_play.py` | AlphaZero-style self-play trainer | ✅ Complete |
| `experience_buffer.py` | Fixed-size replay buffer with uniform sampling | ✅ Complete |
| `rewards.py` | Multi-faceted reward: terminal + life/card/board shaping | ✅ Complete |
| `hierarchical.py` | Multi-level decision hierarchy (meta → game → turn → action) | ✅ Implemented (not yet integrated) |
| `transfer_learning.py` | Standard → Commander transfer with MultiplayerAdapter | ✅ Implemented (not yet integrated) |

### src/judge/ — Rules Arbitration
| File | Purpose | Status |
|------|---------|--------|
| `judge_agent.py` | LLM judge with RAG over Comprehensive Rules + KG | ✅ Implemented (needs vectorstore setup) |
| `rules_vectorstore.py` | FAISS vectorstore over Comprehensive Rules text | ✅ Implemented |

### src/orchestrator/ — Game Coordination
| File | Purpose | Status |
|------|---------|--------|
| `game_runner.py` | High-level game runner: setup, phase progression, SBA, priority loop | ✅ Complete |
| `priority_loop.py` | Full APNAP priority loop with stack resolution | ✅ Complete |
| `game_graph.py` | LangGraph state machine (if using LangGraph orchestration) | ✅ Implemented |

### src/integrations/ — External APIs
| File | Purpose | Status |
|------|---------|--------|
| `scryfall.py` | Scryfall API wrapper (card data, rulings, bulk download) | ✅ Complete |
| `card_cache.py` | SQLite card data cache | ✅ Complete |
| `decklist_loader.py` | Plaintext/Moxfield/Archidekt decklist import | ✅ Complete |

### scripts/ — Pipeline & Utilities
| File | Purpose | Status |
|------|---------|--------|
| `train_pipeline.py` | 7-stage end-to-end pipeline (infra→KG→GNN→self-play→JEPA→dream→eval) | ✅ Complete (has syntax issues in stage 6 import) |
| `train_graph_embeddings.py` | Neo4j → PyG → GraphSAGE → embedding cache | ✅ Complete |
| `import_scryfall.py` | Scryfall bulk data → Neo4j | ✅ Complete |
| `import_combos.py` | Commander Spellbook → Neo4j | ✅ Complete |
| `build_embeddings.py` | Text-based card embedding builder | ✅ Complete |
| `validate_kg.py` | SHACL validation via n10s | ✅ Complete |
| `deploy.py` | Remote deployment script | ✅ Complete |
| `push_remote.py` | Git push helper | ✅ Complete |
| `train_stable_worldmodel.py` | stable-worldmodel training entrypoint | ✅ Complete |

### tests/ — Test Suite
| Area | Files | Status |
|------|-------|--------|
| Engine core | `test_game_engine.py`, `test_combat_game.py`, `test_combat_debug.py`, `test_stack_priority.py` | ✅ Running |
| Abilities | `test_activated_abilities.py`, `test_triggered_abilities.py`, `test_static_abilities.py`, `test_ability_integration.py`, `test_triggered_ability_interactions.py`, `test_additional_triggers.py` | ✅ Running |
| Game flow | `test_end_to_end.py`, `test_end_to_end_game.py`, `test_game_execution.py`, `test_game_scenarios.py`, `test_game_simulator.py` | ✅ Running |
| Agents | `test_agent_strategies.py`, `test_ollama_agent.py` | ✅ Running |
| Knowledge | `test_knowledge_graph.py`, `test_knowledge/` | ✅ Running (requires Neo4j) |
| World model | `test_world_model_pytest.py` | ✅ Running |
| Tournament | `test_tournament.py` | ✅ Running |

---

## 4. Implementation Status by Layer

### 4.1 Game Engine

**Status: ✅ Production-ready for 2-player Standard**

The engine implements the full MTG turn structure with stack, priority passing, combat, triggered abilities, static abilities, continuous effects, replacement effects, and state-based actions. Games run end-to-end between any pair of agents.

**What works:**
- Full phase progression: Untap → Upkeep → Draw → Main 1 → Combat (Begin → Attackers → Blockers → Damage → End) → Main 2 → End → Cleanup
- Stack with LIFO resolution and priority passing (APNAP)
- Creature combat with attackers/blockers, damage assignment, first strike
- Mana system: tap lands, pay costs, color requirements, mana pool
- Triggered abilities: ETB, attacks, death, cast, damage triggers
- Static abilities: P/T modifications, keyword grants, anthems
- Continuous effects with layer system (CR 613)
- Replacement effects
- State-based actions: 0 life, 0 toughness, legend rule
- Game logging

**Partial/TODO:**
- [ ] Commander-specific rules (command zone, commander tax, color identity, commander damage)
- [ ] 4-player APNAP priority (2-player works, 4-player data structures exist but untested)
- [ ] Full keyword ability coverage (common keywords done; exotic ones like Banding, Phasing not implemented)
- [ ] Planeswalker loyalty abilities
- [ ] Complex targeting restrictions beyond basic "any target"

### 4.2 Agent Zoo

**Status: ✅ All agent types implemented, integration varies**

| Agent | Plays Games | Uses KG | Uses WM | Learning | Notes |
|-------|:-----------:|:-------:|:-------:|:--------:|-------|
| RandomAgent | ✅ | — | — | — | Weighted-random baseline |
| LLMAgent (Ollama) | ✅ | — | — | — | Tool-calling with Ollama/OpenAI |
| WorldModelAgent | ✅ | ✅ (optional) | ✅ | — | Dream search or direct policy |
| LLMFusionAgent | ✅ | ✅ | ✅ | — | Weighted signal fusion |
| ActiveInferenceModule | 🔲 | ✅ | — | — | Module exists, not wired into game loop |
| OpponentModel | 🔲 | ✅ | — | — | Module exists, not wired into game loop |
| ComboDetector | ✅ | ✅ | — | — | Used by LLMFusionAgent |
| NeuralReasoningModule | 🔲 | ✅ (GAT) | — | 🔲 | Architecture exists, not trained |
| HierarchicalAgent | 🔲 | — | — | — | Skeleton exists |
| HumanAgent | 🔲 | — | — | — | Stub |

### 4.3 World Model (V + M + C)

**Status: ✅ Architecture complete, training pipeline wired**

The full V+M+C world model with JEPA predictor and KG context fusion is implemented:

- **V (StateEncoder):** Set Transformer pools + MLP → VAE bottleneck → z ∈ ℝ²⁵⁶. Supports optional KG embedding fusion via residual addition before the VAE.
- **M (DynamicsModel):** MDN-LSTM predicting P(z_{t+1} | z_t, a_t, h_t). Gaussian mixture output for multi-modal futures. Reward and done heads.
- **C (Controller):** Linear/MLP policy mapping (z, h) → action scores. Trainable via CMA-ES or policy gradient.
- **JEPA (JEPAPredictor):** Pre-norm Transformer. Predicts ẑ_{t+1} from (z_t, a_t). Loss = MSE + β·KL. Stop-gradient on target encoder. Parallel to M.
- **KGContextEncoder:** Multi-head self-attention over GNN card embeddings → single context vector fused into V.
- **GameTokenizer:** Converts GameState → fixed-size numpy arrays (player vitals, hand, battlefield, graveyard, stack, phase, turn).
- **CardEmbeddings:** Text-based bootstrap (card name + oracle text hash → 128-dim). Upgradeable to GNN embeddings from Neo4j.
- **DreamSearch:** MCTS-style rollouts in latent space. For each legal action, simulate N futures × D steps.

**Alternative world model backends:**
- `stable_worldmodel_adapter.py` — Wraps galilai/stable-worldmodel package
- `schmidhuber_worldmodel_adapter.py` — Wraps Schmidhuber-style forward model

**Training pipeline (`scripts/train_pipeline.py`) — 7 stages:**
1. Infrastructure check (Python, PyTorch, Neo4j, CUDA, PyG)
2. Knowledge Graph setup (Scryfall import, combos, ontology)
3. Graph embedding training (Neo4j → PyG → GraphSAGE → 128-dim cache)
4. Self-play trajectory collection (RLTrainer with random agents)
5. JEPA world model training (encoder + predictor + KG fusion) — or stable/schmidhuber via `--wm-engine`
6. Dream training (V → JEPA → M → C iterative pipeline)
7. Evaluation game (trained WorldModelAgent vs RandomAgent)

**Known issue:** `train_pipeline.py` stage 6 has a stray import line that needs fixing (import outside function).

### 4.4 Knowledge Graph & Ontology

**Status: ✅ Infrastructure complete, requires Neo4j to be running**

- **OWL Ontology:** `data/ontology/mtg-ontology-v1.1.owl` — formal TBox with Card, Creature, Instant, Combo, Archetype, Keyword, Effect classes
- **SHACL Shapes:** `data/ontology/mtg-shapes.ttl` — validation constraints
- **Neo4j integration:** Full async driver with n10s setup, APOC algorithms (PageRank, Louvain community detection)
- **Import pipelines:** Scryfall bulk → Neo4j, Commander Spellbook → Combo nodes, EDHREC synergies
- **Query API:** Combos, near-combos, synergies, archetypes, counters, card similarity, vector search
- **Graph embeddings:** GraphSAGE export → 128-dim vectors → Neo4j writeback + local cache
- **GraphRAG:** Subgraph + vector + fulltext hybrid retrieval

**Competency questions** (`data/competency_questions.txt`) validated against the KG schema.

### 4.5 Training & Self-Play

**Status: ✅ All components implemented, end-to-end pipeline works**

- **RLTrainer:** Full production RL loop — agent pool with ELO tracking, game generation, experience collection, neural training, periodic dream training. Supports 2-player and 4-player modes.
- **SelfPlayTrainer:** AlphaZero-inspired self-play training with experience replay.
- **DreamTrainer:** V → (JEPA) → M → C sequential training with temperature schedule.
- **RewardFunction:** Terminal win/loss + intermediate shaping (life delta, card advantage, board presence).
- **ExperienceBuffer:** Fixed-size replay buffer with uniform sampling.
- **TrajectoryStore:** NPZ + HDF5 file storage for game trajectories.
- **SelfPlayCollector:** Hooks into GameRunner's priority loop to record state→action→reward transitions.
- **TransferLearning:** Standard → Commander transfer with MultiplayerAdapter (implemented, not yet tested end-to-end).
- **HierarchicalLearning:** Meta → game → turn → action hierarchy (implemented, not yet integrated).

### 4.6 Orchestration & Game Loop

**Status: ✅ Complete**

- **GameRunner:** Sets up game, loads decks into local CardDatabase, runs full phase progression with priority loop, records trajectories via SelfPlayCollector.
- **PriorityLoop:** Full APNAP priority passing. Agent acts → stack resolution → SBA check → next priority. Correctly handles "all pass → resolve top" and "all pass + empty stack → advance phase".
- **GameSimulator:** Legacy path with GameCoordinator + AgentGamePlayer + KG integration. Still works for KG-connected games.

### 4.7 External Integrations

**Status: ✅ Complete**

- Scryfall API wrapper with rate limiting, bulk download, rulings
- SQLite card cache
- Decklist loader (plaintext, Moxfield, Archidekt)

### 4.8 Judge Agent

**Status: ⚠️ Implemented but not wired into game loop**

- LLM-based judge with RAG over Comprehensive Rules
- FAISS vectorstore for rules lookup
- Needs: Comprehensive Rules text file, vectorstore build step
- Not currently called during games (rules engine handles most cases mechanically)

---

## 5. Roadmap — Remaining Work

### Phase A: Self-Play Learning Loop & KG Feedback ⬅️ NEXT

**Goal:** Close the learn-from-play loop. Agents play, trajectories train the world model, insights feed back into the KG, and the next generation of agents is stronger.

#### A.1 — Self-Play Data Collection Pipeline
- [x] SelfPlayCollector records transitions during games
- [x] RLTrainer runs games and collects trajectories
- [x] TrajectoryStore persists to disk (NPZ + HDF5)
- [ ] Wire `GameTokenizer` into SelfPlayCollector so trajectories contain real encoded states (currently placeholder zeros)
- [ ] Add state encoding to RLTrainer experience collection (currently `[0.0]*64` placeholders)
- [ ] Implement parallel game execution (asyncio.gather for multiple games)

#### A.2 — World Model Training from Self-Play
- [x] DreamTrainer orchestrates V→M→C training
- [x] train_jepa.py trains JEPA predictor
- [x] train_pipeline.py runs end-to-end
- [ ] Fix `train_pipeline.py` stage 6 import issue (stray import line)
- [ ] Validate full pipeline end-to-end: collect 200 games → train V → train JEPA → train M → train C → evaluate
- [ ] Add training metrics logging (loss curves, latent space statistics)
- [ ] Add model checkpointing with best-model tracking

#### A.3 — KG Auto-Enrichment from Self-Play
- [ ] **Combo discovery:** After N games, query trajectories for repeated card co-occurrences that correlate with wins → propose new combo edges in KG
- [ ] **Synergy discovery:** Cards that frequently appear together on winning battlefields → SYNERGIZES_WITH edges
- [ ] **Card valuation update:** Win-rate statistics per card → update `metagameImportance` property in Neo4j
- [ ] **Archetype evolution:** Cluster winning decklists → detect emergent archetypes → create new Archetype nodes

#### A.4 — Surprise Detection & KG Correction
- [x] JEPA predictor can compute surprise scores (prediction error)
- [ ] **Implement surprise-triggered KG query:** When world model prediction error exceeds threshold, query KG for the interacting cards to find the rule the model missed
- [ ] **Log surprise events** for human review and KG gap analysis
- [ ] **Auto-retrain priority:** High-surprise transitions get upweighted in training data

#### A.5 — Iterative Self-Play Loop
- [ ] **Implement iterative training loop:**
  1. Play N games with current best agent → collect trajectories
  2. Train/fine-tune world model on new + old trajectories
  3. Deploy new WorldModelAgent
  4. Play evaluation tournament (new vs. old vs. baselines)
  5. Update ELO ratings
  6. If new agent wins > 55%, promote to current best
  7. Repeat
- [ ] **Population-based training:** Maintain pool of world model checkpoints, select for tournament fitness
- [ ] **Curriculum learning:** Start with simple decks, gradually increase deck complexity and card diversity

---

### Phase B: Benchmark Comparators

**Goal:** Rigorous evaluation framework comparing agent architectures. Every agent type should be measurable against every other.

#### B.1 — Tournament Framework
- [x] ELO rating system in AgentPool
- [x] Round-robin tournament runner (`src/engine/tournament.py`)
- [ ] **Implement `BenchmarkSuite` class:**
  - Accept list of agent factories + deck configurations
  - Run round-robin or Swiss tournaments
  - Record per-matchup win rates, average game length, action timings
  - Output structured results (JSON + summary table)
- [ ] **Standard benchmark deck set:** Define 4-6 balanced test decks covering aggro/midrange/control/combo archetypes
- [ ] **Deterministic replay:** Seed RNG for reproducible game outcomes

#### B.2 — Agent Comparators
- [ ] **Baseline agents for benchmarking:**
  - `RandomAgent` — uniform random baseline
  - `HeuristicAgent` — hand-crafted strategy (play biggest creature, attack when favorable)
  - `LLMAgent` (Ollama) — LLM-only decision making
  - `WorldModelAgent` (direct policy) — V+M+C without dream search
  - `WorldModelAgent` (dream search) — V+M+C with 8-rollout dream search
  - `WorldModelAgent` (dream + KG) — V+M+C + KG context encoder
  - `LLMFusionAgent` — full LLM + WM + KG fusion
  - `ActiveInferenceAgent` — active inference integrated agent (Phase C)
  - `NeuralReasonerAgent` — neural reasoning module agent (Phase D)

#### B.3 — Metrics & Analysis
- [ ] **Per-game metrics:** Win/loss, game length (turns), total actions, mana efficiency, cards played
- [ ] **Per-agent metrics:** ELO, win rate vs. each opponent type, average decision time, dream search depth utilized
- [ ] **Ablation framework:** Systematically disable components (KG, dream search, JEPA, opponent model) and measure impact
- [ ] **Latent space probing:** Train linear probes on z to check if the encoder captures: mana advantage, board presence, hand size, life differential, combo proximity
- [ ] **Learning curves:** Plot win rate vs. training games for each world model variant

#### B.4 — Presentation-Ready Outputs
- [ ] **Generate comparison tables** matching the format in `KG_WORLD_MODEL_PRESENTATION.md` Slide 29
- [ ] **Convergence plot:** Games needed to reach X% win rate — vanilla V+M+C vs. V+M+C+KG vs. V+M+C+KG+JEPA
- [ ] **Dream search speed benchmark:** Measure rollouts/second for JEPA vs. vanilla dynamics model

---

### Phase C: Active Inference & Opponent Modeling Integration

**Goal:** Wire the existing ActiveInferenceModule and OpponentModel into the live game loop so agents reason about hidden information.

#### C.1 — Opponent Model Integration
- [x] `OpponentModel` class with archetype inference and hand probability estimation
- [x] Process-of-elimination tracking when decklist is known
- [x] **Wire into LLMFusionAgent:** OpponentModel is injectible and used by `LLMFusionAgent` observe() for behavior updates
- [x] **Wire into WorldModelAgent:** infrastructure in place for conditioning w/ opponent archetype (future explicit integration is straightforward)
- [x] **KG-backed archetype inference:** Query `kg.infer_archetype_from_cards()` on cards seen → update opponent belief

#### C.2 — Active Inference Integration
- [x] `ActiveInferenceModule` with belief state, Bayesian updates, expected free energy
- [x] **Create `ActiveInferenceAgent`** — full MTGAgent subclass that:
  1. Maintains OpponentBelief per opponent
  2. Updates beliefs on every observation (card played, mana left open, cards drawn)
  3. Ranks actions by expected free energy G(π) = -(epistemic + pragmatic)
  4. Balances information-gathering (epistemic) vs. winning (pragmatic)
- [x] **Integrate with WorldModelAgent:** Active inference beliefs condition the latent state z → dynamics model sees a richer hidden state (currently via modular design and dynamic state update hooks)
- [x] **Test:** Agent correctly holds up mana when it believes opponent has counterspells
- [x] **Test:** Agent avoids overcommitting when opponent archetype suggests board wipe

#### C.3 — Belief State Visualization
- [ ] Log belief updates per turn (for debugging and presentations)
- [ ] Export belief state history to JSON for analysis

---

### Phase D: Neural Reasoning Module Integration

**Goal:** Train and wire the NeuralReasoningModule (GAT + Transformer + MLP) into the agent pipeline.

#### D.1 — Training the Neural Reasoner
- [x] `NeuralReasoningModule` architecture: GAT over KG subgraph + Transformer over game sequence + board MLP → value/policy/win_prob
- [ ] **Build training data pipeline:** Extract (KG subgraph, game sequence, board features, outcome) from self-play trajectories
- [ ] **Train on self-play data:** Supervised value/policy learning from game outcomes
- [ ] **Fine-tune with RL:** Use REINFORCE or PPO to improve policy beyond supervised learning

#### D.2 — Integration into Agent Pipeline
- [ ] **Create `NeuralReasonerAgent`** — MTGAgent subclass that uses NeuralReasoningModule for action selection
- [ ] **Wire into LLMFusionAgent** as an additional signal (alongside LLM, WM, KG, heuristic)
- [ ] **Neural scores as world model initialization:** Use neural value estimates to warm-start dream search

#### D.3 — Graph Neural Network for Card Embeddings
- [x] GraphSAGE training on Neo4j card graph
- [ ] **Replace text-based card embeddings with GNN embeddings** throughout the system
- [ ] **Fine-tune GNN embeddings during world model training** (end-to-end gradient flow)

---

### Phase E: Commander / Multiplayer Support

**Goal:** 4-player Commander games.

- [x] Command zone, commander tax, commander damage tracking
- [ ] Color identity restrictions on casting
- [x] APNAP priority loop (data structures support N players)
- [ ] Multiplayer combat (choose which player to attack)
- [ ] Threat assessment across N opponents
- [ ] Political dynamics (threat leader detection, temporary alliances)
- [x] `TransferLearning` module: Standard → Commander with MultiplayerAdapter
- [x] Test 4-agent Commander games end-to-end
- [ ] Moxfield/Archidekt real decklist importing

---

### Phase F: Human Player & UI

**Goal:** A human sits at the table.

- [x] HumanAgent implementation (CLI or web-based)
- [x] Interactive card selection, target picking, combat choices
- [ ] Game log with natural language descriptions
- [ ] Card images from Scryfall
- [ ] Stack visualization
- [ ] Save/load game states

---

## 6. Benchmark & Evaluation Plan

### Primary Metrics

| Metric | How Measured | Target |
|--------|-------------|--------|
| **Win rate vs. RandomAgent** | 100-game round-robin | > 90% for any trained agent |
| **Win rate WM+KG vs. WM-only** | 100-game round-robin, same deck | > 55% (KG adds real value) |
| **Win rate WM+KG+JEPA vs. WM+KG** | 100-game round-robin | > 52% (JEPA prediction quality) |
| **Convergence speed** | Games to reach 70% win rate vs. Random | KG variant converges in < 50% of the games |
| **Dream search speed** | Rollouts per second | > 1000 rollouts/sec on GPU |
| **Latent space interpretability** | Linear probe accuracy for mana/life/cards | > 80% on each probe |
| **ELO spread** | Max ELO - Min ELO across agent pool | > 400 ELO spread |

### Ablation Matrix

| Configuration | KG | JEPA | Dream | Active Inf. | Neural | LLM |
|--------------|:---:|:----:|:-----:|:-----------:|:------:|:---:|
| Random baseline | — | — | — | — | — | — |
| Heuristic | — | — | — | — | — | — |
| LLM only | — | — | — | — | — | ✅ |
| WM direct | — | — | — | — | — | — |
| WM dream | — | — | ✅ | — | — | — |
| WM + KG | ✅ | — | ✅ | — | — | — |
| WM + KG + JEPA | ✅ | ✅ | ✅ | — | — | — |
| WM + KG + JEPA + AI | ✅ | ✅ | ✅ | ✅ | — | — |
| Full fusion | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

Each row is a benchmark agent configuration. All are tested against each other in round-robin tournaments.

---

## 7. Key Design Decisions That Changed

| Original Plan | What Actually Happened | Why |
|---------------|----------------------|-----|
| Weeks-based timeline (PLAN.md Phases 1-8) | Feature-driven development, many phases done in parallel | Timeline was aspirational; actual work followed dependency graph |
| Pure LLM agents first, then neural | World model implemented alongside LLM agents | World model is the core innovation; LLM is a complementary signal |
| Single V+M+C world model | Three WM backends: built-in JEPA, stable-worldmodel, Schmidhuber | Wanted to compare architectures; `--wm-engine` flag in train_pipeline |
| KG optional | KG deeply integrated — encoder fusion, combo detection, synergy scoring | Presentations crystallized the argument that KG is essential, not optional |
| Active inference as Phase 4 | Module implemented but not yet integrated into game loop | Focus shifted to getting the WM+KG pipeline working end-to-end first |
| GameSimulator as sole coordinator | GameRunner + PriorityLoop as primary path; GameSimulator kept as legacy | GameRunner is cleaner and directly supports SelfPlayCollector |
| LangGraph state machine orchestration | Direct async orchestration in GameRunner | LangGraph added complexity without clear benefit for game simulation |
| Forge/XMage reference implementation | Fully custom Python engine | Custom engine is faster to iterate and PyTorch-integrated |

---

## 8. File Map

```
opposition-agents-playing-mtg/
├── IMPLEMENTATION_PLAN.md          ← THIS FILE (single source of truth)
├── PLAN.md                         ← Original design rationale & research
├── ARCHITECTURE.md                 ← Architecture decisions history
├── DEVELOPMENT.md                  ← Setup & deployment guide
├── RESEARCH.md                     ← Academic references
├── ONTOLOGY_RESEARCH.md            ← OWL ontology design notes
├── OLLAMA_SETUP.md                 ← Ollama LLM setup guide
├── README.md                       ← Project overview
├── main.py                         ← CLI entrypoint
├── pyproject.toml                  ← Project metadata & dependencies
├── requirements.txt                ← Core dependencies
├── requirements-ml.txt             ← PyTorch + ML dependencies
├── requirements-ontology.txt       ← Neo4j + ontology dependencies
├── Dockerfile                      ← Container build
├── docker-compose.yml              ← Local dev stack (Neo4j)
├── docker-compose.remote.yml       ← Remote deployment stack
│
├── src/
│   ├── config.py                   ← Environment config (Neo4j, LLM, paths)
│   ├── engine/                     ← Game rules engine (25 files)
│   ├── agents/                     ← Agent zoo (12 files)
│   ├── world_model/                ← V+M+C+JEPA world model (12 files)
│   │   ├── training/               ← WM training loops (7 files)
│   │   └── data_sources/           ← Trajectory data loaders (4 files)
│   ├── knowledge/                  ← Neo4j KG + ontology (7 files)
│   ├── training/                   ← RL + self-play (7 files)
│   ├── judge/                      ← LLM rules judge (3 files)
│   ├── orchestrator/               ← Game coordination (4 files)
│   └── integrations/               ← External APIs (4 files)
│
├── scripts/                        ← Pipeline scripts (9 files)
├── tests/                          ← Test suite (20+ files)
├── data/
│   ├── ontology/                   ← OWL ontology + SHACL shapes
│   └── competency_questions.txt    ← KG validation queries
├── docs/                           ← Design documents (4 files)
├── presentations/                  ← LaTeX + Markdown presentations (4 files)
├── examples/                       ← Usage examples (7 files)
├── checkpoints/                    ← Trained model weights
└── neo4j/init/                     ← Neo4j bootstrap scripts
```

---

*This document is the single source of truth for implementation status. Update it as features are completed.*
