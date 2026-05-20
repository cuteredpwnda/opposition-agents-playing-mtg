# Opposition Agents Playing MTG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange.svg)](#known-limitations)

A research framework that combines **knowledge-graph-grounded JEPA world models**, **active-inference LLM agents**, and the **phase-rs Rust MTG engine** to study reasoning under uncertainty in the most combinatorially complex commercial card game.

> **Status (May 2026):** The **phase-rs engine is now the authoritative runtime** for new training/evaluation work. Python agents play end-to-end games via WebSocket bridge (adapter.py), generating structured JSONL traces for offline policy learning. Training pipeline is phase-rs-first: collect traces → JEPA → dream training → evaluation. See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) and [docs/PHASE_RS_INTEGRATION.md](docs/PHASE_RS_INTEGRATION.md) for current status.

> If you use this project in academic work, please cite the [tech report](paper/opposition_agents_mtg.tex) (see [Citation](#citation)).

## TL;DR

We treat **Magic: The Gathering** — the most combinatorially complex commercial card game (≥10⁸⁰⁰ states, ~28k unique cards, partial observability, unbounded action space) — as a testbed for **neuro-symbolic reasoning under uncertainty**. The framework couples four ideas that are usually studied in isolation:

1. **Phase-rs Rust engine as primary runtime** (state-based actions, stack/priority, triggers, replacement effects) — full Comprehensive Rules implementation with 30k+ cards. Python agents drive the phase-rs seat via WebSocket bridge, ensuring we control observations handed to agents.
2. **A Neo4j knowledge graph** built from an OWL 2 ontology of cards/keywords/archetypes/combos, queried by agents for symbolic strategic reasoning (combo detection, archetype inference, GraphRAG context).
3. **Structured JSONL trace collection from phase-rs games** — decision events, legal actions, game outcomes — fed into offline JEPA training for world models.
4. **A V+M+C+JEPA world model** — a Set-Transformer + VAE state encoder, an MDN-LSTM dynamics model, a controller, and a JEPA latent predictor — trained on self-play trajectories for imagination-based planning.
5. **Active-Inference LLM agents** that fuse symbolic graph queries, latent rollouts, and free-energy-minimising action selection, with a per-opponent belief module that does **exact** library/hand inference when decklists are public.
6. **A collective graph-memory loop** where many agents contribute append-only evidence (from self-play traces and outcomes) into a shared knowledge layer, so strategic knowledge accumulates across games rather than being reset per run.

A champion-vs-challenger self-play loop with ELO promotion drives improvement. The full pipeline is described in the [tech report](paper/opposition_agents_mtg.tex) ([build instructions](paper/README.md)).

### What you can do with this repo today

| You want to… | Read | Run |
|---|---|---|
| Play a game with random/heuristic agents on phase-rs | [docs/PHASE_RS_INTEGRATION.md](docs/PHASE_RS_INTEGRATION.md) | `python examples/play_edh_pod.py --max-turns 6 --model none` |
| Play a game with LLM agents on phase-rs | [docs/PHASE_RS_INTEGRATION.md](docs/PHASE_RS_INTEGRATION.md) | `python examples/play_edh_pod.py --max-turns 12 --model ollama:llama3` |
| Collect training traces from phase-rs | [docs/PHASE_RS_INTEGRATION.md](docs/PHASE_RS_INTEGRATION.md) | `python scripts/collect_phase_rs_traces.py --games 16 --picker agent:heuristic` |
| Run ablation suite against phase-rs AI | [docs/PHASE_RS_INTEGRATION.md](docs/PHASE_RS_INTEGRATION.md) | `python scripts/run_phase_rs_ablation.py --games 10 --picker agent:heuristic --autostart` |
| Train JEPA on phase-rs traces (phase-rs-first) | [docs/PHASE_RS_INTEGRATION.md](docs/PHASE_RS_INTEGRATION.md) | `python scripts/train_pipeline.py --phase-rs-traces --num-games 64` |
| Inspect the world-model design | [docs/WORLD_MODEL_DESIGN.md](docs/WORLD_MODEL_DESIGN.md) | `pytest tests/test_world_model_pytest.py` |
| Track a real Commander opponent's hand exactly | [docs/OPPONENT_MODELING_WITH_DECKLIST.md](docs/OPPONENT_MODELING_WITH_DECKLIST.md) | see snippet below |
| Read the science behind it | [paper/opposition_agents_mtg.tex](paper/opposition_agents_mtg.tex) | `cd paper && latexmk -pdf` |

## phase-rs engine (primary runtime, May 2026)

**Phase-rs is now the authoritative runtime engine for all training and evaluation work.** We integrate with the [phase-rs/phase][phase] Rust MTG engine (30k+ cards, full layers/replacement/stack, built-in difficulty-scaled AI) via a WebSocket bridge in [`src/integrations/phase_rs/`](src/integrations/phase_rs). Python agents drive the phase-rs seat by:

1. **Translating** legal actions from phase-rs wire format → internal `Action` objects (adapter.py)
2. **Picking** actions using native MTGAgent implementations (RandomAgent, HeuristicAgent, LLMAgent, WorldModelAgent, etc.)
3. **Collecting** structured JSONL traces for offline JEPA training

**Training pipeline (phase-rs-first):**
- `scripts/collect_phase_rs_traces.py` — Run games on phase-rs, emit decision events
- `scripts/train_pipeline.py --phase-rs-traces` — Stage 4.1 new: collect traces → post-process → feed into JEPA training
- `scripts/run_phase_rs_ablation.py` — Batch evaluation with automatic retry on transient failures
- `scripts/phase_rs_rollout_sweep.py` — Cartesian sweep (picker × difficulty × deck)

See [docs/PHASE_RS_INTEGRATION.md](docs/PHASE_RS_INTEGRATION.md) for detailed setup, protocol reference, and contribution flow.

**Legacy Python engine:** The engine in `src/engine/` remains for back-compat and special cases (custom rulesets), but is no longer the training target.

[phase]: https://github.com/phase-rs/phase

## Non-commercial fan / research project

This is a non-commercial fan / research project built under the spirit of
the [Wizards of the Coast Fan Content Policy][wotc-fcp]. It is not
affiliated with, endorsed by, sponsored by, or approved by Wizards of the
Coast LLC or Hasbro, Inc. This repository does not redistribute MTG card
images, card art, mana symbol artwork, card-frame graphics, or the
Comprehensive Rules document — all such assets are downloaded by the user
at runtime from [Scryfall](https://scryfall.com/) and
[magic.wizards.com](https://magic.wizards.com). See [NOTICE.md](NOTICE.md)
and [DMCA.md](DMCA.md) for full details.

[wotc-fcp]: https://company.wizards.com/en/legal/fancontentpolicy

## Overview

This project builds an agentic framework where multiple AI agents compete in Magic: The Gathering games on the phase-rs Rust engine, using:

- **Game Runtime**: phase-rs (Rust) — full Comprehensive Rules with 30k+ cards, all layers/replacement/stack mechanics. Python agents drive the session via WebSocket bridge in `src/integrations/phase_rs/`.
- **Bridge Architecture**: 
  - `client.py` — WebSocket protocol (v6+)
  - `adapter.py` — GameAction ↔ Action translation
  - `agent_bridge.py` — AsyncActionPicker for native MTGAgent
  - `runner.py` — Trace collection + reconnect/resume logic
- **Trace Collection**: Structured JSONL (decision events, legal actions, game outcomes) → TrajectoryStore → JEPA training
- **Knowledge Graph**: Neo4j + n10s (OWL ontology import) + APOC for card knowledge, combo detection, strategic reasoning
- **Collective Intelligence Layer**: Append-only learned evidence from self-play (`LearnedSynergyEvidence`, `LearnedCardOutcome`, `KGExtensionEvent`)
- **Agent Architecture**: Native MTGAgent protocol (RandomAgent, HeuristicAgent, LLMAgent, WorldModelAgent, ActiveInferenceAgent, LLMFusionAgent, HierarchicalAgent)
- **Training (phase-rs-first)**: 
  1. Collect traces from phase-rs games (Stage 4.1: `scripts/collect_phase_rs_traces.py`)
  2. Post-process into TrajectoryStore
  3. Train JEPA world model on traces (Stage 5: `scripts/train_pipeline.py`)
  4. Dream training for planning (Stage 6)
  5. Evaluate trained agents back on phase-rs
- **Knowledge Graph**: Neo4j card/combo/archetype ontology + GraphSAGE embeddings + RAG query strategies
- **JEPA Integration**: Dual-input state+KG JEPA predictor (2-loss MSE+KL), surprise scoring, hybrid MDN-LSTM + JEPA planning
- **Integration**: Scryfall API for card data, Commander Spellbook combos, rules vectorstore judge, Ollama LLM for agent reasoning

## Documentation

For detailed architecture and design decisions, see:

- **[PLAN.md](PLAN.md)** - Comprehensive project plan with 19 sections covering objectives, architecture, tech stack, implementation roadmap, and open questions
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Module-by-module implementation status and roadmap (canonical status doc)
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Phase-1 design choices and pattern matrix
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development guide and troubleshooting
- **[paper/opposition_agents_mtg.tex](paper/opposition_agents_mtg.tex)** - Tech report describing the system in publication form
- **[docs/RESEARCH.md](docs/RESEARCH.md)** - Survey of open-source MTG engines, GraphRAG, and agent frameworks
- **[docs/ONTOLOGY_RESEARCH.md](docs/ONTOLOGY_RESEARCH.md)** - OWL ontology extension research and best practices
- **[docs/OLLAMA_SETUP.md](docs/OLLAMA_SETUP.md)** - Local LLM (Ollama) installation and configuration
- **[docs/ONTOLOGY_EXTENSION_GUIDE.md](docs/ONTOLOGY_EXTENSION_GUIDE.md)** - Systematic approach to extending the MTG ontology
- **[docs/OPPONENT_MODELING_WITH_DECKLIST.md](docs/OPPONENT_MODELING_WITH_DECKLIST.md)** - Decklist-aware opponent modeling
- **[docs/WORLD_MODEL_DESIGN.md](docs/WORLD_MODEL_DESIGN.md)** - V+M+C+JEPA world model architecture details
- **[docs/HOW_IT_ALL_WORKS.md](docs/HOW_IT_ALL_WORKS.md)** - End-to-end walkthrough of an episode

## Quick Start

### Prerequisites

- Python 3.10+
- Neo4j Community Edition 5.x (optional, for knowledge graph features)
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/cuteredpwnda/opposition-agents-playing-mtg
cd opposition-agents-playing-mtg
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:

On Windows:
```bash
venv\Scripts\activate
```

On macOS/Linux:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

For machine learning features (optional):
```bash
pip install -r requirements-ml.txt
```

Alternatively, with Hatch extras (matches `pyproject.toml`):
```bash
pip install -e .[ml]
```

5. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings (LLM API keys, Neo4j connection, etc.)
```

6. Set up Neo4j (if using knowledge graph features):
```bash
docker-compose up -d neo4j
# Wait for Neo4j to start at http://localhost:7474
# Default credentials: neo4j / neo4j
# Change password on first login
```

## Getting Started

**First time?** Start here:

1. **Read the overview** above to understand the project goals
2. **Set up your environment** following the Quick Start section
3. **Run the test suite** to verify everything works: `pytest tests/`
4. **Explore the code** in `src/` organized by package (see Project Structure below)
5. **Read PLAN.md** for detailed architecture, design decisions, and implementation roadmap

**Want to contribute?** Check [DEVELOPMENT.md](DEVELOPMENT.md) for:
- Running tests and linters
- Setting up the knowledge graph
- Deploying as a server
- Troubleshooting common issues

## Project Structure

```
opposition-agents-playing-mtg/
├── src/
│   ├── engine/              # Game rules engine and state management
│   │   ├── game_state.py    # Core dataclasses (GameState, CardInstance, PlayerState)
│   │   ├── rules_engine.py  # Legal action generation, action execution, SBAs
│   │   ├── zones.py         # Card zone management
│   │   ├── stack.py         # Stack resolution, fizzle logic
│   │   ├── phases.py        # Turn phases and phase advancement
│   │   ├── combat.py        # Combat sequence (declare, resolve, damage)
│   │   ├── mana.py          # Mana costs, payment, pool management
│   │   ├── keywords.py      # Keyword ability implementations
│   │   ├── state_based_actions.py  # CR 704 SBA checking
│   │   ├── triggered_abilities.py  # Trigger detection and queueing
│   │   ├── replacement_effects.py  # Replacement effect handling
│   │   └── continuous_effects.py   # Layer system for continuous effects
│   │
│   ├── knowledge/           # Knowledge graph and reasoning
│   │   ├── n10s_setup.py    # Neo4j n10s initialization
│   │   ├── knowledge_graph.py  # Cypher queries for card combos, synergies, archetypes
│   │   ├── graph_rag.py     # Graph-based RAG strategies
│   │   ├── kg_builder.py    # Bulk card/combo import
│   │   ├── combo_database.py  # Commander Spellbook integration
│   │   └── graph_embedder.py  # PyG graph embeddings (optional)
│   │
│   ├── agents/              # AI agent implementations
│   │   ├── base_agent.py    # MTGAgent abstract base
│   │   ├── llm_agent.py     # LangChain-based LLM agent
│   │   ├── random_agent.py  # Baseline random agent
│   │   ├── human_agent.py   # Human player interface
│   │   ├── active_inference.py  # Free energy principle for uncertainty
│   │   ├── neural_reasoner.py   # Graph neural networks (optional)
│   │   ├── opponent_model.py    # Opponent belief tracking
│   │   ├── combo_detector.py    # Available/near-combo detection
│   │   └── tools.py         # LangChain tools for agents
│   │
│   ├── judge/               # Judge agent for rule adjudication
│   │   ├── judge_agent.py   # RAG-based judge with CR + rulings context
│   │   └── rules_vectorstore.py  # FAISS vectorstore of CR
│   │
│   ├── orchestrator/        # Game orchestration and execution
│   │   ├── game_graph.py    # LangGraph state machine
│   │   ├── priority_loop.py # APNAP priority passing
│   │   └── game_runner.py   # Full game execution loop
│   │
│   ├── training/            # Self-play training
│   │   ├── rewards.py       # Reward function (terminal + shaping)
│   │   ├── self_play.py     # AlphaZero-style self-play trainer
│   │   ├── experience_buffer.py  # Replay buffer
│   │   └── hierarchical.py  # Multi-level strategy hierarchy
│   │
│   ├── integrations/        # External API integrations
│   │   ├── scryfall.py      # Async Scryfall API client
│   │   ├── decklist_loader.py  # Decklist parsing
│   │   └── card_cache.py    # SQLite card cache
│   │
│   └── config.py            # Centralized configuration
│
├── neo4j/
│   └── init/                # Neo4j initialization scripts
│       └── 01_init_n10s.cypher  # n10s plugin setup
│
├── data/
│   ├── ontology/            # OWL ontology and SHACL shapes
│   │   ├── mtg-ontology-v1.0.owl
│   │   └── mtg-shapes.ttl
│   ├── imports/             # Bulk import data
│   └── cache/               # Card cache (SQLite)
│
├── tests/                   # Test suite
├── docs/                    # Documentation
├── pyproject.toml           # Python project configuration
├── requirements.txt         # Core dependencies
├── requirements-ml.txt      # Optional ML dependencies
├── docker-compose.yml       # Neo4j containerization
├── .env.example            # Environment template
└── README.md               # This file
```

## Architecture

### Core Concept

The system treats MTG as a cooperative problem-solving task where agents:

1. **Observe** the game state (their hand, board, opponent threats)
2. **Reason** about available combos, threats, and strategic value
3. **Decide** which legal action minimizes expected free energy (via Active Inference)
4. **Execute** the action and collect experience for training

The architecture separates concerns:

- **Engine**: Pure game rules (what's legal, what happens mechanically)
- **Knowledge**: Strategic information (combos, synergies, archetypes)
- **Agents**: Decision-making (LLM reasoning, neural networks, uncertainty tracking)
- **Training**: Improvement (self-play, reward signals, model updates)

### Collective Graph Intelligence

Beyond single-agent play quality, the framework is designed to support a collective-intelligence loop:

1. Multiple heterogeneous agents play games and emit structured traces.
2. The enrichment pipeline extracts repeated co-occurrence/outcome patterns.
3. Evidence is written into an append-only graph extension layer with provenance.
4. Future agents query the enriched graph and benefit from prior agents' experience.

This makes the KG a shared long-term memory, while keeping source card facts immutable.

### Game Engine

The rules engine implements a subset of the Comprehensive Rules (CR):

- **Legal Action Generation** (`rules_engine.get_legal_actions`): Generates playable spells, land drops, activated abilities based on game phase and stack state
  - Respects timing rules (sorcery-speed only during main phase with empty stack)
  - Instant-speed actions (instants, flash creatures) available anytime
  - Validates mana cost can be paid
  
- **Action Execution** (`rules_engine.execute_action`): Applies state mutations
  - Card zone transitions (hand → stack → battlefield → graveyard)
  - Mana payment and pool management
  - Tapping permanents
  - Tracking damage and counters
  
- **State-Based Actions** (`rules_engine.check_state_based_actions`): CR 704 compliance
  - Creatures with 0 or negative toughness die
  - Creatures with lethal damage die
  - Players at 0 or less life lose
  - Commander damage (21+ = loss in Commander format)
  - Tokens cease to exist when leaving battlefield
  - Game termination (when 1 player remains)
  
- **Commander Rules** (commander format only)
  - command zone for initial commander placement and returns on death
  - commander tax (+2 each extra cast from command zone)
  - color identity restrictions for all owned cards
  - multiplayer attack target selection and APNAP priority implemented

- **Priority loop** (`orchestrator.priority_loop.run_priority_loop`): APNAP order plus infinite-loop guard
  - `max_iterations=1000` abort safety to avoid hung self-play
  - Supports CONCEDE as an early termination action

- **Triggers** (`triggered_abilities.py`): Detects and queues triggered abilities
  - Enters-the-battlefield triggers
  - "Whenever" triggers (attack, damage, discard, etc.)
  - Queues triggers for resolution
  
- **Keywords** (`keywords.py`): Implements keyword abilities with actual game logic
  - **Evasion**: Flying (can only be blocked by flying creatures), Shadow (not blocked by non-shadow creatures), Unblockable
  - **Combat**: Menace (requires 2+ blockers), Vigilance (doesn't tap when attacking)
  - **Life**: Lifelink (damage dealt heals controller)
  - **Removal**: Deathtouch (all damage is lethal)
  - Reach: Can block flying

### Knowledge Graph

All card knowledge and strategic information is stored in Neo4j for queryability:

**Structure** (via OWL ontology):
- **Card class**: Every card's properties (name, type, mana cost, CMC, power/toughness)
- **Combo class**: Sets of cards that together produce unlimited value or instant wins
- **Archetype class**: Deck strategies (control, midrange, combo, aggro) with signature cards
- **Effect class**: Ability types (draw, discard, tutors, removal)

**Queries** (`knowledge_graph.py`):
- `detect_available_combos(available_cards)`: Find combos where ALL pieces are in hand/battlefield
  - Example: Splinter Twin + Pestermite = infinite damage
  
- `detect_near_combos(available_cards)`: Find combos missing exactly 1 piece
  - Example: Have Splinter Twin, need creature with 0-cost activated ability
  
- `infer_archetype_from_cards(cards_seen)`: Classify opponent deck
  - Example: [Counterspell, Snapcaster Mage, Force of Will] strongly suggests Control
  
- `get_strategic_context(hand, battlefield, opponent_cards)`: Aggregate all relevant context
  - What combos can I play?
  - What are the top opponent threats?
  - What removal/answers do I have?

**Algorithms** (via APOC):
- Community detection (find related cards/strategies)
- PageRank (identify most important cards in a combo chain)
- Subgraph expansion (find all variations of a combo)

### Agents

Four agent types with different decision-making strategies:

| Agent Type | Strategy | Use Case | Dependencies |
|-----------|----------|----------|---|
| **LLMAgent** | ChatGPT with tool-calling | Primary playing agent | OpenAI API key |
| **RandomAgent** | Uniformly random legal action | Baseline for evaluation | None |
| **HumanAgent** | Interactive stdin prompts | Manual playtesting | Terminal I/O |
| **NeuralReasoner** | Graph neural network | Advanced player (optional) | PyTorch, PyG |

**Decision-Making Framework**: **Active Inference** (Free Energy Principle)

Each action is scored by minimizing:
```
G(π) = Epistemic Value + Pragmatic Value

Epistemic Value = How much uncertainty does this reduce?
  High for: Discard effects, Thoughtseize, Probe effects (learn opponent's hand)
  Medium for: Draw effects (learn your own deck)

Pragmatic Value = How much does this advance the win condition?
  High for: Combo pieces, lethal damage, key threats
  Medium for: Mana acceleration, card selection
  Low for: Irrelevant cards
```

Agent types implement this differently:
- **LLMAgent**: Includes strategic context in prompt, lets GPT reason about value
- **ActiveInferenceModule**: Explicitly computes epistemic/pragmatic values, ranks actions by free energy
- **NeuralReasoner**: Learns value function from game outcomes (optional)

### Training

**Self-Play Loop** (AlphaZero-inspired):

1. **Self-Play Phase**: Two agent copies play games against each other
   - Collect (state, action, reward, outcome) tuples
   - Store in experience buffer

2. **Training Phase**: Update neural network on batch from buffer
   - Supervised learning on action selection
   - Value function learning (predict game outcome)
   - Policy learning (predict which action leads to best outcome)

3. **Evaluation Phase**: Trained agent vs. baseline/previous checkpoint
   - Win rate metric
   - Decide if new version is better

4. **Update Phase**: If better, new version becomes new baseline
   - Else, keep previous version

**Reward Function**:
- **Terminal rewards**: Win = +1, Loss = -1
- **Shaping rewards**:
  - Life delta: +0.01 per life gained / opponent life lost
  - Card advantage: +0.005 per net card advantage
  - Board presence: +0.005 per creature on battlefield
  - Action cost: -0.001 per action (encourage efficiency)

**Training Configuration** (in `src/training/self_play.py`):
- Number of parallel games: 64
- Experience buffer size: 100,000 transitions
- Batch size: 256
- Learning rate: 1e-4
- Epochs per iteration: 10

## How It Works: Example Scenario

**Scenario**: You're a Blue-Red control player with:
- In hand: Counterspell, Snapcaster Mage, Island, Mountain
- On battlefield: 2 Islands, 1 Mountain
- Opponent: Just cast a 4/4 creature

**LLMAgent Decision Process**:

1. Engine generates legal actions:
   - [CAST_SPELL: Counterspell, ACTIVATE_ABILITY: Island (tap for mana), PASS_PRIORITY, CONCEDE, ...]
   
2. Agent builds strategic context (via knowledge graph):
   - Recognizes opponent pattern as likely creature-based (midrange)
   - Detects no immediately available combos
   - Identifies Counterspell as best answer to creature threat
   - Checks if there's enough mana (yes: 3 available)
   
3. Agent computes active inference scores:
   - Counterspell: Pragmatic value = high (removes threat), Epistemic = low (no info)
   - Island tap: Pragmatic = low, Epistemic = none
   - Pass: Pragmatic = none, Epistemic = none
   
4. LLM selects: Counter the creature

5. State mutation:
   - Pay 1 Blue + 1 generic
   - Move Counterspell hand → stack
   - Mana pool updates
   - Opponent creature is countered (goes to graveyard)
   
6. Reward calculated:
   - +0.01 (removed threat)
   - Game continues

## Performance Characteristics

**Speed**:
- Legal action generation: 5-10ms (50-200 actions)
- SBA checking: 1-2ms
- Combo detection (KB query): 10-50ms
- Full LLM decision: 1-5s (includes API latency)

**Scalability**:
- Card database: 30,000+ Scryfall cards
- Combo database: 5,000+ known combos
- Supports 2-4 player games
- Self-play: 64 parallel games per iteration

**Accuracy**:
- Legal action generation: 100% (rules-compliant)
- Combo detection: 95%+ (known combos)
- Archetype inference: 70-80% (based on cards seen)

## Usage

### Run a Test Game

To verify the game engine works end-to-end:

```bash
python tests/test_game_engine.py
```

This runs a quick integration test that:
- Creates a 2-player game state with mock cards
- Generates legal actions
- Executes an action
- Checks state-based actions
- Verifies the engine doesn't crash

Example output:
```
============================================================
MTG Game Engine Integration Test
============================================================

✓ Game state initialized with 120 cards
  Players: Alice, Bob
  Starting life: 20

✓ Testing legal action generation...
  Legal actions for player_1: 15 total
    - play_land: 3
    - cast_spell: 8
    - activate_ability: 3
    - pass_priority: 1

✓ Testing action execution...
  Executing: play_land
  ✓ Action executed successfully

✓ Testing state-based actions...
  SBA events: 0

============================================================
✅ All tests passed!
============================================================
```

### Play a Full Game with Agents

```python
import asyncio
from src.agents.random_agent import RandomAgent
from src.orchestrator.game_runner import GameRunner, GameConfig

async def play_game():
    # Create agents
    agent1 = RandomAgent("player_1")
    agent2 = RandomAgent("player_2")
    
    # Create simple test decks
    deck_data = [
        {"name": "Plains", "type_line": "Land", "mana_cost": "", "oracle_text": ""},
        {"name": "Goblin Guide", "type_line": "Creature", "mana_cost": "{R}", 
         "power": "2", "toughness": "2", "oracle_text": "Haste"},
        # ... load more cards from Scryfall or use your own format
    ]
    
    decks = {
        "player_1": deck_data,
        "player_2": deck_data,
    }
    
    # Run game
    runner = GameRunner(GameConfig(format="standard", max_turns=50))
    result = await runner.run_game(
        agents={"player_1": agent1, "player_2": agent2},
        decks=decks,
    )
    
    print(f"Winner: {result.winner}")
    print(f"Turns: {result.turns}")

asyncio.run(play_game())
```



```python
import asyncio
from src.knowledge.knowledge_graph import MTGKnowledgeGraph

async def query_kg():
    kg = MTGKnowledgeGraph()
    
    # Find combos with a specific card
    combos = await kg.detect_available_combos(
        available_cards=["Splinter Twin", "Pestermite", "Mountain"]
    )
    print(f"Found {len(combos)} playable combos")
    
    # Infer opponent archetype
    cards_seen = ["Counterspell", "Snapcaster Mage", "Force of Will"]
    archetype = await kg.infer_archetype_from_cards(cards_seen)
    print(f"Opponent likely plays: {archetype}")

asyncio.run(query_kg())
```

### Train an Agent

```python
from src.training.self_play import SelfPlayTrainer, TrainingConfig

config = TrainingConfig(
    num_parallel_games=64,
    buffer_size=100_000,
    num_epochs=10,
    learning_rate=1e-4,
)

trainer = SelfPlayTrainer(config)
asyncio.run(trainer.train(num_iterations=1000))
```

## Recent Enhancements

### 1. Extended MTG Ontology (v1.1)

The knowledge graph now includes a comprehensive 7-layer ontology (~1000 lines):

- **Layer 1**: Core game concepts (Color, Mana, CardType, Zone, Phase)
- **Layer 2**: Game structure (Player, Turn, Stack, Zones as OWL instances)
- **Layer 3**: Ability system (Activated, Triggered, Static, Special abilities + 25 keywords)
- **Layer 4**: Game mechanics (Effects, Replacement, Continuous effects, Triggers, Layers system)
- **Layer 5**: Keywords (Flying, Menace, Vigilance, Deathtouch, Lifelink, Flash, etc.)
- **Layer 6**: Strategic knowledge (Combos, Synergies, Archetypes, Win conditions)
- **Layer 7**: Game state & belief tracking (Decklist, GameState, BeliefState with open/closed zones)

**Systematic Extension via OntologyExtender**:

For controlled, human-in-the-loop ontology evolution, see [docs/ONTOLOGY_EXTENSION_GUIDE.md](docs/ONTOLOGY_EXTENSION_GUIDE.md), which covers:
- OWL design patterns for MTG game rules
- SHACL validation for semantic consistency
- Extracting candidates from Comprehensive Rules (CR) text
- Multi-phase integration roadmap (Weeks 1-9+)
- Tool recommendations (OntologyExtender - MIT licensed)

**Status**: Ontology v1.1 ready. Phase 1-2 (validation & CR extraction) can begin on demand.

### 2. Decklist-Aware Opponent Modeling

The opponent model now uses **process-of-elimination** for exact library composition inference:

```python
# Initialize with opponent's known decklist (public information)
opponent = OpponentModel(
    opponent_id="opponent",
    kg=knowledge_graph,
    known_decklist={
        "Sol Ring": 1,
        "Counterspell": 3,
        "Blue Island": 4,
        # ... full 100-card Commander list
    }
)

# During game, opponent beliefs are computed exactly (not estimated)
threat_assessment = await opponent.get_threat_assessment()
# → probability_has_counterspell: 0.03 (exact, not heuristic 0.60)

# Get exactly what's remaining in opponent's library
remaining = opponent.get_library_composition_remaining()
# → {"Counterspell": 2, "Island": 4, ...}

# Predict draws over next N turns (hypergeometric distribution)
draw_probs = opponent.get_draw_probabilities(num_cards=1)
# → {"Counterspell": 0.033, "Island": 0.133, ...}
```

**Key advantages**:
- ✅ **Exact inference** when decklist is public (Commander, tournament Magic)
- ✅ **Hypergeometric distribution** for accurate draw prediction
- ✅ **Backward compatible** (falls back to heuristics if no decklist)
- ✅ **Fast computation** (arithmetic, no ML needed)

**Implementation details**:
- [src/agents/opponent_model.py](src/agents/opponent_model.py) - Updated with CardInformation tracking and process-of-elimination
- [docs/OPPONENT_MODELING_WITH_DECKLIST.md](docs/OPPONENT_MODELING_WITH_DECKLIST.md) - Complete guide with examples

## Benchmarks

The repo ships with a reproducible benchmark harness ([src/training/benchmark_suite.py](src/training/benchmark_suite.py)) that pits any set of agents against each other across six standard archetype decks ([src/training/archetype_decks.py](src/training/archetype_decks.py)): mono-red aggro, mono-blue control, mono-green ramp, mono-white weenie, mono-black midrange, and Izzet burn. It records per-game CSV (`games.csv`) plus aggregate `summary.json` with win-rates, average decision-time per agent, and incremental ELO updates.

### Run the default benchmark locally

```bash
# Random vs Heuristic on all six archetype pairs, 4 games each, 4 in parallel
python -m src.training.benchmark_suite \
    --games-per-match 4 \
    --parallel 4 \
    --output-dir runs/bench-baseline
```

Add the LLM baseline (requires Ollama running with `gemma4:e2b` — see [docs/OLLAMA_SETUP.md](docs/OLLAMA_SETUP.md)):

```bash
python -m src.training.benchmark_suite --include-llm --parallel 2
```

### Where to run them

| Workload | Hardware target | Wall-clock (rough) |
|---|---|---|
| Random + Heuristic baselines, 6 archetypes × 6 pairings × 10 games | **Laptop / desktop CPU** (1–4 cores) | minutes |
| LLM baseline (Ollama `gemma4:e2b`), 100 games | Single workstation with **8+ GB VRAM** *or* CPU-only Ollama | tens of minutes |
| Full self-play loop ([src/training/self_play.py](src/training/self_play.py)) for 1k iterations | **GPU server** (≥1× consumer GPU, ≥32 GB RAM) | hours–days |
| World-model + neural-reasoner training on stored trajectories | **GPU server** with PyTorch (CUDA) | hours |

In short: **the benchmark suite and all symbolic agents run fine on this laptop today.** Anything that involves repeated LLM calls, JEPA training, or multi-thousand-game self-play should go on a server with a GPU and enough RAM to keep the trajectory store in memory. A single A6000/4090-class GPU is plenty for everything described in the paper; a small node with one such GPU + 64 GB RAM is the recommended setup.

### Default LLM

The default Ollama model has been switched to **`gemma4:e2b`** (Gemma 4 "edge 2B" variant) across [src/agents/llm_agent.py](src/agents/llm_agent.py), [src/agents/llm_fusion_agent.py](src/agents/llm_fusion_agent.py), [src/engine/llm_orchestration.py](src/engine/llm_orchestration.py), and the example scripts. It gives noticeably stronger reasoning than the previous tag at the same VRAM budget. Pull it with `ollama pull gemma4:e2b`.

## Configuration

Set environment variables in `.env`:

```env
# LLM Configuration (for LLMAgent)
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=secure_password

# Game Configuration
DEFAULT_FORMAT=commander
DEFAULT_LIFE_TOTAL=40
MAX_TURNS_PER_GAME=100

# Training Configuration
TRAINING_BATCH_SIZE=256
TRAINING_LEARNING_RATE=0.0001
TRAINING_BUFFER_SIZE=100000
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

### Performance Profiling

```bash
python -m cProfile -s cumulative -m src.orchestrator.game_runner
```

## API Integration

### Scryfall

Async client with rate limiting:

```python
from src.integrations.scryfall import ScryfallClient

async with ScryfallClient() as client:
    card = await client.get_card_by_name("Black Lotus")
    rulings = await client.get_rulings("Counterspell")
    cards = await client.search_cards("o:flying f:vintage")
```

### Commander Spellbook

Fetch combo database:

```python
from src.knowledge.combo_database import fetch_all_combos

combos = await fetch_all_combos()
# Returns: [(piece1, piece2, ...), ...]
```

## Deployment

### Docker Deployment

```bash
docker-compose up
```

This starts Neo4j with the `n10s` plugin pre-loaded. See [docker-compose.yml](docker-compose.yml) for the full stack and [docker-compose.remote.yml](docker-compose.remote.yml) for remote/Ollama-host configurations.

## Citation

If you use this codebase or build on its design, please cite:

```bibtex
@techreport{neuburger2026oppositionagents,
  author      = {Neub\"urger, Felix and Neub\"urger, Jonas},
  title       = {Opposition Agents Playing Magic: The Gathering --
                 Knowledge-Graph-Grounded JEPA World Models and
                 Active-Inference LLM Agents},
  institution = {Independent Research},
  year        = {2026},
  url         = {https://github.com/cuteredpwnda/opposition-agents-playing-mtg}
}
```

## References & Acknowledgments

This project stands on the shoulders of excellent open-source work. See [PLAN.md Section 15](PLAN.md#section-15-related-work-and-references) for detailed references to:

### Existing MTG Engines & Libraries

- **open-mtg** (MIT): Game loop pattern, MCTS framework
- **mtg-python-engine** (MIT): Stack resolution, state-based actions
- **Forge** (GPL-3.0): Behavioral reference for rules correctness
- **Argentum** (MIT): Immutable state pattern

### Agent & AI Frameworks

- **mtg-player** (MIT): Agent architecture, tool-calling pattern
- **LangChain** (MIT): Prompt engineering, tool integration, LLM abstraction
- **LangGraph** (MIT): Graph-based orchestration
- **Scrython** (MIT): Scryfall API client

### Academic Work

- K. Friston et al., "The Free Energy Principle for Action and Perception" — Active Inference framework
- D. Silver et al., "Mastering the game of Go without human knowledge" — AlphaZero self-play method
- A. Vaswani et al., "Attention Is All You Need" — Transformer architecture

For full references including ontology sources, graph databases, and academic papers, see [PLAN.md](PLAN.md).

## License

MIT License. See LICENSE file.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## Authors

- **Felix Neubürger** — design, world-model, knowledge-graph, agents.
- **Jonas Neubürger** ([@cuteredpwnda](https://github.com/cuteredpwnda)) — engine, training infrastructure, ontology.

Project repository: <https://github.com/cuteredpwnda/opposition-agents-playing-mtg>

## Known Limitations

- No network play (local game simulation only)
- Limited CR coverage (basic spells, creatures, simple abilities)
- No GUI (CLI-based or API-based)
- Tournament rules not implemented (no mulligan management, etc.)

## Roadmap

- [ ] Full CR implementation (all ability types)
- [ ] Better neural network architecture
- [ ] Multi-game tournament support
- [ ] Web UI for visualization
- [ ] Performance optimization (Cython compilation)
- [ ] Distributed training across GPUs
- [ ] Integration with MTGAHelper-style telemetry

## Contact

For questions, open an issue or contact the maintainers.

---

**Status**: Experimental. Use for research/education purposes.
