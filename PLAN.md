# Opposition Agents Playing Magic: The Gathering — Master Plan

## Table of Contents

1. [Vision & Goals](#1-vision--goals)
2. [Architecture Overview](#2-architecture-overview)
3. [MTG Ontology & Knowledge Graph](#3-mtg-ontology--knowledge-graph)
4. [Game Engine & Rules Engine](#4-game-engine--rules-engine)
5. [LangGraph State Machine](#5-langgraph-state-machine)
6. [Agent Architecture](#6-agent-architecture)
7. [Active Inference Framework](#7-active-inference-framework)
8. [Neural Reasoning Module](#8-neural-reasoning-module)
9. [Opponent Modeling & Metagame Awareness](#9-opponent-modeling--metagame-awareness)
10. [Judge Agent & Rules Arbitration](#10-judge-agent--rules-arbitration)
11. [External API Integrations](#11-external-api-integrations)
12. [Reinforcement Learning & Training Pipeline](#12-reinforcement-learning--training-pipeline)
13. [Visualization & UI](#13-visualization--ui)
14. [Human Player Integration](#14-human-player-integration)
15. [Supported Formats](#15-supported-formats)
16. [Project Structure](#16-project-structure)
17. [Technology Stack](#17-technology-stack)
18. [Implementation Phases](#18-implementation-phases)
19. [Open Questions & Risks](#19-open-questions--risks)

---

## 1. Vision & Goals

Build a Python-based agentic framework where LLM-powered agents play full games of Magic: The Gathering against each other (and optionally against human players) in Standard (1v1) and EDH/Commander (4-player) formats.

### Core Goals

- **Playable game engine** — Enforce the comprehensive rules of MTG including phases, priority, the stack, combat, mana, zones, triggered/activated abilities.
- **Intelligent agents** — LLM agents that understand card text, evaluate board states, plan multi-turn strategies, and adapt to opponents.
- **MTG Ontology** — A structured knowledge graph encoding card relationships, combos, synergies, archetypes, and interaction patterns to fuel strategic reasoning.
- **Active inference** — Agents that maintain probabilistic beliefs about hidden information (opponent hands, library contents) and act to minimize surprise/free energy.
- **Neural reasoning** — Graph neural networks and transformer-based modules operating over the knowledge graph and game state for deep strategic evaluation.
- **Opponent modeling** — Predict what the opponent is likely holding and planning based on their deck archetype, play patterns, and open mana.
- **Judge agent** — An LLM-based arbiter that resolves ambiguous rules interactions, backed by Scryfall rulings and the Comprehensive Rules.
- **Learning** — Agents that improve over time via reinforcement learning, self-play, and nested/hierarchical learning.
- **Visualization** — A real-time visual interface showing the board state with Scryfall card images.
- **Human-in-the-loop** — A human player can take one of the seats at the table.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          VISUALIZATION LAYER                           │
│   Streamlit / PyGame / Web UI  ←  Scryfall card images                 │
│   Human Player Interface (WebSocket / CLI)                             │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                       LANGGRAPH GAME ORCHESTRATOR                      │
│                                                                        │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│   │ Agent 1  │  │ Agent 2  │  │ Agent 3  │  │ Agent 4  │  (or Human)  │
│   │ (LLM +   │  │ (LLM +   │  │ (LLM +   │  │ (LLM +   │              │
│   │  Active   │  │  Active   │  │  Active   │  │  Active   │              │
│   │ Inference)│  │ Inference)│  │ Inference)│  │ Inference)│              │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│        │              │              │              │                    │
│   ┌────▼──────────────▼──────────────▼──────────────▼────┐              │
│   │              NEURAL REASONING MODULE                  │              │
│   │   GNN over Knowledge Graph + Board State Encoder      │              │
│   │   Strategic Evaluation · Combo Detection              │              │
│   └────────────────────────┬─────────────────────────────┘              │
│                            │                                            │
│   ┌────────────────────────▼─────────────────────────────┐              │
│   │            MTG KNOWLEDGE GRAPH (Neo4j)                │              │
│   │   n10s (OWL ontology import) · APOC (graph algos)     │              │
│   │   Cards · Combos · Synergies · Archetypes · Rulings   │              │
│   │   Cypher + vector search · Community detection         │              │
│   │   KGPlatform tooling for KG build + HITL extension    │              │
│   └──────────────────────────────────────────────────────┘              │
│                                                                        │
│   ┌──────────────────────────────────────────────────────┐              │
│   │              JUDGE AGENT (Rules Arbiter)              │              │
│   │   Comprehensive Rules DB · Scryfall Rulings API       │              │
│   └──────────────────────────────────────────────────────┘              │
│                                                                        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                         GAME ENGINE (Core)                              │
│   Zones · Stack · Priority · Phases · Combat · Mana Pool               │
│   State Transitions · Turn Structure · Triggered Abilities             │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                        EXTERNAL DATA LAYER                              │
│   Scryfall API (cards, images, rulings)                                │
│   Moxfield / Archidekt (decklists)                                     │
│   Local Card DB Cache (SQLite / JSON)                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. MTG Ontology & Knowledge Graph

This is the **brain** that gives agents strategic depth beyond raw LLM reasoning. It encodes the *relationships* between cards, mechanics, and strategies that experienced players internalize over years.

### 3.1 Ontology Schema

```
Card
  ├── name: str
  ├── mana_cost: ManaCost
  ├── type_line: str  (Creature, Instant, Sorcery, Enchantment, Artifact, Planeswalker, Land, Battle)
  ├── subtypes: [str]  (Elf, Wizard, Aura, Equipment, ...)
  ├── oracle_text: str
  ├── keywords: [Keyword]  (Flying, Trample, Flash, Counterspell-ability, ...)
  ├── power / toughness / loyalty / defense
  ├── color_identity: [Color]
  ├── legalities: {format: legality}
  └── image_uris: {size: url}

Keyword (enum/ontology class)
  ├── Evergreen: Flying, First Strike, Deathtouch, Trample, Haste, Flash, Hexproof, ...
  ├── Ability-Words: Landfall, Constellation, Revolt, ...
  └── Mechanic-Specific: Cascade, Storm, Infect, Annihilator, ...

Combo
  ├── cards: [Card]  (2+ cards involved)
  ├── description: str  ("Infinite mana with Devoted Druid + Vizier of Remedies")
  ├── result: [Effect]  (InfiniteMana, InfiniteDamage, InfiniteTokens, ...)
  ├── setup_requirements: str  (cards on battlefield, mana open, etc.)
  └── fragility: float  (how easy it is to disrupt)

Synergy
  ├── cards: [Card, Card]
  ├── strength: float  (0.0–1.0)
  ├── description: str  ("Rhystic Study draws cards when opponents don't pay 1")
  └── synergy_type: enum  (Tribal, Mechanic, Theme, Value, ...)

Archetype
  ├── name: str  ("Blue-White Control", "Mono-Green Stompy", "Grixis Storm", ...)
  ├── format: Format
  ├── color_identity: [Color]
  ├── strategy: enum  (Aggro, Midrange, Control, Combo, Stax, Tempo, ...)
  ├── key_cards: [Card]
  ├── typical_interaction_suite: [Card]  (counterspells, removal, boardwipes)
  ├── win_conditions: [Combo | Card]
  └── weaknesses: [Archetype | Strategy]

CounterPlay
  ├── threat: Card | Combo | Strategy
  ├── answer: Card | Strategy
  ├── effectiveness: float
  └── timing: Phase  (when to deploy the answer)

Interaction
  ├── card_a: Card
  ├── card_b: Card
  ├── interaction_type: enum  (Counters, Removes, Enables, Triggers, Buffs, Taxes, ...)
  └── description: str
```

### 3.2 Knowledge Graph Implementation

```python
# Using NetworkX for the graph + optional Neo4j for persistence at scale
import networkx as nx

class MTGKnowledgeGraph:
    """
    A directed multigraph encoding MTG card relationships.
    
    Node types: Card, Combo, Archetype, Keyword, Effect
    Edge types: PART_OF_COMBO, SYNERGIZES_WITH, COUNTERS, ENABLES, 
                HAS_KEYWORD, BELONGS_TO_ARCHETYPE, ANSWERS, TRIGGERS
    """
    def __init__(self):
        self.graph = nx.MultiDiGraph()
    
    def add_card(self, card_data: dict): ...
    def add_combo(self, combo_id: str, cards: list[str], result: str): ...
    def add_synergy(self, card_a: str, card_b: str, strength: float): ...
    def add_archetype(self, archetype: dict): ...
    
    # Query methods
    def get_combos_containing(self, card_name: str) -> list[Combo]: ...
    def get_synergies_for(self, card_name: str) -> list[Synergy]: ...
    def get_answers_to(self, card_name: str) -> list[Card]: ...
    def get_archetype_signature_cards(self, archetype: str) -> list[Card]: ...
    def infer_archetype_from_cards(self, cards: list[str]) -> Archetype: ...
    def get_likely_cards_in_hand(self, archetype: str, seen_cards: list[str]) -> list[Card]: ...
```

### 3.3 Populating the Knowledge Graph

| Source | What it provides |
|--------|-----------------|
| **Scryfall Bulk Data** | Every card ever printed — oracle text, keywords, types, colors, legalities |
| **EDHREC scraping** | Popular combos, synergies, staple cards per archetype, salt scores |
| **MTGGoldfish / MTGTop8** | Metagame data, tier lists, decklists, archetype classification |
| **Commander Spellbook** | Curated combo database with 10k+ combos, descriptions, results |
| **LLM-assisted extraction** | Parse oracle text to extract structured interaction edges (e.g., "Whenever a creature enters → trigger") |
| **Manual curation** | High-value strategic knowledge, counter-play heuristics |

### 3.4 Graph Embedding

Generate vector embeddings of subgraphs for fast similarity search and neural reasoning:

```python
# Node2Vec or GraphSAGE embeddings
from torch_geometric.nn import SAGEConv

class CardGraphEmbedder(torch.nn.Module):
    """Produces d-dimensional embeddings for each card node,
    capturing combo/synergy/archetype neighborhood structure."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
```

---

## 4. Game Engine & Rules Engine

### 4.1 Core Concepts

The game engine is the source of truth for game state. It must model:

| Concept | Details |
|---------|---------|
| **Zones** | Library, Hand, Battlefield, Graveyard, Exile, Stack, Command Zone |
| **Turn Structure** | Untap → Upkeep → Draw → Main 1 → Combat (Begin → Attackers → Blockers → Damage → End) → Main 2 → End (End Step → Cleanup) |
| **Priority** | Active player gets priority first; each player must pass before stack resolves. In multiplayer, priority passes in turn order (APNAP). |
| **The Stack** | LIFO stack for spells and abilities. Players can respond at instant speed. |
| **Mana System** | Mana pool with WUBRG + colorless + generic. Mana abilities don't use the stack. |
| **Combat** | Attacker/blocker declaration, damage assignment order, first strike / double strike, trample. |
| **State-Based Actions** | Checked whenever a player would receive priority: 0 toughness → dies, 0 life → loses, legend rule, etc. |
| **Triggered Abilities** | "When/Whenever/At" — go on stack next time a player would receive priority. |
| **Replacement Effects** | "If ... would ... instead" — modify events before they happen. |
| **Continuous Effects** | Layers system (1–7) for applying static abilities in correct order. |
| **Commander-specific** | Command zone, commander tax (+{2} per cast), commander damage (21 to lose), color identity restrictions. |

### 4.2 Game State Model

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid

class Zone(Enum):
    LIBRARY = "library"
    HAND = "hand"
    BATTLEFIELD = "battlefield"
    GRAVEYARD = "graveyard"
    EXILE = "exile"
    STACK = "stack"
    COMMAND_ZONE = "command_zone"

class Phase(Enum):
    UNTAP = "untap"
    UPKEEP = "upkeep"
    DRAW = "draw"
    MAIN_1 = "main_1"
    COMBAT_BEGIN = "combat_begin"
    COMBAT_ATTACKERS = "combat_attackers"
    COMBAT_BLOCKERS = "combat_blockers"
    COMBAT_DAMAGE = "combat_damage"
    COMBAT_END = "combat_end"
    MAIN_2 = "main_2"
    END_STEP = "end_step"
    CLEANUP = "cleanup"

@dataclass
class CardInstance:
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    card_data: dict = field(default_factory=dict)  # Scryfall card object
    zone: Zone = Zone.LIBRARY
    owner_id: str = ""
    controller_id: str = ""
    tapped: bool = False
    counters: dict = field(default_factory=dict)
    attached_to: Optional[str] = None
    damage_marked: int = 0
    summoning_sick: bool = True
    face_down: bool = False

@dataclass
class PlayerState:
    player_id: str
    name: str
    life_total: int = 20  # 40 for Commander
    mana_pool: dict = field(default_factory=lambda: {"W":0,"U":0,"B":0,"R":0,"G":0,"C":0})
    commander_tax: int = 0
    commander_damage_received: dict = field(default_factory=dict)  # {commander_id: damage}
    has_drawn_for_turn: bool = False
    land_plays_remaining: int = 1
    passed_priority: bool = False
    is_human: bool = False

@dataclass
class GameState:
    game_id: str
    format: str  # "standard" | "commander"
    turn_number: int = 0
    active_player_index: int = 0
    priority_player_index: int = 0
    phase: Phase = Phase.UNTAP
    players: list[PlayerState] = field(default_factory=list)
    cards: list[CardInstance] = field(default_factory=list)  # Every card in the game
    stack: list = field(default_factory=list)  # Stack items (spells/abilities)
    combat_state: Optional[dict] = None
    game_log: list[str] = field(default_factory=list)
    game_over: bool = False
    winner: Optional[str] = None
```

### 4.3 Rules Engine

```python
class RulesEngine:
    """Validates and executes game actions according to MTG Comprehensive Rules."""
    
    def get_legal_actions(self, game_state: GameState, player_id: str) -> list[Action]:
        """Return all legal actions for a player given current state and priority."""
        ...
    
    def execute_action(self, game_state: GameState, action: Action) -> GameState:
        """Apply an action, update state, check state-based actions, queue triggers."""
        ...
    
    def check_state_based_actions(self, game_state: GameState) -> list[Event]:
        """CR 704 — creatures with 0 toughness die, players at 0 life lose, etc."""
        ...
    
    def resolve_top_of_stack(self, game_state: GameState) -> GameState:
        """Resolve the topmost item on the stack."""
        ...
    
    def advance_phase(self, game_state: GameState) -> GameState:
        """Move to next phase/step, handling turn-based actions."""
        ...
```

### 4.4 Reusable Open-Source Components

Rather than building the entire engine from scratch, we can **accelerate significantly** by adapting existing MIT-licensed Python projects. Here's a concrete mapping of what to take from each:

| Component | Source Project | What to Adapt |
|-----------|---------------|---------------|
| **Game loop** (`get_moves()` / `make_move()`) | [open-mtg](https://github.com/hlynurd/open-mtg) | Core game loop pattern, phase enum, and the stateless `get_moves`→`make_move` interface that plugs cleanly into MCTS or RL agents |
| **Combat system** | [open-mtg](https://github.com/hlynurd/open-mtg) | Attacker/blocker enumeration, damage assignment ordering (CR 509/510), trample, first strike |
| **The Stack** | [mtg-python-engine](https://github.com/wanqizhu/mtg-python-engine) | `play.Play()` objects on stack, LIFO resolution, target legality checking before resolution, spell fizzling |
| **Triggered abilities** | [mtg-python-engine](https://github.com/wanqizhu/mtg-python-engine) | `triggers.triggerConditions` (onETB, onAttack, onControllerLifeGain, etc.), intervening-if clauses |
| **Activated abilities** | [mtg-python-engine](https://github.com/wanqizhu/mtg-python-engine) | `abilities.ActivatedAbility()` with cost parsing, tap symbol, `can_activate()` checks |
| **Static/continuous effects** | [mtg-python-engine](https://github.com/wanqizhu/mtg-python-engine) | `add_effect()` / `add_static_effect()` with toggle functions, effect expiration, power/toughness modification |
| **State-based actions** | [mtg-python-engine](https://github.com/wanqizhu/mtg-python-engine) | `check_state_based_actions()` — creature death, player loss, legend rule |
| **Game state rollback** | [mtg-python-engine](https://github.com/wanqizhu/mtg-python-engine) | Deep-copy based state snapshots for rewinding illegal actions |
| **MCTS AI baseline** | [open-mtg](https://github.com/hlynurd/open-mtg) | `mcts.py` — Monte Carlo Tree Search reference implementation for MTG |
| **Agent architecture** | [mtg-player](https://github.com/theRealMarkCastillo/mtg-player) | LLM tool-calling pattern (get_game_state, get_legal_actions, execute_action), chain-of-thought prompts, heuristic fallback, multi-provider LLM support |
| **Pydantic game models** | [mtg-player](https://github.com/theRealMarkCastillo/mtg-player) | Player state, card models, game state representation using Pydantic v2 |
| **Opponent modeling** | [mtg-player](https://github.com/theRealMarkCastillo/mtg-player) | `OpponentModelingTool` for tracking opponent strategy and threat assessment |
| **Scryfall integration** | [Scrython](https://github.com/NandaScott/Scrython) | Use directly via `pip install scrython` — card lookup, bulk download, rulings, rate limiting, caching |

> **Note on Forge/XMage**: These Java engines (GPL-3.0) have the most complete rules implementations (20k+ cards). We cannot directly port code due to GPL licensing, but they serve as an invaluable **correctness reference** when implementing complex interactions like layer ordering, replacement effect chains, or unusual triggered ability interactions.

---

## 5. LangGraph State Machine

LangGraph models the game as a **cyclic, stateful graph** where nodes represent decision points and edges represent game transitions.

### 5.1 Why LangGraph?

- **Built for cycles** — MTG has deep recursion (stack responses, triggered ability chains).
- **State checkpointing** — Persist and resume games; roll back for training.
- **Human-in-the-loop** — Native `interrupt()` for human player turns.
- **Multi-agent** — First-class support for multiple agents sharing state.
- **Tool calling** — Agents can call Scryfall, knowledge graph, and judge tools.

### 5.2 Graph Topology

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

def build_game_graph(num_players: int):
    graph = StateGraph(GameState)
    
    # Phase nodes
    graph.add_node("untap_step", untap_step)
    graph.add_node("upkeep_step", upkeep_step)
    graph.add_node("draw_step", draw_step)
    graph.add_node("main_phase", main_phase)
    graph.add_node("combat_phase", combat_phase)
    graph.add_node("main_phase_2", main_phase_2)
    graph.add_node("end_step", end_step)
    graph.add_node("cleanup_step", cleanup_step)
    
    # Decision nodes
    graph.add_node("priority_loop", priority_loop)       # Player gets priority, decides action
    graph.add_node("resolve_stack", resolve_stack)        # Resolve top of stack
    graph.add_node("agent_decision", agent_decision)      # LLM agent picks action
    graph.add_node("human_decision", human_decision)      # Human picks action (with interrupt)
    graph.add_node("judge_ruling", judge_ruling)          # Invoked when rules are ambiguous
    graph.add_node("check_game_over", check_game_over)    # Check win/loss conditions
    
    # Turn structure edges
    graph.add_edge(START, "untap_step")
    graph.add_edge("untap_step", "upkeep_step")
    graph.add_edge("upkeep_step", "priority_loop")
    graph.add_edge("draw_step", "main_phase")
    graph.add_edge("main_phase", "priority_loop")
    graph.add_edge("combat_phase", "priority_loop")
    graph.add_edge("main_phase_2", "priority_loop")
    graph.add_edge("end_step", "priority_loop")
    graph.add_edge("cleanup_step", "check_game_over")
    
    # Priority loop — the heart of the game
    graph.add_conditional_edges("priority_loop", route_priority, {
        "agent_turn": "agent_decision",
        "human_turn": "human_decision",
        "all_passed": "resolve_stack",
        "stack_empty": "advance_phase",
    })
    
    graph.add_conditional_edges("agent_decision", route_after_action, {
        "priority_loop": "priority_loop",
        "judge_needed": "judge_ruling",
    })
    
    graph.add_edge("judge_ruling", "priority_loop")
    graph.add_edge("resolve_stack", "priority_loop")  # After resolution, AP gets priority again
    
    graph.add_conditional_edges("check_game_over", lambda s: "end" if s.game_over else "next_turn", {
        "end": END,
        "next_turn": "untap_step",
    })
    
    return graph.compile(checkpointer=SqliteSaver("games.db"))
```

### 5.3 Priority Loop Detail

```
   ┌─────────────────────────────────┐
   │        PRIORITY_LOOP            │
   │  Who has priority?              │
   │  Is the stack empty?            │
   │  Have all players passed?       │
   └─────┬──────┬──────┬──────┬─────┘
         │      │      │      │
    Agent │  Human│  All  │  Stack│
    Turn  │  Turn │ Passed│ Empty │
         ▼      ▼      ▼      ▼
   ┌─────┐ ┌─────┐ ┌──────┐ ┌──────────┐
   │Agent│ │Human│ │Resolv│ │Advance   │
   │Decis│ │Input│ │Stack │ │Phase     │
   └──┬──┘ └──┬──┘ └──┬───┘ └──────────┘
      │       │       │
      └───────┴───────┘
              │
              ▼
       PRIORITY_LOOP (repeat)
```

---

## 6. Agent Architecture

Each agent is a **composite system** combining an LLM backbone with specialized reasoning modules.

### 6.1 Agent Components

```
┌─────────────────────────────────────────────────────┐
│                  MTG AGENT                          │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  LLM Backbone (GPT-4 / Claude / Llama)       │  │
│  │  - Natural language understanding of cards     │  │
│  │  - Strategic reasoning in natural language     │  │
│  │  - Tool calling (Scryfall, KG, Judge)         │  │
│  └────────────────────┬──────────────────────────┘  │
│                       │                             │
│  ┌────────────────────▼──────────────────────────┐  │
│  │  Active Inference Module                      │  │
│  │  - Belief state over hidden info              │  │
│  │  - Expected free energy minimization          │  │
│  │  - Epistemic & pragmatic action selection     │  │
│  └────────────────────┬──────────────────────────┘  │
│                       │                             │
│  ┌────────────────────▼──────────────────────────┐  │
│  │  Neural Reasoning Module                      │  │
│  │  - GNN over knowledge graph subgraph          │  │
│  │  - Board state encoder (transformer)          │  │
│  │  - Action value estimator                     │  │
│  └────────────────────┬──────────────────────────┘  │
│                       │                             │
│  ┌────────────────────▼──────────────────────────┐  │
│  │  Opponent Model                               │  │
│  │  - Archetype classifier                       │  │
│  │  - Hand probability estimator                 │  │
│  │  - Behavioral pattern tracker                 │  │
│  └────────────────────┬──────────────────────────┘  │
│                       │                             │
│  ┌────────────────────▼──────────────────────────┐  │
│  │  Memory & Context                             │  │
│  │  - Cards seen this game                       │  │
│  │  - Graveyard / exile tracking                 │  │
│  │  - Previous games (long-term memory)          │  │
│  │  - Known deck composition (if public info)    │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 6.2 Agent Decision Pipeline

```python
class MTGAgent:
    def __init__(self, player_id, llm, knowledge_graph, active_inference, neural_reasoner):
        self.player_id = player_id
        self.llm = llm  # LangChain ChatModel
        self.kg = knowledge_graph
        self.active_inference = active_inference
        self.neural_reasoner = neural_reasoner
        self.opponent_models = {}  # {player_id: OpponentModel}
        self.memory = AgentMemory()
    
    async def decide_action(self, game_state: GameState, legal_actions: list[Action]) -> Action:
        # 1. Update beliefs about hidden information
        beliefs = self.active_inference.update_beliefs(game_state, self.memory)
        
        # 2. Query knowledge graph for relevant context
        hand_cards = self.get_hand(game_state)
        kg_context = self.kg.get_strategic_context(
            hand=hand_cards,
            battlefield=self.get_battlefield(game_state),
            opponents=self.get_opponent_info(game_state),
        )
        # e.g., "You have Thoracle + Demonic Consultation in hand = instant win combo"
        # e.g., "Opponent is playing UW Control, likely has Counterspell/Force of Will"
        
        # 3. Neural evaluation of candidate actions
        action_scores = self.neural_reasoner.evaluate_actions(
            game_state, legal_actions, beliefs, kg_context
        )
        
        # 4. LLM makes final decision with all context
        prompt = self.build_decision_prompt(
            game_state, legal_actions, beliefs, kg_context, action_scores
        )
        decision = await self.llm.ainvoke(prompt)
        
        # 5. Parse and validate the chosen action
        chosen_action = self.parse_action(decision, legal_actions)
        
        # 6. Update opponent models based on what happened
        self.update_opponent_models(game_state)
        
        return chosen_action
```

### 6.3 LangChain Tool Definitions for Agents

```python
from langchain.tools import tool

@tool
def query_knowledge_graph(query: str) -> str:
    """Query the MTG knowledge graph for combos, synergies, counter-play.
    Example: 'What combos include Thassa\'s Oracle?' or 'What answers Rhystic Study?'"""
    ...

@tool
def lookup_card_rulings(card_name: str) -> str:
    """Fetch official card rulings from Scryfall for a specific card."""
    ...

@tool 
def estimate_opponent_hand(player_id: str) -> str:
    """Based on the opponent's archetype and play pattern, estimate likely cards in hand."""
    ...

@tool
def evaluate_board_position() -> str:
    """Get a neural evaluation of the current board position (advantage/disadvantage)."""
    ...

@tool
def call_judge(situation: str) -> str:
    """Ask the judge agent for a ruling on a complex rules interaction."""
    ...

@tool
def check_combat_math(attackers: list, blockers: list) -> str:
    """Calculate combat outcomes given attacker and blocker assignments."""
    ...
```

---

## 7. Active Inference Framework

Active inference (from Karl Friston's Free Energy Principle) provides a principled framework for **decision-making under uncertainty** — perfect for MTG where agents have incomplete information.

### 7.1 Core Concepts Applied to MTG

| Active Inference Concept | MTG Application |
|--------------------------|-----------------|
| **Generative Model** | Agent's internal model of how the game works — what cards exist, how opponents play, probability distributions over hidden zones. |
| **Hidden States** | Opponent hands, library ordering, face-down cards, unknown deck contents. |
| **Observations** | Cards played, mana tapped, combat declarations, cards revealed, graveyard contents. |
| **Beliefs (Posterior)** | Updated probability distribution over what opponents hold, what they'll draw, and what they're planning. |
| **Expected Free Energy (EFE)** | Combines **epistemic value** (information gain — "if I Thoughtseize them, I learn their hand") with **pragmatic value** (goal achievement — "if I attack, I deal damage"). |
| **Policy Selection** | Choose the action (policy) that minimizes expected free energy — balancing exploration (learning opponent's plan) with exploitation (advancing your own win condition). |

### 7.2 Implementation

```python
import numpy as np
from scipy.special import softmax

class ActiveInferenceModule:
    """
    Maintains probabilistic beliefs about hidden game information
    and selects actions that minimize expected free energy.
    """
    
    def __init__(self, knowledge_graph: MTGKnowledgeGraph):
        self.kg = knowledge_graph
        self.beliefs = {}  # Per-opponent belief states
    
    def initialize_beliefs(self, opponent_id: str, deck_archetype: str):
        """Initialize prior beliefs based on known deck archetype."""
        archetype_cards = self.kg.get_archetype_signature_cards(deck_archetype)
        # P(card in hand) ~ prior from archetype frequency data
        self.beliefs[opponent_id] = {
            "hand_distribution": self._build_hand_prior(archetype_cards),
            "deck_composition": self._build_deck_prior(deck_archetype),
            "strategy_intent": self._build_strategy_prior(deck_archetype),
        }
    
    def update_beliefs(self, observation: Observation) -> dict:
        """
        Bayesian update of beliefs given new game observations.
        
        Observations include:
        - Cards played (now we know they had it, remove from hand distro)
        - Mana left open (blue mana open → higher P(counterspell))
        - Cards drawn (hand size changed)
        - Cards revealed (Thoughtseize, Gitaxian Probe effects)
        - Tutor usage (they searched for something specific)
        """
        for opponent_id, belief in self.beliefs.items():
            # Bayesian update
            likelihood = self._compute_likelihood(observation, opponent_id)
            belief["hand_distribution"] = self._bayesian_update(
                belief["hand_distribution"], likelihood
            )
            # Update strategy intent based on observed play pattern
            belief["strategy_intent"] = self._update_strategy_intent(
                belief["strategy_intent"], observation
            )
        return self.beliefs
    
    def compute_expected_free_energy(self, action: Action, game_state: GameState) -> float:
        """
        G(π) = E_Q[log Q(s) - log P(o,s|π)]
        
        Decomposes into:
        - Epistemic value: How much will this action reduce uncertainty?
        - Pragmatic value: How much does this action advance winning?
        """
        epistemic_value = self._compute_epistemic_value(action, game_state)
        pragmatic_value = self._compute_pragmatic_value(action, game_state)
        return -(epistemic_value + pragmatic_value)  # Negative because we minimize EFE
    
    def _compute_epistemic_value(self, action: Action, game_state: GameState) -> float:
        """Information gain from an action.
        
        High epistemic value actions:
        - Thoughtseize/Duress (reveals opponent's hand)
        - Gitaxian Probe (look + draw)
        - Attacking into unknown blockers (reveals defensive posture)
        - Playing a threat to see if opponent has a counterspell
        """
        if action.reveals_information:
            return self._expected_information_gain(action)
        return 0.0
    
    def _compute_pragmatic_value(self, action: Action, game_state: GameState) -> float:
        """Goal-directed value. How much does this advance the win condition?
        
        High pragmatic value:
        - Playing a combo piece when the other piece is in hand
        - Dealing lethal damage
        - Removing opponent's key threat
        """
        ...
    
    def rank_actions(self, actions: list[Action], game_state: GameState) -> list[tuple[Action, float]]:
        """Rank actions by expected free energy (lower = better)."""
        scored = [(a, self.compute_expected_free_energy(a, game_state)) for a in actions]
        scored.sort(key=lambda x: x[1])
        return scored
    
    def should_hold_open_mana(self, game_state: GameState) -> dict:
        """Decide whether to hold mana open for reactive plays.
        
        E.g., if you have Counterspell in hand and opponent is likely to play
        a high-value spell this turn, hold 2 blue mana open.
        """
        ...
```

### 7.3 Belief Visualization

```
Opponent "Alice" (playing Azorius Control):
  Hand (estimated 5 cards):
    Counterspell:        ████████░░  78%  (2 blue mana open, control archetype)
    Wrath of God:        ██████░░░░  55%  (hasn't played one yet, 4+ creatures out)
    Teferi, Hero:        ███░░░░░░░  30%  (signature card, hasn't appeared)
    Land:                ████████░░  80%  (typically 1-2 lands in hand mid-game)
    Unknown:             ██████████ 100%  (1 unaccounted card)
  
  Strategy Intent:
    Holding for value:   ████████░░  85%  (passing turns, leaving mana up)
    Setting up combo:    ██░░░░░░░░  15%
```

---

## 8. Neural Reasoning Module

### 8.1 Architecture

Combines **graph neural networks** (for knowledge graph reasoning) with **transformers** (for sequential game state encoding).

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class NeuralReasoningModule(nn.Module):
    """
    Multi-modal neural architecture for MTG strategic reasoning.
    
    Inputs:
    1. Knowledge graph subgraph (cards in play + neighbors) → GNN
    2. Game state sequence (actions taken this game) → Transformer
    3. Current board state (numeric features) → MLP
    
    Output:
    - Action value estimates for each legal action
    - Board position evaluation score
    - Win probability estimate
    """
    
    def __init__(self, card_embed_dim=128, hidden_dim=256, num_heads=8):
        super().__init__()
        
        # Graph Attention Network for knowledge graph reasoning
        self.gat1 = GATConv(card_embed_dim, hidden_dim, heads=num_heads)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1)
        
        # Transformer for game sequence encoding
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Board state encoder
        self.board_encoder = nn.Sequential(
            nn.Linear(512, hidden_dim),  # 512 board features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Fusion + heads
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)      # Position evaluation
        self.policy_head = nn.Linear(hidden_dim, 512)    # Action preferences
        self.win_head = nn.Linear(hidden_dim, 1)         # Win probability
    
    def forward(self, graph_data, game_sequence, board_features):
        # Graph reasoning
        graph_embed = self.gat1(graph_data.x, graph_data.edge_index).relu()
        graph_embed = self.gat2(graph_embed, graph_data.edge_index)
        graph_embed = global_mean_pool(graph_embed, graph_data.batch)
        
        # Sequence reasoning
        seq_embed = self.sequence_encoder(game_sequence)
        seq_embed = seq_embed.mean(dim=1)  # Pool over sequence
        
        # Board reasoning
        board_embed = self.board_encoder(board_features)
        
        # Fuse all modalities
        fused = self.fusion(torch.cat([graph_embed, seq_embed, board_embed], dim=-1)).relu()
        
        return {
            "value": self.value_head(fused),
            "policy": self.policy_head(fused),
            "win_prob": torch.sigmoid(self.win_head(fused)),
        }
```

### 8.2 Board State Feature Encoding

```python
class BoardStateEncoder:
    """Converts a GameState into a fixed-size numeric tensor."""
    
    def encode(self, game_state: GameState, player_id: str) -> torch.Tensor:
        features = []
        
        me = self.get_player(game_state, player_id)
        
        # Life totals (normalized)
        features.extend([p.life_total / 40.0 for p in game_state.players])
        
        # Card counts per zone per player
        for p in game_state.players:
            features.append(len(self.cards_in_zone(game_state, p.player_id, Zone.HAND)) / 10.0)
            features.append(len(self.cards_in_zone(game_state, p.player_id, Zone.BATTLEFIELD)) / 20.0)
            features.append(len(self.cards_in_zone(game_state, p.player_id, Zone.GRAVEYARD)) / 40.0)
            features.append(len(self.cards_in_zone(game_state, p.player_id, Zone.LIBRARY)) / 99.0)
        
        # Creature stats on battlefield (total power, total toughness, creature count)
        for p in game_state.players:
            creatures = self.get_creatures(game_state, p.player_id)
            features.append(sum(c.power for c in creatures) / 50.0)
            features.append(sum(c.toughness for c in creatures) / 50.0)
            features.append(len(creatures) / 15.0)
        
        # Mana available
        for color in "WUBRGC":
            features.append(me.mana_pool.get(color, 0) / 10.0)
        
        # Phase encoding (one-hot)
        phase_vec = [0.0] * len(Phase)
        phase_vec[list(Phase).index(game_state.phase)] = 1.0
        features.extend(phase_vec)
        
        # Stack depth
        features.append(len(game_state.stack) / 10.0)
        
        # Turn number (normalized)
        features.append(game_state.turn_number / 20.0)
        
        return torch.tensor(features, dtype=torch.float32)
```

### 8.3 GraphRAG Integration (Neo4j + APOC + n10s)

Rather than running a separate GraphRAG pipeline (e.g., microsoft/graphrag), we implement **GraphRAG-style retrieval natively in Neo4j** using APOC graph algorithms, n10s ontology expansion, full-text indexes, and Neo4j's native vector search. For more sophisticated QA we can optionally layer [GraphQAAgent](https://github.com/DataScienceLabFHSWF/GraphQAAgent) on top.

#### Why Neo4j-native GraphRAG?

| Feature | microsoft/graphrag | Neo4j + APOC + n10s |
|---------|-------------------|---------------------|
| Graph source | Builds its own entity graph from text | Uses **our OWL-driven KG** directly |
| Ontology | None | **n10s** — OWL class hierarchy, synonym expansion, SHACL validation |
| Community detection | Leiden (Python) | **APOC** `apoc.algo.louvain` / GDS Leiden — runs in-database |
| Graph traversal | Python NetworkX | **APOC** `apoc.path.subgraphAll` — native, fast |
| Importance scoring | Community summaries | **APOC** `apoc.algo.pageRank` + centrality |
| Vector search | Separate embeddings store | **Neo4j 5.x native vector index** — no Qdrant needed |
| Full-text search | N/A | **Neo4j full-text index** (Lucene-backed) |
| Cypher generation | N/A | **LLM → Cypher** — query the graph directly |
| One database | ❌ (separate stores) | ✅ Everything in Neo4j |

#### Retrieval Strategies

```python
from neo4j import AsyncGraphDatabase
from langchain_neo4j import Neo4jGraph

class MTGGraphRAG:
    """GraphRAG-style retrieval implemented natively on Neo4j.
    
    Strategies:
    1. subgraph:  APOC path expansion around an entity
    2. cypher:    LLM generates Cypher from natural language
    3. vector:    Neo4j native vector similarity search
    4. fulltext:  Lucene full-text search over card text
    5. community: APOC community detection + summaries
    6. hybrid:    Combine vector + graph + fulltext with RRF
    """
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    
    async def subgraph_retrieval(self, card_name: str, depth: int = 2) -> dict:
        """APOC subgraph expansion — get strategic context around an entity."""
        query = """
        MATCH (start:Card {cardName: $card_name})
        CALL apoc.path.subgraphAll(start, {
          maxLevel: $depth,
          relationshipFilter: 'PART_OF_COMBO|SYNERGIZES_WITH|COUNTERS|ENABLES|BELONGS_TO_ARCHETYPE|HAS_WIN_CONDITION'
        }) YIELD nodes, relationships
        RETURN nodes, relationships
        """
        async with self.driver.session() as session:
            result = await session.run(query, card_name=card_name, depth=depth)
            record = await result.single()
            return self._format_subgraph(record)
    
    async def vector_retrieval(self, query_embedding: list[float], k: int = 10) -> list[dict]:
        """Neo4j native vector index search."""
        query = """
        CALL db.index.vector.queryNodes('cardEmbeddings', $k, $embedding)
        YIELD node, score
        RETURN node.cardName AS card, score
        """
        async with self.driver.session() as session:
            result = await session.run(query, k=k, embedding=query_embedding)
            return [dict(r) async for r in result]
    
    async def fulltext_retrieval(self, search_text: str, limit: int = 10) -> list[dict]:
        """Lucene full-text search over card text."""
        query = """
        CALL db.index.fulltext.queryNodes('cardSearch', $text)
        YIELD node, score
        RETURN node.cardName AS card, node.oracleText AS text, score
        LIMIT $limit
        """
        async with self.driver.session() as session:
            result = await session.run(query, text=search_text, limit=limit)
            return [dict(r) async for r in result]
    
    async def community_retrieval(self, card_name: str) -> dict:
        """Get the strategic community cluster a card belongs to."""
        query = """
        MATCH (c:Card {cardName: $card_name})
        WITH c.strategicCluster AS cluster
        MATCH (member:Card {strategicCluster: cluster})
        RETURN cluster,
               collect(member.cardName) AS members,
               count(member) AS size
        """
        async with self.driver.session() as session:
            result = await session.run(query, card_name=card_name)
            record = await result.single()
            return dict(record) if record else {}
    
    async def find_combos(self, available_cards: list[str]) -> list[dict]:
        """Detect available combos from hand + battlefield."""
        query = """
        MATCH (combo:Combo)
        WHERE ALL(piece IN [(combo)<-[:PART_OF_COMBO]-(c:Card) | c.cardName]
                  WHERE piece IN $available)
        MATCH (combo)<-[:PART_OF_COMBO]-(c:Card)
        MATCH (combo)-[:PRODUCES_EFFECT]->(e:Effect)
        RETURN combo.comboDescription AS description,
               collect(DISTINCT c.cardName) AS pieces,
               collect(DISTINCT e.name) AS effects
        """
        async with self.driver.session() as session:
            result = await session.run(query, available=available_cards)
            return [dict(r) async for r in result]
    
    async def matchup_analysis(self, my_archetype: str, opp_archetype: str) -> dict:
        """Archetype matchup analysis using graph traversal."""
        query = """
        MATCH (me:Archetype {name: $my_arch})
        MATCH (opp:Archetype {name: $opp_arch})
        OPTIONAL MATCH (me)-[w:WEAK_AGAINST]->(opp)
        OPTIONAL MATCH (me)-[s:STRONG_AGAINST]->(opp)
        OPTIONAL MATCH (me)-[:HAS_WIN_CONDITION]->(myWin:Combo)
        OPTIONAL MATCH (opp)-[:HAS_WIN_CONDITION]->(oppWin:Combo)
        OPTIONAL MATCH (answer:Card)-[:COUNTERS]->(oppKey:Card)<-[:BELONGS_TO_ARCHETYPE]-(opp)
        WHERE (answer)-[:BELONGS_TO_ARCHETYPE]->(me)
        RETURN me.name AS myArchetype, opp.name AS oppArchetype,
               w IS NOT NULL AS isUnfavorable,
               s IS NOT NULL AS isFavorable,
               collect(DISTINCT myWin.comboDescription) AS myWinCons,
               collect(DISTINCT oppWin.comboDescription) AS oppWinCons,
               collect(DISTINCT answer.cardName) AS keyAnswers
        """
        async with self.driver.session() as session:
            result = await session.run(query, my_arch=my_archetype, opp_arch=opp_archetype)
            record = await result.single()
            return dict(record) if record else {}
```

#### Hybrid Knowledge Architecture (Neo4j-native)

```
┌──────────────────────────────────────────────────────────────────┐
│              HYBRID KNOWLEDGE LAYER (Neo4j)                      │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                Neo4j (single instance)                      │  │
│  │                                                            │  │
│  │  ┌────────────────┐  ┌──────────────┐  ┌───────────────┐  │  │
│  │  │ ABox (data)    │  │ TBox (schema)│  │ Vector Index  │  │  │
│  │  │ 30k+ Card nodes│  │ n10s OWL     │  │ 384-dim       │  │  │
│  │  │ Combo nodes    │  │ import       │  │ embeddings    │  │  │
│  │  │ Archetype nodes│  │ SHACL shapes │  │ per card      │  │  │
│  │  │ Synergy edges  │  │ Class hier.  │  │               │  │  │
│  │  └────────────────┘  └──────────────┘  └───────────────┘  │  │
│  │                                                            │  │
│  │  Plugins: n10s (ontology) · APOC (algorithms) · GDS (opt) │  │
│  └────────────────────────────┬───────────────────────────────┘  │
│                               │                                  │
│  ┌────────────────────────────▼───────────────────────────────┐  │
│  │  Retrieval Layer                                           │  │
│  │  Cypher queries · APOC subgraph · Vector similarity        │  │
│  │  Full-text search · Community clusters · LLM→Cypher        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  GNN Embedder (PyG GraphSAGE)                              │  │
│  │  Export graph → train → write embeddings back to Neo4j     │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 8.4 Combo Detection via GNN

```python
class ComboDetector:
    """
    Uses the knowledge graph to detect available combos
    from cards across a player's hand + battlefield.
    """
    
    def __init__(self, kg: MTGKnowledgeGraph):
        self.kg = kg
    
    def detect_available_combos(self, hand: list[str], battlefield: list[str]) -> list[Combo]:
        """Find combos where all pieces are in hand + battlefield."""
        available_cards = set(hand + battlefield)
        combos = []
        
        for card in available_cards:
            card_combos = self.kg.get_combos_containing(card)
            for combo in card_combos:
                if all(piece in available_cards for piece in combo.cards):
                    combos.append(combo)
        
        return combos
    
    def detect_near_combos(self, hand: list[str], battlefield: list[str], 
                           library_beliefs: dict) -> list[tuple[Combo, float]]:
        """Find combos that are 1 card away and estimate probability of drawing into them."""
        available = set(hand + battlefield)
        near_combos = []
        
        for card in available:
            for combo in self.kg.get_combos_containing(card):
                missing = [p for p in combo.cards if p not in available]
                if len(missing) == 1:
                    draw_prob = library_beliefs.get(missing[0], 0.0)
                    near_combos.append((combo, draw_prob))
        
        return near_combos
```

---

## 9. Opponent Modeling & Metagame Awareness

### 9.1 Archetype Classification

When the game begins, agents may not know what deck the opponent is playing. As cards are revealed, the agent narrows down the archetype.

```python
class OpponentModel:
    """
    Tracks an opponent's behavior and infers their strategy.
    """
    
    def __init__(self, opponent_id: str, kg: MTGKnowledgeGraph):
        self.opponent_id = opponent_id
        self.kg = kg
        self.cards_seen: list[str] = []
        self.actions_taken: list[Action] = []
        self.archetype_probabilities: dict[str, float] = {}  # {"UW Control": 0.7, ...}
        self.predicted_hand: dict[str, float] = {}
    
    def observe_card(self, card_name: str):
        """Update archetype probabilities when a card is revealed."""
        self.cards_seen.append(card_name)
        self.archetype_probabilities = self.kg.infer_archetype_from_cards(self.cards_seen)
        self._update_hand_predictions()
    
    def observe_behavior(self, action: Action, game_state: GameState):
        """Update model based on opponent behavior patterns.
        
        Heuristics:
        - Leaving blue mana open → likely has counterspells
        - Not attacking with lethal on board → playing around something
        - Tutoring → assembling combo
        - Discarding specific cards → signaling or fixing hand
        """
        self.actions_taken.append(action)
        self._analyze_mana_usage(action, game_state)
        self._analyze_attack_patterns(action, game_state)
    
    def _update_hand_predictions(self):
        """Based on top archetype, predict likely cards in hand."""
        top_archetype = max(self.archetype_probabilities, key=self.archetype_probabilities.get)
        staples = self.kg.get_archetype_signature_cards(top_archetype)
        unseen_staples = [c for c in staples if c not in self.cards_seen]
        
        self.predicted_hand = {}
        for card in unseen_staples:
            # Probability based on archetype frequency, cards drawn, cards remaining
            self.predicted_hand[card] = self._estimate_in_hand_probability(card)
    
    def get_threat_assessment(self) -> dict:
        """What is the opponent threatening?"""
        top_archetype = max(self.archetype_probabilities, key=self.archetype_probabilities.get)
        return {
            "archetype": top_archetype,
            "likely_win_conditions": self.kg.get_archetype_win_conditions(top_archetype),
            "likely_interaction": self.kg.get_archetype_interaction(top_archetype),
            "probability_has_counterspell": self._prob_has_counter(),
            "probability_has_removal": self._prob_has_removal(),
            "probability_has_boardwipe": self._prob_has_boardwipe(),
        }
    
    def _prob_has_counter(self) -> float:
        """Estimate probability opponent has a counterspell.
        
        Factors:
        - Archetype (control decks run 8-12 counters)
        - Open blue mana
        - Cards in hand
        - Counters already seen/used this game
        """
        ...
```

### 9.2 Strategic Decision Examples

```
Scenario: Agent has Thassa's Oracle + Demonic Consultation in hand.
           Opponent is playing Azorius Control with 2 blue mana open.

Knowledge Graph says:
  - Thassa's Oracle + Demonic Consultation = instant win combo
  - Azorius Control typically runs: Counterspell, Force of Will, 
    Dovin's Veto, Mana Drain, Swan Song
  - Opponent has used 1 Counterspell already (seen in graveyard)

Active Inference says:
  - P(opponent has counterspell) = 72% (blue mana open, control deck, 
    hand size 4, only 1 used so far)
  - Epistemic action: Cast a less important spell first to probe for counter
  - Pragmatic action: Wait until opponent taps out, then combo

Agent decision: Cast a bait spell (e.g., Rhystic Study) to force the counter,
then combo next turn. Or wait for opponent's end step when they might 
tap mana for their own plays.
```

---

## 10. Judge Agent & Rules Arbitration

### 10.1 When the Judge Is Called

- Complex stack interactions (multiple triggers, replacement effects layering)
- Ambiguous card text interactions
- State-based action edge cases
- Layer system conflicts
- New/unusual card interactions not hard-coded in the rules engine

### 10.2 Judge Architecture

```python
class JudgeAgent:
    """
    LLM-based rules arbiter with access to:
    1. MTG Comprehensive Rules (full text, ~280 pages)
    2. Scryfall rulings API (per-card official rulings)
    3. Knowledge graph interaction edges
    4. RAG over rules database
    """
    
    def __init__(self, llm, rules_vectorstore, scryfall_client, kg):
        self.llm = llm
        self.rules_rag = rules_vectorstore  # FAISS/Chroma with embedded CR sections
        self.scryfall = scryfall_client
        self.kg = kg
    
    async def rule_on(self, situation: str, game_state: GameState) -> Ruling:
        # 1. Retrieve relevant comprehensive rules sections
        relevant_rules = self.rules_rag.similarity_search(situation, k=10)
        
        # 2. Fetch card-specific rulings from Scryfall
        cards_involved = self.extract_card_names(situation)
        card_rulings = []
        for card in cards_involved:
            rulings = await self.scryfall.get_rulings(card)
            card_rulings.extend(rulings)
        
        # 3. Check knowledge graph for known interactions
        kg_interactions = self.kg.get_interactions_between(cards_involved)
        
        # 4. LLM synthesis
        prompt = f"""You are an MTG Level 3 Judge. Rule on the following situation.

SITUATION: {situation}

RELEVANT COMPREHENSIVE RULES:
{self.format_rules(relevant_rules)}

CARD-SPECIFIC RULINGS:
{self.format_rulings(card_rulings)}

KNOWN INTERACTIONS:
{self.format_interactions(kg_interactions)}

CURRENT GAME STATE:
{self.format_game_state(game_state)}

Provide:
1. Your ruling (what happens)
2. The specific rule numbers that apply
3. Step-by-step resolution order
4. Confidence level (high/medium/low)
"""
        ruling = await self.llm.ainvoke(prompt)
        return self.parse_ruling(ruling)
```

### 10.3 Comprehensive Rules RAG Setup

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def build_rules_vectorstore(comprehensive_rules_path: str) -> FAISS:
    """Build a RAG index over the MTG Comprehensive Rules document."""
    with open(comprehensive_rules_path) as f:
        rules_text = f.read()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". "],
    )
    chunks = splitter.split_text(rules_text)
    
    vectorstore = FAISS.from_texts(
        chunks, 
        OpenAIEmbeddings(),
        metadatas=[{"source": "comprehensive_rules", "chunk_id": i} for i, _ in enumerate(chunks)]
    )
    return vectorstore
```

---

## 11. External API Integrations

### 11.1 Scryfall API Client

> **Use [Scrython](https://github.com/NandaScott/Scrython)** (`pip install scrython`) — a mature, MIT-licensed Python wrapper for the Scryfall API with built-in rate limiting, caching, bulk data download, and full type support. 158 stars, actively maintained (v2.0.2, 2025). See Section 4.4 for full reuse details.

If a custom async client is needed (e.g., for integration with our async game loop), here is the pattern:

```python
import httpx
import asyncio
from functools import lru_cache

class ScryfallClient:
    """Async client for Scryfall API with rate limiting and caching."""
    
    BASE_URL = "https://api.scryfall.com"
    RATE_LIMIT = 0.1  # 100ms between requests
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            headers={
                "User-Agent": "MTGAgentFramework/1.0",
                "Accept": "application/json",
            }
        )
        self._last_request = 0
    
    async def get_card_by_name(self, name: str) -> dict:
        """GET /cards/named?exact={name}"""
        await self._rate_limit()
        resp = await self.client.get(f"{self.BASE_URL}/cards/named", params={"exact": name})
        resp.raise_for_status()
        return resp.json()
    
    async def get_rulings(self, card_name: str) -> list[dict]:
        """Fetch rulings for a card by first resolving its Scryfall ID."""
        card = await self.get_card_by_name(card_name)
        await self._rate_limit()
        resp = await self.client.get(card["rulings_uri"])
        resp.raise_for_status()
        return resp.json()["data"]
    
    async def get_card_image(self, card_name: str, size: str = "normal") -> str:
        """Get image URI for a card. Sizes: small, normal, large, png, art_crop, border_crop."""
        card = await self.get_card_by_name(card_name)
        return card.get("image_uris", {}).get(size, "")
    
    async def search_cards(self, query: str) -> list[dict]:
        """GET /cards/search?q={query} using Scryfall search syntax."""
        await self._rate_limit()
        resp = await self.client.get(f"{self.BASE_URL}/cards/search", params={"q": query})
        resp.raise_for_status()
        return resp.json()["data"]
    
    async def download_bulk_data(self, bulk_type: str = "oracle_cards") -> str:
        """Download bulk data for local processing."""
        await self._rate_limit()
        resp = await self.client.get(f"{self.BASE_URL}/bulk-data")
        resp.raise_for_status()
        for item in resp.json()["data"]:
            if item["type"] == bulk_type:
                return item["download_uri"]
        raise ValueError(f"Bulk data type '{bulk_type}' not found")
    
    async def _rate_limit(self):
        now = asyncio.get_event_loop().time()
        wait = self.RATE_LIMIT - (now - self._last_request)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request = asyncio.get_event_loop().time()
```

### 11.2 Decklist Sources

```python
class DecklistLoader:
    """Load decklists from Moxfield, Archidekt, or plaintext."""
    
    async def from_moxfield(self, deck_url: str) -> Decklist:
        """Parse a Moxfield deck URL and extract the card list.
        Note: Moxfield API is not public — use HTML scraping or ask for API access."""
        # Extract deck ID from URL, fetch page, parse card list
        ...
    
    async def from_archidekt(self, deck_id: int) -> Decklist:
        """Archidekt has a semi-public API.
        GET https://archidekt.com/api/decks/{id}/"""
        ...
    
    def from_text(self, decklist_text: str) -> Decklist:
        """Parse a plaintext decklist (1 Sol Ring\\n1 Command Tower\\n...)."""
        ...
    
    async def resolve_cards(self, decklist: Decklist, scryfall: ScryfallClient) -> list[dict]:
        """Resolve each card name to full Scryfall card data."""
        ...
```

### 11.3 Commander Spellbook (Combo Database)

```python
class ComboDatabase:
    """Integration with Commander Spellbook for combo data.
    API: https://commanderspellbook.com/api/"""
    
    async def fetch_all_combos(self) -> list[dict]:
        """Download the full combo database."""
        ...
    
    async def find_combos_with_card(self, card_name: str) -> list[dict]:
        """Find all combos involving a specific card."""
        ...
    
    def import_into_knowledge_graph(self, kg: MTGKnowledgeGraph, combos: list[dict]):
        """Populate the KG with combo nodes and edges."""
        for combo in combos:
            kg.add_combo(
                combo_id=combo["id"],
                cards=combo["cards"],
                result=combo["result"],
                description=combo["description"],
            )
```

---

## 12. Reinforcement Learning & Training Pipeline

### 12.1 Training Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                           │
│                                                                  │
│  ┌─────────────┐     ┌──────────────┐     ┌──────────────────┐  │
│  │ Self-Play   │────▶│ Experience   │────▶│  Neural Module   │  │
│  │ Engine      │     │ Buffer       │     │  Training        │  │
│  │ (parallel   │     │ (game logs,  │     │  (PPO / AlphaZero│  │
│  │  games)     │     │  state-action│     │   style)         │  │
│  └─────────────┘     │  -rewards)   │     └──────────────────┘  │
│        │              └──────────────┘              │            │
│        │                                            │            │
│        │              ┌──────────────┐              │            │
│        └─────────────▶│  KG Update   │◀─────────────┘            │
│                       │  (new combos │                           │
│                       │   discovered)│                           │
│                       └──────────────┘                           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Hierarchical Learning                                   │    │
│  │                                                          │    │
│  │  Level 0: Card-level (play this card? attack with it?)   │    │
│  │  Level 1: Turn-level (what's my plan this turn?)         │    │
│  │  Level 2: Game-level (aggro, control, combo strategy?)   │    │
│  │  Level 3: Meta-level (deck selection, sideboarding)      │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### 12.2 Reward Design

```python
class RewardFunction:
    """
    Multi-faceted reward for RL training.
    Combines terminal (win/loss) with intermediate shaping rewards.
    """
    
    def compute(self, prev_state: GameState, action: Action, 
                new_state: GameState, player_id: str) -> float:
        reward = 0.0
        
        # Terminal rewards (sparse but critical)
        if new_state.game_over:
            if new_state.winner == player_id:
                reward += 1.0      # Win
            else:
                reward -= 1.0      # Loss
            return reward
        
        me_before = self.get_player(prev_state, player_id)
        me_after = self.get_player(new_state, player_id)
        
        # Shaping rewards (dense, small)
        # Life advantage
        reward += 0.01 * (me_after.life_total - me_before.life_total) / 40.0
        
        # Card advantage (hand + battlefield vs opponents)
        reward += 0.005 * self.card_advantage_delta(prev_state, new_state, player_id)
        
        # Board presence (creatures, permanents)
        reward += 0.005 * self.board_presence_delta(prev_state, new_state, player_id)
        
        # Combo pieces assembled
        reward += 0.02 * self.combo_progress(new_state, player_id)
        
        # Opponent threat removed
        reward += 0.01 * self.threats_removed(prev_state, new_state, player_id)
        
        # Penalty for time/stalling
        reward -= 0.001  # Small per-action cost to encourage efficiency
        
        return reward
```

### 12.3 Self-Play Training Loop

```python
class SelfPlayTrainer:
    """
    AlphaZero-inspired self-play training.
    Agents play each other, generate experience, update neural modules.
    """
    
    def __init__(self, num_parallel_games: int = 64):
        self.num_parallel = num_parallel_games
        self.experience_buffer = ExperienceBuffer(max_size=100_000)
        self.neural_module = NeuralReasoningModule()
        self.optimizer = torch.optim.Adam(self.neural_module.parameters(), lr=1e-4)
    
    async def train(self, num_iterations: int = 1000):
        for iteration in range(num_iterations):
            # 1. Self-play: generate games
            experiences = await self.run_self_play_games(self.num_parallel)
            self.experience_buffer.add(experiences)
            
            # 2. Train neural module on collected experience
            loss = self.train_neural_module(batch_size=256, num_epochs=10)
            
            # 3. Evaluate against previous version
            win_rate = await self.evaluate(num_games=100)
            
            # 4. Update knowledge graph with newly discovered interactions
            self.update_kg_from_experience(experiences)
            
            print(f"Iteration {iteration}: loss={loss:.4f}, win_rate={win_rate:.2%}")
    
    async def run_self_play_games(self, num_games: int) -> list[Experience]:
        """Run multiple games in parallel, collect (state, action, reward) tuples."""
        ...
    
    def train_neural_module(self, batch_size: int, num_epochs: int) -> float:
        """Train on experience buffer using PPO or similar."""
        for epoch in range(num_epochs):
            batch = self.experience_buffer.sample(batch_size)
            
            # Forward pass
            predictions = self.neural_module(
                batch.graph_data, batch.game_sequences, batch.board_features
            )
            
            # Loss: value loss + policy loss + win prediction loss
            value_loss = F.mse_loss(predictions["value"], batch.returns)
            policy_loss = self.compute_policy_loss(predictions["policy"], batch.actions, batch.advantages)
            win_loss = F.binary_cross_entropy(predictions["win_prob"], batch.outcomes)
            
            loss = value_loss + policy_loss + win_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
```

### 12.4 Nested / Hierarchical Learning

```python
class HierarchicalAgent:
    """
    Multi-level decision hierarchy:
    
    Meta-Strategy (Level 3) - Updated between games
      → "I should play aggressively against this slow combo deck"
    
    Game Plan (Level 2) - Updated each turn
      → "Stick an early threat, protect it, close before they combo"
    
    Turn Tactics (Level 1) - Updated each priority pass
      → "Play Ragavan turn 1, hold Spell Pierce for their Sol Ring"
    
    Card Actions (Level 0) - Each individual game action
      → "Tap Mountain, cast Ragavan, pass priority"
    """
    
    def __init__(self):
        self.meta_policy = MetaStrategyNetwork()   # Trained on game-level outcomes
        self.game_policy = GamePlanNetwork()        # Trained on turn-level evaluations
        self.turn_policy = TurnTacticsNetwork()     # Trained on action-level rewards
        self.action_policy = ActionNetwork()         # Low-level card play selection
    
    async def set_game_strategy(self, my_deck: Decklist, opponent_archetype: str):
        """Level 2: Determine overall game plan at start of game."""
        self.current_strategy = self.meta_policy.select_strategy(my_deck, opponent_archetype)
    
    async def plan_turn(self, game_state: GameState) -> TurnPlan:
        """Level 1: Plan the broad strokes of this turn."""
        return self.game_policy.plan_turn(game_state, self.current_strategy)
    
    async def select_action(self, game_state: GameState, legal_actions: list) -> Action:
        """Level 0: Choose specific card/action consistent with turn plan."""
        return self.action_policy.select(game_state, legal_actions, self.current_turn_plan)
```

---

## 13. Visualization & UI

### 13.1 Option A: Streamlit Web App (Recommended for MVP)

```python
import streamlit as st

def render_game(game_state: GameState, scryfall: ScryfallClient):
    st.title("MTG Agent Arena")
    
    # Render each player zone
    for i, player in enumerate(game_state.players):
        with st.container():
            st.subheader(f"🎮 {player.name} — Life: {player.life_total}")
            
            # Battlefield
            battlefield_cards = get_cards_in_zone(game_state, player.player_id, Zone.BATTLEFIELD)
            cols = st.columns(min(len(battlefield_cards), 8))
            for j, card in enumerate(battlefield_cards):
                with cols[j % 8]:
                    img_url = card.card_data.get("image_uris", {}).get("normal", "")
                    if img_url:
                        st.image(img_url, width=150)
                    css_class = "tapped" if card.tapped else ""
                    st.caption(card.card_data.get("name", "Unknown"))
            
            # Hand (hidden for opponents, shown for active player / human)
            if player.is_human or st.session_state.get("show_all_hands"):
                hand = get_cards_in_zone(game_state, player.player_id, Zone.HAND)
                st.write(f"Hand ({len(hand)} cards):")
                hand_cols = st.columns(min(len(hand), 7))
                for j, card in enumerate(hand):
                    with hand_cols[j]:
                        img_url = card.card_data.get("image_uris", {}).get("small", "")
                        if img_url:
                            st.image(img_url, width=100)
    
    # Stack visualization
    if game_state.stack:
        st.sidebar.subheader("📚 The Stack")
        for item in reversed(game_state.stack):
            st.sidebar.write(f"→ {item.description}")
    
    # Game log
    st.sidebar.subheader("📜 Game Log")
    for entry in game_state.game_log[-20:]:
        st.sidebar.write(entry)
```

### 13.2 Option B: PyGame/Pygame-CE (Rich 2D)

For a more polished visual experience with animated card movement, tapping, etc.

```python
import pygame

class MTGRenderer:
    """2D game renderer using PyGame with Scryfall card images."""
    
    ZONES = {
        "battlefield": pygame.Rect(100, 200, 1000, 300),
        "hand": pygame.Rect(100, 550, 1000, 180),
        "graveyard": pygame.Rect(1120, 200, 150, 200),
        "exile": pygame.Rect(1120, 420, 150, 200),
        "command_zone": pygame.Rect(1120, 50, 150, 130),
    }
    
    def __init__(self, width=1300, height=800):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.card_images = {}  # Cache: card_name → pygame.Surface
        self.font = pygame.font.SysFont("Arial", 14)
    
    async def load_card_image(self, card_name: str, scryfall: ScryfallClient):
        """Download and cache card image from Scryfall."""
        ...
    
    def render_frame(self, game_state: GameState):
        """Draw one frame of the game board."""
        self.screen.fill((34, 139, 34))  # Green felt
        
        for player in game_state.players:
            self.render_battlefield(game_state, player)
            self.render_hand(game_state, player)
            self.render_life_total(player)
            self.render_mana_pool(player)
        
        self.render_stack(game_state)
        self.render_phase_indicator(game_state)
        
        pygame.display.flip()
```

### 13.3 Option C: Web-Based (FastAPI + React/HTMX)

Most scalable option; supports remote play and spectating.

```
Backend (FastAPI):
  /api/game/{id}/state    GET   → current game state JSON
  /api/game/{id}/action   POST  → submit player action  
  /ws/game/{id}           WS    → real-time state updates

Frontend:
  React + Canvas   OR   HTMX for simpler server-rendered updates
  Card images from Scryfall CDN (https://cards.scryfall.io/...)
```

---

## 14. Human Player Integration

### 14.1 Architecture

A human player replaces one agent in the seat. The LangGraph `interrupt()` mechanism pauses the game when it's the human's turn.

```python
from langgraph.types import interrupt

async def human_decision(game_state: GameState) -> Action:
    """Pause the graph and wait for human input."""
    player = get_priority_player(game_state)
    legal_actions = rules_engine.get_legal_actions(game_state, player.player_id)
    
    # Interrupt: sends legal_actions to the UI, waits for selection
    chosen = interrupt({
        "type": "human_action_required",
        "player": player.name,
        "legal_actions": [action.to_dict() for action in legal_actions],
        "game_state_summary": summarize_state(game_state),
    })
    
    return Action.from_dict(chosen)
```

### 14.2 Human UI Features

- Display current hand with card images (clickable to select)
- Highlight legal targets (valid attack/block assignments, spell targets)
- Show phase indicator and priority indicator
- "Pass Priority" button
- Stack visualization (what is currently resolving)
- Game log / chat
- Concede button
- Timer (optional, for competitive play)

---

## 15. Supported Formats

### 15.1 Standard (1v1)

| Rule | Value |
|------|-------|
| Players | 2 |
| Starting Life | 20 |
| Deck size | 60 (minimum) |
| Max copies | 4 per card (except basic lands) |
| Sideboard | 15 cards |
| Legal sets | Rotating (recent 2-3 years of Standard-legal sets) |

### 15.2 Commander / EDH (Multiplayer)

| Rule | Value |
|------|-------|
| Players | 4 (typically, 2-6 supported) |
| Starting life | 40 |
| Deck size | 100 (exactly, including commander) |
| Max copies | 1 per card (singleton, except basic lands) |
| Commander | Legendary creature in command zone; determines color identity |
| Commander tax | +{2} for each previous cast from command zone |
| Commander damage | 21 combat damage from a single commander → player loses |
| Color identity | All cards must match commander's color identity |
| Mulligan | Free first mulligan, then London mulligan |
| Politics | Table talk, deals, alliances are part of the format |

### 15.3 Format-Specific Agent Behavior

```python
class CommanderAgent(MTGAgent):
    """Extended agent with multiplayer-specific reasoning."""
    
    async def evaluate_threat_assessment(self, game_state: GameState) -> dict:
        """Who is the biggest threat at the table? Who should I attack?"""
        ...
    
    async def handle_politics(self, game_state: GameState, proposals: list) -> str:
        """Respond to table talk: 'Don't attack me, I'll remove their combo piece.'"""
        ...
    
    async def decide_targets(self, game_state: GameState, spell: Card) -> str:
        """In multiplayer, who to target with removal/interaction?"""
        ...
```

---

## 16. Project Structure

```
opposition-agents-playing-mtg/
├── README.md
├── PLAN.md
├── pyproject.toml                     # Poetry / pip project config
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   │
│   ├── engine/                        # Core game engine
│   │   ├── __init__.py
│   │   ├── game_state.py              # GameState, PlayerState, CardInstance dataclasses
│   │   ├── zones.py                   # Zone management
│   │   ├── stack.py                   # The Stack implementation
│   │   ├── phases.py                  # Phase/step transitions
│   │   ├── combat.py                  # Combat system
│   │   ├── mana.py                    # Mana system
│   │   ├── rules_engine.py            # Legal action generation + execution
│   │   ├── state_based_actions.py     # CR 704
│   │   ├── triggered_abilities.py     # Trigger detection and queueing
│   │   ├── replacement_effects.py     # Replacement effect system
│   │   ├── continuous_effects.py      # Layer system
│   │   └── keywords.py               # Keyword ability implementations
│   │
│   ├── agents/                        # LLM Agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py             # Abstract agent interface
│   │   ├── llm_agent.py              # LLM-powered agent (LangChain)
│   │   ├── random_agent.py           # Random legal action (baseline)
│   │   ├── human_agent.py            # Human player adapter
│   │   ├── active_inference.py       # Active inference module
│   │   ├── neural_reasoner.py        # Neural reasoning module
│   │   ├── opponent_model.py         # Opponent modeling
│   │   ├── combo_detector.py         # Combo detection from KG
│   │   └── tools.py                  # LangChain tools for agents
│   │
│   ├── knowledge/                     # MTG Ontology & Knowledge Graph
│   │   ├── __init__.py
│   │   ├── knowledge_graph.py        # Neo4j-backed KG (Cypher queries, APOC algos)
│   │   ├── graph_rag.py              # GraphRAG retrieval (subgraph, vector, fulltext, hybrid)
│   │   ├── graph_embedder.py         # GNN embeddings (PyG) + Neo4j vector writeback
│   │   ├── kg_builder.py             # Populate KG from various sources
│   │   ├── n10s_setup.py             # n10s graph config + OWL ontology import
│   │   └── combo_database.py         # Commander Spellbook integration
│   │
│   ├── judge/                         # Judge agent
│   │   ├── __init__.py
│   │   ├── judge_agent.py            # LLM judge with RAG
│   │   └── rules_vectorstore.py      # Comprehensive Rules embedding
│   │
│   ├── orchestrator/                  # LangGraph game orchestrator
│   │   ├── __init__.py
│   │   ├── game_graph.py             # LangGraph state machine
│   │   ├── priority_loop.py          # Priority passing logic
│   │   └── game_runner.py            # High-level game runner
│   │
│   ├── training/                      # RL & self-play training
│   │   ├── __init__.py
│   │   ├── rewards.py                # Reward functions
│   │   ├── self_play.py              # Self-play training loop
│   │   ├── experience_buffer.py      # Replay buffer
│   │   └── hierarchical.py           # Hierarchical learning
│   │
│   ├── integrations/                  # External APIs
│   │   ├── __init__.py
│   │   ├── scryfall.py               # Scryfall API client
│   │   ├── decklist_loader.py        # Moxfield / Archidekt / text loaders
│   │   └── card_cache.py             # Local card data cache (SQLite)
│   │
│   └── ui/                           # Visualization
│       ├── __init__.py
│       ├── streamlit_app.py          # Streamlit UI
│       ├── pygame_renderer.py        # PyGame renderer (optional)
│       ├── web_api.py                # FastAPI backend (optional)
│       └── card_renderer.py          # Card image loading/display utils
│
├── data/
│   ├── ontology/
│   │   ├── mtg-ontology-v1.0.owl     # Formal OWL ontology (RDF/XML)
│   │   └── mtg-shapes.ttl            # SHACL shapes for validation
│   ├── competency_questions.txt      # KG competency questions
│   ├── comprehensive_rules.txt       # MTG Comprehensive Rules full text
│   ├── card_cache.db                 # SQLite cache of Scryfall card data
│   └── combos.json                   # Commander Spellbook combo dump
│
├── neo4j/                             # Neo4j configuration
│   ├── plugins/                      # n10s + APOC JARs (mounted into container)
│   ├── init/                         # Cypher scripts run on first start
│   │   └── 01_init_n10s.cypher       # n10s config + ontology import + indexes
│   └── conf/
│       └── neo4j.conf                # Neo4j server config (plugins, memory)
│
├── models/                            # Trained model weights
│   ├── neural_reasoner.pt
│   ├── card_embeddings.pt
│   └── opponent_classifier.pt
│
├── tests/
│   ├── test_engine/
│   │   ├── test_phases.py
│   │   ├── test_combat.py
│   │   ├── test_stack.py
│   │   ├── test_mana.py
│   │   └── test_state_based.py
│   ├── test_agents/
│   │   ├── test_active_inference.py
│   │   ├── test_opponent_model.py
│   │   └── test_combo_detector.py
│   ├── test_knowledge/
│   │   ├── test_knowledge_graph.py
│   │   └── test_ontology.py
│   └── test_integrations/
│       └── test_scryfall.py
│
├── notebooks/                         # Jupyter exploration notebooks
│   ├── 01_scryfall_exploration.ipynb
│   ├── 02_knowledge_graph_build.ipynb
│   ├── 03_active_inference_demo.ipynb
│   └── 04_training_analysis.ipynb
│
└── scripts/
    ├── import_scryfall.py            # Download Scryfall bulk data → Neo4j
    ├── import_combos.py              # Commander Spellbook → Neo4j Combo nodes
    ├── import_edhrec.py              # EDHREC synergies/archetypes → Neo4j
    ├── build_embeddings.py           # Train GNN → write vectors to Neo4j
    ├── run_training.py               # Launch self-play training
    └── validate_kg.py                # Run n10s SHACL validation
```

---

## 17. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM Framework** | LangChain + LangGraph | Agent orchestration, tool calling, game state machine |
| **LLM Providers** | OpenAI GPT-4o / Anthropic Claude / local Llama | Agent brains, judge agent |
| **OWL Ontology** | `mtg-ontology-v1.0.owl` (RDF/XML) | Formal TBox — classes, properties, constraints |
| **Knowledge Graph** | Neo4j 5.x (persistent, single instance) | ABox (data) + TBox (via n10s) + vector indexes + full-text indexes |
| **n10s (neosemantics)** | [neo4j-labs/neosemantics](https://github.com/neo4j-labs/neosemantics) | OWL ontology import into Neo4j, SHACL validation, RDF ↔ LPG mapping |
| **APOC** | [neo4j/apoc](https://github.com/neo4j/apoc) | Graph algorithms (PageRank, Louvain, path expansion), full-text, utilities |
| **KGPlatform** | [KGPlatform](https://github.com/DataScienceLabFHSWF/KGPlatform) (MIT) | KnowledgeGraphBuilder (doc→KG extraction), GraphQAAgent, OntologyExtender |
| **Ontology Extension** | [OntologyExtender](https://github.com/DataScienceLabFHSWF/OntologyExtender) (MIT) | HITL multi-agent debate for ontology evolution, gap detection, OWL export |
| **Graph ML** | PyTorch Geometric (PyG) | GNN embeddings, graph-based reasoning |
| **Neural Networks** | PyTorch | Neural reasoning module, training |
| **Active Inference** | pymdp (partial) + custom | Belief updating, free energy minimization |
| **Vector Store** | Neo4j native vector index (5.x) | Card embeddings, semantic similarity — no separate store needed |
| **Scryfall Wrapper** | [Scrython](https://github.com/NandaScott/Scrython) (158 stars, MIT, `pip install scrython`) | Card data, images, rulings, bulk download with built-in rate limiting + caching |
| **Decklists** | Moxfield / Archidekt / plaintext | Deck importing |
| **Combos** | Commander Spellbook API | Combo database |
| **Visualization** | Streamlit (MVP) / PyGame / FastAPI+React | Game board UI |
| **Data Science** | NumPy, SciPy, pandas | Probability calculations, analysis |
| **Testing** | pytest, pytest-asyncio | Test suite |
| **Async** | asyncio, httpx | Async API calls, parallel game execution |

---

## 18. Implementation Phases

### Phase 1: Foundation (Weeks 1–3)

**Goal:** Playable simplified game with random agents. **Leverage existing open-source projects** (see Appendix C) to accelerate.

- [ ] Set up project structure, dependencies, pyproject.toml
- [ ] **Evaluate and fork/adapt** [open-mtg](https://github.com/hlynurd/open-mtg) game loop + [mtg-python-engine](https://github.com/wanqizhu/mtg-python-engine) stack/triggers (both MIT)
- [ ] **Adopt [Scrython](https://github.com/NandaScott/Scrython)** for Scryfall API access (`pip install scrython`) — cards, rulings, bulk data
- [ ] Study [mtg-player](https://github.com/theRealMarkCastillo/mtg-player) agent architecture and Pydantic models as reference
- [ ] Define core data models (GameState, PlayerState, CardInstance, zones) — adapt from mtg-player's Pydantic models
- [ ] Implement basic turn structure (phases, untap, draw, main, combat, end) — port from open-mtg's `phases.py`
- [ ] Implement mana system (tap lands, pay costs, mana pool)
- [ ] Implement basic creature combat (attack, block, damage) — adapt open-mtg's combat with damage assignment ordering
- [ ] Implement the stack (cast spells, resolve, LIFO order) — port from mtg-python-engine's stack
- [ ] Implement state-based actions (0 life = lose, 0 toughness = die) — port from mtg-python-engine's SBA implementation
- [ ] Build random agent (picks random legal action) — use open-mtg's `random_policy.py` as baseline
- [ ] Run first full game between two random agents with basic decks
- [ ] Decklist loader (plaintext format)

### Phase 2: Intelligence (Weeks 4–6)

**Goal:** LLM agents that make reasonable plays.

- [ ] Integrate LangChain: build LLM agent with game state prompting
- [ ] Build LangGraph game orchestrator (state machine with priority loop)
- [ ] Define LangChain tools: card lookup, rulings lookup, legal actions
- [ ] Implement agent memory (cards seen, game log, opponent tracking)
- [ ] Build judge agent with Comprehensive Rules RAG
- [ ] Test LLM agent vs random agent — verify it wins consistently
- [ ] Basic Streamlit visualization (board state, card images)

### Phase 3: OWL Ontology, Knowledge Graph & Neo4j Integration (Weeks 7–9)

**Goal:** Agents have deep strategic knowledge via a formally validated OWL-driven KG in Neo4j, queryable through Cypher + APOC + vector search.

- [ ] Set up Neo4j 5.x Docker container with **n10s** and **APOC** plugins
- [ ] Initialize n10s graph config and import `data/ontology/mtg-ontology-v1.0.owl` via `n10s.onto.import.fetch()`
- [ ] Write SHACL shapes file (`data/ontology/mtg-shapes.ttl`) and import via `n10s.validation.shacl.import.fetch()`
- [ ] Review & extend OWL ontology (add missing creature types, keywords, mechanics)
- [ ] Write Scryfall bulk data import script (`scripts/import_scryfall.py`): JSON → batch `MERGE` Card nodes
- [ ] Import Commander Spellbook combo data (`scripts/import_combos.py`): Combo nodes + `PART_OF_COMBO` edges
- [ ] Import EDHREC synergy/archetype data: Synergy edges, Archetype nodes
- [ ] Build full-text index: `db.index.fulltext.createNodeIndex('cardSearch', ['Card'], ['cardName','oracleText','typeLine'])`
- [ ] Run APOC community detection (`apoc.algo.louvain`) → set `strategicCluster` on cards
- [ ] Run APOC PageRank → set `metagameImportance` on cards
- [ ] Generate card embeddings (GraphSAGE on exported graph), write back to Neo4j vector index
- [ ] Validate KG with `n10s.validation.shacl.validate()`
- [ ] Build `MTGKnowledgeGraph` Python class (see Section 3.4) — Cypher queries for combos, synergies, answers
- [ ] Build `MTGGraphRAG` retrieval class (see Section 8.3) — subgraph, vector, fulltext, community, hybrid
- [ ] Wire KG into agent decision pipeline as LangChain tools
- [ ] Build combo detector (hand + battlefield → Cypher → available combos)
- [ ] Build archetype classifier (cards seen → Cypher → archetype probabilities)
- [ ] Set up OntologyExtender for HITL extension workflow (your brother extends the ontology)
- [ ] Run first OntologyExtender gap analysis → identify missing classes/properties

### Phase 4: Active Inference & Opponent Modeling (Weeks 10–12)

**Goal:** Agents reason about hidden information and model opponents.

- [ ] Implement belief state (posterior over opponent hands)
- [ ] Implement Bayesian update on observations (card played, mana tapped)
- [ ] Implement expected free energy computation (epistemic + pragmatic)
- [ ] Build opponent archetype classifier (cards seen → deck archetype)
- [ ] Build hand probability estimator (P(opponent has Counterspell))
- [ ] Implement behavioral pattern analysis (mana usage, attack patterns)
- [ ] Integrate active inference into agent decision pipeline
- [ ] Test: agent correctly avoids walking into counterspells

### Phase 5: Neural Reasoning (Weeks 13–15)

**Goal:** Neural networks enhance strategic evaluation.

- [ ] Build board state encoder (game state → tensor)
- [ ] Build GraphSAGE/GAT card embedder over KG
- [ ] Build NeuralReasoningModule (GNN + Transformer + MLP → value/policy/win_prob)
- [ ] Train on self-play game data from Phase 4
- [ ] Integrate neural scores into agent decision pipeline (alongside LLM + active inference)
- [ ] Benchmark: neural-enhanced agent vs LLM-only agent

### Phase 6: Reinforcement Learning & Self-Play (Weeks 16–19)

**Goal:** Agents learn and improve from experience.

- [ ] Implement reward function (win/loss + shaping)
- [ ] Implement experience replay buffer
- [ ] Implement self-play training loop (parallel games)
- [ ] Train with PPO / AlphaZero-style MCTS + neural net
- [ ] Implement hierarchical learning (meta → game → turn → action)
- [ ] KG auto-enrichment: discover new combos/synergies from play data
- [ ] ELO tracking for agent versions
- [ ] Ablation studies: which components contribute most to win rate

### Phase 7: Commander / EDH Support (Weeks 20–22)

**Goal:** Full multiplayer support.

- [ ] Extend game engine: 4 players, APNAP priority, command zone
- [ ] Implement commander tax, commander damage, color identity restrictions
- [ ] Implement multiplayer combat (choose which player to attack)
- [ ] Extend agents: threat assessment, politics, multiplayer targeting
- [ ] Commander-specific KG data (EDHREC staples, commander synergies)
- [ ] Test 4-agent Commander games
- [ ] Moxfield/Archidekt decklist importing

### Phase 8: Human Player & Polish (Weeks 23–25)

**Goal:** A human can sit at the table and play against agents.

- [ ] Implement human player adapter using LangGraph interrupt()
- [ ] Build interactive UI: clickable hand, target selection, combat choices
- [ ] Add game log with natural language descriptions of actions
- [ ] Card hover previews with Scryfall images
- [ ] Stack visualization
- [ ] Concede / undo (within reason) support
- [ ] Save / load game states
- [ ] End-to-end testing: full human vs 3 AI Commander game

---

## 19. Open Questions & Risks

### Technical Risks

| Risk | Mitigation |
|------|-----------|
| **LLM hallucination on rules** | Judge agent with RAG over Comprehensive Rules; structured action validation by rules engine (never trust raw LLM output for game state changes) |
| **Game engine complexity** | **Fork/adapt existing MIT-licensed Python engines** (open-mtg for game loop, mtg-python-engine for stack/triggers, mtg-player for agent tooling). Use Forge/XMage as correctness reference. See Appendix C. |
| **Scryfall rate limits** | Bulk data download + local SQLite cache; only hit API for missing cards |
| **Moxfield API not public** | Fall back to Archidekt API or plaintext decklist importing |
| **Active inference scalability** | Start with heuristic approximations; exact Bayesian inference is intractable over full card space. Use variational methods. |
| **Training compute** | Start with small neural modules; self-play on simplified game states before scaling |
| **Card interactions are combinatorial** | Knowledge graph + LLM in the loop handles novel interactions; hard-code only the most common patterns |

### Design Decisions to Make

1. **How much of the rules engine to hard-code vs. LLM-interpret?**
   - Recommendation: Hard-code turn structure, zones, stack, mana, combat math. LLM-interpret complex abilities and novel interactions.

2. **Which LLM to use for agents vs. judge?**
   - Agents: GPT-4o or Claude (cost-effective, fast)
   - Judge: GPT-4o or Claude Opus (accuracy-critical, less frequent calls)
   - Training: Local model (Llama 3) for high-volume self-play

3. **Knowledge graph persistence?**
   - **Decision: Neo4j from the start** — n10s for OWL ontology, APOC for graph algorithms, native vector indexes. No need for in-memory MVP + later migration.

4. **How to handle cards with extremely complex text?**
   - Parse keywords mechanically; fall back to LLM interpretation with judge validation for novel/complex abilities.

5. **Should agents communicate (table talk in Commander)?**
   - Yes — via structured message passing. Agents can propose deals, warn about threats, bluff. This is a core part of Commander politics.

---

## Appendix A: Key API Endpoints

### Scryfall

| Endpoint | Purpose |
|----------|---------|
| `GET /cards/named?exact={name}` | Fetch card by exact name |
| `GET /cards/search?q={query}` | Search cards with Scryfall syntax |
| `GET /cards/{id}/rulings` | Get rulings for a card |
| `GET /cards/collection` | Batch fetch up to 75 cards |
| `GET /bulk-data` | Download full card database |
| `GET /cards/{code}/{number}` | Get card by set code + collector number |

Image URIs follow the pattern: `https://cards.scryfall.io/{size}/front/{a}/{b}/{uuid}.jpg`
Sizes: `small` (146×204), `normal` (488×680), `large` (672×936), `png` (745×1040), `art_crop`, `border_crop`

### Commander Spellbook

| Endpoint | Purpose |
|----------|---------|
| `GET /api/variants/` | All combos |
| `GET /api/variants/?q={card_name}` | Combos with a specific card |

---

## Appendix B: Example Knowledge Graph Queries

```python
# "I have Thassa's Oracle in hand. What combos can I assemble?"
kg.get_combos_containing("Thassa's Oracle")
# → [Combo("Thassa's Oracle + Demonic Consultation", result="Win the game"),
#    Combo("Thassa's Oracle + Tainted Pact", result="Win the game"),
#    Combo("Thassa's Oracle + Paradigm Shift", result="Win the game")]

# "My opponent played Island, Sol Ring, Arcane Signet. What archetype?"
kg.infer_archetype_from_cards(["Island", "Sol Ring", "Arcane Signet"])
# → {"Blue-based Control": 0.25, "Simic Value": 0.15, "Azorius Stax": 0.12, ...}
#   (Too early to tell — Sol Ring and Arcane Signet are in every deck)

# "Opponent played Rhystic Study. What removes it?"
kg.get_answers_to("Rhystic Study")
# → ["Enchantment removal": [Nature's Claim, Beast Within, Cyclonic Rift, ...],
#    "Counterspell (on cast)": [Counterspell, Swan Song, Dovin's Veto, ...]]

# "What synergizes with Doubling Season in my deck?"
kg.get_synergies_for("Doubling Season")
# → [Synergy(Doubling Season, Planeswalkers, "Enter with double loyalty counters"),
#    Synergy(Doubling Season, token producers, "Double token generation"), ...]
```

---

## Appendix C: Reference Projects & Papers

| Resource | Relevance |
|----------|-----------|
| [mindcrank](https://github.com/dylanlott/mindcrank) | Monte Carlo deck simulation for MTG (Go) — inspiration for probabilistic modeling |
| [Forge MTG](https://github.com/Card-Forge/forge) | Open-source MTG game engine (Java) — reference for rules implementation |
| [XMage](https://github.com/magefree/mage) | Open-source MTG client (Java) — most complete rules engine |
| [pymdp](https://github.com/infer-actively/pymdp) | Python active inference library — foundation for belief updating |
| [AlphaZero paper](https://arxiv.org/abs/1712.01815) | Self-play + MCTS + neural networks for game AI |
| [LangGraph docs](https://langchain-ai.github.io/langgraph/) | LangGraph documentation for state machines |
| [Commander Spellbook](https://commanderspellbook.com/) | MTG combo database |
| [EDHREC](https://edhrec.com/) | Commander card popularity and synergy data |
| [Scryfall API](https://scryfall.com/docs/api) | Card data, images, and rulings |
| [MTG Comprehensive Rules](https://magic.wizards.com/en/rules) | The full rules document (~280 pages) |
| [Active Inference book (Parr, Pezzulo, Friston)](https://mitpress.mit.edu/books/active-inference) | Theoretical foundation for active inference |
| [Graph Neural Networks survey](https://arxiv.org/abs/1901.00596) | GNN foundations for knowledge graph reasoning |
