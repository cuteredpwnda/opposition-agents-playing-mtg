# Opposition Agents Playing MTG — Architecture Guide

## Vision

Build an adversarial MTG game engine where **LLM agents** play Magic against each other using a **tool-calling pattern**, informed by the best open-source MTG engines.

## Design Choices from RESEARCH.md

This document tracks which architectural patterns we adopt from reference implementations and why.

---

## Phase 1: Simplified Game Engine (Current)

### Adopted Patterns

| Reference | Pattern | Status | Why |
|-----------|---------|--------|-----|
| **mtg-player** (MIT) | Tool-calling agent + fallback heuristic | ✅ Adopted | Direct fit for LLM agents with graceful degradation |
| **open-mtg** | `game.get_moves()` returns indexed action list | ⚠️ Partial | Used for legal action generation `get_legal_actions()` |
| **mtg-python-engine** | EventEffect system for triggered abilities | 🔄 Implementing | Foundation for ETB triggers, attack triggers, etc. |
| **Argentum Engine** | Immutable state transitions | ✅ Used | Each action returns new GameState |
| **open-mtg** | Combat damage assignment by CR 509.2 | ❌ TODO | Phase 2: Add proper attack/block targeting |

### Core Files

**`src/engine/game_state.py`** — Game data model
- Zones: Library, Hand, Battlefield, Graveyard, Stack
- CardInstance: Base card with ownership, zone tracking, tapped/summoning-sick flags
- GameState: Current turn, phase, players, all cards
- Action: Indexed legal moves for agent decision trees

**`src/engine/rules_engine.py`** — Legal action generation & execution
- `get_legal_actions()` — Returns list of playable actions (indexed for agents)
- `execute_action()` — Applies action to state, returns new GameState
- Phase-aware: Only offers actions legal in current phase (no instant-speed yet)
- Creature casting:  Go directly to BATTLEFIELD (skip stack for Phase 1)

**`src/engine/combat.py`** — Combat mechanics
- Declare attackers/blockers
- Damage assignment (placeholder: ALL damage unblocked)
- State-based death checking

**`src/engine/triggered_abilities.py`** — Triggered ability system
- `TriggerEvent` enum: ENTERS_BATTLEFIELD, CREATURE_ATTACKS, SPELL_CAST, TURN_BEGINS, etc.
- `detect_enters_battlefield()` — Find cards that entered BF this turn
- `resolve_triggers()` — Execute ETB effects in sequence
- Standard effects library: `_draw_card()`, `_tutors_creature()`, etc.

**`src/orchestrator/game_runner.py`** — Game orchestration
- Turn order: UNTAP → MAIN1 → COMBAT → MAIN2 → CLEANUP
- Summoning sickness reset in UNTAP phase
- Combat action selection during COMBAT_ATTACKERS phase
- State-based actions: Death check, game termination (0 life)

**`src/agents/random_agent.py`** — Baseline agent
- Action selection with strategic bias: CAST_SPELL > PLAY_LAND > DECLARE_ATTACKERS > others
- `get_heuristic_score()` — Fallback evaluation for tree search (MCTS foundation)
- Reference pattern: mtg-player's heuristic fallback

---

## Phase 2: Instant-Speed & Stack (Next Priority)

### What We Need

| Component | Reference | Complexity | Effort |
|-----------|-----------|-----------|--------|
| **Stack resolution** | mtg-python-engine | High | Medium (2-3 days) |
| **Activated abilities** | mtg-python-engine | Medium | Medium (1-2 days) |
| **Instant-speed phases** | open-mtg | Medium | Medium (1-2 days) |
| **Targeted actions** | mtg-python-engine | Medium | Medium (1-2 days) |
| **Static effects (Auras, Enchantments)** | Argentum Engine | High | High (3-5 days) |

### Stack Implementation (mtg-python-engine pattern)

```python
# Each stack item represents spell/ability on stack
@dataclass
class StackItem:
    id: str  # unique ID
    source_card_id: str  # card that created this
    spell_type: SpellType  # SPELL / ABILITY / TRIGGER
    targets: list[str]  # Target card instance IDs
    effects: list[Effect]  # What happens on resolution
```

### Activation Pattern

```python
class ActivatedAbility:
    """Mana abilities, combat abilities, etc."""
    cost: ManaCost  # {R}, {2}{U}, {T}, etc.
    effect: Callable[[GameState], GameState]
```

**Priority**: Correct combat (Phase 2 blocker) > Instant interaction (Phase 3)

---

## Phase 3: Real Card Data & Deck Building

### Scrython Integration

Use [Scrython](https://github.com/NandaScott/Scrython) (158 ⭐, MIT) to load real MTG card data from Scryfall API.

```python
# Load card by name
import scrython
card_data = scrython.cards.Named(fuzzy="Goblin Piker").json()
# Returns: mana_cost, type_line, power/toughness, oracle_text, keywords[], etc.
```

**Benefits**:
- Phase 1: Load test decks from Scryfall instead of hardcoding
- Phase 2: Support any MTG card (20k+ in Scryfall)
- Phase 3: Deck validation, format legality checking

---

## Phase 4: LLM Agent Integration

### Architecture (from mtg-player)

```
LLM Agent
  ↓
Tool Definitions:
  • get_game_state() → serialize visible game state
  • get_legal_actions() → list of playable actions
  • execute_action(action_id) → play card/spell/attack
  ↓
Heuristic Fallback (RandomAgent):
  • If LLM fails or slow: evaluate position, pick best action
  • `get_heuristic_score()` — returns float (-100 to +100)
  ↓
Game Engine (rules_engine.py)
```

### Tool Definitions for Claude/GPT

```json
{
  "name": "get_game_state",
  "description": "View current game state (life totals, hand, board)",
  "output": { "my_life": int, "opp_life": int, "hand": [Card], "board": [Card], ... }
},
{
  "name": "get_legal_actions", 
  "description": "Get list of playable actions this turn",
  "output": [{ "action_id": int, "action_type": str, "target": Card }]
},
{
  "name": "execute_action",
  "parameters": [{ "name": "action_id", "type": "int" }],
  "description": "Play the chosen action"
}
```

---

## Phase 5: Multi-Agent Coordination (Future)

### Pattern: mtg-player + Commander Spellbook API

1. **Combo Detection**: Query [Commander Spellbook](https://commanderspellbook.com/api/) for known combos
2. **Threat Evaluation**: Score opponent's board for combo potential
3. **Coordination**: Agents discuss via tool calls vs. pure tree search

---

## Architecture Principles

### From Open-Source Research

| Principle | Source | Rationale |
|-----------|--------|-----------|
| **Immutable State** | Argentum Engine | Enables state rollback, parallel search, easier debugging |
| **Action as Integer Index** | open-mtg | Fast agent tree search (MCTS), no string parsing |
| **Triggered Ability Event Model** | mtg-python-engine | Extensible for all rules interactions |
| **Tool-Based Agent API** | mtg-player | LLM-native interaction pattern, language-agnostic |
| **Zone-Based Card Tracking** | All engines | Foundation for correct card management |

### NOT Adopting (Why)

| Pattern | Source | Why Skip |
|---------|--------|----------|
| BDD Rule Tests | Forge | Phase 1: Too early, add in Phase 3 |
| Full CR Compliance | Forge | Phase 1: Intentional simplification, add layers in Phase 2 |
| Scripted AI Trees | Magarena/Forge | LLM + heuristic is better, use MCTS for search |
| Complex Effect Layers | Argentum | Phase 1: Simple rules only, add Rule 613 in Phase 2 |

---

## File Organization

```
opposition-agents-playing-mtg/
├── src/
│   ├── engine/
│   │   ├── game_state.py          # Data model
│   │   ├── rules_engine.py        # Action generation/execution
│   │   ├── combat.py              # Combat mechanics
│   │   ├── triggered_abilities.py # ETB/attack/death triggers (mtg-python-engine pattern)
│   │   └── effect_layers.py       # [Phase 2] Static effects (Rule 613)
│   ├── orchestrator/
│   │   ├── game_runner.py         # Game loop, phase sequencing
│   │   └── action_executor.py     # Execute action on state
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── random_agent.py        # Heuristic + bias (fallback for Phase 4)
│   │   └── llm_agent.py           # [Phase 4] Claude/GPT with tool calling
│   └── utils/
│       └── card_loader.py         # [Phase 3] Scrython integration
├── tests/
│   └── test_end_to_end_game.py   # Game progression tests
└── RESEARCH.md                    # Reference implementations
```

---

## Testing Strategy

### Phase 1 (Current)

**Unit Tests**: `tests/test_end_to_end_game.py`
- ✅ Game initialization
- ✅ Multi-turn gameplay progression
- ✅ Game termination (0 life)
- ✅ Card zone integrity

### Phase 2 (Stack Resolution)

**Test**: Stack resolution with instant-speed spells
- Stack puts spell on stack
- Agent can cast instant in response
- Resolution in LIFO order

### Phase 3 (Real Cards)

**Test**: Load 5-10 real MTG decks from Scryfall, verify legality

### Phase 4 (LLM Agent)

**Test**: Claude makes legal moves, plays coherently for 10 turns

---

## Dependencies to Add (Staged)

| Package | Phase | Purpose | Installed |
|---------|-------|---------|-----------|
| `pytest` | 1 | Test framework | ✅ |
| `scrython` | 3 | Scryfall API client | ❌ |
| `langchain` | 4 | LLM tool calling | ❌ |
| `pydantic` | 3+ | Data validation | ❌ |

---

## References

**High-Impact Open Source MTG Engines:**

1. **mtg-player** (2 ⭐, MIT) — LLM agent tool pattern (MOST RELEVANT)
2. **mtg-python-engine** (67 ⭐, MIT) — Stack, triggered abilities framework
3. **open-mtg** (152 ⭐, MIT) — Combat, action indexing, heuristics
4. **Forge** (2.2k ⭐, GPL-3.0) — Complete rules gold standard (behavioral ref)
5. **Argentum Engine** (3 ⭐, GPL-3.0) — Modern Kotlin, immutable state, LLM AI integration
6. **Scrython** (158 ⭐, MIT) — Scryfall API wrapper

All patterns documented in `RESEARCH.md` line numbers added for traceability.
