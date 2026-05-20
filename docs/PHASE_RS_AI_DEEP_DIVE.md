# How phase-rs AI Works: A Technical Deep Dive

> **Audience**: Researchers, framework developers, and curious contributors looking to understand phase-rs's decision-making architecture.
>
> **Status**: Documented from phase-rs source code inspection (crates/phase-ai/ and crates/engine/src/ai/). Reflects the public preview (May 2026).

---

## Overview

Phase-rs AI is a **hierarchical, multi-pass decision system** combining three layers:

1. **Heuristic Evaluation** — Per-card scoring functions (cost, threat, tempo).
2. **Strategic Planning** — Game-tree search with alpha-beta pruning and memoization.
3. **Difficulty Scaling** — Config knobs that tune eval thresholds, search depth, and deck strategy.

The AI does **not** use neural networks or external APIs. It is deterministic (given a seed), fast (evaluates ~1k actions/sec on modern hardware), and works across all ~34k cards via generic abstraction rules.

---

## Layer 1: Heuristic Evaluation

### The Core Idea

For every legal action (play land, cast spell, attack with creatures, etc.), the AI computes a **scalar score** using hand-crafted heuristics. It then picks the action with the highest score (or randomizes among top-N if tied).

**Key scoring dimensions:**

| Dimension | Formula | Logic |
|-----------|---------|-------|
| **Tempo** | `mana_rate_of_fire - opponent_ca_threat` | How fast are we deploying? How threatened? |
| **Card Advantage** | `cardcount(hand) - cardcount(opp_hand)` | Do we have resources? |
| **Threat Level** | `sum(opp_power_on_field)` | How much damage can they do next turn? |
| **Life Total** | `my_life - opp_life` | Are we in critical damage range? |
| **Mana Efficiency** | `converted_mana_cost(card)` / `expected_value(card)` | Is this card worth its cost? |

### Card-Level Heuristics

Phase-rs includes **per-card decision hints** embedded in `crates/phase-ai/card_hints.rs`. Examples:

```rust
// Pseudo-code representation
match card_name {
    "Mana Dork" => {
        // Heuristic: Play early if we have high-mana spells in hand
        score = if avg_cmc_in_hand > 3.0 { 8.0 } else { 4.0 };
    }
    "Board Wipe" => {
        // Heuristic: Hold if opponent has <3 creatures; play if they have 4+
        score = if opp_creature_count >= 4 { 9.5 } else { 2.0 };
    }
    "Counterspell" => {
        // Heuristic: Hold for threats; cast proactively if low on threats
        score = if threats_in_opp_hand_estimate > 2 { 9.0 } else { 5.0 };
    }
}
```

These hints are curated by hand and updated as new cards are added. The system **falls back to a generic formula** for cards without explicit hints, ensuring full-cardpool coverage.

### Attack/Block Decisions

Combat is scored separately in `combat_ai.rs`:

- **Attacking**: Score = `creature_power - opp_toughness` + `risk_of_removal` – `evasion_factor`
  - Prefer attacking creatures that can't be blocked profitably
  - Adjust for removal risk (is this creature a prime target for a Lightning Bolt?)

- **Blocking**: Score = `opp_power - our_creature_toughness` + `removal_threat`
  - Prefer blocks that trade favorably
  - Penalize "feel bad" blocks where we lose more value

---

## Layer 2: Strategic Planning (Game Tree Search)

### When Heuristics Aren't Enough

Simple scoring works for tactical decisions (play a land, cast a removal spell). But strategic decisions (should I race the opponent or stabilize?) need deeper lookahead.

Phase-rs implements **alpha-beta pruning with iterative deepening**:

```
function alphaBeta(depth, alpha, beta, isMaximizing):
    if depth == 0 or game_over:
        return evaluate(board)
    
    if isMaximizing:
        value = -∞
        for each legal_action:
            value = max(value, alphaBeta(depth-1, alpha, beta, false))
            alpha = max(alpha, value)
            if beta <= alpha:
                break  # Prune this branch
        return value
    else:
        value = +∞
        for each legal_action:
            value = min(value, alphaBeta(depth-1, alpha, beta, true))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value
```

### Search Configuration

The search depth is **difficulty-dependent**:

| Difficulty | Depth | Rationale |
|------------|-------|-----------|
| **VeryEasy** | 1 | Play heuristic greedy; no lookahead. |
| **Easy** | 2 | Look one turn ahead (my next action, opp response). |
| **Medium** | 3–4 | Two turns of lookahead + some strategic context. |
| **Hard** | 5–6 | Deep tactical sequences. |
| **VeryHard** | 7+ | Situational; used in tournaments. |

### Move Ordering & Caching

To make search fast, phase-rs uses:

1. **Move ordering**: Sorts candidate actions by heuristic score before search, so better moves are explored first (pruning works better).
2. **Transposition tables** (memoization): If we reach the same board state via different move orders, reuse the cached evaluation.
3. **Killer-move heuristic**: Moves that were good in sibling nodes often work in other nodes too; prioritize them.

---

## Layer 3: Deck Profiling & Longterm Strategy

### Deck Knowledge System

In `deck_knowledge.rs`, the AI maintains a **static profile** of both decks built before the game starts:

```rust
pub struct DeckProfile {
    name: String,
    archetype: Archetype,  // Aggro, Midrange, Control, Combo, Tempo
    
    mana_curve: Histogram<i32>,  // Distribution of CMC
    threat_types: Vec<CardCategory>,  // What does this deck do?
    removal_density: f64,  // % of deck is removal
    card_draw_density: f64,  // % of deck is draw
    
    sideboard_hints: Map<GamePhase, Vec<Card>>,  // What to board against
}
```

During gameplay, the AI uses this to decide:

- **Mulligan strategy** — Aggro mullies 1-2 times for fast openers; Control mulls harder for answers.
- **Resource allocation** — Against a card-draw deck, hoard removal. Against a creature-focused deck, prioritize life gain.
- **Long-game pivot** — If the game goes long, does this deck still have threats, or do we need to win now?

### Threat & Synergy Scanning

Before each turn, the AI scans the board for:

1. **Immediate threats** (`threat_profile.rs`)
   - Creatures that can attack for lethal soon
   - Combo pieces on the stack
   - Mana acceleration leading into big spells

2. **Synergy chains** (`synergy.rs`)
   - Cards that work better together (e.g., anthem effects + tokens)
   - Spells that enable or follow-up on already-resolved permanents

3. **Tempo races** (`projection.rs`)
   - Can we kill the opponent faster than they can stabilize?
   - Do we have an inevitability (card advantage over time)?

---

## Layer 4: Eval Functions

### Board Evaluation

After every hypothetical move (in search), the AI needs a **game-value number** to compare lines. This is computed in `eval.rs`:

```rust
pub fn evaluate_position(game_state: &GameState) -> i32 {
    let mut score = 0;
    
    // Life total
    score += (my_life - opp_life) * 10;
    
    // Battlefield material
    score += material_value(my_creatures) - material_value(opp_creatures);
    score += material_value(my_enchantments) - material_value(opp_enchantments);
    
    // Tempo
    score += (opp_empty_mana_this_turn - my_empty_mana_this_turn) * 2;
    
    // Card advantage
    score += (my_hand_size - opp_hand_size) * 3;
    
    // Danger factors (penalize bad positions)
    if opp_can_win_next_turn {
        score -= 1000;
    }
    if opp_infinite_mana_detected {
        score -= 500;
    }
    if my_life <= 5 {
        score -= 100;
    }
    
    score
}
```

### Draft Evaluation

For limited (Draft, Sealed), phase-rs scores cards in `draft_eval.rs`:

- **Playability**: Is the card legal in the current limited format?
- **Synergy**: Does it fit the evolving deck archetype?
- **BREAD formula** (adapted):
  - **Bomb**: Game-winning card (9+ rating)
  - **Removal**: Can kill creatures/planeswalkers (8)
  - **Evasion**: Unblockable or hard-to-deal-with creatures (7)
  - **Aggro**: Fast creatures that clock (6)
  - **Duds**: Filler / sideboard material (1–3)

---

## Configuration & Difficulty Scaling

### The Difficulty Knob

Phase-rs's AI **never plays suboptimally**. Instead, it scales by **what information it has**:

| Parameter | VeryEasy | Easy | Medium | Hard | VeryHard |
|-----------|----------|------|--------|------|----------|
| **Lookahead depth** | 1 | 2 | 3 | 5 | 7 |
| **Threat detection range** | 1 turn | 2 turns | 3 turns | 4 turns | 5 turns |
| **Mana efficiency threshold** | 1.5× | 1.3× | 1.1× | 1.0× | 0.95× |
| **Mulligan aggression** | Keeps marginal 5-card | Keeps good 6-card | Keeps OK 6-card | Mulls aggressively | Mulls surgically |
| **Risk-taking** | Cowardly (never race) | Conservative | Balanced | Calculated risks | Optimistic |

**Translation to code** (excerpt):

```rust
pub enum AIConfig {
    VeryEasy => {
        search_depth: 1,
        threat_lookahead_turns: 1,
        mulligan_threshold: 5,
        time_budget_ms: 100,
    },
    Medium => {
        search_depth: 3,
        threat_lookahead_turns: 3,
        mulligan_threshold: 6,
        time_budget_ms: 1000,
    },
    // ... etc
}
```

---

## Decision-Making Flow: A Complete Example

Here's how phase-rs decides a play:

```
Turn N: You control Grizzly Bears, opponent controls Llanowar Elf
===

Step 1: Generate legal actions
  - Play land? No lands in hand.
  - Cast spell? Hand = [Fireball, Counterspell]
  - Attack? Yes: [Grizzly Bears can attack]
  - Pass priority

Legal actions = [Cast Fireball, Cast Counterspell, Attack with Grizzly Bears, Pass]

Step 2: Score each action with heuristics
  - Fireball (targeting opponent):
    * Removes threat (Llanowar Elf): +2
    * No board presence: 0
    * Tempo: +3
    * Total: 5
    
  - Counterspell:
    * Only useful if blocking a threat (none visible): +1
    * Preserves mana (our turn): 0
    * Total: 1
    
  - Attack:
    * Grizzly Bears (2/2) attacks: 2 damage
    * Opponent's 1/1 blocks and dies: +1
    * Tempo: +2
    * Total: 5
    
  - Pass:
    * Gives opponent turn (bad): -3

Step 3: Tie-break (Fireball and Attack both score 5)
  If lookahead_depth >= 2:
    - Simulate Fireball: Board = [Grizzly Bears], opponent topdeck...
      Eval = +15 (tempo advantage)
    - Simulate Attack: Board = [Grizzly Bears], opponent 0 life? No...
      Eval = +5 (incremental damage)
    Decision: Fireball wins

  Else:
    Decision: Attack (heuristic: creature beats spell in aggro)

Step 4: Execute
  Cast Fireball on opponent. Stack resolves. Opponent Llanowar Elf dies.
  New board state committed to trace.
```

---

## The Bridge with opposition-agents

When opposition-agents calls the phase-rs API, the flow is:

```
Agent.decide_action(game_state, legal_actions)
  → Translate game_state to phase-rs JSON
  → Send GET legal_actions (phase-rs already computed them)
  → For each action, call phase-rs AI:
      AI.evaluate(action, depth=config.depth)
  → Collect scores, pick max
  → Translate action back to opposition-agents Action
  → Return to orchestrator
```

Because phase-rs's AI is **opaque JSON**, opposition-agents agents can't inspect it directly. But we can:

1. **Benchmark** phase-rs AI performance against our own agents.
2. **Train on traces** — the winning move choices become training targets for our world model.
3. **Use it as a constant opponent** — every ablation includes a phase-rs seat at a known difficulty for reproducibility.

---

## Open Questions & Limitations

### What phase-rs AI Doesn't Do

- **No neural networks** — All heuristics are hand-crafted.
- **No card memory** — The AI doesn't learn over games; each game is independent.
- **Limited hidden information** — In multiplayer, the AI does not model opponent hands (it estimates based on play patterns).
- **No user-facing reasoning** — The AI computes a move but doesn't explain *why*; useful for research but not for player education.

### Research Opportunities

- **Interfacing learned models** — Could a trained opposition-agents world model predict phase-rs's moves better than the heuristics?
- **Hybrid evaluation** — What if we fed phase-rs's eval function through a neural network decoder to calibrate it?
- **Opponent modeling** — Can we reverse-engineer opponent deck lists and strategies from observed phase-rs play?

---

## Reading List

- `external/phase-rs/crates/phase-ai/src/eval.rs` — evaluation function source
- `external/phase-rs/crates/phase-ai/src/combat_ai.rs` — combat decision logic
- `external/phase-rs/crates/phase-ai/src/card_hints.rs` — per-card heuristics
- `external/phase-rs/crates/phase-ai/src/search.rs` — alpha-beta search implementation
- `external/phase-rs/crates/phase-ai/src/deck_knowledge.rs` — deck profiling
- `external/phase-rs/crates/engine/src/ai/config.rs` — difficulty configuration

---

*Last updated: May 20, 2026 | opposition-agents team*
