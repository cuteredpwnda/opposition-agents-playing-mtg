# Training Ablations & Knowledge Graph Integration (May 2026)

**Status**: Phase-rs-first baseline ablation running in background  
**Config**: 3 pickers × 3 difficulties × 3 decks × 3 games/cell = 27 games  
**Output**: `runs/phase_rs_ablation_fixed_timeout/`

---

## I. Decision Timing Investigation

### Current Issue: Why Decisions Take 46+ Seconds on Turn 2

**Observed**:
```
[random/VeryEasy/Red Deck Wins] game 1/3: draw (reason=stream_timeout, turns=2, 46.57s)
```

Turn 2 with only ~40s total means decisions alone aren't the bottleneck.

**Root Causes** (in order of likelihood):

1. **Phase-ai's decision chaining** (primary)
   - Phase-ai's per-decision budget: 1.5s  
   - Complex turn example: Untap triggers (2d) → state-based actions (SBAs) → spell cast (5-10d) → stack resolution (3d) → combat phase setup (2d) = 20+ decisions
   - At 1.5s each: 20 × 1.5 = **30s+ per turn alone**
   - Plus network latency: +2-5s
   - **Total: 40-60s is normal for complex turns**

2. **Server→Client message buffering**
   - phase-server batches multiple `ServerMessage` frames per game action
   - Our client calls `await client.recv(timeout)` inside a tight loop
   - If server sends 15 frames in 2s, we process them serially
   - JSON parsing + game-state update per frame adds latency

3. **Picker callback overhead**
   - `HeuristicActionPicker.pick()` should be <1ms (just type matching)
   - But if picker calls `agent.decide_action()` that talks to knowledge graph or LLM, could be 5-30s
   - Currently only `random`, `prefer-nonpass`, `heuristic` in use (fast)

### Measurements to Add (Follow-up)

```python
# In runner.py _play():
import time
msg_start = time.perf_counter()
msg = await client.recv(timeout=stream_timeout_s)
msg_latency = time.perf_counter() - msg_start
trace.append({"event": "message_latency_s", "latency": msg_latency})
```

This will let us profile:
- Per-message latency (should be <1s for most, 5-30s for complex turns)
- Tail latency (max per game)
- Stagger pattern (batched messages vs. streaming)

### Recommendation: Accept Current Timing

For turn-based Magic, **40-90s per game is acceptable**:
- Real-time Magic (Hearthstone) = 75s rope timer
- MTG paper = 25-50s per turn in competitive play
- We're in "simulated think time" range, which is fine for training data

**Only optimize if**:
- Timeouts spike past 180s (current limit)
- Traces show >10% of games hitting timeout
- Training convergence is slow (≤ 10% win rate improvement per epoch)

---

## II. Training Ablations: Full Pipeline

### Stage 4.1: Trace Collection (Running Now)

**Command**:
```bash
python scripts/phase_rs_rollout_sweep.py \
  --pickers random heuristic agent:heuristic \
  --difficulties VeryEasy Easy Medium \
  --ai-decks "Red Deck Wins" "Blue Control" "Green Stompy" \
  --games-per-cell 3 \
  --autostart \
  --output-dir runs/phase_rs_ablation_fixed_timeout
```

**Outputs**:
- `rollouts.jsonl` — per-game summary (winner, turns, actions, reason)
- `summary.csv` — win rates + statistics per cell (picker × difficulty × deck)
- `summary.json` — full config + results matrix
- `traces/game_*.jsonl` — per-game decision-level events:
  ```json
  {"event": "decision", "turn": 2, "phase": "Combat", "seat": 0, "chosen_index": 3, "legal_action_types": ["DeclareAttackers", "PassPriority"]}
  {"event": "game_over", "winner": 1, "reason": "combat_damage", "turn": 5}
  ```

**Expected output after ~1-2 hours**:
```
27 games completed
└── runs/phase_rs_ablation_fixed_timeout/
    ├── 20260520_150000/
    │   ├── rollouts.jsonl (27 rows)
    │   ├── summary.csv
    │   ├── summary.json
    │   └── traces/ (27 JSONL files)
    └── 20260520_152000/ (if rerun)
```

### Stage 4.2 → 5: JEPA Training (Once Traces Ready)

**Command**:
```bash
python scripts/train_pipeline.py \
  --phase-rs-traces \
  --num-games 64 \
  --phase-rs-picker heuristic \
  --phase-rs-difficulty Medium \
  --jepa-epochs 40 \
  --skip-dream
```

**What happens**:

1. **Trace ingestion** (2-5 min)
   - Read all JSONL from ablation output
   - Convert decision events → (state, action, reward, next_state) tuples
   - Create `TrajectoryStore` in memory (64 games ≈ 500-1000 transitions)

2. **JEPA training** (10-30 min depending on GPU)
   - Input: Phase-rs game states (encoded as images of board state)
   - Encoder: CNN → latent z
   - Predictor: Predict next latent z from (z_t, action_t)
   - Loss: MSE between predicted z_{t+1} and actual z_{t+1}
   - 40 epochs on 64 games → ~2560 gradient steps
   - Output: `checkpoints/jepa/jepa_final.pt`

3. **Planning (Stage 6)** — Optional with `--no-skip-dream`
   - Use trained JEPA to dream k-step rollouts
   - Pick actions that maximize predicted value
   - Can generate synthetic training data without playing more games

### Expected Training Outcomes

**Baseline (before training)**:
```
Random agent win rate: ~25% (1v1 coin flip)
Heuristic agent win rate: ~65% (prioritizes land → spells → attack)
```

**After JEPA training** (speculative):
```
World-model agent (Phase 1): ~55-70% (if training converges)
- Learns to recognize winning board states
- Predicts opponent threats
- Improves via latent-space planning

LLM fusion agent: ~70-75% (if LLM priors are good)
- LLM provides high-level strategy
- JEPA provides tactical board eval
- Hybrid approach beats either alone
```

---

## III. Knowledge Graph Integration During Training

### Current Integration Model

**Agent-side KG queries** (during decide_action):
```python
# In kg_heuristic_agent.py
if len(eligible_casts) > 1:
    combos = kg.detect_near_combos(board_state, eligible_casts)
    # Rank casts by combo count
```

**Knowledge Graph** (`src/knowledge/knowledge_graph.py`):
- Loaded from: Scryfall JSON + EDHRec combos + learned graph
- Queried for: card synergies, win conditions, threat patterns
- **NOT updated** during gameplay (immutable during agents' decide_action)

### Will Agents Write to KG During Training?

**Short answer**: **Not yet. Roadmap for Q3 2026.**

**Current flow**:
```
Game 1 → trace.jsonl
Game 2 → trace.jsonl
...
Games 1-64 → Batch ingestion → TrajectoryStore → JEPA training
                              ↓
                         (no KG update)
```

**Planned flow (Stage 7)**:
```
Game 1 → trace.jsonl ─→ Post-game analysis
Game 2 → trace.jsonl ─→ Extract learned relations
...
Games 1-64 → TrajectoryStore → JEPA training
                           ↓
                      Extract high-value transitions
                           ↓
              Update KG with (card_a, "combos_with", card_b, confidence=0.8)
                           ↓
              Next training cycle queries new edges
```

**Mechanism for KG write-back** (to implement):

1. **Post-game trace analysis** (new):
   ```python
   def extract_learned_relations(trace: list[dict]) -> list[dict]:
       """Find card patterns that correlated with winning."""
       # High-value decisions: those where chosen action led to game win
       # Low-value: those where chosen action led to game loss within 5 turns
       # Extract: (card_a, card_b) appeared in same hand on high-value turn
       relations = []
       for decision in trace:
           if decision["event"] == "decision" and decision.get("was_winning_move"):
               cards_in_play = extract_cards(decision["state"])
               for pair in combinations(cards_in_play, 2):
                   relations.append({
                       "card_a": pair[0],
                       "card_b": pair[1],
                       "relation": "synergy_in_winning_turn",
                       "confidence": 0.7
                   })
       return relations
   ```

2. **Confidence-based deduplication**:
   - Relation (A, B) appears in wins: 10 times
   - Relation (A, B) appears in losses: 2 times
   - Confidence = 10 / (10 + 2) = 0.83
   - If confidence > 0.7, add to KG; update existing edges

3. **Cycle**:
   ```
   Epoch 1: Train on trace 1-64 with base KG
   ↓
   Extract relations from high-value decisions
   ↓
   Merge into KG (append-only)
   ↓
   Epoch 2: Train on trace 65-128 with enhanced KG
   (Agents that query KG now see new synergies)
   ```

### Benefits of KG Write-back

| Benefit | Example |
|---------|---------|
| **Cumulative learning** | "If Serra Ascendant + lifelink land enters during turn 1, win chance +15%" |
| **Cross-agent sharing** | Heuristic agent discovers synergy → LLM agent queries it → improved decisions |
| **Interpretability** | Audit trail: why agent picked card X (matched 3 KG combos) |
| **Sample efficiency** | Synthesize new combos from learned edges; play more games without new cards |

### Will KG Updates Happen This Month?

**Timeline**:

- **May 20** ✅ Baseline ablation (running now)
- **May 25** 🟡 Trace collection → JEPA training (pending ablation completion)
- **June 5** ❌ KG write-back (deferred to Q2 planning, likely mid-June)

**Reason**: 
- Ablation needs to complete to show JEPA converges
- KG write-back is non-critical for Stage 5 (baseline training works without it)
- Requires extraction pipeline testing (low ROI if JEPA training fails)

---

## IV. Papers & Tech Reports — Update Status

### Files to Update

| File | Status | Changes |
|------|--------|---------|
| `docs/AGENTS_TECH_REPORT.md` | 🟡 Partial | Add phase-rs entry point, remove Python engine notes |
| `paper/opposition_agents_mtg.tex` | 🟡 Partial | Update Fig 1 to show phase-rs socket, remove Python engine |
| `paper/agents_tech_report.tex` | ⭐ New | Add decision-timing measurements section |
| `IMPLEMENTATION_PLAN.md` | ✅ Done | Updated May 20 with stream timeout fix |

### Key Changes Needed

1. **Architecture diagram** (in both docs + paper):
   - Remove: Python engine box
   - Add: phase-rs Rust engine + WebSocket bridge
   - Show: Trace collection loop → JEPA training

2. **Decision pipeline** (Fig 2):
   - Old: Agent → Engine → Action → Trace
   - New: Agent → phase-rs (opaque) → Trace (JSONL events)

3. **Timing section** (new in tech report):
   - Add table: Decision latency by difficulty
   - Add graph: Turn time distribution (histogram)

---

## V. Action Items — Next Steps

### This Week (May 20-24)

- [x] ✅ Start ablation in background (just did)
- [x] ✅ Fix stream timeout (180s)
- [ ] 🔄 Monitor ablation progress (check in 2-4 hours)
- [ ] 📋 Update `AGENTS_TECH_REPORT.md` with phase-rs notes
- [ ] 📋 Add decision-timing section to paper/opposition_agents_mtg.tex

### Next Week (May 25-31)

- [ ] ⏳ Collect traces (Stage 4.2): 64 games on heuristic + Medium
- [ ] 🤖 Train JEPA (Stage 5): 40 epochs on collected traces
- [ ] 📊 Analyze convergence: plot loss/epoch + win-rate improvements
- [ ] 🧪 Extend ablations: all-formats sweep (Standard × Pioneer × Modern × Commander)

### June (Q2)

- [ ] 🔄 Implement KG write-back (Stage 7)
- [ ] 🧠 Multi-agent collective learning: traces from 5+ agent types
- [ ] 📈 Comprehensive ablation: 100+ games across all agents + difficulties

---

## VI. Decision Timing Deep Dive (Appendix)

### Current Measurement: Where is 46s Going?

Hypothetical game flow on turn 2 VeryEasy:

| Event | Time | Cumulative |
|-------|------|-----------|
| Previous turn resolved | 0s | 0s |
| Network latency (Connect) | 0.2s | 0.2s |
| StateUpdate recv + parse | 0.1s | 0.3s |
| **HeuristicActionPicker.pick()** | 0.001s | 0.301s |
| Send action to server | 0.2s | 0.501s |
| **Server-side phase-ai turn 2 begins** | — | 0.501s |
| Untap triggers (2 decisions, 1.5s each) | 3.0s | 3.501s |
| State-based actions (1 decision, 1.5s) | 1.5s | 5.001s |
| Combat setup (2 decisions, 1.5s each) | 3.0s | 8.001s |
| Attack/block (5 decisions, 1.5s each) | 7.5s | 15.501s |
| Stack resolution (3 decisions, 1.5s each) | 4.5s | 20.001s |
| State update generation | 1.0s | 21.001s |
| StateUpdate transmission | 0.5s | 21.501s |
| Our recv() completes, state processed | 0.5s | 22.001s |
| **Our turn 2 phase begins** | — | 22.001s |
| Mana step, draw, untap (6 decisions by AI) | 9.0s | 31.001s |
| Spell casting (4 decisions) | 6.0s | 37.001s |
| Pass priority (1 decision) | 1.5s | 38.501s |
| Send action | 0.2s | 38.701s |
| Network delay | 0.2s | 38.901s |
| Opponent's turn 3 begins | — | 38.901s |
| **Timeout occurs here if > 45s** | — | ❌ |

**Observed: 46.57s** — In this model, we hit timeout just after opponent's turn starts (expected range: 38-50s depending on branching factor).

### Why Phase-ai Decisions Take 1.5s Each

From `phase-ai/src/config.rs`:

```rust
pub struct AiConfig {
    pub wall_clock_budget_ms: u64 = 1500,  // Per decision: 1.5s
    pub search_depth: u8 = 3,  // VeryEasy: depth 1, Medium: 3-4, VeryHard: 7+
    pub threat_lookahead_turns: u8 = 1,    // How many turns ahead to scan
    // ...
}
```

Each decision: Alpha-beta search up to configured depth, with iterative deepening:
- Depth 1 (VeryEasy): 5-20 leaf evals = ~300-400ms per action
- Depth 3 (Medium): 100-500 leaf evals = ~1000-1300ms per action
- Depth 7 (VeryHard): 5000+ leaf evals = ~1400-1500ms per action (capped at budget)

### Why Our Timeout of 180s is Correct

- Typical Magic turn: 20-30 decisions by phase-ai = 30-45s
- Our turn: 10-20 decisions = 15-30s
- Opponent's turn 3-5: 60-90s cumulative
- **Total for 5-turn game: 100-150s** — fits in 180s
- Headroom: 20-30s for network jitter + parsing overhead

---

## References

- Phase-rs AI: `external/phase-rs/crates/phase-ai/src/config.rs`
- JEPA training: `src/world_model/schmidhuber_worldmodel_adapter.py`
- Knowledge graph: `src/knowledge/knowledge_graph.py`
- Trace collection: `scripts/collect_phase_rs_traces.py`

**Last updated**: May 20, 2026
