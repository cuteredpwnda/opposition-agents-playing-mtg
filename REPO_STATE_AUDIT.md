# Repository State Audit (May 20, 2026)

## 1. Overall Architecture Status

### Current Situation (Dual-Engine State)
The repo is in **transition from Python engine → phase-rs**:

```
src/
├── engine/           ← LEGACY (Python-only, should deprecate)
├── orchestrator/     ← LEGACY (depends on Python engine)
├── integrations/phase_rs/  ← PRIMARY (WebSocket bridge to Rust)
├── agents/           ← ACTIVE (works with both)
├── knowledge/        ← ACTIVE (KG with write capability)
├── training/         ← MIXED (some legacy, some phase-rs-ready)
└── world_model/      ← MIXED (depends on legacy + new trajectory system)

examples/
├── demo_game_simple.py          ← LEGACY (Python engine)
├── play_with_real_decks.py      ← LEGACY (Python engine)
├── tournament_demo.py           ← LEGACY (Python engine)
├── play_edh_pod.py              ← LEGACY (Python engine)
├── meta_game.py                 ← LEGACY (Python engine)
└── play_phase_rs.py             ← PHASE-RS (but incomplete)

scripts/
├── benchmark_agents.py          ← LEGACY (Python engine)
├── run_tournament.py            ← LEGACY (Python engine)
├── run_empirical_training.py    ← LEGACY (Python engine)
├── deploy.py                    ← LEGACY (uses GameRunner)
├── phase_rs_rollout_sweep.py    ← PHASE-RS (primary ablation, has issues)
├── run_phase_rs_ablation.py     ← PHASE-RS (wrapper around rollout_sweep)
├── collect_phase_rs_traces.py   ← PHASE-RS (trace collection for training)
├── run_kg_enrichment.py         ← KG (writes to Neo4j, ready to integrate)
└── train_pipeline.py            ← MIXED (needs phase-rs traces integration)

tests/
├── test_engine/                 ← LEGACY (Python engine tests)
├── test_agents/                 ← MIXED (some Python, some phase-rs)
└── test_knowledge/              ← ACTIVE (KG tests)
```

### Code Size
- **Python engine** (src/engine/): ~8000 LOC (comprehensive, but redundant)
- **phase-rs integration** (src/integrations/phase_rs/): ~1500 LOC (thin, working)
- **Agents** (src/agents/): ~3000 LOC (production)
- **Knowledge graph** (src/knowledge/): ~1500 LOC (production-ready)
- **Tests**: ~4000 LOC (mostly Python engine, need phase-rs coverage)

---

## 2. Legacy Components (Should Move to `src/engine_legacy/`)

### Keep (with deprecation warnings)
- **src/engine/game_state.py** — 500 LOC, used in tests
- **src/engine/rules_engine.py** — 1200 LOC, comprehensive CR implementation
- **src/engine/combat.py** — 800 LOC
- **src/engine/continuous_effects.py** — 380 LOC
- **src/engine/replacement_effects.py** — 400 LOC
- **src/engine/triggered_abilities.py** — 600 LOC
- **src/engine/stack.py** — 400 LOC
- **src/engine/phases.py** — 300 LOC
- **All other engine files** (~3000 LOC of tactical mechanics)

### Deprecate (move to `src/engine_legacy/`)
- **src/orchestrator/** (3 files, ~600 LOC)
  - `game_runner.py` — uses Python engine
  - `priority_loop.py` — turn structure for Python engine
  - `game_graph.py` — game serialization for Python engine

- **examples/demo_game_simple.py**
- **examples/tournament_demo.py**
- **examples/play_with_real_decks.py**
- **examples/play_edh_pod.py**
- **examples/meta_game.py**

- **scripts/benchmark_agents.py**
- **scripts/run_tournament.py**
- **scripts/run_empirical_training.py**
- **scripts/deploy.py**

- **tests/test_engine/~20 files** (but keep for regression testing)

### Delete (obsolete)
- **scripts/goldfish.py** — old heuristic baseline
- **scripts/benchmark.py** — old benchmark harness
- **scripts/ablation_sweep.py** — old Python-engine ablation
- **examples/demo_phi_agents.py** — incomplete POC

---

## 3. Production Components (Keep in src/)

### Core (No Changes)
- **src/agents/** — All agents, ready for phase-rs
- **src/knowledge/** — KG with write capability ✅
- **src/world_model/** — JEPA + trajectory infrastructure
- **src/training/** — Training harness (needs phase-rs traces integration)
- **src/integrations/phase_rs/** — WebSocket bridge (primary)
- **src/utils/** — Utilities

### Needs Refactoring
- **src/training/benchmark.py** — Update to phase-rs
- **src/training/self_play.py** — Update to phase-rs traces
- **scripts/train_pipeline.py** — Integrate KG enrichment
- **scripts/collect_phase_rs_traces.py** — Already phase-rs, needs KG post-processing

---

## 4. Current Issues (Blocking Ablations)

### Issue A: Phase-rs Server Connection Hanging
**Symptom**: Timeout at 181.59s, then ConnectionResetError
**Root Cause**: phase-rs server (in external/phase-rs) is not responding after ~180s
**Impact**: First game of ablation times out, entire sweep fails
**Solution**: 
- Check phase-rs server process health
- Add connection keep-alive / heartbeat
- Or: reduce stream_timeout back to 45s but with better error handling

### Issue B: Multiple Ablation Processes Contending
**Symptom**: 6 zombie Python processes all listening on port 9374
**Root Cause**: No mutual exclusion; each starts its own phase-rs server
**Impact**: Only one can acquire the port; others hang indefinitely
**Solution**: Use a global lock file or environment variable to ensure single server

### Issue C: Ablation Script Has No Timeout Handling
**Symptom**: ConnectionResetError crashes the sweep, no recovery
**Root Cause**: reconnect logic only handles `asyncio.TimeoutError`, not `ConnectionResetError`
**Solution**: Wrap more exception types (connection errors, socket errors)

---

## 5. KG Writing Infrastructure (Already Exists!)

### Implemented ✅
1. **src/knowledge/kg_enrichment.py** (250 LOC)
   - Analyzes trajectories for card synergies
   - Discovers combos from co-occurrence
   - Writes to Neo4j via `append_learned_synergy()` and `update_card_stats()`

2. **src/knowledge/knowledge_graph.py** methods:
   - `append_learned_synergy(card_a, card_b, confidence)` ✅
   - `update_card_stats(card, wins, games)` ✅
   - Query methods: `get_learned_synergies()`, `get_card_learned_stats()` ✅

3. **scripts/run_kg_enrichment.py** (80 LOC)
   - Standalone enrichment runner
   - Loads trajectories from `data/trajectories/`
   - Writes to Neo4j

### Gap: Not Integrated with phase-rs Pipeline
**Current Flow**:
```
phase_rs_rollout_sweep.py → runs/phase_rs_ablation_*/rollouts.jsonl
                                        ↓
                                    (manual)
                                        ↓
                              run_kg_enrichment.py
                                        ↓
                                   Neo4j writes
```

**Needed Flow**:
```
phase_rs_rollout_sweep.py → runs/phase_rs_ablation_*/traces/
                                        ↓
                              [Auto-trigger KG enrichment]
                                        ↓
                              KGEnrichment.enrich_from_jsonl()
                                        ↓
                              Neo4j writes (LearnedSynergyEvidence)
                                        ↓
                              Agents query enhanced KG
                                        ↓
                              [Next ablation cycle]
```

### Action: Wire up KG enrichment
- Create `src/integrations/phase_rs/kg_enrichment_adapter.py`
  - Parse phase-rs traces JSONL → TrajectoryStore
  - Hook into `phase_rs_rollout_sweep.py` post-game
  - Write to KG automatically

---

## 6. Refactoring Plan (Priority Order)

### Phase 1: Stabilize Ablations (Today)
- [ ] Fix phase-rs connection handling (more exception types)
- [ ] Add process lock to prevent zombie contention
- [ ] Restart single clean ablation run
- [ ] Verify it completes without errors

### Phase 2: Reorganize Code (Tomorrow)
- [ ] Create `src/engine_legacy/` with deprecation warnings
- [ ] Move orchestrator → `src/orchestrator_legacy/`
- [ ] Move legacy examples → `examples_legacy/`
- [ ] Move legacy scripts → `scripts_legacy/`
- [ ] Update all imports in remaining code
- [ ] Tests should still pass

### Phase 3: Integrate KG Writing (Next Day)
- [ ] Create `src/integrations/phase_rs/kg_enrichment_adapter.py`
- [ ] Parse JSONL traces into TrajectoryStore
- [ ] Hook into `phase_rs_rollout_sweep.py` at end
- [ ] Write to Neo4j automatically
- [ ] Test end-to-end: ablation → traces → KG update

### Phase 4: Update Training Pipeline (Next Day)
- [ ] Modify `scripts/train_pipeline.py` to:
  - Start with phase-rs traces (not Python engine)
  - Feed into JEPA training
  - Use KG-enriched agent decisions
- [ ] Modify `scripts/collect_phase_rs_traces.py` to:
  - Auto-run KG enrichment after traces collected
  - Report KG update statistics

### Phase 5: Delete Legacy (After Validation)
- [ ] Archive `examples_legacy/`, `scripts_legacy/` in a branch
- [ ] Delete old orchestrator
- [ ] Delete Python engine tests that are now redundant
- [ ] Keep unit tests for mechanics (for documentation)

---

## 7. Expected Outcomes

### Before Refactoring
```
Repo state: Dual-engine, 35 legacy files, 15K LOC of redundant code
Ablation: Failing (connection issues, zombie processes)
KG writing: Implemented but not integrated
Training pipeline: Partial (no phase-rs traces → JEPA)
```

### After Refactoring
```
Repo state: phase-rs-first, legacy isolated, ~8K LOC removed
Ablation: Running cleanly (81 games, ~2-4 hours)
KG writing: Auto-integrated (traces → KG on each ablation)
Training pipeline: Full (traces → JEPA → eval with KG)
Tests: 100+ tests passing (Python engine + phase-rs integration)
```

---

## 8. File-by-File Changes

### Move to `src/engine_legacy/`
```
src/engine/abilities.py              → src/engine_legacy/
src/engine/cascade.py                → src/engine_legacy/
src/engine/channel.py                → src/engine_legacy/
... (all other engine files)         → src/engine_legacy/
```

### Move to `src/orchestrator_legacy/`
```
src/orchestrator/game_runner.py      → src/orchestrator_legacy/
src/orchestrator/priority_loop.py    → src/orchestrator_legacy/
src/orchestrator/game_graph.py       → src/orchestrator_legacy/
```

### New Files
```
src/integrations/phase_rs/kg_enrichment_adapter.py  ← NEW
examples/play_phase_rs_with_kg.py                   ← NEW (complete example)
scripts/phase_rs_ablation_to_training.py            ← NEW (pipeline wrapper)
```

### Updates
```
scripts/phase_rs_rollout_sweep.py    ← Hook KG enrichment
scripts/collect_phase_rs_traces.py   ← Hook KG enrichment
scripts/train_pipeline.py            ← Use phase-rs traces
src/integrations/phase_rs/runner.py  ← Better exception handling
```

---

## 9. Testing Strategy

### Before
- Run all tests: `pytest tests/ -q`
- Run legacy engine tests: `pytest tests/test_engine/ -q`
- Run phase-rs tests: `pytest tests/integrations/test_phase_rs* -q`

### After
- Run all tests (legacy engine removed from main):
  `pytest tests/ -q`
- Run legacy in isolation (if needed):
  `pytest tests_legacy/test_engine/ -q`
- Run phase-rs (primary):
  `pytest tests/integrations/test_phase_rs* -q`

---

## 10. Timeline

| Phase | Task | Hours | Status |
|-------|------|-------|--------|
| 1 | Fix connection handling + restart ablation | 2h | 🔄 In progress |
| 2 | Reorganize code (move files, update imports) | 3h | 📋 Queued |
| 3 | Integrate KG writing | 2h | 📋 Queued |
| 4 | Update training pipeline | 2h | 📋 Queued |
| 5 | Delete legacy (after validation) | 1h | 📋 Queued |
| **Total** | | **10h** | |

---

**Next action**: Execute Phase 1 (fix connection + restart ablation)
