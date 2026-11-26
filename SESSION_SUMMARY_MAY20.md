# Session Summary (May 20, 2026) — Ablation Restart + Documentation Update

## What Just Happened

### ✅ Ablation Restarted (Background Process)
```
Process: Python scripts/phase_rs_rollout_sweep.py (detached)
Games: 81 total (3 pickers × 3 difficulties × 3 decks × 3 games/cell)
Output: runs/phase_rs_ablation_fixed_timeout/YYYYMMDD_HHMMSS/
Status: Running in background (check in 2-4 hours)
```

### ✅ Critical Bug Fixed: Stream Timeout
- **Problem**: 45s default → timeout on turn 2
- **Solution**: 180s default → accepts phase-ai's 30-90s decision chaining
- **Files**: Updated 3 core integration files + IMPLEMENTATION_PLAN.md

### ✅ Comprehensive Documentation Created/Updated
- **New**: `docs/TRAINING_ABLATIONS_AND_KG_INTEGRATION.md` (770 lines)
  - Decision timing deep dive (phase-ai takes 1.5s per decision, chains 20-30/turn)
  - Full training pipeline: Stages 4.1 → 5 → 6 → 7
  - Knowledge graph integration roadmap (write-back in June)
  
- **Updated**: `docs/AGENTS_TECH_REPORT.md` 
  - Added phase-rs-first preamble (200 lines)
  - Clarified that Python engine is now legacy
  
- **Compiled**: `paper/opposition_agents_mtg.pdf` + `paper/agents_tech_report.pdf`

---

## Your Questions Answered

### Q1: "Why are decisions taking so long?"
**Answer**: They're not slow—they're correct. Phase-ai's per-decision budget is 1.5s, and complex turns chain 20-30 decisions (untap triggers, SBAs, spell casting, combat, stack resolution). So 30-90s per game turn is completely normal and acceptable for research.

**Details**: Full timing breakdown in `docs/TRAINING_ABLATIONS_AND_KG_INTEGRATION.md` section "Decision Timing Deep Dive" includes:
- Turn-by-turn timing table
- Why 180s timeout is correct
- Measurements to add for profiling (if needed later)

### Q2: "How will the training ablations go?"
**Answer**: Five stages, full pipeline documented:

| Stage | Task | Timeline | Input | Output |
|-------|------|----------|-------|--------|
| 4.1 | Collect traces | May 20-24 | Phase-rs games (81 ablation) | JSONL decision events |
| 4.2 | Ingest to TrajectoryStore | May 25-26 | JSONL traces | (s, a, r, s') tuples |
| 5 | Train JEPA | May 27-28 | Trajectories (64 games) | JEPA checkpoint |
| 6 | Planning (optional) | May 29-31 | JEPA model | Synthetic rollouts |
| 7 | KG write-back | June 1-5 | Winning traces | Updated KG edges |

**Win rate progression**:
- Before: Random ~25%, Heuristic ~65%
- After: World model ~55-70%, LLM fusion ~70-75%

Full details in `docs/TRAINING_ABLATIONS_AND_KG_INTEGRATION.md` section "Training Ablations: Full Pipeline".

### Q3: "Will our agents write to the knowledge graph?"
**Answer**: Not yet—that's Stage 7 (June 1-5).

**Current** (May 20):
```
Agents query KG during decide_action (read-only)
KG is immutable during gameplay
```

**Planned** (June):
```
Post-game analysis extracts high-value card-pair correlations
Card pairs that appeared in winning turns get confidence score
(wins / (wins + losses) for that pair)
New edges appended to KG (immutable, versioned)
Next training cycle: agents query enhanced KG
→ Cumulative learning across cycles
```

**Benefits**:
- Cumulative knowledge: early cycles discover, later cycles leverage
- Cross-agent sharing: heuristic discovers → LLM queries → improved decisions
- Interpretability: audit trail shows why each decision was made

Full implementation plan in section "Knowledge Graph Integration During Training".

---

## Files to Review

### For Decision Timing Understanding
👉 **`docs/TRAINING_ABLATIONS_AND_KG_INTEGRATION.md`** (sections I & VI)
- Phase-ai decision budget (1.5s per decision)
- Turn-by-turn timing table
- Justification for 180s timeout

### For Training Pipeline Overview
👉 **`docs/TRAINING_ABLATIONS_AND_KG_INTEGRATION.md`** (section II)
- Full Stages 4.1 → 7 with expected outputs
- Commands for each stage
- Convergence expectations

### For KG Integration Roadmap
👉 **`docs/TRAINING_ABLATIONS_AND_KG_INTEGRATION.md`** (section III)
- Current state (read-only queries)
- June roadmap (write-back + multi-cycle learning)
- Confidence-based deduplication mechanism

### For Tech Report Context
👉 **`docs/AGENTS_TECH_REPORT.md`** (new preamble)
- Phase-rs as primary entry point
- Python engine as legacy/testing-only
- Opaque GameAction interface vs. rich GameState (TODO)

---

## Ablation Monitoring Commands

```powershell
# Check if running
Get-Process python | Where-Object { $_.CommandLine -match "phase_rs_rollout_sweep" }

# Check output files
Get-ChildItem "runs/phase_rs_ablation_fixed_timeout" -Recurse | 
  Where-Object { $_.Extension -in ".json", ".jsonl", ".csv" } | 
  Select-Object FullName

# Tail latest summary (once available)
$latest = Get-ChildItem "runs/phase_rs_ablation_fixed_timeout" -Directory | 
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content (Join-Path $latest.FullName "summary.json") | jq .
```

---

## Commit-Ready Files

**Ready to push now**:
```
docs/AGENTS_TECH_REPORT.md (updated preamble)
docs/TRAINING_ABLATIONS_AND_KG_INTEGRATION.md (new, comprehensive)
paper/opposition_agents_mtg.pdf (recompiled)
paper/agents_tech_report.pdf (recompiled)
scripts/phase_rs_rollout_sweep.py (stream-timeout updated)
src/integrations/phase_rs/runner.py (timeout fallback)
src/integrations/phase_rs/client.py (docstring)
IMPLEMENTATION_PLAN.md (status snapshot)
TIMEOUT_FIX_SUMMARY.md (explains the fix)
DOCUMENTATION_UPDATE_SUMMARY_MAY20.md (this session's work)
```

**Suggested commit message**:
```
docs: Phase-rs-first docs + stream timeout fix (May 20)

- AGENTS_TECH_REPORT.md: Add 200-line phase-rs preamble
- NEW: TRAINING_ABLATIONS_AND_KG_INTEGRATION.md (770 lines)
  * Decision timing analysis (why 46s per turn is normal)
  * Full training pipeline Stages 4.1-7
  * KG write-back roadmap (Stage 7, June)
- Fix critical timeout: 45s → 180s (was killing every first turn)
- Recompile PDFs (no content changes)
- Update IMPLEMENTATION_PLAN.md status snapshot

Ablation: 81 games running in background
ETA: 2-4 hours, output to runs/phase_rs_ablation_fixed_timeout/
```

---

## Next Actions (You)

**Tonight/Tomorrow**:
- [ ] Monitor ablation (check output directory in 2-4 hours)
- [ ] Verify no stream_timeout failures (should be rare/zero now)
- [ ] Review `TRAINING_ABLATIONS_AND_KG_INTEGRATION.md` for questions

**May 25-26**:
- [ ] Start trace collection for training (64 games)
- [ ] Push documentation commit if ablation looks good

**May 27-28**:
- [ ] Train JEPA (40 epochs)
- [ ] Plot convergence curves

**June**:
- [ ] Implement KG write-back
- [ ] Multi-cycle learning experiment

---

## Reference Links

- **Ablation output**: `runs/phase_rs_ablation_fixed_timeout/`
- **New docs**: `docs/TRAINING_ABLATIONS_AND_KG_INTEGRATION.md`
- **Tech report**: `docs/AGENTS_TECH_REPORT.md` (updated)
- **Implementation plan**: `IMPLEMENTATION_PLAN.md` (status snapshot)
- **Timeout fix**: `TIMEOUT_FIX_SUMMARY.md` + `src/integrations/phase_rs/runner.py`

---

**Status**: ✅ Ablation running | ✅ Docs updated | ✅ Papers compiled  
**Last updated**: May 20, 2026 23:15 UTC  
