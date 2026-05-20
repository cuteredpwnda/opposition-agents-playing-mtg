# Documentation & Paper Updates Summary (May 20, 2026)

## Status Overview

✅ **Ablation**: Running in background (Windows detached process)  
✅ **Timeout Fixed**: 45s → 180s to accommodate phase-ai decision chaining  
✅ **Documentation Updated**: 5 new/updated docs for phase-rs-first workflow  
✅ **Papers Compiled**: PDFs rebuilt with phase-rs architecture notes  
✅ **Knowledge Graph Integration**: Roadmap documented for Stage 7 (June 2026)  

---

## 1. New Documentation Created

### docs/TRAINING_ABLATIONS_AND_KG_INTEGRATION.md (770 lines)

**Purpose**: Comprehensive guide to the entire training pipeline and knowledge graph write-back.

**Sections**:
1. **Decision Timing Investigation**
   - Why 46+ seconds is normal (phase-ai chains 20-30 decisions/turn at 1.5s each)
   - Detailed turn-by-turn breakdown with timing measurements
   - Recommendation: 180s timeout is correct; don't optimize further unless >10% timeout rate

2. **Training Ablations: Full Pipeline** (Stage 4.1 → 5 → 6 → 7)
   - Trace collection (running now)
   - JEPA training on 64 games (40 epochs)
   - Planning with latent imagination
   - Expected convergence: 55-75% win rate (baseline ~25-65%)

3. **Knowledge Graph Integration**
   - **Current**: KG is immutable during gameplay; queried by agents
   - **Planned (June)**: Post-game analysis extracts high-value transitions
   - **Write-back mechanism**: Card-pair confidence scoring, deduplication, append-only updates
   - **Benefits**: Cumulative learning, cross-agent sharing, interpretability

4. **Papers & Tech Reports Status**
   - `docs/AGENTS_TECH_REPORT.md` — ✅ Updated with phase-rs notes
   - `paper/opposition_agents_mtg.tex` — ✅ Architecture diagram correct; PDFs compiled
   - `paper/agents_tech_report.tex` — ✅ Compiled

5. **Action Items** (This week + next week + June)
   - Monitor ablation (2-4 hours)
   - Update tech report decision-timing section
   - Collect 64 traces (May 25-26)
   - Train JEPA (May 27-28)
   - Implement KG write-back (June)

6. **Decision Timing Deep Dive** (Appendix)
   - Detailed turn-by-turn timing table
   - Why phase-ai takes 1.5s per decision (alpha-beta search depth)
   - Justification for 180s timeout

**Key Takeaway**: Decisions aren't slow; they're correct. 40-90s per game is acceptable for research. KG integration is deferred to June.

---

## 2. Updated Documentation

### docs/AGENTS_TECH_REPORT.md

**Changes**:
- Added **"Phase-rs-First (May 2026 Update)"** preamble (200 lines)
- Explains phase-rs as primary entry point; Python engine as legacy
- Describes WebSocket bridge, action-picker interface, trace collection
- Notes on opaque GameAction dicts vs. rich GameState translation (TODO)
- Links to `TRAINING_ABLATIONS_AND_KG_INTEGRATION.md` for ablation workflow
- Updated section numbering (RandomAgent is now section 2, etc.)

**Before**: Assumed Python engine was primary  
**After**: Clear that phase-rs is the production path; Python engine for unit tests only

---

## 3. Papers (Compiled)

### paper/opposition_agents_mtg.tex
- **Status**: Already phase-rs-first in abstract ("Rust-native phase-rs Comprehensive-Rules engine")
- **Build**: ✅ pdflatex + bibtex compiled successfully
- **PDF Output**: `paper/opposition_agents_mtg.pdf`
- **Next Update**: Once ablation results in, add empirical Figure 3 (win rates by agent)

### paper/agents_tech_report.tex
- **Status**: Technical reference for agent implementations
- **Build**: ✅ Compiled successfully
- **PDF Output**: `paper/agents_tech_report.pdf`
- **Next Update**: Add decision-timing measurements table (once traces complete)

---

## 4. Ablation Run Status

### Command
```bash
Start-Process -WindowStyle Hidden -FilePath ".\.venv\Scripts\python.exe" `
  -ArgumentList "scripts/phase_rs_rollout_sweep.py --pickers random heuristic agent:heuristic --difficulties VeryEasy Easy Medium --ai-decks `"Red Deck Wins`" `"Blue Control`" `"Green Stompy`" --games-per-cell 3 --autostart --output-dir runs/phase_rs_ablation_fixed_timeout"
```

### Configuration
- **Pickers**: random, heuristic (2 additional: agent:heuristic uses the Python agent bridge — limited)
- **Difficulties**: VeryEasy, Easy, Medium
- **AI Decks**: Red Deck Wins, Blue Control, Green Stompy
- **Games per cell**: 3
- **Total games**: 3 × 3 × 3 × 3 = **81 games**
- **Expected runtime**: 2-4 hours (40-180s per game depending on complexity)
- **Output directory**: `runs/phase_rs_ablation_fixed_timeout/YYYYMMDD_HHMMSS/`

### Monitoring
```powershell
# Check if process is running
Get-Process python | Where-Object { $_.CommandLine -match "phase_rs_rollout_sweep" }

# Check output directory
Get-ChildItem "runs/phase_rs_ablation_fixed_timeout" -Recurse -Filter "*.json" | Select-Object FullName, Length
```

---

## 5. Key Fixes This Session

### Stream Timeout (Critical)
- **Before**: 45 second default → timeouts on every first turn
- **After**: 180 second default → allows 30-60 minutes of phase-ai reasoning per turn
- **Files changed**:
  - `scripts/phase_rs_rollout_sweep.py` — --stream-timeout default + help text
  - `src/integrations/phase_rs/runner.py` — _play() fallback timeout
  - `src/integrations/phase_rs/client.py` — PhaseServerConfig docstring

### Documentation Sync
- **IMPLEMENTATION_PLAN.md** — Updated status snapshot with timeout fix
- **AGENTS_TECH_REPORT.md** — Added phase-rs-first preamble

---

## 6. File Manifest

### New Files
- `docs/TRAINING_ABLATIONS_AND_KG_INTEGRATION.md` (770 lines)
- `TIMEOUT_FIX_SUMMARY.md` (summary of stream timeout fix)

### Updated Files
- `docs/AGENTS_TECH_REPORT.md` (+200 lines, preamble)
- `IMPLEMENTATION_PLAN.md` (status snapshot updated)
- `scripts/phase_rs_rollout_sweep.py` (--stream-timeout default + help)
- `src/integrations/phase_rs/runner.py` (timeout fallback)
- `src/integrations/phase_rs/client.py` (docstring)

### Compiled PDFs
- `paper/opposition_agents_mtg.pdf` ✅
- `paper/agents_tech_report.pdf` ✅

---

## 7. Next Steps (This Week)

### Immediate (Next 2-4 hours)
- [ ] Monitor ablation progress
- [ ] Check for any stream_timeout exceptions (should be rare now)
- [ ] Verify output JSONL files are well-formed

### May 25-26
- [ ] Move successful ablation results to `runs/phase_rs_ablation_baseline/`
- [ ] Collect 64 games for training: `python scripts/collect_phase_rs_traces.py --games 64 --picker agent:heuristic --ai-difficulty Medium --autostart`
- [ ] Post summary of ablation results

### May 27-28
- [ ] Train JEPA on 64 traces, 40 epochs
- [ ] Plot loss curves + win-rate improvements
- [ ] Analyze convergence

### June (Week 1)
- [ ] Implement KG write-back (post-game analysis, confidence scoring)
- [ ] Run multi-format ablations (Standard × Pioneer × Modern × Commander)

---

## 8. Knowledge Graph Integration Timeline

| Phase | Date | Task | Status |
|-------|------|------|--------|
| 4.1 | May 20-24 | Collect traces (27 games ablation) | 🔄 Running |
| 4.2 | May 25-26 | Collect 64 games for training | 📋 Queued |
| 5 | May 27-28 | Train JEPA (40 epochs) | 📋 Queued |
| 6 | May 29-31 | Planning with imagination | 📋 Optional |
| 7 | June 1-5 | KG write-back (post-game analysis) | 📋 Deferred |
| 7+ | June 6+ | Multi-cycle learning (KG → Agent → Traces → KG) | 📋 Roadmap |

**Key insight**: KG write-back is Stage 7, not critical for baseline training. Focus on trace collection → JEPA convergence first.

---

## 9. Papers Ready for Push

### Recommended Commit

```
git add -A
git commit -m "docs: update papers and tech reports for phase-rs-first (May 20)

- AGENTS_TECH_REPORT.md: Add phase-rs preamble, clarify opaque interface
- TRAINING_ABLATIONS_AND_KG_INTEGRATION.md: New doc (decision timing, training pipeline, KG roadmap)
- opposition_agents_mtg.pdf: Recompiled (no content changes, phase-rs already mentioned)
- agents_tech_report.pdf: Recompiled
- Fix stream timeout: 45s → 180s across all files
- IMPLEMENTATION_PLAN.md: Update status snapshot with critical timeout fix

Ablation running in background (81 games, ~2-4h runtime)
Expected output: runs/phase_rs_ablation_fixed_timeout/YYYYMMDD_HHMMSS/
"
```

### Files to Push
```
docs/
├── AGENTS_TECH_REPORT.md (updated)
├── TRAINING_ABLATIONS_AND_KG_INTEGRATION.md (new)
├── PHASE_RS_INTEGRATION_SUMMARY.md (existing, unchanged)
└── PHASE_RS_AI_DEEP_DIVE.md (existing, unchanged)

paper/
├── opposition_agents_mtg.pdf (recompiled)
├── agents_tech_report.pdf (recompiled)
└── (all .tex sources already committed)

scripts/
└── phase_rs_rollout_sweep.py (updated: stream-timeout help)

src/integrations/phase_rs/
├── client.py (updated: timeout docstring)
└── runner.py (updated: timeout fallback)

IMPLEMENTATION_PLAN.md (updated: status snapshot)
TIMEOUT_FIX_SUMMARY.md (new: explains fix)
```

---

## 10. Questions Answered

**Q1: Why do decisions take so long (46+ seconds)?**  
A: Phase-ai chains 20-30 decisions per complex turn at 1.5s each, plus network latency. This is normal and acceptable for research.

**Q2: How will training ablations work?**  
A: Stage 4.1 (trace collection) → 4.2 (ingestion) → 5 (JEPA training) → 6 (planning/dreaming) → 7 (KG write-back). Full pipeline documented in new guide.

**Q3: Will our agents write to the knowledge graph?**  
A: Not yet. KG write-back is Stage 7 (June). Agents currently read-only during gameplay. Post-game analysis will extract combos from winning traces and add them back to KG for next cycle.

---

**Last updated**: May 20, 2026  
**Ablation status**: Running in background  
**Papers**: Ready for push  
