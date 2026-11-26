# opposition-agents: A Collaborative AI Research Framework on phase-rs

## TL;DR

We're building an **AI research framework** where multiple agents play Magic: The Gathering on **phase-rs**, collecting structured game traces, and training deep learning models (JEPA + world models + knowledge graphs) from self-play. We use your engine as our authoritative rules backend—it's a perfect match for reproducible research.

---

## What We Do

**opposition-agents** is a Python+ML framework where:

1. **Agents play on phase-rs** — We wrap native Python decision-makers (heuristic, LLM, world-model, active-inference agents) via a WebSocket bridge that translates between our internal `Action` objects and your `GameAction` protocol.

2. **We collect structured traces** — Every game produces a JSONL file with decision-level events: legal actions, chosen action index, turn/phase, and terminal outcome (win/loss/draw/timeout). These traces become training data.

3. **We train offline models** — We run JEPA (Joint Embedding Predictive Architecture) on traces to learn latent-space world models; train RL policies with these learned dynamics; and feed combo knowledge from our Neo4j graph as priors.

4. **We run competitive ablations** — We benchmark agent vs difficulty vs deck across your AI opponent, generating baseline results for research.

5. **We publish research** — Results feed into papers + tech reports about multi-agent learning, opponent modeling, and knowledge-graph-augmented planning in complex games.

---

## Why phase-rs?

We evaluated three MTG engine options. **phase-rs won** on:

- **Coverage**: 34,300+ cards with comprehensive rules (layers, replacement, stack, all Keywords), supported by active contributors.
- **Multiplayer**: Native Commander/Brawl/Four-Player support with proper game zones, command zone damage, range of influence—essential for our research agenda.
- **Formats**: Standard, Pioneer, Modern, Legacy, Vintage, Pauper, Commander, Brawl, and more—no single-format lock-in.
- **AI system**: Per-card heuristics + game-tree search + difficulty scaling gives us a tunable benchmark opponent.
- **Protocol**: Clean WebSocket v6 interface with typed envelopes (GameCreated, StateUpdate, ActionRejected, GameOver) makes bridge-writing straightforward.
- **Community**: Responsive maintainers, active Discord, and LLM-contributor workflow mean cards get added continuously.

---

## The Bridge

We implemented `src/integrations/phase_rs/` (~800 LOC Python):

- **`client.py`** — WebSocket protocol v6 client with handshake, game creation, action sending, and stream handling.
- **`adapter.py`** — Bidirectional `GameAction` ↔ `Action` translator with metadata preservation for exact round-trip.
- **`agent_bridge.py`** — Wraps MTGAgent (our protocol) to pick legal actions from your opaque JSON list.
- **`runner.py`** — Game loop with reconnect-and-resume on stream timeout, structured trace collection, and retry logic.
- **`decks.py`** — Converts our plain-text decklists into your `DeckData` format.

**Status**: Smoke-tested end-to-end. HeuristicAgent plays full games on phase-rs with correct zone management, mana, stack, and combat.

---

## Immediate Use Case

We're running **comprehensive ablation suites**:

```powershell
# Cartesian sweep: picker × difficulty × deck
python scripts/phase_rs_rollout_sweep.py \
  --pickers random heuristic agent:heuristic agent:world_model \
  --difficulties VeryEasy Medium Hard \
  --ai-decks "Red Deck Wins" "Azorius Control" "Golgari Midrange" \
  --games-per-cell 10 \
  --autostart
```

Output: Per-game JSONL traces + aggregate stats (win rates, turn counts, timeout counts, reason codes).

**Barriers we solved**:
- Stream timeout on idle sessions → reconnect-and-resume logic.
- Transient failures cascading through grid sweeps → per-cell retry loops.
- No phase-rs training entry point → new Stage 4.1 in our training pipeline.

---

## Research Roadmap

**Q3 2026**:
- ✅ Phase-rs bridge (done)
- ✅ Trace collection + retry hardening (done)
- ⏳ Full ablation baseline (running now)
- ⏳ JEPA training on phase-rs traces
- ⏳ Commander pod research (multiplayer game tree in latent space)

**Q4 2026**:
- Multi-format evaluation (Standard, Pioneer, Commander)
- Active inference opponent modeling
- Human player integration (UI overlay)

---

## How to Get Involved

**For phase-rs Maintainers:**
- We're a high-volume test client — real-world stress testing on long sessions, edge cases, large batch jobs.
- If our bridge surfaces protocol ambiguities or missing capabilities, we'll file issues + help debug.
- We'd love to help document the WebSocket contract and AI configuration space for other researchers.

**For Research Collaborators:**
- Looking for co-authors on papers combining world models + KG-augmented planning + multiplayer.
- Open to feedback on trace format, training pipeline design, and experimental rigor.

**For ML/MTG Enthusiasts:**
- We're open-sourcing everything (MIT license coming). Star us on GitHub if interested.
- Contrib guidelines: `CONTRIBUTING.md`, agent templates: `AGENTS.md`, architecture: `ARCHITECTURE.md`.

---

## Contact

- **GitHub**: [opposition-agents-playing-mtg](https://github.com/yourusername/opposition-agents-playing-mtg)
- **This file**: `docs/PHASE_RS_COMMUNITY_INTRO.md` (in our repo)

We're not asking for anything beyond what you're already doing—we're just routing our research traffic through your fantastic engine. 🎉

---

*Last updated: May 20, 2026*
