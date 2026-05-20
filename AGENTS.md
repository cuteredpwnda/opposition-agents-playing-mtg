# AGENTS.md — Agent Zoo

This document describes agents that play Magic in this repository on the
**phase-rs Rust engine runtime** via the WebSocket bridge. For engineering
conventions see `.github/copilot-instructions.md`; for deep design see
`IMPLEMENTATION_PLAN.md` (single source of truth) and `PLAN.md`.

## The contract every agent satisfies

All agents implement the `MTGAgent` protocol:

```python
class MTGAgent(Protocol):
    name: str
    
    async def decide_action(
        self,
        game_state: GameState,
        legal_actions: list[Action],
    ) -> Action: ...
```

Agents receive a (controlled-perspective filtered) `GameState` and a list of
legal `Action` values, then return exactly one. They are **free of side
effects** on game state — only the rules engine (phase-rs) mutates it.

When used with phase-rs, the bridge in `src/integrations/phase_rs/adapter.py`
translates between phase-rs wire format and internal `Action` objects. Agents
can be sync (`decide_action`) or async (`decide_action_async`); the runner
detects and handles both.

## The lineup

| Agent | Module | What it does |
|---|---|---|
| `RandomAgent` | `src/agents/random_agent.py` | Uniform random over legal actions. Baseline / smoke test. |
| `HeuristicAgent` | `src/agents/heuristic_agent.py` | Hand-tuned priorities (play land → cast curve → attack with profitable creatures → pass). Deterministic with a seed. |
| `LLMAgent` | `src/agents/llm_agent.py` | Prompts a chat-completions LLM (OpenAI-compatible) with a textual board state and asks for an action index. Cheap, no fine-tune. |
| `OllamaAgent` | `src/agents/ollama_agent.py` | Local LLM via Ollama — same interface as `LLMAgent` but talks to `http://localhost:11434`. See `docs/OLLAMA_SETUP.md`. |
| `WorldModelAgent` | `src/agents/world_model_agent.py` | V+M+C: encodes state → simulates k-step rollouts in latent space → picks action with highest predicted value. |
| `LLMFusionAgent` | `src/agents/llm_fusion_agent.py` | Combines an LLM critic with the world model's value estimate; LLM picks among the top-N world-model candidates. |
| `ActiveInferenceAgent` | `src/agents/active_inference_agent.py` | Maintains a Bayesian belief over hidden information (opponent hand, library top) and minimises expected free energy. |
| `HierarchicalAgent` | `src/agents/hierarchical_agent.py` | High-level "plan" (e.g. *race*, *stabilise*, *combo*) selected by an outer policy; low-level move chosen by an inner policy. |

## Collective Intelligence Layer

The agent zoo is designed to write into and read from one shared strategic
memory: the knowledge graph extension layer.

- Individual agents stay side-effect free on `GameState`.
- Self-play traces are post-processed into graph evidence.
- Learned evidence is append-only and provenance-tagged.
- Later agents query these learned relations as priors.

This yields cumulative cross-agent learning while preserving rules-engine
determinism and immutable source card data.

## Running games — the easy way (phase-rs-first)

**All agents now run on phase-rs (Rust engine)** via the WebSocket bridge. The fastest way to watch agents play:

```powershell
# 4-player Commander pod on phase-rs (deterministic)
.\.venv\Scripts\python.exe examples\play_edh_pod.py --max-turns 6 --seed 7 --model none

# Same with Ollama LLM in seat 0
.\.venv\Scripts\python.exe examples\play_edh_pod.py --max-turns 12 --seed 7 --model ollama:llama3

# Collect training traces
.\.venv\Scripts\python.exe scripts\collect_phase_rs_traces.py --games 16 --picker agent:heuristic --autostart

# Run ablation suite
.\.venv\Scripts\python.exe scripts\run_phase_rs_ablation.py --games 10 --picker agent:heuristic --autostart

# Cartesian sweep (picker x difficulty x deck)
.\.venv\Scripts\python.exe scripts\phase_rs_rollout_sweep.py --pickers random agent:heuristic --difficulties VeryEasy Medium --games-per-cell 3 --autostart
```

Output:

- Per-game human-readable log → `runs/edh_pod/pod_game_NNN.log` (phase-rs-first only)
- Structured JSONL traces → `runs/<script>/<date>_<time>/traces/game_NNNN.jsonl`
  - Decision events: turn, phase, legal_action_types, chosen_index, chosen_type
  - Outcomes: winner, reason (normal_play | stream_timeout | action_rejected | etc.)

Legacy note: `python examples/demo_game_simple.py` still works on the Python engine for compatibility, but is not the training target.

## Adding a new agent

1. Create `src/agents/my_agent.py`. Keep imports inside `decide_action` if
   they are heavy (torch, transformers) so the random/heuristic path stays
   import-cheap.
2. Subclass nothing — just provide `name: str` and `decide_action(...)`.
3. Add a 5-line test under `tests/test_my_agent.py` that constructs a
   minimal `GameState`, asks for a decision, and asserts the action is in
   the legal set.
4. Register the agent in `src/agents/__init__.py` so the runner CLI can
   look it up by name (`--seat0 my_agent`).
5. Update the table above and add a queue entry in `IMPLEMENTATION_PLAN.md`
   if there is follow-up work.

## Debugging an agent

- **It never plays a land** — usually a colour-identity / commander
  bookkeeping bug. Check `state.commander_ids(player_id)` returns what
  you expect; the older `state.commanders[pid]` was a *single* string and
  is preserved as a back-compat property only.
- **It loops `PASS_PRIORITY`** — `legal_actions` probably contains only
  `PASS_PRIORITY` and `CONCEDE`; check whether triggered abilities went
  on the stack but no resolution path is firing.
- **It plays but never attacks** — combat phase legal actions come from
  phase-rs's combat engine. Confirm the creature state (haste, summoning sickness, tapped) via the phase-rs state snapshot.
- **Rules questions** — consult phase-rs's comprehensive rules implementation
  (30k+ cards, all layers/replacement/stack) or point the judge at `data/rules/latest.txt`
  (download with `python scripts/fetch_rules.py`) for text-based lookups.

## Determinism

All agents that take a seed **must** use it for any RNG they own. The
runner threads a single seed through `random.Random(seed).randint(...)`
draws so a given `(seed, decklists, agent-types)` triple replays
identically on phase-rs. If your agent calls `random.random()` without a seeded
`Random` instance you have introduced a bug.
