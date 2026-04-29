# AGENTS.md — Agent Zoo

This document is the entry point for **agents that play Magic** in this
repository, both for humans onboarding and for AI coding tools.

For *engineering* conventions (file layout, plan-file workflow, test
commands) see `.github/copilot-instructions.md`. For deep design see
`IMPLEMENTATION_PLAN.md` (single source of truth) and `PLAN.md`.

## The contract every agent satisfies

```python
class Agent(Protocol):
    name: str

    def decide_action(
        self,
        game_state: GameState,
        legal_actions: list[Action],
    ) -> Action: ...
```

That's the whole interface. An agent receives a (controlled-perspective
filtered) `GameState` and a list of currently legal `Action` values, and
must return exactly one of them. Agents are **free of side effects** on
the game state — only the rules engine mutates it.

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

## Running games — the easy way

The fastest way to watch agents play is the EDH pod sim:

```powershell
# 4-player pod with the bundled commander decks, fully deterministic
.\.venv\Scripts\python.exe examples\play_edh_pod.py --max-turns 6 --seed 7 --model none

# Same thing but with an Ollama-hosted LLM in seat 0
.\.venv\Scripts\python.exe examples\play_edh_pod.py --max-turns 12 --seed 7 --model ollama:llama3
```

Output:

- Per-turn human-readable log → `runs/edh_pod/pod_game_NNN.log`
- JSONL action trace (when enabled) → `runs/edh_pod/pod_game_NNN.jsonl`

For a 1v1 quick smoke test:

```powershell
.\.venv\Scripts\python.exe examples\demo_game_simple.py
```

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
  `src/engine/combat.py`'s `legal_attackers`. Confirm summoning sickness
  / haste / tap state on the creatures in question.
- **Rules questions** — point the judge at `data/rules/latest.txt`
  (download with `python scripts/fetch_rules.py`) and ask via
  `src/judge/`.

## Determinism

All agents that take a seed **must** use it for any RNG they own. The
runner threads a single seed through `random.Random(seed).randint(...)`
draws so a given `(seed, decklists, agent-types)` triple replays
identically. If your agent calls `random.random()` without a seeded
`Random` instance you have introduced a bug.
