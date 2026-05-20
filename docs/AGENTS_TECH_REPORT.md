# Agent Zoo — Technical Implementation Report

## Phase-rs-First (May 2026 Update)

**As of May 20, 2026**, all agents now run on the **phase-rs Rust engine** via WebSocket bridge.
The integration is defined in [`src/integrations/phase_rs/`](../src/integrations/phase_rs/):

- **Protocol**: WebSocket v6 with typed envelopes (phase-server format)
- **Bridge**: Opaque action picker → phase-rs GameAction dict → index selection (no state translation yet)
- **Entry point**: `scripts/phase_rs_rollout_sweep.py` for benchmarking; `examples/play_edh_pod.py` for demos
- **Trace collection**: Per-game JSONL with decision-level events (turn, phase, action index, legal types)

The Python engine (`src/engine/`) remains for:
- Unit testing agent decision logic in isolation
- Debugging rules without network latency
- Research on rule-space changes (not currently used for production training)

For new ablation runs, see [`docs/TRAINING_ABLATIONS_AND_KG_INTEGRATION.md`](TRAINING_ABLATIONS_AND_KG_INTEGRATION.md).

---

## 1. Agents on Phase-rs

This document is the engineering reference for every agent in
[src/agents/](../src/agents/). It is paired with the scientific writeup in
[paper/opposition_agents_mtg.tex](../paper/opposition_agents_mtg.tex)
(section "Agent Zoo: Algorithmic Specification"), but where the paper
focuses on objectives and pseudocode, this report focuses on *exactly
what the code does*: class signatures, control flow, configuration, and
failure modes.

All agents implement the same one-method protocol declared in
[src/agents/base_agent.py](../src/agents/base_agent.py):

```python
class MTGAgent:
    name: str
    player_id: str

    async def decide_action(
        self,
        game_state: GameState,
        legal_actions: list[Action],
    ) -> Action: ...
```

The engine is the only thing that mutates `GameState`; agents are
side-effect free on the public game state, but they do mutate their own
internal state (caches, beliefs, hidden RNN state) and write a
`ReasoningTrace` accessible via `agent.last_reasoning` for logging.

### Phase-rs Integration Note

On phase-rs, agents do not see a full Python `GameState`. Instead, they receive:
- **Raw phase-rs state dict** (board cards, hand, mana pool, etc.) — opaque to agent logic
- **Legal actions** as a list of JSON dicts with `type` and optional `data`
- **Expected output**: index into the legal actions list (0-based)

Simple agents like `RandomAgent` and `HeuristicAgent` work directly on this opaque interface via
[`src/integrations/phase_rs/agent_bridge.py`](../src/integrations/phase_rs/agent_bridge.py).
Complex agents (LLM, world model) still see a full Python `GameState` by using a thin translation layer
(TODO: finish `phase_action_to_engine_action` in adapter.py).

The full registry lives in [src/agents/__init__.py](../src/agents/__init__.py).
You can construct any agent by short name:

```python
from src.agents import make_agent
agent = make_agent("llm_fusion", "player1", llm_model="gemma4:e2b")
```

Registered names: `random`, `heuristic`, `kg_heuristic`, `human`,
`ollama`, `llm` (alias), `world_model`, `active_inference`,
`llm_fusion`, `fusion` (alias).

Across the zoo, the intended learning loop is collective: self-play traces
from many agent types are post-processed into append-only graph evidence,
and later agents query that shared graph memory as an additional strategic
prior.

---

## 2. RandomAgent

**File:** [src/agents/random_agent.py](src/agents/random_agent.py)
**Role:** baseline / smoke test. Stronger than uniform random — uses
fixed action-type weights to avoid pathological play.

### Decision flow

`decide_action` builds a *weighted bag* and uniform-samples from it:

| ActionType         | Weight | Notes |
|--------------------|--------|-------|
| `CAST_SPELL`       | 10     | +20 if the source card is in `Zone.COMMAND_ZONE` (commander cast bonus). |
| `PLAY_LAND`        | 25     | Highest because dropping a land is almost always correct. |
| `DECLARE_ATTACKERS` | 8      | Plus a target bonus: `max(0, 40 - target_life) + 2 * commander_damage_received`. |
| `ACTIVATE_ABILITY` | 1      | Below pass — protects against burning once-per-turn taps. |
| `PASS_PRIORITY`    | 1      | Last resort. |
| `CONCEDE`          | 0      | Never picked unless it is the only action. |

Sampling is `random.choice(weighted)` over the expanded multiset, so it
is **not seedable per agent** — all `RandomAgent` instances share the
process-global `random` state. The orchestrator sets that seed once at
game start.

### Heuristic helper

`get_heuristic_score(game_state) -> float` returns
`(my_life - opp_life) + 3 * (my_creatures - opp_creatures)` and is used
by the `LLMFusionAgent` and `OllamaAgent` fallback paths. It does not
participate in `decide_action` itself.

### Failure modes

- Empty `legal_actions` → returns a synthesised `PASS_PRIORITY` action.
- Process-global RNG → if you need full determinism use `HeuristicAgent`
  or seed `random.seed(...)` at the harness level.

---

## 2. HeuristicAgent

**File:** [src/agents/heuristic_agent.py](src/agents/heuristic_agent.py)
**Role:** deterministic priority cascade. The strongest baseline that
needs no learned model. Used as the floor in every benchmark.

### Decision flow

```
legal_actions
   ├─ filter "counter target spell" casts unless an opponent
   │  spell is on top of the stack
   ├─ sort by (-PRIORITY[type], stable_tiebreak(action))
   ├─ candidates = top-priority slice
   ├─ if prefer_aggressive and any DECLARE_ATTACKERS in candidates
   │     → return that one
   ├─ if top is DECLARE_BLOCKERS
   │     → chump-block: smallest blocker vs largest attacker
   └─ else return candidates[0]
```

Priority table (`_PRIORITY` at module top):

```
PLAY_LAND          100
CAST_SPELL          90
DECLARE_ATTACKERS   80
DECLARE_BLOCKERS    70
ACTIVATE_ABILITY    60
PASS_PRIORITY        1
CONCEDE              0
```

### Configuration

```python
HeuristicAgent(
    player_id: str,
    name: str = "",
    seed: int | None = None,         # threads through self._rng
    prefer_aggressive: bool = True,  # picks DECLARE_ATTACKERS over PASS at top tier
)
```

### Determinism

Fully deterministic given `seed`. The tiebreak hashes the action's repr
through `self._rng.random()` so identical action lists produce identical
choices.

### Reasoning trace

`self.last_reasoning` is a dict written via `_record_reasoning`:

```jsonc
{
  "agent_kind": "heuristic",
  "rationale": "highest-priority action class (CAST_SPELL, prio=90)",
  "chose_via": "top_priority" | "aggressive_override" | "chump_block",
  "priority": 90,
  "candidate_count": 3,
  "filtered_counters": 1,
  "legal_action_count": 12
}
```

---

## 3. KGHeuristicAgent

**File:** [src/agents/kg_heuristic_agent.py](src/agents/kg_heuristic_agent.py)
**Role:** `HeuristicAgent` with Neo4j-driven re-ranking on `CAST_SPELL`
ties. Showcases how the knowledge graph plugs into a classical agent
without disturbing the priority skeleton.

### Decision flow

```
casts = [a for a in legal_actions if a.action_type == CAST_SPELL]
if len(casts) <= 1 or no kg or kg disabled:
    return super().decide_action(legal_actions)        # pure heuristic

# 1. one-shot near-combo lookup
near = await kg.detect_near_combos(my_battlefield ∪ my_hand)
near_combo_pieces = {entry["missingPiece"] for entry in near}

# 2. score each cast
for cast in casts:
    score  = 10.0 if cast.card_name in near_combo_pieces else 0.0
    score += await kg.get_synergies_for(cast.card_name)
              .filter(partner in my_battlefield).sum("strength")

# 3. let the parent pick from {non_casts ∪ {best_cast}}
best_cast = argmax(scored)
pruned    = [a for a in legal_actions if a.type != CAST_SPELL] + [best_cast]
return super().decide_action(pruned)
```

The KG handle (`self._kg`) is injected at construction time
(`make_agent("kg_heuristic", pid, kg=...)`). If any Cypher query raises,
`self._kg_disabled = True` is set and the agent silently degrades to the
plain heuristic for the rest of the game.

### Caches

Two per-game caches keep traffic to one Cypher query per
(card, battlefield-snapshot) pair:

```python
self._combo_cache:   dict[str, float]
self._synergy_cache: dict[tuple[str, frozenset[str]], float]
```

### Reasoning trace adds

```jsonc
{
  "agent_kind": "kg_heuristic",
  "top_candidates": [
    {"action": "CAST_SPELL(Goblin Recruiter)",
     "score": 12.5, "reason": "near-combo closer"}
  ],
  "scores": [12.5, 3.0, 0.0],
  "beliefs": {
    "kg_active": true,
    "best_cast_score": 12.5,
    "cast_options": 3,
    "kg_chose_cast": true
  }
}
```

---

## 4. OllamaAgent (`llm` alias)

**File:** [src/agents/llm_agent.py](src/agents/llm_agent.py)
**Role:** chat-completions LLM as a posterior over the legal-action set,
served locally by [Ollama](https://ollama.ai/) at `http://localhost:11434`.
Default model `gemma4:e2b`. Falls back to `RandomAgent` if Ollama is
unreachable or the model is not pulled.

### Decision flow

1. `_check_ollama()` runs once at construction: hits `/api/tags`, sets
   `self._ollama_available` if the model is loaded, else builds a
   `RandomAgent` fallback.
2. On `decide_action`:
   - if not available → fallback agent decides, increment
     `stats["fallback_invocations"]`, return.
   - else build a textual context describing the board state (life
     totals, hand sizes, lands, creatures), enumerate
     `legal_actions[i]` as `f"[{i}] {a.action_type.value} ..."`, send a
     prompt that ends with "respond with ONLY the option number".
   - parse the integer from the response (regex). On any error
     increment `stats["llm_calls_failed"]` and return the first playable
     action.
3. The model is kept warm via `keep_alive: "30m"` on `/api/generate`,
   `temperature: 0.2`, timeout 120 s.

### Stats (visible at `agent.stats`)

```python
{
  "llm_calls_total": int,
  "llm_calls_success": int,
  "llm_calls_failed": int,
  "fallback_invocations": int,
  "actions_chosen": {"PASS_PRIORITY": int, "CAST_SPELL": int, ...},
  "mulligan_decisions": list[tuple[int, bool]],   # (mulligans_taken, kept)
}
```

### Failure modes

| Symptom | Cause | Mitigation |
|---|---|---|
| All decisions are `PASS_PRIORITY` | Model returned non-integer text | Check `stats["llm_calls_failed"]`; re-pull model. |
| `httpx.ConnectError` on every turn | Ollama daemon down | The agent prints `Ollama unavailable` and falls back to `RandomAgent`. |
| Cold start ~30–60 s | First call loads model into VRAM | `keep_alive: "30m"` keeps it resident across turns. |
| LLM picks an out-of-range index | Hallucinated number | Clamped to `[0, len(legal_actions)-1]` in `_parse_action_index`. |

### Configuration

```python
OllamaAgent(
    player_id: str,
    name: str = "Ollama Agent",
    model: str = "gemma4:e2b",
    base_url: str = "http://localhost:11434",
)
```

---

## 5. ActiveInferenceAgent

**File:** [src/agents/active_inference_agent.py](src/agents/active_inference_agent.py)
**Role:** picks actions by minimising expected free energy
$\mathcal{F}(a)$ over the legal set, using
`src/agents/active_inference.py::ActiveInferenceModule` for the
free-energy computation and
[src/agents/opponent_model.py](src/agents/opponent_model.py) for
opponent-hand belief tracking (a particle filter constrained by the
known decklist).

### Decision flow

```
if no kg / module unavailable           → fallback to RandomAgent
for each opponent first seen this game  → ai_module.initialize_beliefs()
ranked = ai_module.rank_actions(legal_actions, game_state)
                       # returns list[(Action, free_energy)]

if opponent_model.threat.probability_has_counterspell > 0.5:
    drop CAST_SPELL(non-mana) actions from ranked, retry if empty

# tactical override: among DECLARE_ATTACKERS, pick the highest-threat target
if any(a.type == DECLARE_ATTACKERS for a, _ in ranked):
    chosen = argmax_a score_attack_target(a.targets[0])
else:
    chosen = ranked[0][0]
```

`score_attack_target(target_id)` is the same formula as in `RandomAgent`:
`max(0, 40 - life) + 2 * commander_damage_received`.

### Configuration

```python
ActiveInferenceAgent(
    player_id: str,
    kg: MTGKnowledgeGraph | None = None,
    known_opponent_decklist: dict[str, int] | None = None,
    name: str = "ActiveInferenceAgent",
)
```

If `kg` is `None` the agent tries `MTGKnowledgeGraph()` on construction;
on connection failure it logs a warning, sets `self.ai_module = None`,
and behaves as a `RandomAgent` for the rest of the game.

### Metadata stamped on chosen action

```python
chosen_action.metadata["decision_mode"]        = "active_inference"
chosen_action.metadata["expected_free_energy"] = float | None
chosen_action.metadata["target_score"]         = float   # only for attacks
```

### Belief state

Maintained by `ActiveInferenceModule`:

- `beliefs[opponent_id]`: particle distribution over
  `{ hand : Counter[str], library_top : list[str] }` reduced to weight
  by Boltzmann re-weighting after each observed opponent action.
- `threat.probability_has_counterspell`: scalar in `[0, 1]`, computed by
  the opponent model from `n10s`-typed `:Counter` cards in the
  decklist.

---

## 6. WorldModelAgent

**File:** [src/agents/world_model_agent.py](src/agents/world_model_agent.py)
**Role:** plays via the V+M+C+JEPA world model defined in
[src/world_model/world_model.py](src/world_model/world_model.py).
Encodes the real state, then either calls the controller directly (fast
path) or runs `dream_search` (Monte-Carlo rollouts in latent space).

### Decision flow

```python
with torch.no_grad():
    z, h = self._encode_state(game_state)
    action_encodings, action_mask = self._encode_actions(legal_actions)

    if mode == "dream_search" and len(legal_actions) > 1:
        action_idx = self._dream_search(z, h, action_encodings, action_mask)
    else:
        action_idx = self._direct_policy(z, h, action_encodings, action_mask)

return legal_actions[action_idx]
```

`_encode_state`:
1. `tokens = self.tokenizer.encode(game_state)` — Set-Transformer tokens.
2. `z = world_model.encoder(tokens)` — VAE latent, $z \in \mathbb{R}^{256}$.
3. `g = kg_encoder(game_state)` if `kg_encoder` was supplied, else
   zeros.
4. `h = self._hidden` (carried from previous turn) → fed into the
   `MDN-LSTM` dynamics on the next observation.

`_direct_policy(z, h, A, mask)`:
1. `logits, value = controller(z, h, A)` — controller is a small
   transformer head over the action embeddings.
2. mask out invalid actions (impossible since we only feed legal ones,
   but cheap defence in depth).
3. `argmax(logits)` if `deterministic` else
   `Categorical(logits=logits / T).sample()`.

`_dream_search(z, h, A, mask)`:
- For each candidate action $a_i$, simulate `dream_rollouts` rollouts of
  depth `dream_depth` using `world_model.dynamics.predict(z, a)` (the M
  module) followed by `controller.policy_logits` to pick subsequent
  actions.
- Score = mean `value` head readout over rollouts.
- Pick `argmax`.

### Configuration

```python
WorldModelAgent(
    player_id: str,
    world_model: WorldModel,
    tokenizer: GameTokenizer,
    kg_encoder: KGContextEncoder | None = None,
    name: str = "WorldModelAgent",
    mode: str = "direct",          # or "dream_search"
    dream_rollouts: int = 8,
    dream_depth: int = 10,
    deterministic: bool = False,
    device: str = "cpu",
)
```

### Checkpoint loading

`make_agent("world_model", ...)` loads `checkpoints/jepa/jepa_final.pt`
via `WorldModel.load(path)`. The loader is tolerant of three checkpoint
shapes:

1. `{"config": WorldModelConfig, "state_dict": OrderedDict}` — full
   model, written by `WorldModel.save`.
2. `{"encoder": ..., "predictor": ..., "kg_encoder": ...}` — JEPA-only
   training checkpoint from `train_pipeline.py` stage 5; the `dynamics`
   and `controller` modules stay default-initialised.
3. Anything else → loads any matching sub-state-dicts and warns about
   the rest, never raises.

If the checkpoint file is missing, the registry falls back to a
default-initialised `WorldModel()`. The agent will play with random
weights, useful for smoke-testing the harness before training.

### Persistent hidden state

`self._hidden: tuple[torch.Tensor, torch.Tensor] | None` is the LSTM
`(h, c)` tuple. `decide_action` overwrites it after every encode so the
M module sees the full game-step trajectory across turns.

---

## 7. LLMFusionAgent (`fusion` alias)

**File:** [src/agents/llm_fusion_agent.py](src/agents/llm_fusion_agent.py)
**Role:** the strongest agent. Combines four signals through a
weighted sum and returns the argmax. Designed so any single signal can
be ablated by setting its weight to zero.

### Decision flow

```
heur  = self._heuristic_scores(game_state, legal_actions)
kg    = await self._kg_scores(game_state, legal_actions)
wm    = self._world_model_scores(game_state, legal_actions)

if not self._is_obvious(heur, wm):       # one action dominates by > 0.8
    llm = await self._llm_scores(game_state, legal_actions)
else:
    llm = [0.0] * len(legal_actions)

scores = (cfg.heuristic_weight * heur
        + cfg.kg_weight        * kg
        + cfg.world_model_weight * wm
        + cfg.llm_weight       * llm)
return legal_actions[argmax(scores)]
```

### Signal sources

| Signal | Source | Default weight | Cost |
|---|---|---|---|
| heuristic | `RandomAgent.get_heuristic_score` evaluated post-hoc on every action's expected board | 0.15 | µs |
| kg | `MTGKnowledgeGraph.detect_near_combos` + `get_synergies_for` (same as `KGHeuristicAgent` § 3) | 0.15 | one Cypher round-trip |
| world_model | `WorldModel` dream rollouts: 8 rollouts × depth 10, scored by the controller's value head | 0.40 | hundreds of ms on CPU |
| llm | `OllamaAgent` posterior over `legal_actions` indices | 0.30 | seconds |

### `FusionConfig` defaults

```python
@dataclass
class FusionConfig:
    llm_weight: float            = 0.3
    world_model_weight: float    = 0.4
    kg_weight: float             = 0.15
    heuristic_weight: float      = 0.15

    dream_rollouts: int          = 8
    dream_depth: int             = 10
    dream_temperature: float     = 1.15

    llm_provider: str            = "ollama"
    llm_model: str               = "gemma4:e2b"
    llm_timeout: float           = 15.0

    skip_llm_if_obvious: bool    = True
    obvious_threshold: float     = 0.8
```

### Ablations

The matchup harness wires each ablation by overriding weights through
the registry kwargs:

```python
make_agent("llm_fusion", "p1", llm_weight=1.0,        world_model_weight=0.0, kg_weight=0.0, heuristic_weight=0.0)  # LLM-only
make_agent("llm_fusion", "p1", llm_weight=0.0,        world_model_weight=1.0, kg_weight=0.0, heuristic_weight=0.0)  # WM-only
make_agent("llm_fusion", "p1", llm_weight=0.0,        world_model_weight=0.0, kg_weight=1.0, heuristic_weight=0.0)  # KG-only
make_agent("llm_fusion", "p1")                                                                                      # all four
```

`docs/EXPERIMENTS.md` enumerates these as the canonical four-arm study.

### Lazy initialisation

`_ensure_llm()` and `_ensure_fallback()` defer expensive imports
(`httpx` for the LLM, `RandomAgent` for the fallback) until the first
call. Tests can inject mocks by passing `world_model=...`,
`tokenizer=...`, `knowledge_graph=...` at construction time.

### `observe(game_state, action)`

Called by the orchestrator after every public action. Updates the LSTM
hidden state via `world_model.dynamics.step(z, a, h)` and forwards to
the opponent model when `action.player_id != self.player_id`.

---

## 8. HumanAgent

**File:** [src/agents/human_agent.py](src/agents/human_agent.py)
**Role:** interactive CLI for tournaments where one seat is human.
Prints `legal_actions` numbered, prompts on stdin, validates the index,
re-prompts on parse error. No reasoning trace; not used in benchmarks.

---

## Sequence diagram — a single turn through `LLMFusionAgent`

```
priority_loop ────► agent.decide_action(state, legal)
                        │
                        ├── self._heuristic_scores()           [≈ 50 µs]
                        │       └── RandomAgent.get_heuristic_score
                        │
                        ├── await self._kg_scores()            [≈ 5–20 ms]
                        │       └── kg.detect_near_combos
                        │       └── kg.get_synergies_for
                        │
                        ├── self._world_model_scores()         [≈ 100–400 ms CPU]
                        │       └── tokenizer.encode(state)
                        │       └── world_model.encoder.forward
                        │       └── world_model.dynamics.dream_rollout × 8
                        │       └── controller.value_head
                        │
                        ├── self._is_obvious(h, wm)?
                        │      no
                        │       └── await self._llm_scores()   [≈ 1–4 s]
                        │             └── OllamaAgent.decide_action
                        │
                        ├── argmax(0.15 h + 0.15 kg + 0.40 wm + 0.30 llm)
                        │
                        └── ReasoningTrace(agent_kind="llm_fusion",
                              top_candidates=[…], scores=[…])
```

---

## Reasoning trace contract

Every agent writes a `ReasoningTrace`-typed dict to
`agent.last_reasoning` before returning. The orchestrator's JSONL writer
([scripts/run_matchups.py](scripts/run_matchups.py),
[scripts/run_tournament.py](scripts/run_tournament.py)) flushes one line
per decision to `runs/<out>/logs/game_NNNN.jsonl` with this schema:

```jsonc
{
  "turn": int,
  "phase": "MAIN1" | "COMBAT_DECLARE_ATTACKERS" | …,
  "player_id": "player1",
  "agent_kind": "random" | "heuristic" | … | "llm_fusion",
  "rationale": "human-readable one-liner",
  "legal_action_count": 14,
  "chosen_index": 7,
  "top_candidates": [{"action": "CAST_SPELL(Lightning Bolt)", "score": 0.62}, ...],
  "scores": [0.62, 0.41, 0.0, …],
  "beliefs": { /* agent-specific */ }
}
```

This is what the post-hoc analysis notebooks consume. See
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the full pipeline.

---

## Testing each agent

Per-agent regression tests under `tests/`:

| Agent | Test |
|---|---|
| Registry | [tests/test_agent_registry.py](tests/test_agent_registry.py) — registry lookup + factory smoke |
| `RandomAgent` | [tests/test_random_agent.py](tests/test_random_agent.py) |
| `HeuristicAgent` | [tests/test_heuristic_agent.py](tests/test_heuristic_agent.py) |
| `KGHeuristicAgent` | covered indirectly via [tests/test_jsonl_trace.py](tests/test_jsonl_trace.py) |
| `OllamaAgent` | [tests/test_llm_agent.py](tests/test_llm_agent.py) (mocks `httpx.Client`) |
| `ActiveInferenceAgent` | [tests/test_active_inference_agent.py](tests/test_active_inference_agent.py) |
| `WorldModelAgent` | [tests/test_world_model_agent.py](tests/test_world_model_agent.py) |
| `LLMFusionAgent` | [tests/test_llm_fusion_agent.py](tests/test_llm_fusion_agent.py) |

The full smoke command for an EDH pod:

```powershell
.\.venv\Scripts\python.exe examples\play_edh_pod.py `
    --max-turns 6 --seed 7 --model none
```

The full ablation harness (4-arm × 8 games):

```powershell
.\.venv\Scripts\python.exe scripts\run_matchups.py `
    --format standard `
    --decks data/decks/modern/modern_mono_red_burn.txt `
            data/decks/modern/modern_azorius_control.txt `
    --agents random heuristic kg_heuristic world_model llm llm_fusion `
    --games 8 --max-turns 100 --seed 7 `
    --out runs\ablation\full
```
