# Full Working Vision — Roadmap & Implementation Map

> Status: April 2026. This document is the **single source of truth** for the
> "everything turned on" milestone: agentic play (LLM + world-model + active
> inference + KG agents) playing real games end-to-end, with self-play
> training of the world model and a controlled ablation harness over both
> world-model sizes and LLM sizes (≤ 7 B parameters).

---

## 0. Definition of Done

The vision is reached when **all four** of the following are true on a single
GPU workstation (24 GB-class):

1. **Agentic play** — Any pair of (`RandomAgent`, `HeuristicAgent`,
   `OllamaAgent`, `WorldModelAgent`, `LLMFusionAgent`,
   `ActiveInferenceAgent`, `NeuralReasonerAgent`) plays a complete game
   end-to-end through the async `GameRunner` and the sync `GameSimulator`,
   using the strategy-aware mulligan hook on `MTGAgent`.
2. **Self-play training** — `scripts/train_pipeline.py` runs all stages 1–7
   without manual intervention given a populated Neo4j and an Ollama
   instance, producing a checkpointed `WorldModel` whose challenger ELO has
   improved over the random baseline by ≥ 100 points.
3. **World-model size ablation** — `scripts/ablation_sweep.py
   --sweep wm_size` produces a CSV/JSON in `runs/ablation/` with one row per
   `WorldModelPreset` (tiny / small / medium / large / xl), reporting
   parameter count, training wall-clock, peak VRAM, and round-robin win
   rates against the baseline.
4. **LLM size ablation** — `scripts/ablation_sweep.py --sweep llm_size`
   does the same over the entries in `LLM_REGISTRY` (≤ 7 B parameters,
   Ollama-served), reporting per-decision latency, ELO, and VRAM headroom.

---

## 1. Status Snapshot

| Pillar | State | Files |
|---|---|---|
| Engine (sync `GameSimulator`) | ✅ End-to-end games, mulligan, hand cap, deck-out, timeout tiebreak | [src/engine/game_simulator.py](src/engine/game_simulator.py) |
| Engine (async `GameRunner`) | ✅ Same rules, async priority loop | [src/orchestrator/game_runner.py](src/orchestrator/game_runner.py) |
| Agents (8 classes) | ✅ All have `decide_action`; mulligan hook NEW (this milestone) | [src/agents/](src/agents/) |
| World model (V+M+C+JEPA) | ✅ Modules exist; presets NEW (this milestone) | [src/world_model/](src/world_model/) |
| Self-play / RL | ✅ `RLTrainer` + parallel `asyncio.gather`; pipeline stages 1–4 ok, 5–7 partial | [src/training/](src/training/), [scripts/train_pipeline.py](scripts/train_pipeline.py) |
| Knowledge graph | ✅ Builder, RAG, enrichment, GraphSAGE | [src/knowledge/](src/knowledge/) |
| Benchmark / ablation | ⚠️ `BenchmarkSuite` exists; ablation sweep NEW (this milestone) | [src/training/benchmark_suite.py](src/training/benchmark_suite.py), [scripts/ablation_sweep.py](scripts/ablation_sweep.py) |
| LLM integration | ✅ Ollama; per-agent model swap NEW (`LLM_REGISTRY`) | [src/agents/llm_agent.py](src/agents/llm_agent.py), [src/agents/llm_registry.py](src/agents/llm_registry.py) |
| Determinism | ⚠️ Per-test only; global seeding NEW | [src/utils/seeding.py](src/utils/seeding.py) |

---

## 2. Implementation Map (this milestone)

The following files are added or modified as part of this roadmap. Each is a
minimal, runnable stub that the rest of the system can build on.

### 2.1 Strategy-aware, agent-driven mulligan
- [src/agents/mulligan.py](src/agents/mulligan.py) — pure functions
  `should_keep(hand, strategy, mulligans_taken, max_mulligans)` and
  `select_bottom_cards(hand, n, strategy)`. Per-strategy heuristics for
  AGGRESSIVE / CONTROL / COMBO / REACTIVE.
- [src/agents/base_agent.py](src/agents/base_agent.py) — `MTGAgent` gains
  `decide_mulligan(...)` and `select_bottom_cards(...)` methods with
  default implementations that delegate to `mulligan.py`. Subclasses may
  override (LLM agents will, in a follow-up).
- [src/engine/game_simulator.py](src/engine/game_simulator.py) and
  [src/orchestrator/game_runner.py](src/orchestrator/game_runner.py) —
  `_apply_london_mulligan` accepts an optional `keep_fn` / `bottom_fn`
  callable, defaulting to the agent's hooks.

### 2.2 World-model size presets (ablation knob A)
- [src/world_model/presets.py](src/world_model/presets.py) — five named
  configurations:

  | Preset | latent | hidden | layers | ~params | target VRAM |
  |---|---|---|---|---|---|
  | `tiny`   | 64  | 128 | 1 | ~0.3 M | < 1 GB |
  | `small`  | 128 | 256 | 2 | ~2 M   | < 2 GB |
  | `medium` | 256 | 512 | 2 | ~12 M  | 4 GB   |
  | `large`  | 384 | 768 | 3 | ~40 M  | 10 GB  |
  | `xl`     | 512 | 1024| 4 | ~120 M | 20 GB  |

  Each returns a fully-configured `WorldModelConfig`.

### 2.3 LLM model registry (ablation knob B, ≤ 7 B)
- [src/agents/llm_registry.py](src/agents/llm_registry.py) —
  `LLM_REGISTRY: dict[str, LLMSpec]` with Ollama tag, family, parameter
  count, expected VRAM at q4_K_M quantisation, recommended use case.
  Initial entries: `gemma2:2b`, `phi3:mini` (3.8 B), `qwen2.5:3b`,
  `llama3.2:3b`, `mistral:7b`, `qwen2.5:7b`, `llama3.1:8b-instruct-q4_K_M`
  (only the q4 8B fits the ≤ 7 B effective-size budget when quantised;
  marked accordingly).

### 2.4 Ablation sweep runner
- [scripts/ablation_sweep.py](scripts/ablation_sweep.py) — CLI entry point.
  Modes:
  - `--sweep wm_size` — instantiate each `WorldModelPreset`, train for
    `--train-games N`, then play `--eval-games M` round-robin.
  - `--sweep llm_size` — for each entry in `LLM_REGISTRY`, run an
    `OllamaAgent` against a fixed baseline (`HeuristicAgent`).
  - `--sweep cross` — Cartesian product of the two.
  Outputs `runs/ablation/<timestamp>/{config.json, results.csv,
  summary.md}`.

### 2.5 Determinism
- [src/utils/seeding.py](src/utils/seeding.py) — `set_global_seed(seed)`
  seeds `random`, `numpy.random`, `torch.manual_seed`, `torch.cuda`, and
  the Python hash seed. Called at the top of every script and benchmark
  entry point.

### 2.6 Tests
- [tests/test_mulligan_agent_hook.py](tests/test_mulligan_agent_hook.py) —
  per-strategy keep behaviour, agent override, end-to-end through
  `GameSimulator.setup_game`.
- [tests/test_world_model_presets.py](tests/test_world_model_presets.py) —
  every preset instantiates, parameter counts in expected band,
  `forward()` shape sanity.
- [tests/test_llm_registry.py](tests/test_llm_registry.py) — every entry
  is well-formed and ≤ 7 B effective params, lookup by name works.

---

## 3. Outstanding Work After This Milestone

Numbered by execution priority. Items 1–3 are needed for the "GPU runs the
ablations" demo; items 4–7 are quality and breadth.

1. **Train-pipeline stages 5–7 wiring.** Stage 5 (JEPA training), stage 6
   (full V → M → C dream training), stage 7 (eval game) must all be
   callable from `train_pipeline.py` against the new presets. The current
   `WorldModel.dream()` and `dream_search()` need a single end-to-end
   integration test.
2. **LLM-agent mulligan override.** `OllamaAgent.decide_mulligan` should
   serialise the hand to text and ask the model "keep or mulligan?" with a
   strict JSON schema; `LLMFusionAgent` should weight the LLM answer with
   the world-model rollout estimate.
3. **Active-inference mulligan.** `ActiveInferenceAgent.decide_mulligan`
   uses the particle filter's expected free energy of the resulting belief
   state — keeping a hand is a free-energy minimisation over imagined
   first-3-turn rollouts.
4. **Priority-loop instants in `GameSimulator`.** Combat steps in the sync
   simulator do not currently expose a response window; thread the
   existing async priority loop into combat so counterspells and combat
   tricks fire mid-attack.
5. **Smarter combat heuristics.** Replace the greedy attacker / blocker
   default with the value-based logic already present in
   `AgentStrategist`.
6. **Persistent benchmark output.** Add `--out` flag and structured
   CSV/JSON writer to `BenchmarkSuite`; promote
   `scripts/ablation_sweep.py` to write the same schema.
7. **Commander rules.** Free first mulligan, command zone, commander tax,
   colour identity, commander damage. Required only for the multiplayer
   demo, not for the ablation sweeps.

---

## 4. Hardware Profile (Reference)

The ablation harness is calibrated for a single workstation GPU
(24 GB-class such as RTX 4090 / 3090 / A6000). Approximate budgets:

- **WorldModel `xl`** training step: ~9 GB activation + 4 GB optimiser ≈
  16 GB peak; safe with batch 32.
- **LLM `mistral:7b` q4_K_M**: ~5 GB resident, ~7 GB peak with KV cache
  for 4 k context.
- **Concurrent**: world-model `medium` + `mistral:7b` + Neo4j fits in
  16 GB; `large` + `mistral:7b` requires ≥ 20 GB.

`scripts/ablation_sweep.py` honours `--max-vram-gb` and skips any
combination whose static-estimated footprint exceeds the cap.
