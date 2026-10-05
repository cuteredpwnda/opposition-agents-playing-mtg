# phase-rs integration

[phase-rs/phase][phase] is a Rust-native Magic: The Gathering rules engine
with a WebSocket server, ~34,300+ implemented cards (sourced from MTGJSON),
and a per-difficulty AI opponent. We vendor it as a git submodule under
`external/phase-rs/` and talk to it from Python via a small adapter in
`src/integrations/phase_rs/`.

> **Status — October 5, 2026:** Pinned to `1191bba65`, protocol v106.
> Release build, refreshed native data, and a complete live Modern game
> with `HeuristicActionPicker` are verified. Tev1 0.8B's local decision API
> works; an isolated same-weight 8192-context alias passes a terminal game
> without truncation. The original tag's 2050-token context setting is not
> a hard endpoint ceiling. See [the experiment runbook](EXPERIMENTS.md).
> Controlled strength measurements remain pending. Native viewer
> interactions remain queued; keyed reconnect is now live-verified in
> [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md).

### Session restore and casting-model limits

The client retains the server-issued `full_key` (game code + generation) and
player credential. A stream reconnect sends the native `Reconnect` message and
requires a matching `GameStarted`, controlled seat and non-stale state revision
before recording success. It restores legal actions/terminal state and keeps
credentials out of traces. A handshake alone is not session recovery. A forced
native socket closure has been verified; this does not qualify stalled native
AI progression or long Commander pods.

KL casting subdecisions forecast one pending-cast completion rather than treating
mode/target choices as identical to cancellation. In two-player snapshots,
simple fixed-damage targets use native effect and player/object identities;
multiplayer snapshots retain coarse forecasts because the latent opponent-life
minimum cannot represent a particular opponent's life. Other effects remain coarse.
Two fresh mirror games completed after the repair, both losses. This is a
functional repair, not a card-/mode-exact transition or strength result.

## Why phase-rs

Our hand-written Python engine in `src/engine/` covers Comprehensive-Rules
mechanics deeply enough to train a JEPA world model, but its card-text
coverage is intentionally narrow (a few hundred cards). phase-rs ships
proper layers, replacement effects, the stack, combat, state-based
actions, and 30k+ cards — enough to play arbitrary Standard / Modern /
Commander decks. Using it as the rules backend lets our agent research
operate on the full card pool while our Python engine stays alive as the
authoritative substrate for the trajectory + JEPA pipeline and as a
differential-conformance check.

## Architecture

```
+-----------------------+       WebSocket (JSON envelopes)       +---------------------+
| Python agent process  |  <----------------------------------->  | phase-server (Rust) |
|                       |                                          |                     |
| ActionPicker          |  ClientMessage::CreateGameWithSettings   | engine + phase-ai   |
| RandomActionPicker    |  ----------------------------------->    |                     |
| PreferNonPassPicker   |                                          | Difficulty:         |
| (HeuristicPicker WIP) |  <-----   ServerMessage::StateUpdate     |   VeryEasy / Easy / |
|                       |  <-----   GameStarted / GameOver         |   Medium / Hard /   |
| run_game_sync(...)    |                                          |   VeryHard          |
+-----------------------+                                          +---------------------+
```

Protocol source: [`external/phase-rs/crates/server-core/src/protocol.rs`][protocol].
Wire version: `PROTOCOL_VERSION = 106` (upstream `1191bba65`, October 2026;
the constant now lives in `lobby-broker/src/protocol.rs` and is re-exported
by `server-core`). Our parser is validated against
phase-rs's own `fixtures/adapter-contract/*.json` files via
`tests/integrations/phase_rs/test_protocol_envelopes.py`.

## Running a game

Install the optional extra:

```powershell
pip install -e ".[phase_rs]"     # adds websockets>=12
```

Start phase-server in one terminal (from inside the submodule):

```powershell
cd external/phase-rs
cargo serve                       # binds ws://127.0.0.1:9374/ws (release)
```

Run a Python-controlled seat vs phase-ai in another terminal:

```powershell
.\.venv\Scripts\python.exe examples\play_phase_rs.py `
    --deck "Blue Control" --picker random --seed 7 --format Modern `
    --our-deck-file data\decks\modern\modern_mono_red_burn.txt
```

Useful flags:

| Flag | Default | What it does |
|---|---|---|
| `--deck NAME` | `"Red Deck Wins"` | Starter-deck name for the AI seat (one of phase-rs's bundled decks). If you pass a path with `--our-deck-file`, this still drives the opponent. |
| `--our-deck-file PATH` | unset | Path to a `data/decks/*.txt` decklist for our seat; required for a valid live host deck. |
| `--format NAME` | unset (Standard) | Both host and AI decks must be legal in this format. |
| `--picker {random,prefer-nonpass,heuristic,tev1,ollama}` | `prefer-nonpass` | Which `ActionPicker` to install. `tev1` uses the local decision API; `ollama` uses chat/generation. |
| `--picker agent:<name>` | unset | Run one of our native `src/agents` implementations through the phase-rs bridge (for example `agent:heuristic`). |
| `--ollama-model NAME` | `gemma4:e2b` | Ollama model tag for `--picker ollama`. |
| `--ollama-url URL` | `http://localhost:11434` | Ollama HTTP endpoint for `--picker ollama`. |
| `--tev1-model NAME` | `tev1:0.8b` | Installed decision model for `--picker tev1` (Ollama 0.35+). |
| `--tev1-candidates N` | `24` | Shortlist cap, between 2 and 24; seeded heuristic tiers with pass retained. |
| `--tev1-timeout N` | `120` | Decision API timeout in seconds; errors are explicit incomplete runs. |
| `--ai-difficulty NAME` | `Medium` | One of `VeryEasy`, `Easy`, `Medium`, `Hard`, `VeryHard`. |
| `--uri URI` | `ws://127.0.0.1:9374/ws` | phase-server WebSocket URI. |
| `--autostart` | off | Starts a local `phase-server` subprocess automatically (or adopts an already-running one on the same port). |
| `--seed N` | `7` | RNG seed for the picker. |
| `--max-actions N` | `2000` | Safety cap. |
| `--ai-deck-file PATH` | unset | Repeat for explicit native AI decks; three repetitions plus a host Commander list create a four-player pod. |
| `--max-game-seconds N` | unset | Cooperative wall-clock game limit; incomplete if reached. |
| `--max-turns N` | unset | Global native turn-number cap, not pod rounds; incomplete in benchmark output. |
| `--max-message-mib N` | `16` in CLIs | Bounded transport frame limit; Python config retains a 2 MiB default. |

For batched experiments against phase-rs AI, use:

```powershell
.\.venv\Scripts\python.exe scripts\run_phase_rs_ablation.py `
  --games 8 --picker ollama --ai-difficulty Hard `
  --our-deck-file data\decks\modern\modern_mono_red_burn.txt `
  --format Modern --ai-deck "Blue Control" --autostart
```

This emits `summary.json` + `games.jsonl` under `runs/phase_rs_ablation/<timestamp>/`.
Each game also writes a structured trace JSONL file under
`runs/phase_rs_ablation/<timestamp>/traces/` (decision events, legal action
types, chosen action type, terminal reason).
See [EXPERIMENTS.md](EXPERIMENTS.md) for the three-objective sweep,
configuration provenance, full Commander commands, bounded publication pilot
and schema-v2 completion-aware metrics. See
[RESEARCH_PIPELINES.md](RESEARCH_PIPELINES.md) for the decision/control diagrams.

For rollout grids (picker x difficulty x deck), use:

```powershell
.\.venv\Scripts\python.exe scripts\phase_rs_rollout_sweep.py `
  --games-per-cell 3 --autostart
```

This emits per-game rollouts (`rollouts.jsonl`) plus aggregate tables
(`summary.csv`, `summary.json`) under
`runs/phase_rs_rollout_sweep/<timestamp>/`.
Per-game traces are additionally written to
`runs/phase_rs_rollout_sweep/<timestamp>/traces/`.

For training-trace collection (flat event stream + per-game metadata), use:

```powershell
.\.venv\Scripts\python.exe scripts\collect_phase_rs_traces.py `
  --games 16 --picker agent:heuristic --ai-difficulty Medium --autostart
```

This writes `games.jsonl` and `trace_events.jsonl` under
`runs/phase_rs_training_traces/<timestamp>/`.

These event summaries do **not** contain full numeric transition observations.
Do not turn game/deck metadata into dummy world-model training states. For
actual native graph and model learning, use the direct recorder:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_native_learning `
  --train-games 64 --validation-games 16 --test-games 8 `
  --training-seeds 0 1 2 3 4 --epochs 20 --graph-epochs 100 --eval-games 20
```

The [learning runbook](EXPERIMENTS.md#native-graph-and-model-learning) documents
data boundaries, strict model bundles, graph activation and limitations.
The old `train_pipeline.py --phase-rs-traces` conversion is retired and fails
explicitly, rather than mixing placeholders or legacy self-play into native data.

Note: these are **episode rollouts** (full-game Monte Carlo), not
per-decision branch dreaming from arbitrary in-game states. True
``dream_search`` on phase-rs requires the pending Action/GameState
translator tracked in `IMPLEMENTATION_PLAN.md`.

No GUI is required — the browser/Tauri client is for human players. The
server can run headless and our agents only need the WebSocket.

## Engine update checklist

1. Build the pinned server using the MSVC discovery command under K1 in
   [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md).
2. If old cached card data no longer deserializes, let the server refresh it
   through its signed bootstrap. For this release, set
   `PHASE_DATA_MANIFEST_URL=https://data.phase-rs.dev/desktop/release-server-v0.102.0.json`
   in the environment of the server you launch. The native bootstrap verifies
   hashes and quarantines incompatible data as `.unusable`; do not edit card JSON.
3. Regenerate `src/integrations/phase_rs/format_defaults.json` after engine
   updates. Protocol v106 requires complete `FormatConfig` payloads: passing
   only `{"format": "Modern"}` is rejected. The snapshot is exported from
   `GameFormat::registry()`, not guessed Python defaults.

The exporter needs the engine `.rlib` and **the exact serde_json dependency
linked into that engine**, not whichever serde_json file is newest. Inspect
the engine with `rustc -Z ls=root <engine-rlib>` in the pinned nightly toolchain.
After the current release build, the verified libraries are:

```powershell
# Run in the repository root with the same initialized MSVC environment.
.\.venv\Scripts\python.exe -m scripts.export_phase_rs_formats `
  --engine-rlib external\phase-rs\target\release\deps\libengine-996fedad16fa4767.rlib `
  --serde-json-rlib external\phase-rs\target\release\deps\libserde_json-5f0d4f6fc327e23f.rlib
```

Library hashes change with builds; rediscover them instead of reusing stale
paths. The exporter uses release-compatible thin LTO and emits revision/protocol
provenance. Commit the regenerated snapshot with the submodule update.

4. Run the protocol/format tests and a complete live game. The October smoke
   used Modern burn versus `Blue Control` at VeryEasy and finished in 14 turns
   and 150 controlled actions. This verifies integration, not playing strength.

The engine exposes only five named AI starter decks: `Red Deck Wins`,
`White Weenie`, `Blue Control`, `Green Stompy`, and `Azorius Flyers`.
Names are not a guarantee of format legality: several contain Legacy-only
cards. Python's exported list is checked against the engine source.

## phase-rs's built-in AI (the opponent)

phase-rs ships its own AI in the `phase-ai` crate. The headline difficulty
levels (`AiDifficulty`) map to qualitatively different policies:

| Difficulty | Search | Threat awareness | Roughly |
|---|---|---|---|
| `VeryEasy` | disabled | None | Heuristic + softmax over candidate scores. |
| `Easy` | disabled | None | Same, slightly tighter softmax temperature. |
| `Medium` | beam search (shallow), no rollouts | Archetype-only | Default opponent. |
| `Hard` | beam + rollouts | Full (per-card hypergeometric) | Reasons about what opponent might hold. |
| `VeryHard` | wider beam + deeper rollouts | Full | Same regime, larger budgets. |

Wall-clock budget for any single decision is capped at
`AI_SEARCH_TIME_BUDGET_MS = Some(1500)` (see
`external/phase-rs/crates/phase-ai/src/config.rs`). Their deterministic
test runs disable this cap; on the WebSocket protocol you get the
production regime. Knobs include `OpponentModel`
(`DeterministicBestReply` / `ThreatWeightedReply` / `SampledReply`),
`PlannerMode` (`BeamOnly` / `BeamPlusRollout`), and CMA-ES-tuned
`PolicyPenalties` (gift-card penalty, overkill penalty, ward / hexproof
respect, etc.).

For our purposes:

- **Baseline opponent for agent benchmarks:** `Medium` is the right
  default — closest to "a human who reads their hand but doesn't
  metagame your decklist."
- **Stress-testing belief modules:** `Hard` / `VeryHard` give an
  opponent that actually models your hidden information, so our
  `ActiveInferenceAgent`'s belief updates have something non-trivial to
  fight against.
- **Quick smoke / unit tests:** `VeryEasy` — search disabled means
  reproducible-ish moves at minimal latency.

## What's wired today

`src/integrations/phase_rs/`:

| File | Purpose |
|---|---|
| `client.py` | Protocol-v6 WebSocket client. Typed dataclasses for `ServerHello`, `GameCreated`, `GameStarted`, `StateUpdate`, `GameOver`, `ActionRejected`. `PhaseServerClient.handshake()`, `.create_game_with_ai(...)`, `.send_action(...)`, `.stream()`. |
| `decks.py` | `Decklist` → phase-server `DeckData` JSON. `STARTER_DECK_NAMES` mirror. |
| `agent_bridge.py` | `ActionPicker` Protocol + `RandomActionPicker` + `PreferNonPassPicker` (+ `HeuristicActionPicker`). |
| `adapter.py` | phase-rs `GameAction` <-> our `Action` conversion and minimal snapshot -> `GameState` projection used by agent-backed pickers. |
| `runner.py` | `run_game` / `run_game_sync` — drives one phase-rs game to completion. |
| `README.md` | One-page quick reference (mirrors this doc). |

## What's not yet wired

- **Full-fidelity `Action ↔ GameAction` translator.** A practical bridge now
  exists for common actions (including native-agent picker support), but full
  coverage of all rare/complex variants still needs expansion before claiming
  complete parity for every policy module.
- **Differential conformance harness.** Replay a Python-engine trajectory
  through phase-rs and diff. Our chosen sync strategy between engines.
- **`--engine phase_rs` flag on `examples/play_edh_pod.py`** and the
  orchestrator.

## Keeping the submodule in sync

```powershell
git submodule update --remote --merge external/phase-rs
.\.venv\Scripts\python.exe -m pytest tests/integrations/phase_rs -q
```

If the parser tests fail after a sync, phase-rs likely bumped
`PROTOCOL_VERSION`. Update `PROTOCOL_VERSION` in `client.py` and any
renamed fields surfaced by the failing tests.

## Contributing back to phase-rs

phase-rs accepts card implementations and bug fixes from LLM-assisted
contributors with no Rust toolchain required. See their guide:
<https://github.com/phase-rs/phase/blob/main/docs/AI-CONTRIBUTOR.md>.

If we contribute, prefer cards that appear in our own pod decks
(`data/decks/`) and show up as unimplemented in their public coverage
feed: <https://pub-fc5b5c2c6e774356ae3e730bb0326394.r2.dev/staging/coverage-data.json>.

[phase]: https://github.com/phase-rs/phase
[protocol]: https://github.com/phase-rs/phase/blob/main/crates/server-core/src/protocol.rs
