# phase-rs integration

[phase-rs/phase][phase] is a Rust-native Magic: The Gathering rules engine
with a WebSocket server, ~34,300+ implemented cards (sourced from MTGJSON),
and a per-difficulty AI opponent. We vendor it as a git submodule under
`external/phase-rs/` and talk to it from Python via a small adapter in
`src/integrations/phase_rs/`.

> **Status — May 2026:** Adapter beachhead on branch `feat/phase-rs-engine`.
> A `RandomActionPicker` finishes games against phase-rs's AI end-to-end
> over WebSocket. Full state translation (so `HeuristicAgent`,
> `WorldModelAgent`, `ActiveInferenceAgent` can drive phase-rs) is in
> progress — tracked in [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md)
> under the "phase-rs engine adapter" queue entry.

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
Wire version: `PROTOCOL_VERSION = 6`. Our parser is validated against
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
    --deck "Red Deck Wins" --picker random --seed 7
```

Useful flags:

| Flag | Default | What it does |
|---|---|---|
| `--deck NAME` | `"Red Deck Wins"` | Starter-deck name for the AI seat (one of phase-rs's bundled decks). If you pass a path with `--our-deck-file`, this still drives the opponent. |
| `--our-deck-file PATH` | unset | Path to a `data/decks/*.txt` decklist for our seat. |
| `--picker {random,prefer-nonpass}` | `prefer-nonpass` | Which `ActionPicker` to install in our seat. |
| `--ai-difficulty NAME` | `Medium` | One of `VeryEasy`, `Easy`, `Medium`, `Hard`, `VeryHard`. |
| `--uri URI` | `ws://127.0.0.1:9374/ws` | phase-server WebSocket URI. |
| `--seed N` | `7` | RNG seed for the picker. |
| `--max-actions N` | `2000` | Safety cap. |

No GUI is required — the browser/Tauri client is for human players. The
server can run headless and our agents only need the WebSocket.

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
| `runner.py` | `run_game` / `run_game_sync` — drives one phase-rs game to completion. |
| `README.md` | One-page quick reference (mirrors this doc). |

## What's not yet wired

- **`Action ↔ GameAction` translator.** Right now pickers receive opaque
  JSON dicts. Mapping each phase-rs `GameAction` variant to our
  `src/engine.Action` (and vice versa) is the gating step for plugging
  `HeuristicAgent` / `WorldModelAgent` / `ActiveInferenceAgent` into
  phase-rs. Tracked in `IMPLEMENTATION_PLAN.md`.
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
