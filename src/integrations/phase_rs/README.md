# phase-rs runtime integration

[phase-rs](https://github.com/phase-rs/phase) is the authoritative rules
runtime for native training traces and evaluation. Python agents select from
engine-authored legal choices through the WebSocket bridge; the legacy Python
engine remains a compatibility path, not the native rules authority.

## Current contract

The repository pins revision `1191bba65048c83fbbd5a56dfd5f419ee0ead67a`,
protocol **106**. The client validates the handshake and translates tagged
wire envelopes, runner budgets/failures, viewer interactions and reconnect
credentials. Complete format configurations come from the pinned registry.
Contract and runner tests live in `tests/integrations/phase_rs/`.

Build the pinned release `phase-server` using the MSVC instructions in
[the integration guide](../../../docs/PHASE_RS_INTEGRATION.md), then run from
the repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_phase_rs_ablation `
    --picker heuristic --format Modern `
    --our-deck-file data\decks\modern\modern_mono_red_burn.txt `
    --ai-deck-file data\decks\modern\modern_mono_red_burn.txt `
    --ai-difficulty VeryEasy --games 1 --autostart
```

An existing local server uses `ws://127.0.0.1:9374/ws` by default. Supply
`--uri` and omit `--autostart` to connect explicitly. Implicit startup is
process-locked; caller-supplied server processes retain caller ownership.
The local lock file is runtime state and must not be committed.

## Boundaries

- Picker seeds do **not** seed native shuffle or AI RNG; complete deterministic
  replay and controlled seat rotation remain pending.
- Only one Python-controlled seat is currently driven per game; Commander
  pods use three native opponents. Terminal pod qualification is separate.
- Completion-aware reports retain caps and operational failures as incomplete
  runs, not draws. An API smoke or legal move is not a playing-strength result.
- Tev1, KG grounding and learned native dynamics have separate evaluation
  gates; see [the plan](../../../IMPLEMENTATION_PLAN.md#6-benchmark--evaluation-plan)
  and [experiment runbook](../../../docs/EXPERIMENTS.md).

## Attribution

phase-rs is dual-licensed MIT / Apache-2.0. See the upstream notices and
repository [NOTICE](../../../NOTICE.md). Downloaded card data remains separate
from the bridge source; acquire it following the pinned runtime instructions.
