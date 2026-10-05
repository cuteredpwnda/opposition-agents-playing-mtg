# Experiments — Overnight Runs, Ablations, Tournaments

This is the runbook for **everything that takes more than five minutes**:
ablation studies, head-to-head tournaments, and overnight training
pipelines.  None of these scripts pipe their output through PowerShell
filters — they all write a top-level ``run.log`` plus per-game logs and
machine-readable summaries.  Use ``Get-Content -Wait <path>`` to follow.

---

## Current supported benchmark — phase-rs (October 2026)

The older sections below describe the legacy Python-engine experiments.
Several entry points moved to `scripts_legacy/`; do not use the weekend
campaign as the current phase-rs benchmark. The authoritative execution
plan is [IMPLEMENTATION_PLAN.md, section 6](../IMPLEMENTATION_PLAN.md).

First build the pinned `external/phase-rs` server using the MSVC setup
documented under K1 in the implementation plan. Python expects protocol
v106; the contract test checks the upstream constant. See the integration
guide's [engine update checklist](PHASE_RS_INTEGRATION.md#engine-update-checklist)
for data-schema refresh and complete format-registry regeneration.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_phase_rs_ablation `
    --picker kl_control --objectives kl efe_infogain efe_ambiguity `
    --our-deck-file data\decks\modern\modern_mono_red_burn.txt `
    --format Modern --ai-deck "Blue Control" --ai-difficulty Medium `
    --games 3 --seed 7 --max-actions 2000 --stream-timeout 180 --autostart
```

`--games` is games **per arm** (this pilot attempts nine scheduled games).
For a baseline use `--picker heuristic` or `--picker random` and omit
`--objectives`. An explicit nonempty host deck is required: the opponent's
starter-deck name does not populate our deck.
Both decks must be legal in the selected format. In the pinned engine,
`Red Deck Wins` contains Jackal Pup and is not Modern legal; `Blue Control`
was validated with the Modern burn host deck.

Outputs under `runs/phase_rs_ablation/<timestamp>/`:

- `config.json`: full arguments, checkout revisions, protocol, deck hash,
  seed scope and the objective/planning confound.
- `games.jsonl`: written after each scheduled game; includes objective,
  picker seed, outcome, attempt count and total wall time including retries.
- `traces/`: final game traces plus timed-out attempt traces. The initial
  event records the actual server version/build from the handshake, which
  may differ from the checkout if an existing or remote server is used.
- `summary.json`: per-arm and aggregate completion rate, win/loss/draw,
  incomplete counts, reason counts and Wilson 95% win-rate intervals;
  decision-latency p50/p95, mean candidate coverage and shortlisted-decision count.

**Schema v2:** `win_rate` is wins / completed games (draws included).
Timeouts, rejections, AI-driver faults and action caps are incomplete runs,
not draws. With no completed games, win rate and interval are `null`.
Any incomplete scheduled game makes the command exit with code 1.
Transport/handshake errors propagate rather than being reported as games.
Only Python picker RNG is seeded; these are not deterministic paired
engine replays. KL horizon planning vs one-step legacy scoring is a
policy-stack comparison, not a controlled objective-only result.

### Deck variety and promotion from preliminary to full results

The bounded pilot now supports an explicit deck pool and recorded matchups.
The default burn mirror is unchanged. The `modern-diverse` pool adds
hand-authored creature-aggro and control fixtures; see
[the deck-pool notes](../data/decks/benchmark/README.md). It is not a claim
that older example lists are current tournament-legal decks.

```powershell
# Three deck mirrors, one heuristic game each: bounded development smoke.
.\.venv\Scripts\python.exe -m scripts.run_paper_pilot `
    --format Modern --deck-pool modern-diverse --arms heuristic `
    --games 1 --budget-seconds 900

# Nine ordered matchups, two baselines: 18 scheduled games.
# The budget may stop the campaign; unstarted cells remain explicit.
.\.venv\Scripts\python.exe -m scripts.run_paper_pilot `
    --format Modern --deck-pool modern-diverse --matchups round-robin `
    --arms random heuristic --games 1 --budget-seconds 1800

# An explicitly selected pool; --deck-files and --deck-pool are exclusive.
.\.venv\Scripts\python.exe -m scripts.run_paper_pilot `
    --format Modern --deck-files data\decks\benchmark\modern_green_stompy.txt `
    data\decks\benchmark\modern_azorius_control.txt --arms heuristic --games 1
```

`--games` is replicates **per arm and matchup**. Policies are interleaved
within each matchup/replicate; the seed controls the Python picker, not native
shuffle or opponent RNG. A round-robin changes deck roles, not player seats.
Schema-v3 manifests freeze all selected deck files, and each cell names its
host and opponents. Custom pools currently apply to Modern; Commander retains
its existing explicit four-deck pod. The pilot is deliberately capped at
30 minutes and is not the full-campaign runner.

Moving to paper-ready results requires these runs and gates, in order:

1. **Freeze and audit inputs:** record source/data/deck/checkpoint hashes,
   protocol, environment/hardware and dirty diff. Validate selected decks in
   the native format and audit study-card/action semantics.
2. **Qualify every intended condition:** at least 20 calibration games per
   deck/difficulty/policy condition, at least 95% completion, no protocol or
   illegal-choice failures. Preserve all failures. Small budgeted batches can
   be repeated, but aggregation must not silently select successful batches.
3. **Resolve claim-specific confounds:** objective-only claims need matched
   candidates, horizon, models and compute; current KL/EFE stacks differ.
   Native RNG controls/seat rotation remain work, not implemented flags.
   Tev1 input budgeting, KG grounding and native learned dynamics are separate
   blockers. Extra games cannot fix them.
4. **Lock a held-out schedule and power analysis:** freeze the primary
   endpoint, strata, exclusions, sample size and stopping rules before tuning
   ends. The plan's initial 2,000-game budget is exploratory: about 1,570
   independent games per arm are needed to detect a five-percentage-point
   difference near 50% with 80% power at two-sided 5%, before multiplicity.
5. **Run eligible held-out cells:** use the completion-aware ablation runner,
   explicit host/opponent decks and difficulty, and the prespecified schedule.
   It is a policy-stack study until controlled objective comparisons exist.
6. **Analyse and release:** per-stratum win/completion/failure rates, Wilson
   intervals, stratified game-level effect intervals, declared multiple-test
   adjustment, latency and budget; release manifests, all attempts, analysis
   code and regenerated tables. Commander additionally needs terminal pod
   and elimination qualification, not turn-capped smokes.

The authoritative details and outstanding gates are in
[Implementation Plan §6](../IMPLEMENTATION_PLAN.md#6-benchmark--evaluation-plan).
No currently available command alone turns the entire proposed architecture
into a validated full-results study.

**Downloaded coverage:** `--deck-pool modern-expanded` adds four pinned,
source-attributed historical Forge lists (Boros/Atarka burn, Eldrazi Tron,
Affinity), for seven decks and 49 ordered matchups. Source license, hashes,
counts and eligibility boundaries are in the
[sourced-deck README](../data/decks/benchmark/sourced/README.md).
Reacquire with `python -m scripts.fetch_benchmark_decks`.
Native qualification of these additions must precede strength comparisons.
Do not confuse 49 matchup configurations with 49 qualified experimental cells.

### Tev1 decision-model baseline

The dedicated `--picker tev1` uses `/v1/systemone`, not the chat picker.
Ollama 0.35+ is required. The 0.8B model was installed and its real local
API contract validated on Oct 5; this is not an MTG-strength result.
Pull `tev1:4b` separately before a 4B run.
**Full-game evaluation is not yet validated:** local Ollama 0.35.1 enforces
64 KiB requests and a 2050-token prompt ceiling. Raw engine snapshots exceeded
the byte limit; compact pilots reached turns 18/22 but still failed at the
token ceiling. Do not launch a strength campaign until K6's budgeted-context
policy is implemented. The command below is for functional investigation.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_phase_rs_ablation `
    --picker tev1 --tev1-model tev1:0.8b --tev1-candidates 24 `
    --our-deck-file data\decks\modern\modern_mono_red_burn.txt `
    --format Modern --ai-deck "Blue Control" --ai-difficulty VeryEasy `
    --games 3 --seed 7 --autostart
```

The same picker/flags are available in `examples/play_phase_rs.py`,
`scripts/collect_phase_rs_traces.py` and `scripts/phase_rs_rollout_sweep.py`.
`--ollama-url` must point to a local server for Tev1;
`--tev1-timeout` defaults to 120 seconds.

Candidate selection reuses heuristic action tiers, uses a seeded RNG for
ties, excludes Concede when alternatives exist and preserves a pass action.
One eligible action bypasses inference. Model option keys map to the original
legal indices; action data/targets are not reconstructed from model text.
Traces record all offered indices, types, coverage, probabilities and
confidence. Coverage is offered / eligible action count, **not optimal-move
recall**. Confidence is probability concentration, **not chance of winning**.

Context projection `compact_card_aware_v3` retains visible card IDs (grouping
equivalent rows), public characteristics, life/mana, combat and decision state.
Hidden placeholders, library order, internal journals and compiled rule ASTs
are omitted; it is not a complete rules-state representation. No visible
card rows or candidate payloads are silently cropped. Traces record projection
version and actual request bytes. Requests above 64 KiB fail before HTTP;
native token-budget failures retain the server diagnostic as incomplete runs.

API errors and invalid answers are distinct incomplete reasons; no random
fallback occurs. Latency aggregates include attempted decisions in the retained
game trace (including forced decisions and explicit inference errors).
Separate forced decisions and cold starts when interpreting performance.

This initial deployment comparison is not a clean decision-tuning ablation:
the existing chat picker receives a compact context and silently permits
fallback, whereas Tev1 uses a card-aware filtered projection and explicit failures.
Match context/candidates, account for chat fallback, add a size-matched
general Qwen3.5 comparator and control engine RNG/seat assignment before
making causal claims. The paper's decision-model benchmark subsection
distinguishes these experiments.

---

## Publication campaign and four-player Commander

The full staged paper plan, claim-to-evidence matrix, sample-size caveats,
holdouts and stopping rules are in
[IMPLEMENTATION_PLAN.md, section 6](../IMPLEMENTATION_PLAN.md#6-benchmark--evaluation-plan).
Five Mermaid diagrams explain
[data lineage, decisions, pods, learning and publication gates](RESEARCH_PIPELINES.md).

### Data freeze first

```powershell
.\.venv\Scripts\python.exe -m scripts.fetch_card_data --refresh
```

This refreshes **all oracle-card records and rulings**, not every printing/image.
The October 5 snapshot contains 38,706 oracle records, 79,706 rulings and
38,201 indexed names. The current API's gzip JSONL is validated and converted
to the existing JSON-array cache. All downloads and the dependent name index
are prepared before publishing; each file replacement is atomic and the
SHA-256/source-date manifest is published last. Consumers must verify hashes:
the multi-file replacement itself is not a filesystem transaction.

The engine's compiled card definitions were already refreshed separately
through the signed phase-rs bootstrap. Scryfall data does not replace native
mechanic implementations. Refreshing these files does not automatically
re-import Neo4j or retrain checkpoints.

### Bounded initial pilot

```powershell
.\.venv\Scripts\python.exe -m scripts.run_paper_pilot `
    --format Modern --games 3 --seed 7 --budget-seconds 900
```

This attempts 15 games: random, heuristic and three objective **stacks**,
interleaved over three picker seeds. Host and native opponent use the same
explicit burn list. It starts an isolated server with a temporary persistence
database, verifies source hashes, captures per-cell configs/traces/summaries,
and has a hard campaign timeout. Caps and process errors are incomplete cells,
not draws; unscheduled cells are reported separately.

Schema v2 requires an already built pinned release server, selects that binary
explicitly, records its SHA-256, deck-file/Python-source hashes and Python/package
versions, verifies the dependent name index, and rechecks frozen inputs after
the campaign. It refuses adopted servers; changed or deleted inputs invalidate
the campaign with the affected paths recorded. Exact dirty-patch
archival and native auxiliary-input coverage still require P0 review.

The initial schema-v1 October 5 feasibility run attempted two games per stack:
**8/10 completed, all eight losses**; both KL games reached the 1,500-action cap.
Their traces show repeated `CastSpell` / `CancelCast` decisions at Upkeep.
Fix modeled casting decisions and cancellation before another strength run;
this does not pass the calibration completion/semantic gate. This initial
pilot does not retroactively gain schema v2's source/binary freeze.

The pilot is not powered evidence, does not seed native RNG or rotate seats,
and does not include unsupported learning arms or Tev1/chat until their gates
pass. Full studies need a fresh holdout and an approved compute budget.

### Commander smoke and terminal qualification

```powershell
# Explicit six-turn development smoke; not a completed pod or benchmark draw.
.\.venv\Scripts\python.exe -m examples.play_edh_pod `
    --max-turns 6 --max-game-seconds 120 --seed 7 --model none --autostart

# Host Krenko; three native AI seats get real 99 + commander lists.
.\.venv\Scripts\python.exe -m scripts.run_phase_rs_ablation `
    --picker heuristic --games 1 --format Commander --ai-difficulty VeryEasy `
    --our-deck-file data\decks\edh\krenko-mob-boss_core.txt `
    --ai-deck-file data\decks\edh\atraxa-praetors-voice_core.txt `
    --ai-deck-file data\decks\edh\meren-of-clan-nel-toth_core.txt `
    --ai-deck-file data\decks\edh\urza-lord-high-artificer_core.txt `
    --max-message-mib 16 --max-game-seconds 420 --max-retries 1 --autostart
```

The pod wrapper currently supplies **one Python policy + three native AI
policies**, not four Python zoo agents. All four decks have 99 main-deck cards
plus a commander; the older `edh_krenko_goblins.txt` has only 75 main-deck cards.
The first real pod reached turn 34/483 host actions and exercised elimination,
but its transport closed; that run is incomplete, not a successful terminal pod.
The 16 MiB follow-up reached turn 40/507 actions before `game_timeout` at
420 seconds. A host-Atraxa/native-Krenko follow-up reached turn 39/418 actions,
then `stream_timeout`, recording a WebSocket keepalive-ping timeout (1011,
no close frame received). No terminal pod is verified; investigate native
progression and reconnect as well as transport limits before pod studies.

`play_edh_pod` treats an explicitly requested turn-capped smoke as successful
execution while writing `turn_cap` in its trace. The benchmark always counts
turn/time/action caps as incomplete and exits nonzero if a scheduled game fails.
Global native `turn_number` counts each player's turn, not complete pod rounds.
Wall-clock checks inside the async runner are cooperative and can overrun during
a blocking picker call; the pilot subprocess timeout is the hard outer bound.

---

## TL;DR

| Goal | Script | Output |
|---|---|---|
| Train V → M → C → eval overnight | `scripts/overnight_run.py` | `runs/overnight_<ts>/stage{4..7}_*.log`, `SUMMARY.txt` |
| 1v1 ablation (every pair, both seats) | `scripts/run_matchups.py` | `runs/<out>/run.log`, `games.csv`, `summary.json`, `logs/game_NNNN.log` |
| 4-player EDH pod (rotating seats) | `scripts/run_matchups.py --pod` | same as above |
| Swiss-paired tournament (1v1) | `scripts/run_tournament.py --mode swiss` | `run.log`, `rounds.csv`, `standings.csv`, `summary.json` |
| Pod tournament (multi-round, reseating) | `scripts/run_tournament.py --mode pod` | same as Swiss |
| LLM-vs-LLM smoke (Ollama only) | `examples/ablation_llm_only.py` | per-game JSONL + summary |

All scripts default to writing logs.  None of them require ``--save-logs``.

---

## 1. Overnight training run

`scripts/overnight_run.py` chains four stages of `train_pipeline.py`:

| Stage | What it does |
|---|---|
| 4 | Self-play data collection — agents play themselves into transition tuples |
| 5 | JEPA encoder + V-model training |
| 6 | Dream training (M-model rollouts + C-model controller) |
| 7 | Evaluation games against baselines |

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED  = '1'

# Recommended overnight invocation (no Neo4j dependency):
.\.venv\Scripts\python.exe -u -m scripts.overnight_run `
    --num-games 50 `
    --jepa-epochs 30 `
    --jepa-batch-size 8 `
    --dream-iters 3 `
    --eval-games 4 `
    --no-kg
```

The runner writes everything under
``runs/overnight_<YYYY-MM-DDTHH-MM>/``:

```
runs/overnight_2026-04-28T22-15/
  config.json          ← exact CLI arguments used
  stage4_selfplay.log
  stage5_jepa.log
  stage6_dream.log
  stage7_eval.log
  SUMMARY.txt          ← per-stage exit codes + total wall time
```

Useful flags:

* ``--skip-stage 4`` — repeat to skip multiple (e.g. ``--skip-stage 4 --skip-stage 5``)
* ``--out-root runs`` — change output directory
* The runner bails on first non-zero stage exit; check the matching ``stage*_*.log`` for the failure.

### Backgrounding on Windows

```powershell
$env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
$p = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList "-u","-m","scripts.overnight_run","--num-games","50","--jepa-epochs","30","--no-kg" `
    -RedirectStandardOutput "runs\overnight_dispatch.log" `
    -RedirectStandardError  "runs\overnight_dispatch.err" `
    -NoNewWindow -PassThru
"overnight_pid=$($p.Id)"
```

Tail with:

```powershell
Get-Content runs\overnight_dispatch.log -Wait -Tail 30
```

---

## 2. Ablation harness — `scripts/run_matchups.py`

Use this when you want **direct head-to-head data** between agents on
fixed decks.  It plays every ordered pair (seat-swapped) for ``--games``
games each and reports per-agent win rate.

### The four canonical 1v1 ablations

The paper's claim is that combining *LLM*, *world model*, and *KG*
beats any single component.  Run them all into one folder per ablation:

```powershell
# A) LLM-only (Ollama)
python scripts/run_matchups.py --format standard `
    --decks data/decks/modern/modern_mono_red_burn.txt data/decks/modern/modern_azorius_control.txt `
    --agents llm heuristic --games 8 --max-turns 80 `
    --out runs/ablation/A_llm_only

# B) World-model only
python scripts/run_matchups.py --format standard `
    --decks data/decks/modern/modern_mono_red_burn.txt data/decks/modern/modern_azorius_control.txt `
    --agents world_model heuristic --games 8 --max-turns 80 `
    --out runs/ablation/B_world_model_only

# C) KG-only (kg_heuristic = heuristic + KG combo bias)
python scripts/run_matchups.py --format standard `
    --decks data/decks/modern/modern_mono_red_burn.txt data/decks/modern/modern_azorius_control.txt `
    --agents kg_heuristic heuristic --games 8 --max-turns 80 `
    --out runs/ablation/C_kg_only

# D) Full fusion (LLM + world model + KG)
python scripts/run_matchups.py --format standard `
    --decks data/decks/modern/modern_mono_red_burn.txt data/decks/modern/modern_azorius_control.txt `
    --agents llm_fusion heuristic --games 8 --max-turns 80 `
    --out runs/ablation/D_fusion
```

Each run writes:

```
runs/ablation/A_llm_only/
  run.log              ← live stdout, tail with Get-Content -Wait
  games.csv            ← one row per game (winner, turns, seats, decks)
  summary.json         ← aggregate win rate per agent
  logs/game_0001.log   ← full readable game log (engine + state.log lines)
  logs/game_0002.log
  ...
```

### EDH pod ablation

```powershell
python scripts/run_matchups.py --format commander --pod `
    --decks data/decks/edh/krenko-mob-boss_core.txt `
            data/decks/edh/atraxa-praetors-voice_core.txt `
            data/decks/edh/urza-lord-high-artificer_core.txt `
            data/decks/edh/meren-of-clan-nel-toth_core.txt `
    --agents heuristic kg_heuristic world_model llm_fusion `
    --games 4 --max-turns 200 `
    --out runs/ablation/pod_full
```

In ``--pod`` mode each agent plays every seat (cyclic rotation).
``--games 4`` means four full rotations.

---

## 3. Tournament harness — `scripts/run_tournament.py`

For **multi-agent leaderboards** (more than two agents in one run, with
points and tiebreakers).

### Swiss (1v1 modern/standard)

```powershell
python scripts/run_tournament.py --mode swiss --rounds 6 `
    --agents random heuristic kg_heuristic world_model active_inference llm_fusion `
    --decks data/decks/modern/modern_mono_red_burn.txt data/decks/modern/modern_azorius_control.txt `
    --format standard --max-turns 80 `
    --out runs/tournament/standard_swiss
```

* 3 / 1 / 0 points per win / draw / loss.
* Buchholz tiebreak (sum of opponents' points).
* No agent plays the same opponent twice (until the pool is exhausted).

### Pod tournament (EDH, multi-round, reseating)

```powershell
python scripts/run_tournament.py --mode pod --rounds 8 `
    --agents heuristic kg_heuristic world_model llm_fusion `
    --decks data/decks/edh/krenko-mob-boss_core.txt `
            data/decks/edh/atraxa-praetors-voice_core.txt `
            data/decks/edh/urza-lord-high-artificer_core.txt `
            data/decks/edh/meren-of-clan-nel-toth_core.txt `
    --format commander --max-turns 200 `
    --out runs/tournament/edh_pod
```

Each round randomly reseats the four agents, plays one pod game,
awards 3 points to the winner.

---

## 4. Watching a long run

Every script writes ``<out>/run.log`` and individual ``logs/game_NNNN.log``
files.  Tailing in PowerShell:

```powershell
# Top-level dispatch (matchup CLI lines, summary)
Get-Content runs\tournament\edh_pod\run.log -Wait -Tail 40

# Latest game in progress
Get-ChildItem runs\tournament\edh_pod\logs -Filter game_*.log |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 | ForEach-Object { Get-Content $_.FullName -Wait -Tail 30 }
```

---

## 5. Reproducibility

All harnesses honour a single ``--seed`` integer and thread it through
the runner so every game with the same ``(seed, decks, agent-types)``
triple replays identically.  Per-game seeds are derived as
``seed + game_idx`` so games within one run are still distinguishable.

---

## 6. Available agents

The ``--agents`` flag in every harness accepts names registered in
[src/agents/__init__.py](src/agents/__init__.py):

| Name | Class | Notes |
|---|---|---|
| `random` | `RandomAgent` | uniform baseline |
| `heuristic` | `HeuristicAgent` | hand-tuned priorities |
| `kg_heuristic` | `KGHeuristicAgent` | heuristic + KG combo bias (needs Neo4j) |
| `human` | `HumanAgent` | interactive CLI input |
| `ollama` / `llm` | `OllamaAgent` | local LLM via Ollama |
| `world_model` | `WorldModelAgent` | trained V/M/C model |
| `active_inference` | `ActiveInferenceAgent` | belief-state EFE planner |
| `llm_fusion` / `fusion` | `LLMFusionAgent` | LLM + world model + KG combined |

---

## 7. Known caveats

* **KG-dependent agents** (``kg_heuristic``, ``llm_fusion``,
  ``active_inference``) silently fall back if Neo4j isn't running.
  Make sure ``mtg-neo4j`` is up before launching tournaments meant to
  test KG contributions.  Boot with: ``docker compose up -d neo4j``.
* **Ollama agents** require a running daemon on ``localhost:11434``
  with the chosen model pulled.  See [docs/OLLAMA_SETUP.md](docs/OLLAMA_SETUP.md).
* **Large pods with LLM agents** are slow — count on ~1–5 minutes per
  game depending on model size.  Run those overnight.
