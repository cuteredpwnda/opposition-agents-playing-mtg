# Experiments — Overnight Runs, Ablations, Tournaments

This is the runbook for **everything that takes more than five minutes**:
ablation studies, head-to-head tournaments, and overnight training
pipelines.  None of these scripts pipe their output through PowerShell
filters — they all write a top-level ``run.log`` plus per-game logs and
machine-readable summaries.  Use ``Get-Content -Wait <path>`` to follow.

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
