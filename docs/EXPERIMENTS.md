# Experiments — Overnight Runs, Ablations, Tournaments

This is the runbook for **everything that takes more than five minutes**:
ablation studies, head-to-head tournaments, and overnight training
pipelines.  None of these scripts pipe their output through PowerShell
filters — they all write a top-level ``run.log`` plus per-game logs and
machine-readable summaries.  Use ``Get-Content -Wait <path>`` to follow.

---

## Current supported benchmark — phase-rs (October 2026)

For the scientific questions, controls and staged learning/graph factorial,
see [the agent learning study](AGENT_LEARNING_STUDY.md). This runbook describes
execution, not a replacement for that protocol or the plan's status tracker.

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

### Model-stack qualification before the learning study

The bounded model pilot attempts wire random/heuristic, installed chat models, Tev1, explicit-checkpoint
world-model direct/dream modes, explicit-checkpoint LLM fusion and the three
analytic objective stacks. It owns a private server, retains subprocess logs,
attempted-condition summaries and unstarted conditions, and checks source/
checkpoint/deck hashes after the run.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_model_qualification `
    --checkpoint checkpoints\jepa\jepa_final.pt `
    --models llama3.2:1b qwen2.5-coder:1.5b gemma4:e2b `
    --budget-seconds 1800 --condition-seconds 150
```

This schedules twelve conditions at one game each, with a hard per-process limit.
`--games N` schedules replicates sharing the condition deadline;
increase `--condition-seconds` accordingly, subject to the campaign budget.
`--ai-difficulty` selects the native anchor explicitly.
`--only world_model_direct world_model_dream_search fusion` permits a
bounded repair rerun without overwriting earlier failure evidence.
The archived checkpoint is **not native-trained evidence**. A completed
fusion episode does not prove all components were used; graph grounding is
not supplied by this command. Tev1+world-model prior/critic and JEPA-head
dreaming remain unwired/unqualified combinations, recorded as blocked rather
than fabricated arms. No timed-out or fallback-driven condition establishes
a strength result. Read the model manifest's qualification limitations.

The ablation CLI now accepts `--agent-checkpoint`, `--agent-mode`,
`--agent-device`, `--agent-deterministic`, `--dream-rollouts` and
`--dream-depth`. Explicit missing checkpoints fail rather than silently
initialising a replacement. Checkpoint fingerprints accompany configs.
Loaded-key coverage, fallback and model-service identity still require
further audit before confirmatory use.

### What the built-in opponent does

`phase-ai` is upstream's Rust tactical evaluator/search policy, not our
uniform wire-random or action-type wire-heuristic picker. At the pinned
revision, VeryEasy disables tree search but retains tactical scoring and
temperature-4 softmax decisions; Medium enables depth-2 beam/rollout search
with 24 nodes, branching cap 5 and one depth-1 rollout sample, bounded by
a 1.5-second interactive search deadline. Dedicated decision paths can
choose deterministically. Presets load fixed, turn-dependent fitted evaluation
weights; the opponent does not learn from this campaign.

The upstream AI accepts internal `GameState`, and shipped search presets
disable hidden-zone resampling (`determinization_samples=0`). Our policies
receive perspective-filtered observations. Treat native-AI games as
anchor-opponent deployment comparisons, not matched-information rankings.
The paper's baseline and ablation tables state this distinction explicitly.

### Recorded initial local execution

`runs/model_qualification/20261005_172945_548058` attempts all twelve
conditions after the structured-stack repair. **10/12 rules-terminal games,
all ten losses**, zero unstarted conditions, and unchanged frozen input hashes.
Random, heuristic, Llama 3.2 1B, Qwen2.5-Coder 1.5B, direct/recurrent model,
explicit-model fusion and all three analytic objective stacks finish.
Gemma4 E2B reaches the cooperative game timeout; Tev1 receives an explicit
HTTP 400 for a 2,697-token prompt above its 2,050-token ceiling.

This initial configuration is superseded by the replicated campaign below,
but its failures are retained rather than relabelled as repaired results.
One game per condition is feasibility evidence only. These cells do not
satisfy the 20-game/95%-completion calibration gate. The ten losses are
not a policy ranking. Native-trained components, graph interventions,
matched information/compute, model contribution/fallback audits and true
multi-policy control remain K12 prerequisites; this campaign cannot launch
or substantiate unwired RQ2--RQ5 arms.

### Tev1 decision-model baseline

The dedicated `--picker tev1` uses `/v1/systemone`, not the chat picker.
Ollama 0.35+ is required. The 0.8B model was installed and its real local
API contract validated on Oct 5; this is not an MTG-strength result.
Pull `tev1:4b` separately before a 4B run.
The original `tev1:0.8b` tag specifies `num_ctx=2050`; this is not a hard
decision-API token ceiling. Request-level `options.num_ctx` is ignored by
the structured endpoint. Create an isolated context alias instead, without
overwriting the original model or truncating observations:

```powershell
.\.venv\Scripts\python.exe -m scripts.configure_tev1_context
```

This derives `tev1-mtg-8k:0.8b` from the same installed weights with
`num_ctx=8192`, verifies the parameter via `/api/show`, and leaves the
original tag unchanged. A real 2,906-token structured-choice request
completes, whereas the original tag rejects it. The 64 KiB request-body
guard remains. Extended-context performance and model activation still
need qualification; this changes deployment configuration, not learned weights.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_phase_rs_ablation `
    --picker tev1 --tev1-model tev1-mtg-8k:0.8b --tev1-candidates 24 `
    --our-deck-file data\decks\modern\modern_mono_red_burn.txt `
    --format Modern --ai-deck "Blue Control" --ai-difficulty VeryEasy `
    --games 3 --seed 7 --autostart
```

To rerun all twelve implemented conditions under the repaired configuration:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_model_qualification `
    --checkpoint checkpoints\jepa\jepa_final.pt `
    --models llama3.2:1b qwen2.5-coder:1.5b gemma4:e2b `
    --tev1-model tev1-mtg-8k:0.8b `
    --games 3 --condition-seconds 360 --budget-seconds 1800
```

Native chat generation now disables thinking and caps generated output at
16 tokens (the answer is an index). It skips single-action inference, places
temperature in Ollama's `options`, and records model invocation or a specific
random-fallback reason in the decision trace. The later semantic repair also
uses the shared card-aware perspective projection and includes complete
legal-action JSON: previously `ChooseOption` alternatives such as Cancel and
Cast were represented by identical action types. Report fallback counts rather than calling
every completed game a language-model success.

Generated `runs/` outputs are ignored by Git and retained locally. Previously
tracked research artifacts are not automatically removed by ignore rules;
publish deliberately reviewed aggregate evidence rather than raw traces.

The isolated full-game repair check in
`runs/model_qualification/20261005_174657_499980` completes both Gemma and
8K-context Tev1 games with rules-terminal losses, unchanged source hashes,
and elapsed times of 20.5 and 65.3 seconds. Gemma's trace has 18 successful
model decisions, 71 forced decisions, and zero random fallback events.
This confirms the repaired execution path, not a policy-strength advantage.
The replicated campaign additionally freezes model digests, parameters and
quantization before/after execution; Tev1 traces retain actual token usage.

The three-replicate twelve-condition campaign in
`runs/model_qualification/20261005_174927_211167` completes **35/36 games**,
with four wins and one Gemma timeout. Every condition is attempted; source
hashes and model snapshots are unchanged. Llama, fusion, KL and the ambiguity
proxy each win once; the other terminal outcomes are losses. Tev1 completes
3/3 games. These are still small feasibility cells, not policy rankings or
qualified learning ablations.

Gemma's timeout accompanies repeated subdecisions (604 successful model
choices across its three attempts), motivating the complete-action prompt
repair rather than merely increasing the deadline. The separately frozen
repair campaign `runs/model_qualification/20261005_182558_589785` completes
**3/3 Gemma games**, all losses, with 168 total decisions in 116.4 seconds.
Its activation audit records 30 successful model choices, 137 forced choices
and one explicitly traced 30-second request-timeout fallback.
This uses the same 115-second cooperative per-game and 360-second process
budgets as the replicated campaign. Native RNG is not controlled, so it is
execution evidence, not a causal estimate of the prompt change.

The full-context Llama/Qwen check in
`runs/model_qualification/20261005_183103_493572` records two game timeouts
under a 165-second cooperative/180-second process budget, while turns
continue advancing. Larger observations increase inference cost; these are
not the earlier indistinguishable-choice loops. A separately frozen,
declared-budget check `20261005_183955_919512` uses a 585-second cooperative/
600-second process budget and completes 2/2 terminal losses:
Llama 194.6 seconds (18 model choices, 140 forced, six invalid-response
fallbacks), Qwen 48.1 seconds (six model choices, 133 forced, one invalid-response
fallback). Do not pool these runs with the earlier shorter-budget results.
Completion does not establish an unaided LLM contribution or robust reply
formatting. Subsequent parser regressions reject negative/fractional indices
rather than silently extracting their positive/integer parts; strict structured
chat replies and per-model calibration remain follow-up work.

### Decision 2.0: native local classifier, not a vLLM requirement

The [Decision 2.0 collection](https://huggingface.co/collections/vllm-sr/decision-20)
releases 0.6B, 0.8B, 2B, 4B, 9B and 27B variants. The published choice contract
allows **2–255 options**. Eos 0.8B declares **16,384 input tokens**; Kai 0.6B
declares **8,192**. This is a model/runtime change, not a drop-in weight swap
inside Ollama's original Tev1 adapter.

No migration of the engine or existing agents to vLLM is required.
The official local Transformers `system_one` runtime scores a classifier head
without generating chat text. Our `--picker decision2` calls that interface,
reuses the Tev1 candidate/context/answer-validation helpers, and preserves
original legal-action indices. It is wired into ablation, trace collection
and rollout sweep commands. GPU inference remains optional; this machine's
Python Torch build is CPU-only. Do not equate upstream GPU timing with local
CPU timing.

Install the optional declared dependency group:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[decision2]"
```

The default model is Eos 0.8B pinned to
`3594047d69f476f1d01cf84c593e213fc3a4dfe0`, including the custom inference code.
It uses the upstream manifest-verified package and retains the decision head.
Other releases require both their model identifier and a full commit revision;
never leave custom-code or weight revisions floating during evaluation.

**Current Windows qualification blocker:** the real pinned Eos load fails with
`Model identity differs from the scored checkpoint`. All six actual
model/tokenizer file hashes match the published manifest. The upstream
fingerprint serializes `str(file.relative_to(path))`, producing backslash keys
on Windows instead of the manifest's slash keys. Hashing the same verified
file hashes with slash keys reproduces published identity
`d97127991870ae202017a99eb496bdc56a62afd038594a67febfab0aea9d76fe`;
backslash keys produce
`7aa5272a9814966a893e6ed0944c7a60ecc210510d33989d839af81387968023`.
We do not disable the identity check, edit the Hub cache or silently substitute
a chat model. The adapter's 255-option mapping and error paths pass unit tests.
The bounded qualification harness records this as a pre-game model-load failure,
not a completed episode. Direct Windows deployment needs a portable upstream
fingerprint release; the command below does not currently pass that Windows gate.

**Local Linux reference qualification passes:** an owned, isolated Linux
container loads the unchanged, manifest-verified Eos package from the pinned
snapshot using Torch 2.11.0+cpu and Transformers 5.17.0, four CPU threads and a
6 GiB container limit. No vLLM, GPU or inference service is used. Cold model
preparation takes 53.7 seconds, excluding environment installation.

| Synthetic choice set | Input tokens | Returned original key | Probabilities | Forward time |
|---|---:|---|---:|---:|
| 30 options | 1,136 | `action_25` | 30 finite, normalized values | 12.2 s |
| 255 options | 9,771 | `action_222` | 255 finite, normalized values | 136.1 s |

Both requests complete without truncation or generated output tokens. These
are single-request interface checks, not MTG games or a latency benchmark.
The fixture asks for the largest JSON-encoded amount; neither output selects
the intended maximum (`action_29` / `action_254`). Valid shapes therefore do
not certify decision quality. Native MTG episodes, larger family variants,
GPU deployment and candidate-order/coverage ablations remain unmeasured.
The temporary container and model-only hard-link view are removed afterwards.

The equivalent reference check on a Linux Python environment is:

```python
import json
from transformers import AutoModel

revision = "3594047d69f476f1d01cf84c593e213fc3a4dfe0"
model = AutoModel.from_pretrained(
    "vllm-sr/Decision-2.0-Eos-0.8B", revision=revision,
    code_revision=revision, trust_remote_code=True, device="cpu", threads=4,
)
for count in (30, 255):
    criteria = {
        f"action_{i}": json.dumps({"type": "ChooseOption", "data": {"amount": i}})
        for i in range(count)
    }
    result = model.system_one(
        state={"goal": "Choose the largest offered amount."},
        questions={"move": {
            "type": "choice", "instructions": "Choose the largest amount.",
            "criteria": criteria,
        }},
    )
    answer = result["answers"]["move"]
    assert not answer.get("error")
    assert answer["choice"] in criteria
    assert set(answer["probabilities"]) == set(criteria)
    print(result["usage"], answer["choice"])
```


```powershell
.\.venv\Scripts\python.exe -m scripts.run_phase_rs_ablation `
    --picker decision2 --decision2-candidates 255 --decision2-device cpu `
    --decision2-threads 4 `
    --our-deck-file data\decks\modern\modern_mono_red_burn.txt `
    --ai-deck-file data\decks\modern\modern_mono_red_burn.txt `
    --format Modern --ai-difficulty VeryEasy --games 1 --seed 7 `
    --max-retries 1 --max-game-seconds 600 --max-actions 1500 --autostart
```

The CLI prepares the model before creating a live game and reuses its loaded
runtime across replicate picker seeds; preparation time is retained separately
in model decision telemetry. The qualification harness supports
`--include-decision2 --only decision2`, the same pinned model arguments, and a
process deadline that also bounds preparation. Candidate
caps 24/64/128/255 are proposed matched-input ablations, not established
MTG improvements. Sets above the cap still use seeded heuristic tiers and
retain pass; token-budget overflow fails rather than cropping card state.
Choice confidence is normalized entropy in Decision 2.0's runtime and is
not calibrated winning probability. Precision differs from the Q8 Ollama
baseline; report this confound rather than attributing all differences to
new training.

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

This deployment comparison is not a clean decision-tuning ablation:
chat and Tev1 now share the card-aware projection, but chat sees all offered
actions and permits explicitly traced fallback, while Tev1 applies a shortlist
and surfaces inference errors.
Match context/candidates, account for chat fallback, add a size-matched
general Qwen3.5 comparator and control engine RNG/seat assignment before
making causal claims. The paper's decision-model benchmark subsection
distinguishes these experiments.

---

## Native graph and model learning

Use `scripts/run_native_learning.py`, not the retired
`train_pipeline.py --phase-rs-traces` path. That path previously constructed
empty states and one-dimensional dummy actions from game/deck summaries,
then silently reused old trajectory directories. It now refuses to train.
Event-only JSONL files are still useful for diagnostics, but cannot recover
observations that were never recorded.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_native_learning `
  --train-games 64 --validation-games 16 --test-games 8 `
  --training-seeds 0 1 2 3 4 --epochs 20 --graph-epochs 100 `
  --eval-games 20 --threads 4 --max-game-seconds 300 --max-actions 2000
```

This performs the entire implemented **offline native development learning**
cycle on CPU, without requiring or modifying a shared Neo4j service:

1. Capture actual numeric observations at each controlled-seat decision,
   complete original legal actions, chosen indices and a genuine terminal
   observation. Hidden opponent hands, library identities and face-down names
   are excluded; card text is signed token-hashed, not a pretrained embedding.
   Public hand/library counts survive that filtering and are independently
   checked; removing identities must not replace known counts with zeros.
   Pending-decision context is encoded separately without hidden card names.
   Producers and deployment share a non-conceding candidate policy.
   Fixed tokenizer capacities are an explicit compression limitation.
2. Preassign independent whole-game train/validation splits to burn and green
   stompy; reserve blue-white control as a held-out deck family. Shared staple
   cards remain possible; this is not a zero-overlap split. Native AI has
   privileged internal state and engine RNG is not seeded. The opponent uses
   the same burn deck throughout (`--opponent-deck`), rather than changing
   between family-specific mirrors.
3. Build a frozen graph only from completed **training** observations:
   `CO_VISIBLE` edges, exposed-game denominators, terminal win/loss/draw counts
   and per-game producer/deck/source provenance. Co-visibility is an association,
   never a proven combo or curated rules fact. No normative ontology is mutated.
4. Train GraphSAGE on that observation graph, then encoder/JEPA, recurrent
   latent/reward/terminal prediction and a controller imitating producer
   decisions. JEPA uses a stop-gradient EMA target encoder (momentum 0.99),
   a 0.05 batch latent-standard-deviation floor (weight 10), and KL weight
   0.001; held-out latent variance is retained as a collapse diagnostic.
   Targets run from one controlled decision to the next, including
   intervening opponent activity; these are not engine microstep dynamics.
   No dream-policy or reward-maximizing self-play promotion is claimed.
5. Train five independent initializations on the same frozen data. Save
   complete initial/trained bundles containing model configuration, all
   parameter states, frozen graph embeddings and component digests.
   Verify each requested component changes, and load every parameter strictly.
6. Report validation/test predictive diagnostics and a **2×2** deployment
   intervention: initial/trained weights × no/induced graph context. Twenty
   fresh control-deck episodes per cell run for the first prespecified seed;
   `--evaluate-all-seeds` runs playing evaluations for every trained seed.
   This is not the planned 2×3 no/curated/induced study: curated relational
   retrieval remains unwired. No graph-free retraining is implied by a
   frozen-model graph-context ablation.

Each run owns a private native server/database and writes its manifest,
individual tensor datasets, exposure/evidence graph, graph/model checkpoints,
training histories, held-out diagnostics, evaluation traces and summary under
`runs/native_learning/<timestamp>/`. No graph/weight cache is overwritten.
The manifest freezes source/decks, the actual server binary and native
card/AI input files; `progress.json` records the current stage.
Collection requires >=95% terminal completion in each split; operational
failures are retained and stop the training gate rather than becoming draws.
Evaluation retains incomplete episodes and traceable controller/dynamics
activation, retrieved graph cards, graph identity and a local graph-score
ablation delta. Parameter updates or nonzero score influence alone do not
prove better decisions.

The development policy samples from its learned categorical distribution
using an owned seeded generator, not a heuristic/random fallback. This matches
the imitation objective and avoids turning a moderate cancellation probability
into an endless greedy cast/cancel cycle. `--selection argmax` (or the separate
ablation CLI's `--native-selection argmax`) retains that explicit comparison.
Report non-forced imitation metrics separately: the headline all-decision
accuracy is dominated by mandatory passes and is not a playing-strength measure.

Qualification `20261005_193213_820367` completed all eight collection games,
induced 241 co-visible edges with 407 provenance records and updated all four
components, with unchanged frozen inputs. Its greedy evaluations completed
both initial-model games but timed out in both trained-model games, including
a recorded cast/cancel loop. These failures remain part of the evidence;
the separate seeded-sampling deployment qualification
`20261005_194343_832447` completes all four episodes (four losses), with
identical original checkpoint hashes and no heuristic fallback.
The initial full attempt `20261005_194651_374203` was stopped before optimization
after a tensor audit exposed zeroed public library/opponent-hand counts; its
27 terminal datasets and stopped summary remain retained. Feature v3 restores
public counts while continuing to exclude hidden identities, rejects malformed
counts and passes the live tensor check.
Fresh full campaign `runs/native_learning/20261005_200225_435675` is running the
64/16/8-game collection, five training seeds and 80 first-seed intervention
episodes; results are pending, not inferred from the small qualification.

The `W0` bundle has untrained model weights but shares the training-derived
frozen graph with `W1` in graph-enabled conditions. Diagnostic latent MSE is
per-model, not comparable across changing latent coordinate systems. Validation
does not select hyperparameters in this development run; final test predictions
are computed once per seed. Playing outcomes still need game/producer-run
uncertainty, controls and qualification before scientific improvement claims.

Deploy a strict bundle separately:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_phase_rs_ablation `
  --picker native_learned --native-checkpoint <run>\seed_0\trained.pt `
  --native-graph induced --games 20 --ai-difficulty VeryEasy `
  --our-deck-file data\decks\benchmark\modern_azorius_control.txt `
  --ai-deck-file data\decks\benchmark\modern_azorius_control.txt `
  --format Modern --autostart
```

Full confirmatory RQ1–RQ5 work still requires native multi-seat/information
controls, shuffled/curated graph arms, larger training batches and replication,
calibrated dynamics/planning, validated strategic evidence and powered
held-out comparisons. Do not turn a completed development pipeline into a
claim that all those gates passed.

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
