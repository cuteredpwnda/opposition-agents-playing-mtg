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

The checkout now expects Full protocol v374; the campaigns documented below
used the earlier v106 runtime. An official signed Windows build of the new
pin completed one game each in Standard, Pioneer, Modern, Legacy and
four-player Commander. These checks do not qualify the earlier results on the
new runtime. Build or verify the pinned `external/phase-rs` server and follow
the integration guide's
[engine update checklist](PHASE_RS_INTEGRATION.md#engine-update-checklist)
before running a new campaign. Protocol mismatch fails at handshake.

### Protocol-v374 format smokes

Each terminal check used one heuristic-controlled seat, VeryEasy native AI
and picker seed 7. Standard and Pioneer used the refreshed mono-red Standard
deck in a mirror, Modern used burn versus the built-in Blue Control starter,
Legacy used a burn mirror, and Commander used Krenko versus Atraxa, Meren and
Urza. One additional Commander attempt was deliberately capped at six turns.

| Format | Outcome | Final trace turn | Controlled actions |
|---|---|---:|---:|
| Standard | Seat 0 win | 22 | 322 |
| Pioneer | Seat 1 win | 13 | 143 |
| Modern | Seat 1 win | 15 | 153 |
| Legacy | Seat 1 win | 7 | 81 |
| Commander | Seat 3 win | 40 | 483 |

The [manifest and JSONL traces](../runs/phase_rs_v374_smoke_20261125/manifest.json)
retain outcomes, input hashes and engine identity. The manifest also records
the runner's observed turn-transition counter, which can differ from the final
trace turn. The capped Commander episode is incomplete, not a draw. Single
games are execution checks, not reliability estimates or strength comparisons;
the other registry formats remain unqualified and native shuffle/AI randomness
is not seeded.

### Protocol-v374 Modern deck benchmark (Nov 25, 2026)

Each controlled deck played Modern against the native AI on VeryEasy, using
`data/decks/benchmark/modern_azorius_control.txt`. The picker seed was 1000,
with five games per arm, a 300-second game limit, a 2,000-action cap and one
retry. Win counts are wins out of five scheduled games unless stated. KL-family
rows are the final `analytic_pending_cast_v4` reruns in
`runs/v374_deck_benchmark_20261125_v4/`. All 60 games were rules-terminal,
with no action caps.

| Controlled deck | Random | Heuristic | KL | EFE-infogain | EFE-ambiguity |
|---|---:|---:|---:|---:|---:|
| Mono-Red Burn | 1/5 | 1/5 | 3/5 | 3/5 | 2/5 |
| Death's Shadow | 0/5 | 1/5 | 0/5 | 2/5 | 3/5 |
| Hardened Scales | 4/5 | 3/5 | 5/5 | 5/5 | 4/5 |
| Mono-Green Tron | 5/5 | 5/5 | 3/5 | 2/5 | 1/5 |
| **Total** | 10/20 | 10/20 | 11/20 | 12/20 | 10/20 |

The Hardened Scales baseline cells are the original five-game runs. An
identical rerun, not included in the table, won 4/5 for random and 4/5 for
heuristic. The first Death's Shadow attempt was rejected before play because its
list contained Bayou and Underground Sea. The list is now Modern-legal; the
table reports only the legal rerun.

Retained failures are part of the evidence. The original KL-family cells used
`v2` of the transition model. Hardened Scales had 5/15 games reach the action
cap through a `PayCost` (`SelectCards`) → `CancelCast` loop. After that fix,
`v3` still reached the cap in 2/15 Death's Shadow games and 2/15 Tron games
through convoke and mana-payment → `CancelCast` loops. A concurrent `v3`
Hardened Scales rerun had 1/15 capped. The `v2` Burn cells had won 5/5 for KL
and EFE-infogain; with five games per arm and unseeded native RNG, the `v4`
3/5 results are not distinguishable from those earlier results. These loops
came from a model error, not missing loop suppression. While an announcement
was pending, confirming payment, mana abilities, convoke taps and cost-card
selection were scored as independent actions that also gave the opponent a
clock tick. That made `CancelCast` look strictly cheaper. Retained original,
v3-rerun and v4 directories:
`runs/v374_deck_benchmark_20261125/`,
`runs/v374_deck_benchmark_20261125_rerun/` and
`runs/v374_deck_benchmark_20261125_v4/`.

Interpretation: this is one weak opponent with five games per cell, so every
95% interval overlaps. Native shuffle and AI RNG were not seeded, and KL
horizon planning is compared with one-step EFE scoring. The benchmark verifies
that every arm completes games across four archetypes. It does not rank
objectives or show that active inference beats the baselines. Tron is the
clearest warning sign: each baseline won 5/5, while the KL family won 6/15.

### Diagnosing the native-AI gap

The Standard round-robin and ladder below compare random, a hand-written
heuristic and a hand-built KL-control planner with phase-rs's native AI;
they do **not** compare the LLM or decision-model agents. In the 18-game
ladder cells, KL won 8, 6 and 3 games against Easy, Medium and Hard,
respectively. The opponent won most games in those particular conditions,
but their confidence intervals are wide; they do not establish an overall
ranking across decks, seats or model families.

The KL picker is not a trained policy: its analytic transition model uses
generic cast gains for most card effects and targets, and its rollouts reuse
the root action candidates rather than generating rules-exact future legal
actions. The native AI instead has card-aware policies and, from Medium
onward, searches simulated engine states. Its shipped search presets use
perfect-information states, so they can exploit hidden cards; this does not
apply to the VeryEasy and Easy search-disabled presets. These are plausible
mechanistic disadvantages, not measured causal contributions to the gap.
Seat assignment is not rotated and the native RNG is not seeded.

The separately trained native world-model controller is an imitation learner
trained on collected policy actions, not a policy optimized directly for
winning games. The earlier evaluation won 2/100 games across mixed initial
and trained conditions, but that floor and the small development training set
do not establish why it lost or whether more training would suffice. Chat
and decision-model execution/strength qualifications have different
limitations and cannot be inferred from these scripted-arm results. Isolating
training effects requires matched initial/trained checkpoints, independent
training seeds, held-out decks, and controls for information, compute and
seat position.

### Modern and Pioneer fixed-opponent pilots (Nov 26)

Each format used nine controlled meta decks against one fixed native VeryEasy
opponent deck, excluding the mirror: Modern Boros Energy or Pioneer Nykthos
Ramp. Random, heuristic and KL each played two games per controlled deck
(54 scheduled per format). The opponent and seat were not rotated, and the
picker seed does not seed native game randomness. These are exploratory
pilots, not cross-deck strength rankings or tests of trained models.

| Format | Random wins | Heuristic wins | KL wins | Incomplete |
|---|---:|---:|---:|---:|
| Modern | 1/18 | 7/18 | 2/18 | 0/54 |
| Pioneer | 1/18 | 5/17 | 6/18 | 1/54 (heuristic action rejected) |

The excluded Pioneer game is not a loss or draw. The ridge-adjusted analysis
is retained alongside the raw results, but only nine matchup clusters and two
games per arm/matchup make uncertainty substantial; no opponent-deck effect
can be estimated with one opposing deck. The different KL outcomes in these
two pilots are not enough to identify a cause or generalize across formats.
See `runs/campaigns/modern_vs_boros_v374_20261126/` and
`runs/campaigns/pioneer_vs_nykthos_v374_20261126/`.

As a trace-level diagnostic, when both `CastSpell` and `PassPriority` were
legal, neither scripted picker passed: heuristic 0/226 and KL 0/125 such
decisions in Modern; heuristic 0/202 and KL 0/175 in Pioneer. They sometimes
chose a land rather than a spell. This exposes a possible timing weakness
(for example, holding up interaction), not proof that any particular pass
would have been better or that this explains the win-rate gap. The per-game
decision traces used for this diagnostic remain local, under the ignored
campaign `cells/` directories; they are not included in the result summaries.

### Five-deck Modern and Pioneer round-robins (Nov 26)

Each format used the first five meta decks in all 20 ordered non-mirror
matchups against VeryEasy. Random, heuristic and KL each played two games per
matchup: 120 scheduled games per format. The controlled seat remained fixed
and native shuffle/AI randomness was not seeded. These scripted baselines
do not test trained models, LLMs or decision models.

| Format | Random wins | Heuristic wins | KL wins | Incomplete |
|---|---:|---:|---:|---:|
| Modern | 4/40 | 14/40 | 14/40 | 0/120 |
| Pioneer | 8/37 | 14/39 | 20/37 | 7/120 |

Pioneer exclusions comprise four `action_rejected` games and three KL
`action_cap` games. All three capped KL games used Azorius Control; their
traces show repeated `CastSpell` followed by a prompt offering only
`CancelCast`. The reason the cast cannot be completed is not established.
The detailed traces are local under the ignored campaign `cells/` directory.
The runner printed `ok` for these cells because a summary existed, even
though the subprocess exited nonzero; the raw incomplete outcomes are
retained and excluded from the scored denominators. In particular, Pioneer
KL completed only 37/40 games, below the 95% reliability gate.

The deck-adjusted KL-minus-random estimates are +0.170 [0.032, 0.308] for
Modern and +0.211 [0.083, 0.337] for Pioneer, using the existing ridge model
and 400 matchup-cluster bootstrap replicates. These exploratory intervals
do not account for the informative incomplete games, lack of seat rotation,
or native unseeded randomness. Two games per arm/matchup are insufficient to
settle detailed agent rankings. The separate campaign summaries and results
are in `runs/campaigns/modern_5deck_rr_v374_20261126/` and
`runs/campaigns/pioneer_5deck_rr_v374_20261126/`.

### Protocol-v374 native shared-graph learning campaign

`scripts/run_native_learning.py` used the signed v374 server and a private data
directory. It collected 32 training, 8 validation and 8 test games, all 48
rules-terminal, with Mono-Red Burn and Green Stompy training decks. It trained
seeds 0, 1 and 2 for 20 epochs and 100 graph epochs. The induced graph had 58
edges with 1,423 evidence records. The curated Modern Spellbook graph had
5,868 edges from 3,841 combos, and the union had 5,926 edges. Every component
updated for every seed, and the frozen inputs did not change.

Seed 0 was evaluated across 20 cells with five games each: initial or trained
weights × `none`, curated, induced, curated-plus-induced or identity-permuted
graph × Azorius or Hardened Scales test deck, against Mono-Red Burn. All 100
games were rules-terminal, but the learned policy won only 2/100. One win came
from `initial_identity_shuffled_hardened_scales` and one from
`trained_curated_hardened_scales`. Because of this floor effect, the graph
conditions cannot be compared, so no graph or learning benefit is shown.
Run: `runs/native_learning_v374/20261125_150818_767626/`. Seeds 1 and 2 remain
frozen and unevaluated. A useful graph study first needs a policy above the
floor, such as a stronger base policy, a weaker opponent or a graph consumer
inside the KL prior.

### Agent audit and Standard deck round-robin (v375b, Nov 26, 2026)

**Audit.** Before the wider benchmark, we reviewed traces of both scripted
pickers. Four shared defects made earlier comparisons uninformative:
- Upkeep activation spam.
- Random any-colour mana choices that made casts unpayable.
- Heuristic `CancelCast` during targeting, plus random mulligans: 13
  mulligans against 9 keeps in 9 games.
- The KL model scored every attack declaration identically, so the
  argmin tie-break always declared no attackers. It declared none in 128 of
  128 attack steps.

The fixes are described in [ACTIVE_INFERENCE.md §3.4](ACTIVE_INFERENCE.md):
the policy tag is `action_hygiene_v2` and the transition model is
`analytic_combat_v5`. The `heuristic_legacy` and `kl_legacy` arms reproduce
the pre-fix agents. The 40 cells of the first `standard_rr_v375` attempt
used pre-fix agents and are superseded.

**Design.**
- Ten MTGGoldfish Standard meta decks (`data/decks/meta/standard/`) in a
  full round-robin without mirrors: 90 ordered matchups × 6 arms × 3 games,
  1,620 games in total.
- The opponent is the native AI on VeryEasy, and every arm plays the same
  matchups.
- `kl_graph_learn` starts from an empty shared graph and writes to it after
  every game, using Thompson-sampling exploration.
- 1,524 games were scored. Of the 96 excluded games, 59 ended with an
  engine `action_rejected` for an action it had listed as legal, spread
  over every arm including random; 37 hit the action cap.
- Analysis uses `scripts/analyze_phase_rs_campaign.py`: a ridge fractional
  logistic model with terms for arm, our deck and opponent deck, and
  matchup-cluster bootstrap CIs.

Run directory: `runs/campaigns/standard_rr_v375b/`.

| Arm | Games | Raw win rate | Deck-adjusted [95% CI] | Δ vs random [95% CI] |
|---|---:|---:|---|---|
| random | 265 | 0.174 | 0.179 [0.137, 0.221] | — |
| heuristic_legacy | 247 | 0.211 | 0.218 [0.162, 0.275] | +0.038 [−0.020, +0.093] |
| heuristic | 257 | 0.296 | 0.284 [0.226, 0.350] | +0.105 [+0.044, +0.174] |
| kl_legacy | 239 | 0.063 | 0.072 [0.045, 0.105] | −0.107 [−0.151, −0.061] |
| kl | 257 | 0.377 | 0.362 [0.306, 0.421] | +0.183 [+0.119, +0.251] |
| kl_graph_learn | 259 | 0.328 | 0.318 [0.259, 0.381] | +0.139 [+0.076, +0.198] |

Deck strength is large. Our-deck logit effects span 2.05 (Boros Dwarves
+0.36, 4c Control −1.69); opponent-deck effects span 2.89 (Azorius Control
+2.10, Azorius Momo −0.79). This is why raw pooled rates are not reported
as the result.

**Findings.**
- After the audit, the KL planner is the strongest scripted arm against
  VeryEasy (+0.18 over random). It also finishes games faster: 17.4 turns
  on average against 19.9 for random.
- The audit fixes are the largest effect in the study: +0.29 for KL and
  +0.07 for the heuristic. Before the fixes, KL was worse than random.
- `kl_graph_learn` shows no within-run learning. Its score was 0.333 in the
  first half and 0.323 in the second (permutation p = 0.90). It is no better
  than `kl` without the graph.

### Frozen learned-graph evaluation (v375b)

This test reuses the graph learned in the round-robin: 259 games and
443 KB of evidence. A new seed (2027) and five decks (01, 03, 05, 07 and
09) give 20 matchups × 4 arms × 3 games, all 240 games scored. Frozen arms
read the graph and never write to it. The heuristic uses the graph only to
break ties among actions of equal priority. KL uses it to tilt the passive
prior.

Run directory: `runs/campaigns/standard_frozen_v375b/`.

| Arm | Wins | Deck-adjusted [95% CI] | Δ vs heuristic [95% CI] |
|---|---:|---|---|
| heuristic | 16/60 | 0.275 [0.172, 0.413] | — |
| heuristic_graph_frozen | 26/60 | 0.423 [0.311, 0.542] | +0.132 [−0.011, +0.239] |
| kl | 23/60 | 0.376 [0.293, 0.463] | +0.085 [−0.032, +0.190] |
| kl_graph_frozen | 23/60 | 0.376 [0.265, 0.504] | +0.085 [−0.060, +0.220] |

Interpretation:
- **Cross-agent transfer:** graph evidence collected by the KL agent made
  the heuristic's tie-breaks better. This is suggestive but not conclusive:
  the interval crosses zero.
- **KL with the graph:** no effect. Its own analytic forecast already
  dominates the tilted prior, and its earlier graph-guided choices are
  what produced the graph in the first place.
- **Why there is little learning:** the graph credits first-visit
  `(context, card/action)` keys with Beta win evidence. With about 260
  games spread over 10 decks and 90 matchups, most keys have only a few
  observations, and deck strength confounds the credit.
- **Next tests:** a learning claim needs more games per deck, credit that
  conditions on the matchup, and replicate graphs from independent seeds.

### Native-AI difficulty ladder (v375b)

This campaign used three Standard decks (01 Mono-Green Landfall, 03 Dimir
Midrange and 05 Lifegain), all six ordered matchups, the arms random,
heuristic and KL, and native difficulties Easy, Medium and Hard, with three
games per cell and seed 2028. All 162 games were scored. Medium and Hard
use beam search with rollouts and perfect information (see "What the
built-in opponent does").

Run directory: `runs/campaigns/standard_ladder_v375b/`.

| Opponent | random | heuristic | KL |
|---|---:|---:|---:|
| Easy | 2/18 | 7/18 | 8/18 |
| Medium | 0/18 | 4/18 | 6/18 |
| Hard | 0/18 | 3/18 | 3/18 |

Adjusted for deck and difficulty:
- heuristic: +0.150 [+0.063, +0.239] over random;
- KL: +0.200 [+0.088, +0.312] over random.

Random play stops winning once native search is enabled. Both scripted
agents still win some games against search, but with 18 games per cell the
heuristic-versus-KL difference is not resolved at any level. VeryHard and
cEDH were not run.

### Running an objective ablation

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

- `config.json`: arguments except the vLLM API key, checkout revisions,
  protocol, deck hash, seed scope and the objective/planning confound.
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

The separate `scripts/phase_rs_rollout_sweep.py` also counts only traced
rules-terminal games as wins, losses or draws. Each cell reports attempted
and completed games, incomplete games and attempts, completion rate, and
wins per completed game (`null` if none complete). A retry is a new native
game, not a matched replay: rows retain every attempt reason, prior attempt
traces are saved separately when available, and elapsed time includes retries.
Incomplete scheduled games make the command fail. Requested KG enrichment
is skipped and the command fails if any attempt was incomplete. A successful
final retry must not be interpreted as first-attempt reliability.
Older sweep summaries may have labelled interrupted games as draws and
discarded retry details; reanalyse retained traces where possible or rerun
before using those summaries.

### Native-learning graph-control qualification

The v4 native-learning path records controlled-seat visible-zone evidence and
supports five graph conditions (`none`, curated, induced, curated-plus-induced,
and card-identity-permuted combined graph) across two distinct Modern
evaluation strata (Azorius Control and Hardened Scales). The permutation
preserves the learned graph's embedding rows, topology and degree sequence;
it is not an edge-rewired null graph. Curated relations come from the
status-`OK`, Modern-legal Spellbook records; training negatives sampled from
missing curated pairs are not proof that those pairs are non-combos.

Provision the pinned server's data first and set `PHASE_DATA_DIR` to that
directory (relative paths are resolved from the phase-rs checkout). Native
learning records the required card-data hash and the presence and hashes of
the optional draft, combo and Commander-deck files, then starts the server
bound to localhost with data downloading disabled. Evaluation reuses and
checks the recorded directory; older campaigns without this identity cannot
be resumed safely.
Use a private data directory when qualifying a new server version.

Run a bounded end-to-end qualification in a **new** output directory before
any larger campaign:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_native_learning `
    --train-games 4 --validation-games 2 --test-games 2 `
    --training-seeds 7 --epochs 2 --graph-epochs 10 `
    --eval-games 1 --threads 2 --max-game-seconds 90 `
    --max-actions 2000 --seed 7 `
    --output-dir runs\native_learning_v4_qualification
```

This smoke schedules eight collection games plus 20 one-attempt evaluation
games for the single training seed. It verifies schema, graph construction,
strict checkpoint use, intervention telemetry and deck-stratum wiring only.
It is not powered evidence, a paired engine replay, a reliability qualification,
or a playing-strength estimate. Do not continue to a large run if collection,
graph-support, timeout, or frozen-input gates fail.

The v4 qualification at
`runs/native_learning_v4_qualification/20261007_093832_702012` completed all
eight collection games and updated all four model components. Its induced
graph had 13 repeated battlefield pairs over 24 nodes and 220 provenance
records; the curated Modern graph had 3,841 source hyperedges. All 20
single-game weight × graph × deck-stratum cells ran: 19 terminal losses and
one 90-second timeout in `seed7_initial_none_azorius`. The overall 19/20
completion rate does not make every cell reliable. JEPA validation MSE
worsened from 0.0107 to 0.0279, and no game-strength or graph-benefit claim is
supported.

Frozen-checkpoint follow-up uses the resumable evaluator; it verifies campaign
input hashes and preserves every attempt rather than overwriting a timeout:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_native_learning `
    --campaign runs\native_learning_v4_qualification\20261007_093832_702012 `
    --training-seeds 7 --arms seed7_initial_none_azorius `
    --max-game-seconds 300 --max-actions 2000 `
    --output-dir runs\native_learning_reliability
```

That separate one-game budget-sensitivity run completed as a loss with frozen
inputs unchanged. It is a new episode under an unseeded native engine, not a
replay or a replacement of the original timeout. Use `--resume-dir <existing
evaluation-directory>` to continue an interrupted schedule; schedule and
input hashes must match. `--retry-incomplete` appends a separately counted
attempt and never erases the prior failure.

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

`phase-ai` is upstream's Rust tactical evaluator and search policy. It is not
our uniform wire-random picker or action-type wire-heuristic picker. All
evaluation weights are fixed, and the opponent does not learn from our
campaigns.

Each decision scores the legal candidates with a hand-built tactical evaluator:
- fitted, turn-dependent evaluation weights (`EvalWeightSet::learned()`);
- policy penalties and bonuses from about 120 tactical policy modules in
  `crates/phase-ai/src/policies/` (mana efficiency, counter-magic tap-out,
  wipe overextension, and others);
- deck and archetype features.

It then samples from a softmax at the preset temperature. Dedicated decision
paths, such as mulligans and some targets, choose deterministically. From
Medium up, a beam search with rollouts re-ranks the candidates. The presets in
`crates/phase-ai/src/config.rs::create_config` at the pinned revision are:

| Preset | Softmax T | Search | Depth / nodes | Rollouts (depth × samples) | Opponent model | Threat awareness | Lookahead |
|---|---:|---|---|---|---|---|---|
| VeryEasy | 4.0 | off | — | — | — | none | — |
| Easy | 2.0 | off | — | — | — | none | play |
| Medium | 1.0 | beam + rollout | 2 / 24 | 1 × 1 | deterministic best reply | archetype only | play |
| Hard | 0.5 | beam + rollout | 3 / 48 | 2 × 1 | threat-weighted reply | full (hypergeometric) | play |
| VeryHard | 0.3 | beam + rollout | 3 / 64 | 2 × 2 | threat-weighted reply | full | play |
| cEDH | 0.2 | beam + rollout | 3 / 96 | 2 × 2 | threat-weighted reply | full | play + combat |

All search presets use branching cap 5 and a 1.5-second interactive budget.
Depth and nodes scale down in three- and four-player pods, except for cEDH.

**Information asymmetry.** Every shipped search preset sets
`determinization_samples = 0`, which means *perfect-information* search.
Medium and above therefore read our real hidden hand and library inside the
search. Upstream made this a product decision on 2026-07-18 because
determinised sampling broke the monotonicity of the difficulty ladder. VeryEasy
and Easy do not search, but their evaluator still receives the internal
`GameState`.

Our policies receive perspective-filtered observations. Native-AI results are
therefore anchor-opponent deployment comparisons, not matched-information
rankings. A difficulty ladder measures robustness against an increasingly
strong, and from Medium up omniscient, opponent. The paper's baseline and
ablation tables state this distinction explicitly.

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
and rollout sweep commands. GPU inference is optional; the local environment
used for the later gameplay qualification has CUDA-enabled Torch. The earlier
Linux reference timings below used CPU Torch and are not GPU timings.

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
not certify decision quality. The later two-candidate Linux-CUDA MTG episode
establishes only minimal terminal feasibility; larger family variants and
candidate-order/coverage ablations remain unmeasured.
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

API errors and invalid answers are distinct incomplete reasons. The phase-rs
benchmark CLI defaults chat to strict failure, a 120-second request budget
and Llama rather than Gemma; legacy random fallback is opt-in only through
`--chat-failure-policy random`. Strict Ollama and vLLM requests use a JSON
schema enumerating eligible original action indices, and validate the returned
object locally. Concede is excluded when a non-concession action exists; a
missing model or malformed reply cannot produce a random action. A sole legal
action is forced without model inference. Traces distinguish request attempts,
accepted model choices, server diagnostics and fallback decisions. Latency
aggregates include attempted decisions in the retained game trace. Separate
forced decisions and cold starts when interpreting performance.

This deployment comparison is not a clean decision-tuning ablation:
chat and Tev1 share the card-aware projection, but strict chat sees all offered
non-concession actions while Tev1 applies a shortlist.
Match context/candidates, account for chat fallback, add a size-matched
general Qwen3.5 comparator and control engine RNG/seat assignment before
making causal claims. The paper's decision-model benchmark subsection
distinguishes these experiments.

### Liquid d1 and local vLLM serving

`--picker liquid_d1` supports pinned `LiquidAI/d1-omni-600M` and
`LiquidAI/d1-3B` releases through their native Transformers `system_one`
classification head. They declare 16,384- and 32,768-token contexts and
return choice probabilities with zero generated output tokens. Stock vLLM's
OpenAI-compatible generation endpoint does not expose that custom head, so
d1 is not routed through chat completions. The optional `liquid_d1` dependency
group declares the required Transformers/Torch/Torchvision runtime.

The pinned d1-omni-600M revision
`02b55d7076f15129e59ab3f94783f32c4b088674` passes a real local CPU smoke:
197 input tokens, zero output tokens and a complete two-action distribution.
After the weights were cached, first load plus decision took 51.1 seconds and
the next warm decision 0.52 seconds. On the RTX 3060 with 128 candidates, the
same pinned 600M model completed a 29-turn Modern game with 256 decisions.
The pinned d1-3B revision
`051bcc464b01b9f92942b364d9586b0ef5912432` also loaded on CUDA and returned
valid decisions. Its 128-candidate attempt reached 767 decisions before the
600-second cap; a 24-candidate attempt reached 196 decisions before CUDA
out-of-memory. A two-candidate run completed a 25-turn game with 269
decisions. Thus d1-3B has passed the valid-move and minimal terminal gates,
but its larger candidate configurations are not yet viable on a 12 GiB GPU.
These are execution/throughput results, not quality comparisons.

The verified `tev1-mtg-8k:0.8b` alias completed a 26-turn game with 220
decisions. The original `tev1:0.8b` alias failed at 2,146 prompt tokens
against its 2,050-token limit, confirming that the two aliases must remain
separate experimental conditions.

The prior strict Ollama chat qualification was negative: Llama 3.2 1B
returned an invalid action index after 71 decisions, and Qwen2.5-Coder 1.5B
returned one after three decisions. Both runs were recorded as incomplete.
Those runs predate JSON-schema-constrained requests and do not qualify the
repaired protocol; rerun each model before including it in a comparison.

`--picker vllm` is a separate strict local OpenAI-compatible path for
generative Llama/Qwen controls. It requests JSON-schema output and never
falls back to random. The observed vLLM 0.31.0 Docker failure occurred before
inference: the process requested 92% of 12 GiB (11.04 GiB) while only
10.98 GiB was free. Start with a lower `--gpu-memory-utilization` such as
`0.85`; for MTG prompts, also lower the Qwen default 40,960-token allocation
with `--max-model-len 8192` unless a larger frozen context is required.
Those corrected settings successfully served Qwen3-0.6B on the RTX 3060.
A Burn-versus-Tron attempt produced 240 accepted model choices and 368 valid
actions before its 600-second cap. A Burn mirror then reached a rules-terminal
loss at turn 14 with 10 model choices, 140 valid actions and no picker error.
The first result exposes throughput/deck sensitivity; the second establishes
minimal terminal feasibility. Neither is a strength estimate.

Pinned Decision 2.0 Eos 0.8B was also run unchanged in a Linux CUDA container
using the Windows phase-rs server over the local Docker host bridge. With a
two-candidate shortlist it made 64 native `system_one` choices, recorded 406
valid actions without picker errors and reached a rules-terminal loss at turn
45. This closes the MTG interface/terminal gate for that configuration while
leaving 24--255-candidate GPU throughput, other family sizes and direct
Windows loading unresolved.

The aggregate verification artifact
`runs/model_variant_qualification/verification_20261008.json` scans retained
decision events, checks that `chosen_index` is in range and that its indexed
legal type equals `chosen_type`, and records each trace SHA-256. It does not
replay engine rules, validate every action payload or establish tactical quality.

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

### Completed full development campaign

`runs/native_learning/20261005_200225_435675` completed **88/88 collection
games** (64 train, 16 validation, 8 held-out test), all rules-terminal. The
training-only graph contains **290 co-visibility edges and 5,977 per-game
evidence records**. All four components updated for each of five independent
training seeds. Final source, deck, native executable and native data hashes
match the original manifest.

Five-seed held-out diagnostics (mean ± sample SD over seeds):

| Metric | Initial validation | Trained validation | Trained test |
|---|---:|---:|---:|
| JEPA MSE | 0.0130 ± 0.0037 | 0.0062 ± 0.0021 | 0.0040 ± 0.0010 |
| Non-forced imitation accuracy | 0.346 ± 0.031 | 0.418 ± 0.008 | 0.472 ± 0.010 |
| Non-forced imitation NLL | 1.106 ± 0.004 | 1.166 ± 0.014 | 0.986 ± 0.012 |

Test values are trained-model diagnostics; an initial-model test baseline was
not calculated. Accuracy increases while validation NLL worsens, so predictive
evidence is mixed. All-decision accuracy is omitted because it is inflated by
mandatory actions. Lower JEPA error and greater action agreement do not measure
policy strength.

The first training seed's frozen W0/W1 × no/induced graph-policy cells each
attempt 20 fresh control-deck games against the fixed VeryEasy burn anchor:

| Frozen policy / graph context | Rules-terminal | Wins among terminal games | Incomplete |
|---|---:|---:|---:|
| W0, no induced graph | 17/20 (85%) | 0/17 | 3 timeouts |
| W0, induced graph | 20/20 (100%) | 1/20 | 0 |
| W1, no induced graph | 20/20 (100%) | 0/20 | 0 |
| W1, induced graph | 18/20 (90%) | 0/18 | 2 timeouts |

Completed-game win-rate Wilson 95% intervals are 0--18.4%, 0.9--23.6%,
0--16.1%, and 0--17.6%, respectively. Timeouts are retained as incomplete,
never recoded as losses or draws. Evaluation completion is below 95% in two
cells; the table is a feasibility/development result, not a powered causal
comparison. Engine RNG is uncontrolled, only one of five trained policy seeds
was evaluated in play, and the native anchor has privileged internal state.
The results provide **no reliable evidence that training or induced graph
context improves playing strength**.

The final induced-graph cell was interrupted after 15 attempts. The five
remaining games were appended as attempts 16--20 with the original checkpoint,
picker-seed schedule and unchanged frozen-input hashes. The complete aggregate
summary and per-game traces are preserved in the run directory.

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
The earlier host-Krenko, host-Atraxa and 16-MiB runs reached turns 34, 40 and
39, respectively, before transport or time limits; these are incomplete
attempts, not terminal games.

An un-capped Krenko-host/heuristic qualification against Atraxa, Meren and Urza
then completed 3/3 pods at turns 53, 41 and 47 (all losses). The first pod
required one retry after a stream timeout, so reliability is not yet
established. A separate random-host attempt stopped at turn 55 after a
120-second stream timeout; a repeat with a 600-second stream window also
timed out at turn 50, after two reconnect attempts. A Tev1 host pilot with an
eight-candidate cap and 120-second inference timeout made nine real model
decisions (maximum observed call 71.4 seconds), then timed out at turn 33 in
`DeclareBlockers`. All three traces report WebSocket keepalive-ping failure
(1011, no close frame received). This is evidence of transport loss during
multiplayer progression, not proof that the native AI is the root cause.
Retain all incomplete episodes in the record. Do not advance to the 160-pod
study until the progression failure is investigated and reliability and
per-cell completion qualify.

Modern 1v1 model qualification does not imply Commander qualification.
Commander model pilots must be separate, one controlled Python policy at a
time, against the same three native seats and deck schedule. Chat fallback,
decision-model input limits, loaded checkpoints and component activation must
be retained in their traces. Decision 2.0 now has a terminal Linux-CUDA
Modern episode, but remains unqualified for Commander; a Commander arm must
still complete under the same frozen multiplayer protocol. This interface
does not run four Python zoo agents or control
native engine randomness/seats.

Current Commander host-policy qualification:

| Host policy | Attempts | Terminal outcomes | Qualification status |
|---|---:|---:|---|
| Heuristic | 3 | 3 losses | Terminal play demonstrated; one attempt needed a transport-timeout retry; completion reliability not established |
| Random | 2 | 0 | Two stream timeouts (turns 55 and 50); not qualified |
| Tev1 | 1 | 0 | Nine model decisions; stream timeout at turn 33; not qualified |
| Chat LLM, direct/dream world model, fusion, active inference, KL/EFE, native-learned | 11 configuration screens | 1 host win (native-learned); 10 incomplete | One attempt per configuration only; no comparative or reliability qualification; see the detailed grid below |
| Decision 2.0 | 0 | 0 | Blocked before gameplay by local model loading |

The successful heuristic pods are all losses and do not support comparative
strength claims. The single native-learned host win is feasibility evidence
only. The random, Tev1 and ten incomplete grid episodes remain failures, not
draws or replacements. Further Commander model-family pilots are paused
pending diagnosis of progression and transport reliability; Modern 1v1
results cannot substitute for that gate.

### One-attempt Commander model-family screen

To identify which runnable families can finish a real pod, a sequential
one-attempt screen ran on Oct 7, 2026. Every cell used the same Krenko host
deck, Atraxa/Meren/Urza native opponents, VeryEasy native-AI difficulty,
picker seed 7, one attempt, a 900-second cooperative game cap and a
180-second WebSocket stream timeout. Chat models used Ollama; direct/dream
world-model and fusion arms used the archived `checkpoints/jepa/jepa_final.pt`
checkpoint, not the native-trained v4 checkpoint. The last cell used
`runs/native_learning_v4_qualification/20261007_093832_702012/seed_7/trained.pt`
with induced graph and seeded sampling. These are feasibility outcomes, not a
matched strength comparison; Python picker seed does not seed native shuffle
or native-AI randomness.

| Host policy / configuration | Outcome | Turns / decisions | Activation and failure evidence |
|---|---|---:|---|
| Chat: Llama 3.2 1B | Incomplete: stream timeout | 33 / 339 | 11 model invocations; 6 trace decisions include fallback/error evidence |
| Chat: Qwen 2.5 Coder 1.5B | Incomplete: game timeout | 29 / 317 | 22 model invocations; 14 trace decisions include fallback/error evidence |
| Chat: Gemma4 E2B | Incomplete: stream timeout | 49 / 580 | 0 model invocations; 61 trace decisions include fallback/error evidence; not an LLM gameplay result |
| World model, direct (legacy checkpoint) | Incomplete: stream timeout | 40 / 513 | No trained/native checkpoint qualification |
| World model, dream search (legacy checkpoint) | Incomplete: stream timeout | 25 / 238 | No trained/native checkpoint qualification |
| Fusion (legacy checkpoint + Gemma4) | Incomplete: game timeout | 33 / 1,196 | 0 model invocations; no fusion-component activation demonstrated |
| Active inference | Incomplete: native AI driver fault | 37 / 374 | Native AI batch reached its 200-action safety cap |
| KL control | Incomplete: game timeout | 37 / 4,055 | Picker completed decisions; no terminal outcome |
| EFE information-gain proxy | Incomplete: game timeout | 38 / 1,127 | Picker completed decisions; no terminal outcome |
| EFE ambiguity proxy | Incomplete: game timeout | 45 / 2,119 | Picker completed decisions; no terminal outcome |
| Native-learned v4, trained + induced graph | **Terminal host win** | 39 / 587 | One completed pod; single-game feasibility evidence only |

All ten incomplete cells remain recorded with their original reasons; there
were no draws and no retry/replacement. The native-learned result establishes
that this checkpoint/policy combination can complete one pod, not that
learning, the graph, or the policy caused a strength improvement. It does not
qualify completion reliability or remove the need for replicated controls.
Gemma and fusion are not counted as invoked-model arms. The overall runtime
pattern still blocks a multi-game family matrix and the prespecified 160-pod
Commander study; first diagnose the native driver/transport stalls and run
replicated completion calibration on cells that can finish.

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
