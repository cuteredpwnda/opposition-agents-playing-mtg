# Implementation Plan — Single Source of Truth

> **opposition-agents-playing-mtg**
> Last updated: 2026-04-29

This document is the **single source of truth** for what has been implemented,
what is in progress, and what remains. It supersedes the phase descriptions in
`PLAN.md`, `ARCHITECTURE.md`, and the presentation slides for tracking purposes
— those documents retain their value as design rationale and research context.

## Status Snapshot — April 2026

Full two-player games of Magic now run end-to-end through both the synchronous
`GameSimulator` and the async `GameRunner`. Recent engine hardening:

- **End-to-end gameplay**: lands, mana, casting, stack resolution, attacks,
  blocks, damage, life loss, elimination, and game termination all wired
  through the simulator main loop.
- **Mulligans (London)**: opening hands draw 7, optionally mulligan up to a
  configured cap, then bottom cards equal to mulligans taken. Configurable via
  `setup_game(mulligan_enabled=..., max_mulligans=...)` and `GameConfig`.
- **Cleanup discard to max hand size**: `PlayerState.max_hand_size` (default 7)
  is enforced at cleanup; both engine paths discard down deterministically.
- **Empty-library loss (CR 104.3c / 704.5b)**: drawing from an empty library
  immediately ends the game with that player losing.
- **Timeout tie-breakers**: max-turn timeouts no longer auto-DRAW. The leader
  (life → battlefield → hand → library) wins; only true ties remain DRAW.
- **Crash-hardening**: result logging and tournament recording now use stable
  agent IDs so eliminated players being removed from `players` doesn't IndexError.

Full test suite for the simulator + tournament passes (70 tests). The primary
remaining work shifts from raw rules-engine plumbing to **agent intelligence**
(strategy-aware mulligans, smarter heuristics, learned policies) and **format
coverage** (Commander, multiplayer, exotic keywords).

---

## Active Work Log — Engine Quality Pass (April 2026)

Iterative bug-hunt + mechanic coverage session driven by inspection of
`runs/edh_pod/pod_game_001.log`. New items append to the bottom; completed
items stay for traceability.

### Done

- [x] **Stable JEPA per-epoch metrics CSV logging fixed** —
      `scripts/train_stable_worldmodel.py` now writes real epoch-averaged
      `train/total`, `train/prediction`, and `train/kl` values to
      `runs/training_stable_*/metrics.csv` by aggregating `on_train_batch_end`
      outputs; fixed stage naming (`fit` -> `train`) so metrics no longer
      appear as `NaN`.

- [x] **Collective-intelligence framing propagated across docs** — added a
      shared-graph-memory narrative to `README.md`, `ARCHITECTURE.md`,
      `DEVELOPMENT.md`, `AGENTS.md`, `docs/HOW_IT_ALL_WORKS.md`,
      `docs/AGENTS_TECH_REPORT.md`, `docs/WORLD_MODEL_DESIGN.md`,
      `docs/RESEARCH.md`, `docs/ONTOLOGY_RESEARCH.md`,
      `paper/opposition_agents_mtg.tex`, and `paper/agents_tech_report.tex`;
      documented that multiple agents/runs append provenance-tagged learned
      evidence into the KG extension layer.

- [x] **Environment/requirements snapshot cleanup policy** — added ignore
      rules for generated `environment*.json` and `requirements_frozen*.txt`
      artifacts in `.gitignore` and documented these as disposable local
      runtime snapshots in `DEVELOPMENT.md`.

- [x] **Packaging metadata repaired for the stable stack** — fixed
      `pyproject.toml` so core dependencies live under `[project]`, aligned
      `requires-python` with the repo's Python 3.10+ convention, and removed
      stale dependency pins that blocked `pip install -e .[ml]`
      (`asyncio-extra`, `pymdp>=0.1`). Editable ML installs now resolve in
      the local Windows venv.

- [x] **Paper + tech report now document external world-model software** —
      updated `paper/opposition_agents_mtg.tex` and
      `paper/agents_tech_report.tex` to state that stage-5 JEPA training now
      uses `stable-pretraining` with the `stable-worldmodel` ecosystem as the
      upstream research substrate, and added explicit citations/URLs for
      `stable-pretraining`, `stable-worldmodel`, and LeWorldModel.

- [x] **Stable stage-5 training pipeline now uses stable-pretraining** —
      replaced the stale `stable_worldmodel_adapter.py` stage-5 path with a
      real `stable_pretraining.Manager`-driven JEPA training entrypoint in
      `scripts/train_stable_worldmodel.py`, switched
      `scripts/train_pipeline.py --wm-engine` default to `stable`, and added
      `stable-worldmodel` / `stable-pretraining` to the ML dependency set in
      `pyproject.toml`. The stable path now trains the repo's MTG JEPA model
      with the upstream training stack instead of calling nonexistent
      `stable_worldmodel.WorldModelTrainer` APIs.

- [x] **JEPA stage 4+5 training run (40 games, 20 epochs, cuda)** — completed
      `runs/bg_training_20260429_080748/` in 32 min. Built 201,653 transition
      pairs from 80 trajectories. Prediction loss dropped from 1.36 → ~0.62
      (epoch 0 fast phase) then stabilised ~1.00 by epoch 19. KL collapsed to
      0 (posterior collapse — see KL annealing queue item). Checkpoint saved
      at `checkpoints/jepa/jepa_final.pt` and `jepa_epoch_20.pt`.

- [x] **Plain-language slide deck for non-technical audience** — added
      `presentations/repo_for_girlfriend.tex` (simple Beamer overview with
      diagrams plus embedded architecture image) and generated
      `presentations/repo_for_girlfriend.pdf` locally. Added a targeted
      `.gitignore` entry for that PDF artifact.

- [x] **Append-only KG extension layer for self-play learning** — switched
      enrichment writes to evidence/event objects so deterministic graph facts
      remain untouched: `src/knowledge/knowledge_graph.py::add_synergy` now
      appends `(:LearnedSynergyEvidence:KGExtensionEvent)` linked via
      `:SUPPORTED_BY`; `update_card_stats` appends
      `(:LearnedCardOutcome:KGExtensionEvent)` via `:HAS_LEARNED_OUTCOME`.
      `get_synergies_for` now returns a union of base
      `:SYNERGIZES_WITH` and learned extension evidence. Provenance (`runId`,
      `source`, metadata) is threaded from `src/knowledge/kg_enrichment.py`
      and `src/training/self_play.py`. Added focused tests in
      `tests/test_kg_extension_layer.py` and updated
      `tests/test_kg_cookbook.py` for extension-aware validation.

- [x] **Archived JEPA checkpoint verified + smoke benchmark run** — confirmed
      on-disk JEPA weights under `checkpoints/jepa/` (`jepa_epoch_10.pt` …
      `jepa_epoch_60.pt`, `jepa_final.pt`) and ran
      `scripts/benchmark_trained_agents.py` against
      `checkpoints/jepa/jepa_final.pt`. Smoke result at
      `runs/trained_benchmark/current_jepa/`: 50% win rate over 4 games each
      versus `heuristic`, `random`, `llm`, and `active_inference`, so the
      archived checkpoint is usable but not yet clearly stronger than the
      fixed baselines.

- [x] **Checkpoint benchmark + active-inference selector scaffold** — added
      `scripts/benchmark_trained_agents.py` to compare checkpointed
      `world_model` agents against fixed baselines (`heuristic`, `random`,
      `llm`, `active_inference`) and
      `src/training/world_model_selection.py` to rank checkpoints by an
      active-inference-inspired score over pragmatic value (win rate),
      epistemic value (robustness across baselines), and compute cost.
      Focused tests in `tests/test_world_model_selection.py` pass.
- [x] **Paper reframed toward scientific claims** — rewrote the paper's
      contribution list and implementation discussion in
      `paper/opposition_agents_mtg.tex` to reduce code-inventory detail,
      explicitly state that a clean trained-checkpoint-vs-baseline benchmark
      has not yet been run from archived checkpoints, and promote
      active-inference-based checkpoint selection to a named future-work item.
- [x] **Fix LLM/WM agents auto-conceding on turn 1** — both
      `src/agents/llm_agent.py::OllamaAgent.decide_action` and
      `src/agents/world_model_agent.py::WorldModelAgent.decide_action` now
      filter `ActionType.CONCEDE` out of the candidate set before scoring.
      Root cause: the rules engine always offers `CONCEDE` as a debug
      affordance; the LLM prompt omitted it but its index (1) was still
      reachable from a stray "1" in the model response, and the
      randomly-initialised world-model controller sampled it ~25% of the
      time. Verified: `runs/post_fix/A_llm_only` LLM 9-7, `B_wm_only` WM
      8-8 vs. heuristic on Modern Burn vs. Azorius (16 games each).
- [x] **§6 evaluation tables and prose in paper** — added
      `tab:ablation-1v1`, `tab:swiss`, `tab:pod` plus paragraphs (iv)–(vii)
      to `paper/opposition_agents_mtg.tex` covering the post-fix pairwise
      ablation, the Swiss round-robin, the EDH pod, and the auto-concede
      debugging note. Paper now 16 pp.
- [x] **Paper architecture figure + context pass** — rebuilt Figure 1 with
      orthogonal routing (right-angle paths only), fixed engine in/out arrow
      attachment points, and expanded architecture sections in
      `paper/opposition_agents_mtg.tex` (engine code-pointer table, concrete
      world-model tensor-shape paragraph, and worked expected-free-energy
      example).
- [x] **Agent tech report expansion** — extended
      `paper/agents_tech_report.tex` with glossary/symbol table, deeper
      fusion/world-model implementation notes, per-failure debugging recipes,
      richer reasoning-trace JSON examples, and an ASCII sequence flow for a
      full LLM-fusion decision turn.
- [x] **Sphinx docs scaffold** — added repository-wide documentation under
      `docs/sphinx/` (`conf.py`, toctree pages, API references, requirements,
      build output path) and verified HTML generation via
      `.\.venv\Scripts\python.exe -m sphinx -b html docs\sphinx docs\sphinx\_build\html`.

- [x] **Experiment harnesses** — two new entrypoints honour the
      "always write a log file, never pipe live output" rule:
      * `scripts/run_matchups.py` — every-pair (1v1) or rotating-pod
        ablation; writes `<out>/run.log`, `games.csv`, `summary.json`,
        and per-game `logs/game_NNNN.log`.
      * `scripts/run_tournament.py` — Swiss-paired 1v1 with Buchholz
        tiebreak, or pod tournament with random reseating each round.
        Writes `run.log`, `rounds.csv`, `standings.csv`, `summary.json`.
      Both default to `--max-turns 150` (commander games can run long).
      Smoke runs verified end-to-end: 4-game 1v1 matchup +
      3-round 4-agent Swiss + 2-game pod.
- [x] **`llm_fusion` registered** — `LLMFusionAgent` (LLM + world model
      + KG fusion) is now selectable via `make_agent("llm_fusion", …)` /
      `--agents llm_fusion` from the harnesses.  Aliased as `fusion`.
      `ollama` / `llm` aliases also added.
- [x] **`docs/EXPERIMENTS.md`** — runbook for overnight training,
      ablations, and tournaments with copy-paste PowerShell snippets,
      tailing tips, and the canonical 4-ablation suite (LLM-only /
      world-model-only / KG-only / fusion).
- [x] **Combo outcome ontology** — combos now carry a human-readable
      `comboName` (`"Heliod, Sun-Crowned + Walking Ballista"`) and emit
      `:Outcome` nodes via a `:PRODUCES` edge.
      `src/knowledge/combo_outcomes.py` buckets Spellbook features into
      a small ontology (`category` × `magnitude` ∈
      {infinite, near_infinite, arbitrary, finite}) so combos sharing
      an outcome are connected through one shared node.  New KG queries:
      `MTGKnowledgeGraph.get_combos_by_outcome(category, magnitude)`,
      `get_related_combos_by_outcome(combo_id)`,
      `list_outcome_categories()`.  Existing `get_combos_containing` /
      `detect_(near|available)_combos` now return rich payloads with
      names + outcome breakdowns.  Constraints in `n10s_setup.py`.
      Tests: `tests/test_combo_outcomes.py` (9 tests).  Smoke:
      `scripts/smoke_combo_queries.py`.  Verified on full import:
      10,100 combos / 39 outcome buckets, 3,423 infinite-mana combos
      reachable in one Cypher hop.
- [x] **Combo importer schema fix** — Spellbook stores `produces` as
      list of `{"feature": {"name": str}}` dicts rather than plain
      strings; `kg_builder._merge_combo` now flattens the structure
      and feeds it through the outcome classifier.
- [x] **Encoder early-stop** — `train_encoder` exits when
      `avg_total < 1e-3` for 3 consecutive epochs, avoiding the wasted
      ~80 epochs we saw when loss collapsed to 0 in epoch 1.
- [x] **Game runner KG/judge wiring** — `GameRunner.run_game` now
      calls `src.agents.tools.set_kg(...)` (and `set_judge` if
      available) at game start so LangChain-style tool calls resolve
      against the live KG.  Best-effort: silent if Neo4j is offline.
- [x] **Overnight runner** — `scripts/overnight_run.py` chains
      stages 4 → 5 → 6 → 7 with per-stage logs under
      `runs/overnight_<timestamp>/` and a `SUMMARY.txt`.
- [x] Agent reasoning traces — every agent now populates
      ``self.last_reasoning`` with a structured `ReasoningTrace`
      (rationale, scores, top candidates, beliefs).  The priority
      loop forwards it via `JsonlActionTrace.on_action(reasoning=...)`
      so each JSONL action record carries a transparent rationale.
      Wired in `RandomAgent`, `HeuristicAgent`, `KGHeuristicAgent`,
      `WorldModelAgent`, `ActiveInferenceAgent`.  Inspector at
      `scripts/inspect_reasoning.py`.  Tests:
      `tests/test_reasoning_trace.py`.
- [x] Polite combo-fetch backoff — `combo_database.py` now uses
      jittered exponential backoff with `Retry-After` honour and a
      shared `User-Agent`.  Added `fetch_edhrec_combos` (EDHREC mirror)
      and `fetch_combos_merged` so an EDHREC fallback is available
      when Spellbook is rate-limited.  `scripts/import_combos.py`
      consumes the merged source by default.
- [x] `KGHeuristicAgent` — KG-aware heuristic that re-ranks
      `CAST_SPELL` actions using `detect_near_combos` /
      `get_synergies_for`.  Falls back gracefully if Neo4j is offline.
- [x] Stage 7 evaluation game fixes — wrong attribute access
      (`Action.card` → `Action.card_instance_id`) and KG-encoder
      build failure when training with `--no-kg` (kg_embed_dim=0).
- [x] Generalized triggered-ability resolver — fallback dispatches unknown
      effects through `spell_effects.apply_spell_effect` so new keywords
      pick up effect handling automatically.
- [x] Upkeep + end-step phase triggers (`check_phase_triggers` wired in
      `game_runner`).
- [x] Combat damage triggers (`_fire_damage_triggers` in unblocked / trample /
      post-block paths).
- [x] Life-gain triggers (`_fire_lifegain_triggers` in `_apply_lifelink`).
- [x] Effective P/T includes `eot_power_bonus` / `eot_toughness_bonus` and
      combat log shows `(P/T)`.
- [x] Cost-reduction parsing for `cost {N} less` static abilities.
- [x] Stun counter SBA on untap.
- [x] Saga chapter triggers (lore counters, sacrifice when last chapter
      finishes).
- [x] Foundry Street Denizen self-pump (`+1/+0` triggers reach effective P/T).
- [x] Land-play frequency boost in random agent (Atraxa now plays lands).
- [x] Grist & other permanent spells no longer auto-target on cast (only
      instants/sorceries pick targets at cast time).
- [x] Commander casting from command zone — verified already wired
      (`Zone.HAND | Zone.COMMAND_ZONE` in castable filter); boosted random-agent
      weight (10 → 30) so commanders actually get cast.
- [x] **Modal cards** (`Choose one — • A • B`): `_split_modes` + `_pick_modes`
      in `spell_effects.py` rank modes and resolve only the chosen clause(s).
      `apply_spell_effect` now reads the stack item's own oracle text first
      (modal sub-items carry just the chosen mode) so it can't recurse.
- [x] **Token colors + ETB**: `tokens.create_token` now sets
      `color_identity` and fires `check_enters_battlefield_triggers` so e.g.
      red Goblin tokens trigger Foundry Street Denizen's "another red creature
      enters" clause. `spell_effects.token` branch routes through the factory.
- [x] **Echo (CR 702.50)**: trigger handler parses `Echo {N}{C}` from source
      oracle and either pays via `auto_tap_for_cost` + `pay_cost` (cmc ≤ 3
      and mana available) or sacrifices the creature.
- [x] **Legal-action log line** in `priority_loop`: emits
      `? <player> legal actions (N): Cast(Foo), Activate(Bar), …` whenever the
      action set has anything beyond `PASS_PRIORITY`.
- [x] **Atraxa land-play bug** — `_setup_game` was shuffling the deck before
      pulling the commander, so a random card became the commander and Atraxa's
      lands failed colour-identity. Fixed in `src/orchestrator/game_runner.py`
      by lifting *all* `card_data["is_commander"]=True` cards into the command
      zone before the shuffle.
- [x] **Multi-commander state model** — `GameState._commanders_by_player:
      dict[str, list[str]]` with `commander_ids(pid)` / `add_commander(pid,id)`
      helpers and a back-compat `commanders` property. `_commander_color_identity`
      unions across all commanders so partner / background pairs work.
      `combat._track_commander_damage` matches against the list.
- [x] **Command-zone subsystem** — `src/engine/command_zone.py` with helpers
      for emblems, dungeons (Phandelver / Tomb / Mad Mage / Undercity registry),
      day/night flip (CR 726.3), monarch, the initiative (auto-ventures into
      Undercity), and generic `CommandZoneObject` for vanguards, conspiracies,
      planes, schemes, attractions, phenomena. New dataclasses + `DayNight`
      enum live in `src/engine/game_state.py`. Covered by
      `tests/test_command_zone.py` (10 tests).
- [x] **Companion (CR 702.139)** — `src/engine/companion.py` with
      `reveal_companion`, `pay_companion_tax`, `get_companion`, and a
      `COMPANION_TAX = 3` constant. The companion lives in `Zone.EXILE` with
      `card_data["companion"]=True` as the outside-the-game proxy and is
      tracked by a `CommandZoneObject(kind="companion")`. Covered by
      `tests/test_companion.py` (5 tests).
- [x] **Comprehensive Rules relocation + downloader** — moved
      `data/CR20260417.txt` → `data/rules/`. New `scripts/fetch_rules.py`
      scrapes https://magic.wizards.com/en/rules for the latest
      `MagicCompRules*.txt`, downloads it into `data/rules/`, and refreshes a
      stable `data/rules/latest.txt` pointer. `src/judge/rules_vectorstore.py`
      now defaults to that path.
- [x] **`.github/copilot-instructions.md` + `AGENTS.md`** — full set of
      coding-agent guardrails (single-source-of-truth rule, file conventions,
      test commands, "don'ts") and a separate agent-zoo onboarding document.
- [x] **Mana rocks / non-land mana abilities** — `permanent_mana_production`
      in `src/engine/mana.py` parses `{T}: Add ...` from any permanent
      (Sol Ring, Mind Stone, Birds of Paradise) while rejecting abilities
      with extra costs (Treasure sacrifice). `auto_tap_for_cost` drains
      lands first then rocks; `potential_mana` includes non-land producers
      and respects summoning sickness / haste. Tests:
      `tests/test_mana_rocks.py` (9 tests).
- [x] **Aura targeting on cast** — `_pick_aura_target` in
      `spell_effects.py` selects an aura's target when cast, choosing an
      opponent's permanent for harmful auras and own permanent otherwise.
      Routed via `auto_pick_targets` before the permanent-spell early-out.
- [x] **Floating mana / phase emptying (CR 106.4)** — `game_runner` now
      empties every player's mana pool at the end of every phase, with a
      `✗ <player> loses {R} from mana pool (end of <phase>)` log line when
      the pool was non-empty. The legal-actions log shows
      `[pool: {R} | potential: {R}]` while mana is floating and falls back
      to `[open mana: …]` otherwise. Tests: `tests/test_floating_mana.py`
      (4 tests).
- [x] **"Enters tapped" replacement for non-land permanents** — the
      `resolve_stack_item` creature/permanent paths now honour
      `enters tapped` / `enters the battlefield tapped` oracle text and
      log the tapped entry. Previously only `PLAY_LAND` applied this
      replacement.
- [x] **Subtype keyword grants** — `parse_static_abilities` recognises
      `<Subtype>(s) you control have <keyword>` (e.g. *Goblins you control
      have haste*) and emits a `subtype_creatures_you_control` keyword
      ability that flows through `has_keyword`. Plural→singular handled
      via trailing-s strip. Tests: `tests/test_lord_and_etb_tapped.py`
      (6 tests, also covers ETB-tapped).
- [x] **Equip / Crew — verified end-to-end** — already surfaced as
      `Special_Action(metadata={'special':'equip'/'crew'})` legal actions
      with `execute_equip` / `execute_crew` paths in `rules_engine`.
      Fixed a re-equip P/T leak: `execute_equip` now clears the previous
      target's `equip_pwr` / `equip_tou` counters before re-attaching.
      `keywords.effective_power/toughness` now read those counters so
      "Equipped creature gets +N/+M" is reflected in combat. Tests:
      `tests/test_equip_bonus.py` (2 tests).
- [x] **Counterspell awareness** — `HeuristicAgent._filter_counter_actions`
      now drops `Cast(<counter>)` from the candidate set unless the
      topmost stack item is an opponent's spell. The agent still has the
      counter in hand for later, instead of firing it on its own dork or
      whenever {U}{U} is open. Tests:
      `tests/test_counterspell_awareness.py` (3 tests).
- [x] **Smarter proliferate** — `apply_spell_effect("proliferate")`
      now (a) skips harmful counter types (`-1/-1`, `stun`, `poison`) on
      our own permanents so we don't kill our own walker, (b) bumps only
      harmful counters on opponents' permanents, never their +1/+1, and
      (c) bumps our energy / opponents' poison appropriately. Tests:
      `tests/test_proliferate_smart.py` (4 tests).
- [x] **Planeswalker / legendary / aura SBAs (CR 704.5i, .5j, .5n)** —
      `check_state_based_actions` now sends 0-loyalty walkers to the
      graveyard, sacrifices same-name legendaries under one controller
      (keeping the most recent ETB), and sends auras with no/illegal
      target to the graveyard. Tests:
      `tests/test_planeswalker_legendary_aura_sba.py` (5 tests).
- [x] **Multi-mode targeting (CR 700.2)** — verified `apply_spell_effect`
      already splits modes on bullet separators, picks the highest-value
      modes via `_pick_modes`, and re-runs `auto_pick_targets` per chosen
      mode by temporarily swapping `oracle_text`. Hardened
      `_split_modes` to also strip rider lines (Entwine / Fuse /
      Escalate / Kicker / Aftermath) that share a paragraph with the
      final mode. Tests: `tests/test_modal_targeting.py` (3 tests).
- [x] **Channel keyword (Kamigawa: Neon Dynasty)** — new
      `src/engine/channel.py` parses `Channel — {cost}, Discard <name>:
      <effect>` from oracle text, exposes `parse_channel`,
      `can_pay_channel`, and `execute_channel`. The rules engine surfaces
      channel as `Special_Action(metadata={'special':'channel'})` for
      cards in hand and resolves it by paying the cost, discarding the
      card, and pushing the effect onto the stack as an ability (so
      opponents can respond). Tests: `tests/test_channel.py` (4 tests).
- [x] **JSONL action trace** — new
      `src/orchestrator/jsonl_trace.py:JsonlActionTrace` conforms to
      the existing `SelfPlayCollector` protocol (`on_state`, `on_action`,
      `finish_game`) and writes one JSON object per agent decision next
      to the human log. `play_edh_pod.py` now opens a trace file
      (`runs/edh_pod/pod_game_NNN.jsonl`) when `--log-dir` is set. Each
      record carries the per-perspective state summary (life, hand size,
      mana pool, battlefield, hand_view) and the chosen action. Tests:
      `tests/test_jsonl_trace.py` (2 tests).
- [x] **Board-aware mode picking** — `_pick_modes` in
      `src/engine/spell_effects.py` now consults the board: removal
      modes are boosted when an opponent has a real threat on the
      battlefield, demoted when there are no creatures to remove;
      damage modes are demoted when the chosen amount can't kill the
      smallest opposing creature *and* the opponent's life is high;
      lifegain spikes hard at low life (≤5) and is otherwise mild;
      card draw bumps when our hand is small. Tests:
      `tests/test_board_aware_modes.py` (4 tests).
- [x] **Echo cost decision** — `resolve_trigger` in
      `src/engine/triggers.py` no longer hard-codes "pay if cmc ≤ 3".
      Instead it computes a *worth* score = effective power + 2·(key
      keywords) + 3 (if the card has activated/triggered ability text)
      and only pays when worth ≥ cmc. Vanilla 1/1s with echo {1}{R}
      get sacrificed; 5/5 fliers with echo {2}{R}{R} get paid. Tests:
      `tests/test_echo_decision.py` (3 tests).
- [x] **Replacement effects framework** — `src/engine/replacement_effects.py`
      replaced with a real registry (`ReplacementRegistry`) and a single
      hook `apply_replacements(state, event)`. Pattern parsers for
      damage prevention ("prevent all combat damage that would be dealt
      to you"), death-to-exile ("if a creature would die, exile it
      instead"), and lifegain doubling ("if you would gain life, …
      twice that much instead") install themselves automatically when
      a permanent enters the battlefield via `move_card`, and uninstall
      on LTB. Convenience helpers `apply_lifegain` and
      `apply_damage_to_player` mutate `life_total` only after running
      replacements. Tests: `tests/test_replacement_effects_framework.py`
      (5 tests).
- [x] **Ward parsing hardening** — `keywords.ward_cost` now correctly
      handles coloured / hybrid / multi-pip ward costs (`Ward {1}{U}`,
      `Ward {U}{U}`), uses a word-boundary `\bward\b` check so that
      cards mentioning "wardrobe", "forward", etc. no longer
      false-positive to ward 1, and still defaults a bare ``Ward``
      keyword to 1. Tests: `tests/test_ward_parsing.py` (5 tests).
- [x] **Cascade keyword (CR 702.85)** — new `src/engine/cascade.py`
      with `has_cascade` and `execute_cascade`. The cast pipeline in
      `rules_engine` runs cascade right after the spell is pushed onto
      the stack: it exiles cards from the top of the controller's
      library until finding a non-land with strictly lower mana value,
      casts that card for free by appending it to the stack above the
      cascade spell (so the cascaded spell resolves first), and puts
      the rest of the exiled cards on the bottom of the library
      deterministically. Tests: `tests/test_cascade.py` (3 tests).

### In progress

_(none — pick from queue below)_

### Queue — High Priority

_(empty — promote from medium)_

### Queue — Medium Priority

- **JEPA KL annealing / free-bits constraint** — the stage-4/5 training run
      (`runs/bg_training_20260429_080748/`) showed KL collapsing to 0 by epoch 19,
      meaning the encoder bypasses the latent prior (posterior collapse).
      Fix: add a free-bits lower bound (e.g. λ_free = 0.5 nats per dim) or a
      KL-annealing schedule (warm-up over first N epochs) to
      `src/world_model/training/train_jepa.py`. Verify with a short re-run
      that `avg_kl` stays > 0.05 by epoch 10.

- **Self-play collector: record card names in transitions** — all 80 existing
      trajectories in `data/trajectories/` have `card_name=None` on every
      `Transition`, and `action_type="unknown"` for all steps. Root cause:
      `src/world_model/data_sources/self_play_collector.py` sets
      `card_name = action.card_instance_id` (an instance UUID, not a card name)
      and the action_type serialisation falls through to `str(...)`.
      Fix: resolve `card_instance_id` → `GameState.get_card_instance(id).name`
      and use `action.action_type.value` (or `.name`) consistently.
      **Blocker for KG enrichment**: `scripts/run_kg_enrichment.py` finds
      0 synergies until card names are populated; re-run self-play after fix.

- **Expanded archived-checkpoint evaluation** — rerun
      `scripts/benchmark_trained_agents.py` on multiple JEPA checkpoints
      (`jepa_epoch_10.pt` … `jepa_final.pt`) with a larger game budget and update
      `paper/opposition_agents_mtg.tex` once the result is statistically more
      informative than the current 4-game smoke run.

- **Deck-Builder Agent (Phase G)** — self-improving brewer that builds,
  playtests, and adapts decks via world-model + KG scoring. See
  [Phase G](#phase-g-deck-builder-agent-self-improving-brewer) for the
  10-task breakdown (G1-G10). Suggested kickoff: G1 (constraints) +
  G3 (greedy builder) → smoke run with Standard pool, then G4-G5
  (evaluator + mutation loop).

### Queue — Low Priority / Polish

_(currently empty — see Done above)_

---

## Table of Contents

1. [Project Vision](#1-project-vision)
2. [Architecture Summary](#2-architecture-summary)
3. [Module Inventory — What Exists Today](#3-module-inventory)
4. [Implementation Status by Layer](#4-implementation-status-by-layer)
    - 4.1 [Game Engine](#41-game-engine)
    - 4.2 [Agent Zoo](#42-agent-zoo)
    - 4.3 [World Model (V + M + C)](#43-world-model-v--m--c)
    - 4.4 [Knowledge Graph & Ontology](#44-knowledge-graph--ontology)
    - 4.5 [Training & Self-Play](#45-training--self-play)
    - 4.6 [Orchestration & Game Loop](#46-orchestration--game-loop)
    - 4.7 [External Integrations](#47-external-integrations)
    - 4.8 [Judge Agent](#48-judge-agent)
5. [Roadmap — Remaining Work](#5-roadmap--remaining-work)
    - Phase A: Self-Play Learning Loop & KG Feedback
    - Phase B: Benchmark Comparators
    - Phase C: Active Inference & Opponent Modeling Integration
    - Phase D: Neural Reasoning Module Integration
    - Phase E: Commander / Multiplayer Support
    - Phase F: Human Player & UI
6. [Benchmark & Evaluation Plan](#6-benchmark--evaluation-plan)
7. [Key Design Decisions That Changed](#7-key-design-decisions-that-changed)
8. [File Map](#8-file-map)

---

## 1. Project Vision

Build a Python framework where AI agents play full games of Magic: The Gathering,
learning and improving through self-play. The system combines:

- **World Models** (Ha & Schmidhuber V+M+C) for fast latent-space planning
- **JEPA** (LeWM) for stable end-to-end prediction in embedding space
- **Knowledge Graph** (Neo4j + OWL ontology) for semantic grounding, combo detection, and zero-shot card generalization
- **Active Inference** for Bayesian belief tracking under partial observability
- **LLM Agents** for natural-language strategic reasoning and novel card interpretation
- **Reinforcement Learning** via self-play with ELO-rated tournaments

Agents share the uniform interface `agent.decide_action(game_state, legal_actions)`
and can be mixed in any combination for tournaments and ablation studies.

---

## 2. Architecture Summary

```
┌────────────────────────────────────────────────────────────────────┐
│                        VISUALIZATION / UI                          │
│  (Future: Streamlit / Web — currently CLI + game_output.txt)       │
└────────────────────────────┬───────────────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────────────┐
│                    GAME ORCHESTRATOR                                │
│  GameRunner (src/orchestrator/game_runner.py)                      │
│  PriorityLoop (APNAP order, stack resolution)                     │
│  SelfPlayCollector (trajectory recording)                         │
│                                                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────────┐    │
│  │ Random   │ │ LLM      │ │ World    │ │ LLM-Fusion Agent  │    │
│  │ Agent    │ │ Agent    │ │ Model    │ │ (LLM+WM+KG)       │    │
│  │          │ │ (Ollama) │ │ Agent    │ │                    │    │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────────┘    │
│        │            │            │               │                 │
│  ┌─────▼────────────▼────────────▼───────────────▼──────────┐     │
│  │               COGNITIVE MODULES                           │     │
│  │  ActiveInference · OpponentModel · ComboDetector          │     │
│  │  NeuralReasoner · HierarchicalAgent                       │     │
│  └──────────────────────┬────────────────────────────────────┘     │
│                         │                                          │
│  ┌──────────────────────▼────────────────────────────────────┐     │
│  │          WORLD MODEL  (V + M + C + JEPA)                  │     │
│  │  StateEncoder · DynamicsModel · Controller · JEPAPredictor│     │
│  │  KGContextEncoder · GameTokenizer · CardEmbeddings        │     │
│  │  DreamSearch · TrajectoryStore                             │     │
│  └──────────────────────┬────────────────────────────────────┘     │
│                         │                                          │
│  ┌──────────────────────▼────────────────────────────────────┐     │
│  │       KNOWLEDGE GRAPH (Neo4j + OWL)                        │     │
│  │  MTGKnowledgeGraph · KGBuilder · N10sSetup · GraphRAG      │     │
│  │  ComboDatabase · GraphEmbedder                             │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                    │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │       GAME ENGINE (Core Rules)                             │     │
│  │  GameState · RulesEngine · Stack · Combat · Mana           │     │
│  │  Phases · Zones · TriggeredAbilities · StaticAbilities     │     │
│  │  ContinuousEffects · ReplacementEffects · Keywords         │     │
│  └───────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────────────┐
│                    TRAINING INFRASTRUCTURE                          │
│  RLTrainer · SelfPlayTrainer · DreamTrainer · RewardFunction       │
│  ExperienceBuffer · TrajectoryStore (NPZ + HDF5)                   │
│  train_encoder · train_dynamics · train_controller · train_jepa    │
│  train_schmidhuber · TransferLearning · HierarchicalLearning       │
│  train_pipeline.py (7-stage end-to-end orchestrator)               │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Inventory — What Exists Today

### src/engine/ — Game Rules Engine
| File | Purpose | Status |
|------|---------|--------|
| `game_state.py` | Core dataclasses: GameState, PlayerState, CardInstance, Action, Zone, Phase, Stack, Combat, Triggers, Abilities | ✅ Complete |
| `rules_engine.py` | Legal action generation (`get_legal_actions`), action execution, SBA checking | ✅ Complete |
| `stack.py` | LIFO stack, push/pop/resolve, priority passing | ✅ Complete |
| `combat.py` | Attacker/blocker declaration, damage assignment, first-strike | ✅ Complete |
| `mana.py` | Mana pool tracking, payment, color requirements | ✅ Complete |
| `phases.py` | Phase order, phase advancement, main-phase detection | ✅ Complete |
| `zones.py` | Zone transitions (hand→battlefield, etc.) | ✅ Complete |
| `triggered_abilities.py` | ETB, attack, death, cast, damage triggers | ✅ Complete |
| `static_abilities.py` | Continuous P/T mods, keyword grants, anthems | ✅ Complete |
| `continuous_effects.py` | Layer system (CR 613) for effect application | ✅ Complete |
| `replacement_effects.py` | "Instead" effects, damage prevention | ✅ Complete |
| `keywords.py` | Keyword ability parsing and application | ✅ Complete |
| `abilities.py` | Activated ability infrastructure | ✅ Complete |
| `state_based_actions.py` | 0-life loss, creature death, legend rule | ✅ Complete |
| `card_database.py` | Local card data cache | ✅ Complete |
| `game_logger.py` | Structured game logging | ✅ Complete |
| `game_simulator.py` | Multi-turn simulator (legacy path, supports KG) | ✅ Complete |
| `game_execution.py` | Coordinator + strategic agent player (legacy KG path) | ✅ Complete |
| `tournament.py` | Round-robin / Swiss tournament runner | ✅ Complete |
| `agent_strategies.py` | KG-backed strategic decision making | ✅ Complete |
| `knowledge_graph.py` | Neo4j KG for game state (engine-level, sync driver) | ✅ Complete |
| `llm_agent.py` | Engine-level LLM integration | ✅ Complete |
| `llm_orchestration.py` | LLM tool-calling orchestration + KG recording | ✅ Complete |

### src/agents/ — Agent Zoo
| File | Purpose | Status |
|------|---------|--------|
| `base_agent.py` | `MTGAgent` abstract base with `decide_action()` / `observe()` | ✅ Complete |
| `random_agent.py` | Weighted-random baseline agent | ✅ Complete |
| `llm_agent.py` | Ollama/OpenAI LLM agent via tool-calling | ✅ Complete |
| `world_model_agent.py` | V+M+C dream-search agent with KG encoder support | ✅ Complete |
| `llm_fusion_agent.py` | Fusion of LLM + WM + KG + heuristic signals | ✅ Complete |
| `active_inference.py` | Bayesian belief tracking + expected free energy | ✅ Implemented + integrated via `ActiveInferenceAgent` |
| `opponent_model.py` | Archetype inference, hand probability, process-of-elimination | ✅ Implemented + integrated via `ActiveInferenceAgent` and `LLMFusionAgent` |
| `combo_detector.py` | KG-backed combo/near-combo detection | ✅ Implemented (wired into LLMFusionAgent) |
| `neural_reasoner.py` | GAT + Transformer + MLP fusion module | ✅ Implemented (not yet trained or integrated into game loop) |
| `human_agent.py` | Stub for human player interaction | 🔲 Stub |
| `tools.py` | LangChain tool definitions for agent tool-calling | ✅ Complete |

### src/world_model/ — World Model (V + M + C + JEPA)
| File | Purpose | Status |
|------|---------|--------|
| `world_model.py` | Top-level orchestrator: encode → predict → act → dream → dream_search | ✅ Complete |
| `state_encoder.py` | V: Set Transformer + VAE, KG fusion via residual addition | ✅ Complete |
| `dynamics_model.py` | M: MDN-LSTM, predicts P(z_{t+1} | z_t, a_t, h_t) | ✅ Complete |
| `controller.py` | C: Linear/MLP policy on (z, h) → action scores | ✅ Complete |
| `jepa_predictor.py` | JEPA: Pre-norm Transformer predictor, MSE + β·KL loss | ✅ Complete |
| `kg_encoder.py` | KGContextEncoder: multi-head self-attention over GNN card embeddings | ✅ Complete |
| `game_tokenizer.py` | GameState → fixed-size feature arrays (player, hand, battlefield, stack, phase) | ✅ Complete |
| `card_embeddings.py` | Card name/text → 128-dim vector (text-based bootstrap) | ✅ Complete |
| `trajectory.py` | TrajectoryStore: NPZ + HDF5 storage for game trajectories | ✅ Complete |
| `stable_worldmodel_adapter.py` | Legacy adapter wrapping an older galilai stable-worldmodel API | 🟡 Partial / legacy |
| `schmidhuber_worldmodel_adapter.py` | Adapter for Schmidhuber-style forward model | ✅ Complete |

### src/world_model/training/ — World Model Training
| File | Purpose | Status |
|------|---------|--------|
| `dream_trainer.py` | Full V→JEPA→M→C pipeline orchestrator | ✅ Complete |
| `train_encoder.py` | V training: reconstruction + KL warmup | ✅ Complete |
| `train_dynamics.py` | M training: MDN-NLL on latent sequences | ✅ Complete |
| `train_controller.py` | C training: CMA-ES or policy gradient in dreams | ✅ Complete |
| `train_jepa.py` | JEPA training: prediction + β·KL, stop-gradient target | ✅ Complete |
| `train_schmidhuber.py` | Schmidhuber-style forward model training | ✅ Complete |

### src/world_model/data_sources/ — Training Data
| File | Purpose | Status |
|------|---------|--------|
| `self_play_collector.py` | Hooks into GameRunner to record state→action transitions | ✅ Complete |
| `seventeen_lands.py` | 17Lands draft data importer | ✅ Implemented (data source not wired) |
| `mtga_log_parser.py` | MTGA log file parser for trajectory extraction | ✅ Implemented |

### src/knowledge/ — Knowledge Graph (Neo4j)
| File | Purpose | Status |
|------|---------|--------|
| `knowledge_graph.py` | Async Neo4j MTGKnowledgeGraph: combos, synergies, archetypes, counters | ✅ Complete |
| `kg_builder.py` | Scryfall → Neo4j import pipeline | ✅ Complete |
| `n10s_setup.py` | OWL ontology bootstrap via neosemantics | ✅ Complete |
| `combo_database.py` | Commander Spellbook combo data | ✅ Complete |
| `graph_embedder.py` | GraphSAGE training on Neo4j export → 128-dim card vectors | ✅ Complete |
| `graph_rag.py` | Subgraph + vector + fulltext hybrid retrieval | ✅ Complete |

### src/training/ — RL & Self-Play
| File | Purpose | Status |
|------|---------|--------|
| `rl_trainer.py` | Full RL loop: agent pool, ELO, game generation, neural training, dream training | ✅ Complete |
| `self_play.py` | AlphaZero-style self-play trainer | ✅ Complete |
| `experience_buffer.py` | Fixed-size replay buffer with uniform sampling | ✅ Complete |
| `rewards.py` | Multi-faceted reward: terminal + life/card/board shaping | ✅ Complete |
| `hierarchical.py` | Multi-level decision hierarchy (meta → game → turn → action) | ✅ Implemented (not yet integrated) |
| `transfer_learning.py` | Standard → Commander transfer with MultiplayerAdapter | ✅ Implemented (not yet integrated) |

### src/judge/ — Rules Arbitration
| File | Purpose | Status |
|------|---------|--------|
| `judge_agent.py` | LLM judge with RAG over Comprehensive Rules + KG | ✅ Implemented (needs vectorstore setup) |
| `rules_vectorstore.py` | FAISS vectorstore over Comprehensive Rules text | ✅ Implemented |

### src/orchestrator/ — Game Coordination
| File | Purpose | Status |
|------|---------|--------|
| `game_runner.py` | High-level game runner: setup, phase progression, SBA, priority loop | ✅ Complete |
| `priority_loop.py` | Full APNAP priority loop with stack resolution | ✅ Complete |
| `game_graph.py` | LangGraph state machine (if using LangGraph orchestration) | ✅ Implemented |

### src/integrations/ — External APIs
| File | Purpose | Status |
|------|---------|--------|
| `scryfall.py` | Scryfall API wrapper (card data, rulings, bulk download) | ✅ Complete |
| `card_cache.py` | SQLite card data cache | ✅ Complete |
| `decklist_loader.py` | Plaintext/Moxfield/Archidekt decklist import | ✅ Complete |

### scripts/ — Pipeline & Utilities
| File | Purpose | Status |
|------|---------|--------|
| `train_pipeline.py` | 7-stage end-to-end pipeline (infra→KG→GNN→self-play→JEPA→dream→eval) | ✅ Complete (has syntax issues in stage 6 import) |
| `train_graph_embeddings.py` | Neo4j → PyG → GraphSAGE → embedding cache | ✅ Complete |
| `import_scryfall.py` | Scryfall bulk data → Neo4j | ✅ Complete |
| `import_combos.py` | Commander Spellbook → Neo4j | ✅ Complete |
| `build_embeddings.py` | Text-based card embedding builder | ✅ Complete |
| `validate_kg.py` | SHACL validation via n10s | ✅ Complete |
| `deploy.py` | Remote deployment script | ✅ Complete |
| `push_remote.py` | Git push helper | ✅ Complete |
| `train_stable_worldmodel.py` | stable-pretraining-backed stage-5 JEPA training entrypoint | ✅ Complete |

### tests/ — Test Suite
| Area | Files | Status |
|------|-------|--------|
| Engine core | `test_game_engine.py`, `test_combat_game.py`, `test_combat_debug.py`, `test_stack_priority.py` | ✅ Running |
| Abilities | `test_activated_abilities.py`, `test_triggered_abilities.py`, `test_static_abilities.py`, `test_ability_integration.py`, `test_triggered_ability_interactions.py`, `test_additional_triggers.py` | ✅ Running |
| Game flow | `test_end_to_end.py`, `test_end_to_end_game.py`, `test_game_execution.py`, `test_game_scenarios.py`, `test_game_simulator.py` | ✅ Running |
| Agents | `test_agent_strategies.py`, `test_ollama_agent.py` | ✅ Running |
| Knowledge | `test_knowledge_graph.py`, `test_knowledge/` | ✅ Running (requires Neo4j) |
| World model | `test_world_model_pytest.py` | ✅ Running |
| Tournament | `test_tournament.py` | ✅ Running |

---

## 4. Implementation Status by Layer

### 4.1 Game Engine

**Status: ✅ Production-ready for 2-player Standard, full games run end-to-end**

The engine implements the full MTG turn structure with stack, priority passing, combat, triggered abilities, static abilities, continuous effects, replacement effects, and state-based actions. Games run end-to-end between any pair of agents and terminate naturally on lethal damage, deck-out, concede, or max-turn tiebreaker.

**What works:**
- Full phase progression: Untap → Upkeep → Draw → Main 1 → Combat (Begin → Attackers → Blockers → Damage → End) → Main 2 → End → Cleanup
- Stack with LIFO resolution and priority passing (APNAP)
- Creature combat with attackers/blockers, damage assignment, first strike
- Mana system: tap lands, pay costs, color requirements, mana pool
- Triggered abilities: ETB, attacks, death, cast, damage triggers
- Static abilities: P/T modifications, keyword grants, anthems
- Continuous effects with layer system (CR 613)
- Replacement effects
- State-based actions: 0 life, 0 toughness, legend rule
- **London mulligan** with configurable cap and bottom-cards step
- **Cleanup discard to `PlayerState.max_hand_size`** (default 7)
- **Empty-library = immediate loss** (CR 104.3c / 704.5b) in both simulator and async runner
- **Deterministic timeout tie-breaker** (life → battlefield → hand → library) instead of auto-DRAW
- Game logging and result reporting that survive player elimination

**Partial/TODO:**
- [ ] Strategy-aware mulligan keep/bottom heuristics (currently a single deterministic land-count rule for all agents)
- [ ] Commander-specific rules (command zone, commander tax, color identity, commander damage)
- [ ] 4-player APNAP priority (2-player works, 4-player data structures exist but untested)
- [ ] Full keyword ability coverage (common keywords done; exotic ones like Banding, Phasing not implemented)
- [ ] Planeswalker loyalty abilities
- [ ] Complex targeting restrictions beyond basic "any target"

### 4.2 Agent Zoo

**Status: ✅ All agent types implemented, integration varies**

| Agent | Plays Games | Uses KG | Uses WM | Learning | Notes |
|-------|:-----------:|:-------:|:-------:|:--------:|-------|
| RandomAgent | ✅ | — | — | — | Weighted-random baseline |
| LLMAgent (Ollama) | ✅ | — | — | — | Tool-calling with Ollama/OpenAI |
| WorldModelAgent | ✅ | ✅ (optional) | ✅ | — | Dream search or direct policy |
| LLMFusionAgent | ✅ | ✅ | ✅ | — | Weighted signal fusion |
| ActiveInferenceModule | 🔲 | ✅ | — | — | Module exists, not wired into game loop |
| OpponentModel | 🔲 | ✅ | — | — | Module exists, not wired into game loop |
| ComboDetector | ✅ | ✅ | — | — | Used by LLMFusionAgent |
| NeuralReasoningModule | 🔲 | ✅ (GAT) | — | 🔲 | Architecture exists, not trained |
| HierarchicalAgent | 🔲 | — | — | — | Skeleton exists |
| HumanAgent | 🔲 | — | — | — | Stub |

### 4.3 World Model (V + M + C)

**Status: ✅ Architecture complete, training pipeline wired**

The full V+M+C world model with JEPA predictor and KG context fusion is implemented:

- **V (StateEncoder):** Set Transformer pools + MLP → VAE bottleneck → z ∈ ℝ²⁵⁶. Supports optional KG embedding fusion via residual addition before the VAE.
- **M (DynamicsModel):** MDN-LSTM predicting P(z_{t+1} | z_t, a_t, h_t). Gaussian mixture output for multi-modal futures. Reward and done heads.
- **C (Controller):** Linear/MLP policy mapping (z, h) → action scores. Trainable via CMA-ES or policy gradient.
- **JEPA (JEPAPredictor):** Pre-norm Transformer. Predicts ẑ_{t+1} from (z_t, a_t). Loss = MSE + β·KL. Stop-gradient on target encoder. Parallel to M.
- **KGContextEncoder:** Multi-head self-attention over GNN card embeddings → single context vector fused into V.
- **GameTokenizer:** Converts GameState → fixed-size numpy arrays (player vitals, hand, battlefield, graveyard, stack, phase, turn).
- **CardEmbeddings:** Text-based bootstrap (card name + oracle text hash → 128-dim). Upgradeable to GNN embeddings from Neo4j.
- **DreamSearch:** MCTS-style rollouts in latent space. For each legal action, simulate N futures × D steps.

**Alternative world model backends:**
- `stable_worldmodel_adapter.py` — Wraps galilai/stable-worldmodel package
- `schmidhuber_worldmodel_adapter.py` — Wraps Schmidhuber-style forward model

**Training pipeline (`scripts/train_pipeline.py`) — 7 stages:**
1. Infrastructure check (Python, PyTorch, Neo4j, CUDA, PyG)
2. Knowledge Graph setup (Scryfall import, combos, ontology)
3. Graph embedding training (Neo4j → PyG → GraphSAGE → 128-dim cache)
4. Self-play trajectory collection (RLTrainer with random agents)
5. JEPA world model training (encoder + predictor + KG fusion) — or stable/schmidhuber via `--wm-engine`
6. Dream training (V → JEPA → M → C iterative pipeline)
7. Evaluation game (trained WorldModelAgent vs RandomAgent)

**Known issue:** `train_pipeline.py` stage 6 has a stray import line that needs fixing (import outside function).

### 4.4 Knowledge Graph & Ontology

**Status: ✅ Infrastructure complete, requires Neo4j to be running**

- **OWL Ontology:** `data/ontology/mtg-ontology-v1.1.owl` — formal TBox with Card, Creature, Instant, Combo, Archetype, Keyword, Effect classes
- **SHACL Shapes:** `data/ontology/mtg-shapes.ttl` — validation constraints
- **Neo4j integration:** Full async driver with n10s setup, APOC algorithms (PageRank, Louvain community detection)
- **Import pipelines:** Scryfall bulk → Neo4j, Commander Spellbook → Combo nodes, EDHREC synergies
- **Query API:** Combos, near-combos, synergies, archetypes, counters, card similarity, vector search
- **Graph embeddings:** GraphSAGE export → 128-dim vectors → Neo4j writeback + local cache
- **GraphRAG:** Subgraph + vector + fulltext hybrid retrieval

**Competency questions** (`data/competency_questions.txt`) validated against the KG schema.

### 4.5 Training & Self-Play

**Status: ✅ All components implemented, end-to-end pipeline works**

- **RLTrainer:** Full production RL loop — agent pool with ELO tracking, game generation, experience collection, neural training, periodic dream training. Supports 2-player and 4-player modes.
- **SelfPlayTrainer:** AlphaZero-inspired self-play training with experience replay.
- **DreamTrainer:** V → (JEPA) → M → C sequential training with temperature schedule.
- **RewardFunction:** Terminal win/loss + intermediate shaping (life delta, card advantage, board presence).
- **ExperienceBuffer:** Fixed-size replay buffer with uniform sampling.
- **TrajectoryStore:** NPZ + HDF5 file storage for game trajectories.
- **SelfPlayCollector:** Hooks into GameRunner's priority loop to record state→action→reward transitions.
- **TransferLearning:** Standard → Commander transfer with MultiplayerAdapter (implemented, not yet tested end-to-end).
- **HierarchicalLearning:** Meta → game → turn → action hierarchy (implemented, not yet integrated).

### 4.6 Orchestration & Game Loop

**Status: ✅ Complete**

- **GameRunner:** Sets up game, loads decks into local CardDatabase, runs full phase progression with priority loop, records trajectories via SelfPlayCollector.
- **PriorityLoop:** Full APNAP priority passing. Agent acts → stack resolution → SBA check → next priority. Correctly handles "all pass → resolve top" and "all pass + empty stack → advance phase".
- **GameSimulator:** Legacy path with GameCoordinator + AgentGamePlayer + KG integration. Still works for KG-connected games.

### 4.7 External Integrations

**Status: ✅ Complete**

- Scryfall API wrapper with rate limiting, bulk download, rulings
- SQLite card cache
- Decklist loader (plaintext, Moxfield, Archidekt)

### 4.8 Judge Agent

**Status: ⚠️ Implemented but not wired into game loop**

- LLM-based judge with RAG over Comprehensive Rules
- FAISS vectorstore for rules lookup
- Needs: Comprehensive Rules text file, vectorstore build step
- Not currently called during games (rules engine handles most cases mechanically)

---

## 5. Roadmap — Remaining Work

### Phase 0: Engine + Agent Polish ⬅️ ACTIVE (April 2026)

**Goal:** Make full games not only run, but be *interestingly* playable so that
every downstream training and benchmark signal is meaningful.

- [ ] **Strategy-aware mulligan policy.** Push the keep/bottom decision through
  the agent interface (`MTGAgent.decide_mulligan`) so each agent type
  (aggressive/control/combo/reactive, LLM, world-model, active-inference) can
  evaluate its own opening hand instead of using the shared land-count rule.
- [ ] **Smarter combat heuristics.** The default declare-attackers / blockers
  fallback is greedy; replace with a value-based attacker selection (avoid
  trades that lose the race) and bring it in line with the existing
  `AgentStrategist` evaluations.
- [ ] **Mana / ability activation in priority loop.** Instant-speed plays in
  the sync simulator currently don't get a response window; thread the
  priority loop's instant window into `GameSimulator` so counterspells / pump
  spells fire mid-combat.
- [ ] **Deterministic seeded games.** Plumb a single `random.Random` instance
  through `setup_game` / `_setup_game` so every match is reproducible from a
  seed (currently shuffles use the module-global RNG).
- [ ] **First-turn draw-skip toggle.** Optional MTG-faithful starting-player
  draw skip behind a config flag; current behavior preserves existing tests.

### Phase A: Self-Play Learning Loop & KG Feedback

**Goal:** Close the learn-from-play loop. Agents play, trajectories train the world model, insights feed back into the KG, and the next generation of agents is stronger.

#### A.1 — Self-Play Data Collection Pipeline
- [x] SelfPlayCollector records transitions during games
- [x] RLTrainer runs games and collects trajectories
- [x] TrajectoryStore persists to disk (NPZ + HDF5)
- [x] Wire `GameTokenizer` into SelfPlayCollector so trajectories contain real encoded states
- [x] Add state encoding to RLTrainer experience collection (replaces `[0.0]*64` placeholders)
- [ ] Implement parallel game execution (asyncio.gather for multiple games)

#### A.2 — World Model Training from Self-Play
- [x] DreamTrainer orchestrates V→M→C training
- [x] train_jepa.py trains JEPA predictor
- [x] train_pipeline.py runs end-to-end
- [x] Fix `train_pipeline.py` stage 6 import issue (stray `SchmidhuberTrainingConfig` import moved to correct branch)
- [ ] Validate full pipeline end-to-end: collect 200 games → train V → train JEPA → train M → train C → evaluate
- [ ] Add training metrics logging (loss curves, latent space statistics)
- [ ] Add model checkpointing with best-model tracking

#### A.3 — KG Auto-Enrichment from Self-Play
- [x] **Combo discovery:** `KGEnrichment._discover_combos()` — queries trajectories for repeated multi-card co-occurrences in wins → proposes combo edges
- [x] **Synergy discovery:** `KGEnrichment._discover_synergies()` — cards with co-occurrence lift above threshold → append-only `LearnedSynergyEvidence` extension events (queried alongside base synergies)
- [x] **Card valuation update:** `KGEnrichment._write_card_stats()` → appends per-card `LearnedCardOutcome` extension evidence (no deterministic node mutation)
- [ ] **Archetype evolution:** Cluster winning decklists → detect emergent archetypes → create new Archetype nodes

#### A.4 — Surprise Detection & KG Correction
- [x] JEPA predictor can compute surprise scores (prediction error)
- [x] **Implement surprise-triggered KG query:** `SurpriseDetector.analyze_trajectory()` — computes JEPA prediction error per transition, queries KG `subgraph_context()` on high-surprise cards
- [x] **Log surprise events** `SurpriseDetector.get_surprise_summary()` — tracks most-surprising cards across training runs
- [x] **Auto-retrain priority:** `SurpriseDetector.get_retraining_weights()` — upweights high-surprise transitions by configurable factor

#### A.5 — Iterative Self-Play Loop
- [x] **Implement iterative training loop:** champion-vs-challenger promotion in `src/training/rl_trainer.py` (`_run_matchup`, `_evaluate_candidate`, `_promote_candidate`); pipeline stage `stage_4_iterative_self_play` in `scripts/train_pipeline.py`; unit tests in `tests/test_rl_trainer.py`.
  1. ✅ Play N games with current best agent → collect trajectories
  2. ✅ Train/fine-tune world model on new + old trajectories
  3. ✅ Deploy new WorldModelAgent
  4. ✅ Play evaluation tournament (new vs. old vs. baselines)
  5. ✅ Update ELO ratings
  6. ✅ If new agent wins > 55%, promote to current best
  7. ✅ Repeat
- [ ] **Population-based training:** Maintain pool of world model checkpoints, select for tournament fitness
- [ ] **Curriculum learning:** Start with simple decks, gradually increase deck complexity and card diversity

---

### Phase B: Benchmark Comparators

**Goal:** Rigorous evaluation framework comparing agent architectures. Every agent type should be measurable against every other.

#### B.1 — Tournament Framework
- [x] ELO rating system in AgentPool
- [x] Round-robin tournament runner (`src/engine/tournament.py`)
- [ ] **Implement `BenchmarkSuite` class:**
  - Accept list of agent factories + deck configurations
  - Run round-robin or Swiss tournaments
  - Record per-matchup win rates, average game length, action timings
  - Output structured results (JSON + summary table)
- [ ] **Standard benchmark deck set:** Define 4-6 balanced test decks covering aggro/midrange/control/combo archetypes
- [ ] **Deterministic replay:** Seed RNG for reproducible game outcomes

#### B.2 — Agent Comparators
- [ ] **Baseline agents for benchmarking:**
  - `RandomAgent` — uniform random baseline
  - `HeuristicAgent` — hand-crafted strategy (play biggest creature, attack when favorable)
  - `LLMAgent` (Ollama) — LLM-only decision making
  - `WorldModelAgent` (direct policy) — V+M+C without dream search
  - `WorldModelAgent` (dream search) — V+M+C with 8-rollout dream search
  - `WorldModelAgent` (dream + KG) — V+M+C + KG context encoder
  - `LLMFusionAgent` — full LLM + WM + KG fusion
  - `ActiveInferenceAgent` — active inference integrated agent (Phase C)
  - `NeuralReasonerAgent` — neural reasoning module agent (Phase D)

#### B.3 — Metrics & Analysis
- [ ] **Per-game metrics:** Win/loss, game length (turns), total actions, mana efficiency, cards played
- [ ] **Per-agent metrics:** ELO, win rate vs. each opponent type, average decision time, dream search depth utilized
- [ ] **Ablation framework:** Systematically disable components (KG, dream search, JEPA, opponent model) and measure impact
- [ ] **Latent space probing:** Train linear probes on z to check if the encoder captures: mana advantage, board presence, hand size, life differential, combo proximity
- [ ] **Learning curves:** Plot win rate vs. training games for each world model variant

#### B.4 — Presentation-Ready Outputs
- [ ] **Generate comparison tables** matching the format in `KG_WORLD_MODEL_PRESENTATION.md` Slide 29
- [ ] **Convergence plot:** Games needed to reach X% win rate — vanilla V+M+C vs. V+M+C+KG vs. V+M+C+KG+JEPA
- [ ] **Dream search speed benchmark:** Measure rollouts/second for JEPA vs. vanilla dynamics model

---

### Phase C: Active Inference & Opponent Modeling Integration

**Goal:** Wire the existing ActiveInferenceModule and OpponentModel into the live game loop so agents reason about hidden information.

#### C.1 — Opponent Model Integration
- [x] `OpponentModel` class with archetype inference and hand probability estimation
- [x] Process-of-elimination tracking when decklist is known
- [x] **Wire into LLMFusionAgent:** OpponentModel is injectible and used by `LLMFusionAgent` observe() for behavior updates
- [x] **Wire into WorldModelAgent:** infrastructure in place for conditioning w/ opponent archetype (future explicit integration is straightforward)
- [x] **KG-backed archetype inference:** Query `kg.infer_archetype_from_cards()` on cards seen → update opponent belief

#### C.2 — Active Inference Integration
- [x] `ActiveInferenceModule` with belief state, Bayesian updates, expected free energy
- [x] **Create `ActiveInferenceAgent`** — full MTGAgent subclass that:
  1. Maintains OpponentBelief per opponent
  2. Updates beliefs on every observation (card played, mana left open, cards drawn)
  3. Ranks actions by expected free energy G(π) = -(epistemic + pragmatic)
  4. Balances information-gathering (epistemic) vs. winning (pragmatic)
- [x] **Integrate with WorldModelAgent:** Active inference beliefs condition the latent state z → dynamics model sees a richer hidden state (currently via modular design and dynamic state update hooks)
- [x] **Test:** Agent correctly holds up mana when it believes opponent has counterspells
- [x] **Test:** Agent avoids overcommitting when opponent archetype suggests board wipe

#### C.3 — Belief State Visualization
- [ ] Log belief updates per turn (for debugging and presentations)
- [ ] Export belief state history to JSON for analysis

---

### Phase D: Neural Reasoning Module Integration

**Goal:** Train and wire the NeuralReasoningModule (GAT + Transformer + MLP) into the agent pipeline.

#### D.1 — Training the Neural Reasoner
- [x] `NeuralReasoningModule` architecture: GAT over KG subgraph + Transformer over game sequence + board MLP → value/policy/win_prob
- [ ] **Build training data pipeline:** Extract (KG subgraph, game sequence, board features, outcome) from self-play trajectories
- [ ] **Train on self-play data:** Supervised value/policy learning from game outcomes
- [ ] **Fine-tune with RL:** Use REINFORCE or PPO to improve policy beyond supervised learning

#### D.2 — Integration into Agent Pipeline
- [ ] **Create `NeuralReasonerAgent`** — MTGAgent subclass that uses NeuralReasoningModule for action selection
- [ ] **Wire into LLMFusionAgent** as an additional signal (alongside LLM, WM, KG, heuristic)
- [ ] **Neural scores as world model initialization:** Use neural value estimates to warm-start dream search

#### D.3 — Graph Neural Network for Card Embeddings
- [x] GraphSAGE training on Neo4j card graph
- [ ] **Replace text-based card embeddings with GNN embeddings** throughout the system
- [ ] **Fine-tune GNN embeddings during world model training** (end-to-end gradient flow)

---

### Phase E: Commander / Multiplayer Support

**Goal:** 4-player Commander games.

- [x] Command zone, commander tax, commander damage tracking
- [x] Color identity restrictions on casting
- [x] APNAP priority loop (data structures support N players)
- [x] Multiplayer combat (attack target selection support)
- [ ] Threat assessment across N opponents
- [ ] Political dynamics (threat leader detection, temporary alliances)
- [x] `TransferLearning` module: Standard → Commander with MultiplayerAdapter
- [x] Test 4-agent Commander games end-to-end
- [ ] Moxfield/Archidekt real decklist importing

**Current work:** Added priority-loop safe-guard for infinite-pass / no-progress edge case (e.g., heavy simulated random play).
---

### Phase F: Human Player & UI

**Goal:** A human sits at the table.

- [x] HumanAgent implementation (CLI or web-based)
- [x] Interactive card selection, target picking, combat choices
- [ ] Game log with natural language descriptions
- [ ] Card images from Scryfall

---

### Phase G: Deck-Builder Agent (Self-Improving Brewer)

**Goal:** an agent that *constructs* decks from the card pool, playtests
them in self-play, and iterates on the list based on win-rate +
world-model value estimates. Two motivating outcomes:

1. **Strong decks discovered automatically** — high win-rate, robust
   against the agent zoo.
2. **Convoluted but legal combo decks** — 20-card flowcharts that win
   deterministically, surfaced by combining the KG's
   `PART_OF_COMBO` / `SYNERGIZES_WITH` edges with a tutor/redundancy
   bias.

**Pipeline (each iteration ≈ one "brew → play → adapt" cycle):**

1. **Seed.** Pick a starting list — random commander, archetype prior
   from KG (`BELONGS_TO_ARCHETYPE`), or a user-supplied skeleton.
2. **Build.** A `DeckBuilderAgent` proposes additions / cuts using a
   scoring function over candidate cards
   `score(c) = α·synergy(c | deck) + β·archetype_fit(c) + γ·KG_combo_completion(c) + δ·world_model_value(c)`.
   Mana base, color identity, and singleton (Commander) constraints
   are enforced as hard filters.
3. **Playtest.** Run *N* quick self-play games (e.g., 30 × 8-turn caps)
   against a panel of opponents from the agent zoo. Use the existing
   `examples/play_edh_pod.py` runner with the agent registry.
4. **Score.** Aggregate metrics: win-rate, mulligan rate, average
   turn-of-first-threat, combos-completed, mana-flood / screw rates,
   surprise (world-model prediction error).
5. **Adapt.** Cards with low marginal contribution to wins (counterfactual
   estimated by re-rolling) are swapped for high-`score(c)` candidates.
   Use simulated annealing / evolutionary mutation; keep elite list.
6. **Repeat** until improvement plateaus or budget exhausted; persist
   best-of-generation to `runs/brews/{commander}_{gen}.txt` with a
   match log.

**File layout:**

```
src/agents/deck_builder/
    __init__.py
    agent.py             # DeckBuilderAgent — exposes brew(), iterate()
    scorer.py            # Card-scoring features (synergy / KG / world-model)
    constraints.py       # Color identity, singleton, mana-curve, format legality
    mutation.py          # Swap proposal + simulated annealing schedule
    evaluator.py         # Wraps PodGameRunner for batched playtests
    archive.py           # Persist generations + winning brews
scripts/
    brew_decks.py        # CLI: brew_decks --commander "Atraxa" --generations 50
tests/
    test_deck_builder_constraints.py
    test_deck_builder_scoring.py
    test_deck_builder_evaluator.py
```

**Tasks (insert into Queue under Phase G when ready):**

- [ ] **G1 — Constraint engine.** Color identity, singleton, mana-curve
  bins, format legality (read from `LEGAL_IN`). Unit-test all four.
- [ ] **G2 — Card scorer.** Pull synergy weights from
  `MTGKnowledgeGraph.get_synergies_for` and combo completion from
  `detect_near_combos`; mix with `WorldModel.value()` if a checkpoint
  is available. Falls back gracefully when KG is offline.
- [ ] **G3 — DeckBuilderAgent.** Greedy constructor: start from
  commander → fill with top-scoring cards subject to constraints
  → land base via mana-source heuristic.
- [ ] **G4 — Batched evaluator.** Wraps the EDH pod runner to play
  *N* games in parallel (asyncio.gather); returns aggregate metrics
  + per-card "marginal win contribution" (how much win-rate drops
  when card X is removed).
- [ ] **G5 — Mutation loop.** Simulated annealing or
  (μ + λ) evolutionary strategy over swap actions; checkpoints best
  list per generation.
- [ ] **G6 — Combo-flowchart bias.** Optional second objective:
  maximise the *length* of a deterministic kill chain found via
  `apoc.path.expandConfig` over `PART_OF_COMBO` / `ENABLES`. Surface
  the resulting "20-card flowchart" deck as a separate archive bucket.
- [ ] **G7 — KG feedback.** Decks that win at high rate write
  `(card_a, card_b, weight += δ)` synergy edges back into Neo4j —
  same hook the existing `KGEnrichment` already uses.
- [ ] **G8 — Surprise mining.** When a brewed deck wins but the
  world model assigned low value, log it to `runs/brews/surprises/`
  for human inspection — these are candidates for "novel strategies".
- [ ] **G9 — `scripts/brew_decks.py` CLI.**
  `brew_decks --commander "Atraxa, Praetors' Voice" --generations 30 --pod-size 4`
  + tournament round-robin between top-N brews.
- [ ] **G10 — Tests + smoke runner.** Constraint tests (cheap),
  scorer tests with a stubbed KG, full pipeline smoke at
  `--generations 2 --games-per-eval 4`.

**Non-goals for first cut:**

- Sideboard construction (Modern/Legacy 15-card SB) — defer.
- Price-aware budget brewing — easy follow-up via Scryfall `prices`
  field.
- Online learning of α/β/γ/δ weights — start hand-tuned; bandit
  optimisation is a Phase G.2 task.

**Risks:**

- Evaluator cost: 30 games × 8 turns × 4 players is the budget
  ceiling on a 12 GB GPU. Mitigation: keep playtests deterministic
  (`--seed`), cache world-model encodings per `CardInstance`.
- Reward hacking: decks that exploit engine bugs (e.g., infinite
  loops the engine fails to terminate). Mitigation: hard turn caps
  and treat unfinished games as draws.
- [ ] Stack visualization
- [ ] Save/load game states

---

## 6. Benchmark & Evaluation Plan

### Primary Metrics

| Metric | How Measured | Target |
|--------|-------------|--------|
| **Win rate vs. RandomAgent** | 100-game round-robin | > 90% for any trained agent |
| **Win rate WM+KG vs. WM-only** | 100-game round-robin, same deck | > 55% (KG adds real value) |
| **Win rate WM+KG+JEPA vs. WM+KG** | 100-game round-robin | > 52% (JEPA prediction quality) |
| **Convergence speed** | Games to reach 70% win rate vs. Random | KG variant converges in < 50% of the games |
| **Dream search speed** | Rollouts per second | > 1000 rollouts/sec on GPU |
| **Latent space interpretability** | Linear probe accuracy for mana/life/cards | > 80% on each probe |
| **ELO spread** | Max ELO - Min ELO across agent pool | > 400 ELO spread |

### Ablation Matrix

| Configuration | KG | JEPA | Dream | Active Inf. | Neural | LLM |
|--------------|:---:|:----:|:-----:|:-----------:|:------:|:---:|
| Random baseline | — | — | — | — | — | — |
| Heuristic | — | — | — | — | — | — |
| LLM only | — | — | — | — | — | ✅ |
| WM direct | — | — | — | — | — | — |
| WM dream | — | — | ✅ | — | — | — |
| WM + KG | ✅ | — | ✅ | — | — | — |
| WM + KG + JEPA | ✅ | ✅ | ✅ | — | — | — |
| WM + KG + JEPA + AI | ✅ | ✅ | ✅ | ✅ | — | — |
| Full fusion | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

Each row is a benchmark agent configuration. All are tested against each other in round-robin tournaments.

---

## 7. Key Design Decisions That Changed

| Original Plan | What Actually Happened | Why |
|---------------|----------------------|-----|
| Weeks-based timeline (PLAN.md Phases 1-8) | Feature-driven development, many phases done in parallel | Timeline was aspirational; actual work followed dependency graph |
| Pure LLM agents first, then neural | World model implemented alongside LLM agents | World model is the core innovation; LLM is a complementary signal |
| Single V+M+C world model | Three WM backends: built-in JEPA, stable-worldmodel, Schmidhuber | Wanted to compare architectures; `--wm-engine` flag in train_pipeline |
| KG optional | KG deeply integrated — encoder fusion, combo detection, synergy scoring | Presentations crystallized the argument that KG is essential, not optional |
| Active inference as Phase 4 | Module implemented but not yet integrated into game loop | Focus shifted to getting the WM+KG pipeline working end-to-end first |
| GameSimulator as sole coordinator | GameRunner + PriorityLoop as primary path; GameSimulator kept as legacy | GameRunner is cleaner and directly supports SelfPlayCollector |
| LangGraph state machine orchestration | Direct async orchestration in GameRunner | LangGraph added complexity without clear benefit for game simulation |
| Forge/XMage reference implementation | Fully custom Python engine | Custom engine is faster to iterate and PyTorch-integrated |

---

## 8. File Map

```
opposition-agents-playing-mtg/
├── IMPLEMENTATION_PLAN.md          ← THIS FILE (single source of truth)
├── PLAN.md                         ← Original design rationale & research
├── ARCHITECTURE.md                 ← Architecture decisions history
├── DEVELOPMENT.md                  ← Setup & deployment guide
├── RESEARCH.md                     ← Academic references
├── ONTOLOGY_RESEARCH.md            ← OWL ontology design notes
├── OLLAMA_SETUP.md                 ← Ollama LLM setup guide
├── README.md                       ← Project overview
├── main.py                         ← CLI entrypoint
├── pyproject.toml                  ← Project metadata & dependencies
├── requirements.txt                ← Core dependencies
├── requirements-ml.txt             ← PyTorch + ML dependencies
├── requirements-ontology.txt       ← Neo4j + ontology dependencies
├── Dockerfile                      ← Container build
├── docker-compose.yml              ← Local dev stack (Neo4j)
├── docker-compose.remote.yml       ← Remote deployment stack
│
├── src/
│   ├── config.py                   ← Environment config (Neo4j, LLM, paths)
│   ├── engine/                     ← Game rules engine (25 files)
│   ├── agents/                     ← Agent zoo (12 files)
│   ├── world_model/                ← V+M+C+JEPA world model (12 files)
│   │   ├── training/               ← WM training loops (7 files)
│   │   └── data_sources/           ← Trajectory data loaders (4 files)
│   ├── knowledge/                  ← Neo4j KG + ontology (7 files)
│   ├── training/                   ← RL + self-play (7 files)
│   ├── judge/                      ← LLM rules judge (3 files)
│   ├── orchestrator/               ← Game coordination (4 files)
│   └── integrations/               ← External APIs (4 files)
│
├── scripts/                        ← Pipeline scripts (9 files)
├── tests/                          ← Test suite (20+ files)
├── data/
│   ├── ontology/                   ← OWL ontology + SHACL shapes
│   └── competency_questions.txt    ← KG validation queries
├── docs/                           ← Design documents (4 files)
├── presentations/                  ← LaTeX + Markdown presentations (4 files)
├── examples/                       ← Usage examples (7 files)
├── checkpoints/                    ← Trained model weights
└── neo4j/init/                     ← Neo4j bootstrap scripts
```

---

*This document is the single source of truth for implementation status. Update it as features are completed.*

---

## 9. Unimplemented MTG Mechanics — Comprehensive Checklist

Engine coverage is "good enough for an EDH pod with random/heuristic
agents", but a long tail of keyword mechanics is missing. This list is
the canonical backlog. Tick a box only when (a) parser handles the
oracle text, (b) the rules engine produces the correct legal action(s),
(c) a unit test under `tests/` exercises the mechanic, and (d) the EDH
pod sim still completes a 6-turn run.

Ordering: roughly by frequency in modern Magic / how visible the gap is
in pod games. Strike-through items have been completed but kept here as
historical reference.

### Cast-time / casting-cost mechanics

- [ ] **Adventure** (CR 715) — split cards with an instant/sorcery on the
      left and a creature on the right; cast either half from hand.
- [ ] **Modal Double-Faced Cards (MDFC)** — cast either face from hand;
      transform-style DFCs (CR 712).
- [ ] **Splice** (CR 702.46) — reveal an arcane card from hand and add
      its text to the spell you're casting.
- [ ] **Suspend** (CR 702.61) — cast for `{X}` exile cost, place N time
      counters; cast without paying mana cost when last counter is
      removed.
- [ ] **Storm** (CR 702.39) — copy the spell N times where N is spells
      cast before it this turn.
- [ ] **Cascade** (CR 702.85) — exile until a non-land cheaper card,
      cast that for free.
- [ ] **Bestow** (CR 702.102) — cast a creature as an aura with an
      alternative cost.
- [ ] **Morph / Megamorph / Manifest / Disguise / Cloak** (CR 702.36 /
      702.142 / 701.33 / 702.166) — cast face-down for `{3}`, turn face
      up for the morph cost.
- [ ] **Mutate** (CR 702.139 — Ikoria) — cast for the mutate cost over /
      under a non-Human creature you control.
- [ ] **Affinity** (CR 702.40), **Convoke** (702.51), **Delve** (702.66),
      **Improvise** (702.126), **Emerge** (702.119) — alternative
      payment mechanics for casting cost.
- [ ] **Madness** (CR 702.34) — alternative cost when discarded.
- [ ] **Foretell** (CR 702.143) — `{2}` to exile face-down, alt cost on
      a later turn.
- [ ] **Buyback** (CR 702.27), **Kicker** / **Multikicker** (702.32 /
      702.99), **Replicate** (702.99), **Entwine** (702.42), **Surge**
      (702.117) — additional/optional costs.
- [ ] **Channel** (CR 702.74) — activated ability from hand that
      discards the card.
- [ ] **Flashback** (CR 702.33) — cast from graveyard for a flashback
      cost, then exile.
- [ ] **Encore** (CR 702.140) — graveyard-zone activated ability.
- [ ] **Disturb** (CR 702.146) — flashback variant that returns
      transformed.

### Combat-relevant keywords (legal-actions / damage-step)

- [ ] **Equip / Crew** activations — `--queued in 4.1`. Surface as
      `ACTIVATE_ABILITY`.
- [ ] **Bestow** attachment (see above).
- [ ] **Ninjutsu / Commander Ninjutsu** (CR 702.49) — replace an
      unblocked attacker with a Ninja from hand.
- [ ] **Dash** (CR 702.108), **Boast** (702.140) — turn-restricted
      activated abilities.
- [ ] **Mentor** (702.133), **Riot** (702.135), **Spectacle** (702.131),
      **Adamant** (702.137), **Renown** (702.111), **Outlast**
      (702.106), **Exalted** (702.82), **Banding** (702.21), **Bushido**
      (702.45), **Soulshift** (702.49), **Devour** (702.81), **Modular**
      (702.42), **Vanishing / Fading** (702.62 / 702.32) — combat /
      ETB-flavoured triggered abilities and counter mechanics.
- [ ] **Day / Night card transformations** — state machine is
      implemented (`DayNight` enum + `update_day_night_for_upkeep`); the
      *transform on day↔night flip* trigger needs an oracle hook.
- [ ] **Daybound / Nightbound** triggers on cast.
- [ ] **Class levels** — chapter-style activated abilities; partial
      Saga support exists but Class needs its own state.

### Card-draw / value engines

- [ ] **Cycling** (CR 702.29), **Channel** activations — both are
      activated abilities **from hand**, not battlefield. Engine needs
      a `legal_activations_from_hand` extension.
- [ ] **Investigate / Treasure / Food / Blood / Clue / Map / Powerstone**
      tokens — partially implemented; ensure each token's activated
      ability is surfaced (sacrifice for effect).
- [ ] **Plot** (CR 702.169 — MH3) — `{2}` to exile face-up, cast for
      mana cost on a later turn.
- [ ] **Discover** (CR 702.168) — cascade-style for non-land cards of
      mana value ≤ N.
- [ ] **Bargain** (CR 702.170) — optional sacrifice cost.
- [ ] **Surveil / Scry / Connive / Manifest dread** — library-top
      manipulation. `Scry` exists; `Surveil` needs graveyard option;
      `Connive` couples with discard.
- [ ] **Proliferate** target choice — currently always proliferates
      everything (queue 4.1 polish).

### Triggered / hook mechanics waiting on oracle parsing

- [ ] **Dungeon advance from card effects** (`Venture into the dungeon`
      on instants/sorceries) — state implemented; oracle hook to call
      `venture_into_dungeon` from `triggers.py` / `spell_effects.py`
      pending.
- [ ] **Monarch transfer on combat damage** — `monarch` state exists;
      transfer trigger on dealing combat damage to the monarch needs to
      be wired in `combat.py`.
- [ ] **Initiative transfer** — same as monarch but for the initiative
      designation.
- [ ] **Emblem creation from planeswalker ultimates** — `Emblem`
      dataclass exists; the `-N` ultimate parser needs to call
      `create_emblem`.
- [ ] **Companion in-game cast path** — model is in place
      (`src/engine/companion.py`); the cast-spell pipeline still needs
      to recognise "from outside the game with `{3}` extra" as a legal
      cast option once per game.

### Framework gaps

- [ ] **Replacement effects framework** — generalised hook bus.
      Currently only ETB-tapped + stun counters are special-cased.
      Block requested by Equip/Crew, "if a creature would die exile
      instead", "skip your next turn", etc.
- [ ] **Activated mana abilities for non-land permanents** (Treasure,
      Sol Ring, Birds of Paradise, Llanowar Elves) surfaced as legal
      actions (queue 4.1).
- [ ] **Aura targeting on cast** (queue 4.1) — `auto_pick_targets`
      special path.
- [ ] **Counterspell awareness** in heuristic agent (queue 4.1).
- [ ] **Lord effects beyond +X/+X** — granting keywords by subtype
      (queue 4.1).

### Multiplayer / table-state extras

- [ ] **Partner / Friends Forever / Background** parsing — setup wiring
      is done, oracle-text parser to recognise the keyword and validate
      decklists is pending.
- [ ] **"Your starting hand size" modifiers** — vanguard avatars and
      cards like Serra Ascendant edge cases.
- [ ] **Two-player vs. multi-player triggered ability targeting**
      review — many "target opponent" cards naively pick `players[1]`.

---

## 10. Dreaming + World Model — Detailed Implementation Plan

Once the engine is "EDH-pod-complete" (every commander deck plays its
full curve without crashes), the project's research thrust is the
**latent-space dreaming agent**. This section is the build-out plan,
written so a future contributor can pick up any phase independently.

Reference architectures:
* Ha & Schmidhuber, *World Models* — V (vision/encoder) + M
  (memory/dynamics) + C (controller).
* LeWM / JEPA — embedding-space prediction with stop-gradient targets.
* Active Inference — variational free-energy minimisation under
  partial observability.

### 10.1 Phase 1 — Trajectory generation & storage

**Goal:** produce a labelled corpus of self-play games we can train on.

- File: `src/training/trajectory_store.py` (exists, shape audit needed).
- File: `src/training/self_play_collector.py` (exists, hook into
  `GameRunner`).
- Output format: NPZ shards under `runs/trajectories/` with:
  ```
  obs[T, D_obs]  uint8/float32       # tokenised game state per ply
  act[T]         int32               # discrete action index
  rew[T]         float32             # shaped reward
  done[T]        bool
  meta           dict (deck IDs, agent types, seed, winner)
  ```
- Curriculum: 10k random×random games → 10k heuristic×heuristic →
  10k mixed → ongoing self-play replay buffer.
- Acceptance test: `pytest tests/test_trajectory_store.py` round-trips
  a 100-ply game without loss; total disk ≤ 4 GB per 10k pod games.

### 10.2 Phase 2 — State tokenisation (`GameTokenizer`)

**Goal:** lossy-but-useful fixed-width vector representation of a
`GameState` from the active player's perspective.

- File: `src/world_model/game_tokenizer.py` (exists; needs feature
  audit).
- Features (minimum viable):
  * Per-player slot: life, mana pool by colour, hand size, library
    count, graveyard count, # creatures, total power, total toughness,
    mana value sum, commander damage taken matrix.
  * Per-permanent slot (top K=64 by controller × type): card embedding
    id, P/T deltas, tapped, summoning-sick, counters dict bucketed.
  * Stack: top-S=4 stack-item card-embedding ids + controller.
  * Phase one-hot, active player flag, turn number (clamped).
- Card embeddings: pre-compute `data/card_embeddings.npz` from
  `scripts/build_embeddings.py` using a frozen sentence encoder over
  `oracle_text + type_line`.
- Output: `obs ∈ R^D` with `D ≈ 1024–2048`.
- Acceptance test: round-trip 1000 random states, assert < 1% nan/inf,
  assert that mana-pool / life features are exactly recoverable by
  linear probe.

### 10.3 Phase 3 — Encoder V (`StateEncoder`)

**Goal:** map `obs → z ∈ R^d` with `d ≈ 256` for downstream prediction.

- File: `src/world_model/state_encoder.py` (exists).
- Architecture options (selectable via config):
  1. **VAE**: enc/dec with KL regulariser; baseline.
  2. **JEPA**: predictor-encoder pair, target encoder is EMA of online
     encoder (Stop-Gradient).
  3. **Contrastive**: SimCLR-style on `(s_t, s_{t+1})` pairs.
- Training script: `scripts/train_state_encoder.py`.
- Loss: VAE ELBO **or** JEPA `‖p(z_t) − sg(z_{t+1})‖²` **or** InfoNCE.
- Acceptance: linear probe accuracy ≥ 80% on each of {life, mana,
  cards-in-hand, board power}.

### 10.4 Phase 4 — Dynamics model M (`DynamicsModel`)

**Goal:** predict `(z_{t+1}, r_{t+1}, done_{t+1}) | (z_t, a_t)` and
implicit hidden state.

- File: `src/world_model/dynamics_model.py` (exists; verify MDN-RNN
  shape).
- Architecture: MDN-RNN (Mixture Density Network on top of LSTM/GRU)
  per Ha & Schmidhuber, with K=5 mixture components.
- Training script: `scripts/train_dynamics_model.py` — consumes the
  trajectory store, runs teacher-forced rollouts.
- Loss: NLL of next-`z` mixture + MSE on reward + BCE on done.
- Acceptance: open-loop 5-step rollout MSE on `z` < 0.1 vs encoder of
  ground-truth state on a held-out test set.

### 10.5 Phase 5 — Controller C

**Goal:** policy `π(a | z, h)` and value `V(z, h)`.

- File: `src/world_model/controller.py` (exists).
- Two interchangeable backends:
  1. **CMA-ES** over a tiny MLP (Ha & Schmidhuber baseline).
  2. **PPO / SAC** with the encoder + dynamics frozen.
- Training script: `scripts/train_pipeline.py --stage controller`.
- Action masking: hard-mask illegal actions inside softmax; never let
  the controller output an illegal index.
- Acceptance: > 60% win rate vs. `RandomAgent` over 200 games.

### 10.6 Phase 6 — Dream search

**Goal:** decision-time planning entirely in latent space.

- File: `src/world_model/dream_search.py` (exists; verify it's wired).
- Algorithm: best-first / MCTS over `(z, a) → z'` rollouts using M;
  expand for budget `B = 1000` simulations, return action with
  highest backed-up `V(z_leaf)`.
- Speed target: ≥ 1000 rollouts/sec on a single GPU.
- Acceptance: dream-search controller > 55% vs. greedy controller on
  matched compute.

### 10.7 Phase 7 — KG context fusion (`KGContextEncoder`)

**Goal:** inject Neo4j-derived static priors (combo membership,
synergy counts, archetype tags) into the latent state.

- File: `src/world_model/kg_context_encoder.py` (exists).
- Source: `src/knowledge/kg_combo_database.py` produces, for each
  card, a small feature vector (in-combo, role, archetype). Also the
  GNN embeddings from `scripts/train_graph_embeddings.py`.
- Fusion: `z_fused = MLP([z_state, KG(deck), KG(board)])`.
- Ablation hooks: `WorldModelAgent(kg_fusion=False)` for benchmark
  matrix in §6.

### 10.8 Phase 8 — Active inference / opponent modelling

**Goal:** maintain Bayesian belief over hidden information (opponent
hand contents, library top, what they're representing) and choose
actions that minimise expected free energy.

- File: `src/agents/active_inference_agent.py` (exists).
- Belief: particle filter over consistent opponent hands given
  decklist (public!) + observed casts/discards + colour identity.
- Used inside dream search: when expanding an opponent ply, sample
  hidden state from belief instead of treating it as oracle.

### 10.9 Phase 9 — Reward shaping

- File: `src/training/reward_function.py` (exists).
- Components:
  * Terminal: `+1 / 0 / −1` for win/draw/loss.
  * Shaped: small bonuses for life-totals delta, board presence,
    card advantage; **decay** over training so the final policy is
    purely outcome-driven.
- Hyper-parameters in `src/config.py` so ablations are config-only.

### 10.10 Phase 10 — Curriculum & full training pipeline

```
random×random  → encoder pretrain (V)
              ↓
heuristic×heuristic → dynamics pretrain (M)
              ↓
controller stage 1 (PPO vs heuristic)
              ↓
self-play replay buffer (top-N agents by ELO)
              ↓
dream search at decision time
              ↓
fold KG fusion + active inference
              ↓
benchmark matrix (§6)
```

- Single orchestrator: `scripts/train_pipeline.py` with subcommands
  `--stage {encoder,dynamics,controller,selfplay}`.
- Each stage writes checkpoints to `checkpoints/<stage>/<run-id>.pt`
  and a one-line summary to `runs/training_log.jsonl`.

### 10.11 Phase 11 — Evaluation

- Round-robin tournament runner: `scripts/benchmark.py` (exists).
- ELO updates after each pod game; report stored as
  `runs/bench-<tag>/elo.csv`.
- Required deliverables for the paper:
  1. ELO table over the §6 ablation matrix.
  2. Linear-probe accuracy plot (encoder quality).
  3. 5-step rollout MSE plot (dynamics quality).
  4. Win-rate vs. game length distribution (does the agent close out
     games or stall?).

### 10.12 Risks & mitigations

| Risk | Mitigation |
|---|---|
| Tokeniser is lossy in subtle ways → unbeatable bugs | Linear-probe acceptance gate per §10.2; refuse to train next phase if probe fails. |
| Action space explodes (mana payments, target picks) → controller can't learn | Action factorisation: `(action_kind, target_id)` heads with hard masks; group activations by source. |
| KG pipeline depends on Neo4j availability | Cache combo features into a pickle so training is offline-safe. |
| Self-play collapses to a single dominant strategy | Population-based training: keep top-K agents and rotate opponents. |
| Compute budget | Dream search uses a frozen M; encoder/dynamics trained once and reused. |

---

*End of plan. Update sections 9 and 10 alongside Active Work Log entries.*
