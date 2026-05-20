# =============================================================================
# Makefile -- Unix counterpart to scripts/run_overnight_benchmarks.bat
#
# Trains the world model in three configurations, then runs ablations,
# WM head-to-head, and tournaments.  Each stage is a separate target so
# you can resume from any point:
#
#     make smoke                 # tiny params, end-to-end dry run
#     make overnight             # full pipeline (hours)
#     make overnight SKIP_TRAINING=1
#     make train-A train-B train-C
#     make ablations wm-compare tournaments
#
# All artefacts land under runs/overnight_<TS>/.
# =============================================================================

SHELL          := /bin/bash
PY             ?= ./.venv/bin/python
TS             := $(shell date +%Y%m%d_%H%M%S)
ROOT           ?= runs/overnight_$(TS)

# ---- knobs -------------------------------------------------------------
GAMES_PER_PAIRING ?= 8
MATCH_TURNS       ?= 100
POD_GAMES         ?= 4
POD_TURNS         ?= 150
TOURNEY_ROUNDS    ?= 6
POD_TOURNEY_ROUNDS?= 8

TRAIN_GAMES_A     ?= 30
TRAIN_EPOCHS_A    ?= 20
TRAIN_BATCH_A     ?= 8

TRAIN_GAMES_B     ?= 80
TRAIN_EPOCHS_B    ?= 60
TRAIN_BATCH_B     ?= 16

TRAIN_GAMES_C     ?= 80
TRAIN_EPOCHS_C    ?= 60
TRAIN_BATCH_C     ?= 16

DECK_BURN     := data/decks/modern/modern_mono_red_burn.txt
DECK_CONTROL  := data/decks/modern/modern_azorius_control.txt
POD_DECKS     := data/decks/edh/krenko-mob-boss_core.txt \
                 data/decks/edh/atraxa-praetors-voice_core.txt \
                 data/decks/edh/urza-lord-high-artificer_core.txt \
                 data/decks/edh/meren-of-clan-nel-toth_core.txt

CKPT_A := $(ROOT)/wm/A_small/jepa_final.pt
CKPT_B := $(ROOT)/wm/B_medium/jepa_final.pt
CKPT_C := $(ROOT)/wm/C_kg/jepa_final.pt

# When SKIP_TRAINING=1, fall back to the live default checkpoint everywhere.
ifeq ($(SKIP_TRAINING),1)
CKPT_A := checkpoints/jepa/jepa_final.pt
CKPT_B := checkpoints/jepa/jepa_final.pt
CKPT_C := checkpoints/jepa/jepa_final.pt
TRAIN_TARGETS :=
else
TRAIN_TARGETS := train-A train-B train-C
endif

.PHONY: help overnight smoke train-A train-B train-C ablations wm-compare \
        tournaments pod-ablation tourney-swiss tourney-pod clean-runs \
        phase-rs-ablation phase-rs-commander phase-rs-all-formats phase-rs-traces phase-rs-training

help:
	@echo "Targets:"
	@echo "  make smoke                  Tiny-parameter dry run."
	@echo "  make overnight              Full pipeline (hours)."
	@echo "  make overnight SKIP_TRAINING=1   Skip WM training, reuse checkpoint."
	@echo "  make train-A | train-B | train-C"
	@echo "  make ablations wm-compare tournaments"
	@echo ""
	@echo "  Phase-RS Ablations (NEW — May 2026):"
	@echo "  make phase-rs-ablation      Standard 1v1 (picker × difficulty × deck)"
	@echo "  make phase-rs-commander     Commander pod format"
	@echo "  make phase-rs-all-formats   Cartesian sweep (4 formats × 3 difficulties)"
	@echo "  make phase-rs-traces N=16   Collect traces for JEPA training"
	@echo "  make phase-rs-training      Full pipeline (traces → JEPA → eval)"
	@echo ""
	@echo "  ROOT=$(ROOT)"

# ----- smoke = quick, tiny -----------------------------------------------
smoke: GAMES_PER_PAIRING := 2
smoke: MATCH_TURNS       := 20
smoke: POD_GAMES         := 2
smoke: POD_TURNS         := 20
smoke: TOURNEY_ROUNDS    := 2
smoke: POD_TOURNEY_ROUNDS:= 2
smoke: TRAIN_GAMES_A     := 4
smoke: TRAIN_EPOCHS_A    := 2
smoke: TRAIN_BATCH_A     := 2
smoke: TRAIN_GAMES_B     := 4
smoke: TRAIN_EPOCHS_B    := 2
smoke: TRAIN_BATCH_B     := 2
smoke: TRAIN_GAMES_C     := 4
smoke: TRAIN_EPOCHS_C    := 2
smoke: TRAIN_BATCH_C     := 2
smoke: overnight
	@echo ">>> smoke run completed under $(ROOT)"

# ----- the pipeline -------------------------------------------------------
overnight: $(TRAIN_TARGETS) ablations wm-compare tournaments
	@echo
	@echo "============================================================"
	@echo " ALL BENCHMARKS COMPLETED.  Output under $(ROOT)"
	@echo " Compare configs: cat $(ROOT)/wm_compare/*/summary.json"
	@echo "============================================================"

# ----- training -----------------------------------------------------------
train-A:
	@echo "=== [1/6] WM training A (small, no-KG) ==="
	@mkdir -p $(ROOT)/wm/A_small
	$(PY) -u scripts/train_pipeline.py --stage 1 --end-stage 5 \
	    --num-games $(TRAIN_GAMES_A) --jepa-epochs $(TRAIN_EPOCHS_A) \
	    --jepa-batch-size $(TRAIN_BATCH_A) --no-kg --skip-eval --skip-dream \
	    > $(ROOT)/wm/A_small/train.log 2>&1
	@cp -f checkpoints/jepa/jepa_final.pt $(CKPT_A) 2>/dev/null || true

train-B:
	@echo "=== [2/6] WM training B (medium, no-KG) ==="
	@mkdir -p $(ROOT)/wm/B_medium
	$(PY) -u scripts/train_pipeline.py --stage 1 --end-stage 5 \
	    --num-games $(TRAIN_GAMES_B) --jepa-epochs $(TRAIN_EPOCHS_B) \
	    --jepa-batch-size $(TRAIN_BATCH_B) --no-kg --skip-eval --skip-dream \
	    > $(ROOT)/wm/B_medium/train.log 2>&1
	@cp -f checkpoints/jepa/jepa_final.pt $(CKPT_B) 2>/dev/null || true

train-C:
	@echo "=== [3/6] WM training C (medium + KG) ==="
	@mkdir -p $(ROOT)/wm/C_kg
	$(PY) -u scripts/train_pipeline.py --stage 1 --end-stage 5 \
	    --num-games $(TRAIN_GAMES_C) --jepa-epochs $(TRAIN_EPOCHS_C) \
	    --jepa-batch-size $(TRAIN_BATCH_C) --skip-eval --skip-dream \
	    > $(ROOT)/wm/C_kg/train.log 2>&1
	@cp -f checkpoints/jepa/jepa_final.pt $(CKPT_C) 2>/dev/null || true

# ----- ablations ----------------------------------------------------------
ablations:
	@echo "=== [4/6] 1v1 ablations (LLM / WM / KG / fusion) ==="
	$(PY) scripts/run_matchups.py --format standard \
	    --decks $(DECK_BURN) $(DECK_CONTROL) \
	    --agents llm heuristic \
	    --games $(GAMES_PER_PAIRING) --max-turns $(MATCH_TURNS) --seed 7 \
	    --out $(ROOT)/ablation/A_llm_only
	$(PY) scripts/run_matchups.py --format standard \
	    --decks $(DECK_BURN) $(DECK_CONTROL) \
	    --agents world_model heuristic \
	    --games $(GAMES_PER_PAIRING) --max-turns $(MATCH_TURNS) --seed 7 \
	    --wm-checkpoint $(CKPT_B) \
	    --out $(ROOT)/ablation/B_wm_only
	$(PY) scripts/run_matchups.py --format standard \
	    --decks $(DECK_BURN) $(DECK_CONTROL) \
	    --agents kg_heuristic heuristic \
	    --games $(GAMES_PER_PAIRING) --max-turns $(MATCH_TURNS) --seed 7 \
	    --out $(ROOT)/ablation/C_kg_only
	$(PY) scripts/run_matchups.py --format standard \
	    --decks $(DECK_BURN) $(DECK_CONTROL) \
	    --agents llm_fusion heuristic \
	    --games $(GAMES_PER_PAIRING) --max-turns $(MATCH_TURNS) --seed 7 \
	    --wm-checkpoint $(CKPT_B) \
	    --out $(ROOT)/ablation/D_fusion

# ----- WM config head-to-head --------------------------------------------
wm-compare:
	@echo "=== [5/6] World-model config head-to-head ==="
	$(PY) scripts/run_matchups.py --format standard \
	    --decks $(DECK_BURN) $(DECK_CONTROL) \
	    --agents world_model heuristic \
	    --games $(GAMES_PER_PAIRING) --max-turns $(MATCH_TURNS) --seed 7 \
	    --wm-checkpoint $(CKPT_A) \
	    --out $(ROOT)/wm_compare/A_small
	$(PY) scripts/run_matchups.py --format standard \
	    --decks $(DECK_BURN) $(DECK_CONTROL) \
	    --agents world_model heuristic \
	    --games $(GAMES_PER_PAIRING) --max-turns $(MATCH_TURNS) --seed 7 \
	    --wm-checkpoint $(CKPT_B) \
	    --out $(ROOT)/wm_compare/B_medium
	$(PY) scripts/run_matchups.py --format standard \
	    --decks $(DECK_BURN) $(DECK_CONTROL) \
	    --agents world_model heuristic \
	    --games $(GAMES_PER_PAIRING) --max-turns $(MATCH_TURNS) --seed 7 \
	    --wm-checkpoint $(CKPT_C) \
	    --out $(ROOT)/wm_compare/C_kg

# ----- tournaments --------------------------------------------------------
tournaments: pod-ablation tourney-swiss tourney-pod

pod-ablation:
	@echo "=== [6/6a] EDH pod ablation ==="
	$(PY) scripts/run_matchups.py --format commander --pod \
	    --decks $(POD_DECKS) \
	    --agents random heuristic kg_heuristic llm_fusion \
	    --games $(POD_GAMES) --max-turns $(POD_TURNS) --seed 7 \
	    --wm-checkpoint $(CKPT_B) \
	    --out $(ROOT)/pod_ablation

tourney-swiss:
	@echo "=== [6/6b] Swiss tournament ==="
	$(PY) scripts/run_tournament.py --mode swiss \
	    --format standard \
	    --decks $(DECK_BURN) $(DECK_CONTROL) \
	    --agents random heuristic kg_heuristic active_inference world_model llm llm_fusion \
	    --rounds $(TOURNEY_ROUNDS) --max-turns $(MATCH_TURNS) --seed 7 \
	    --out $(ROOT)/tournament_swiss

tourney-pod:
	@echo "=== [6/6c] Pod tournament ==="
	$(PY) scripts/run_tournament.py --mode pod \
	    --format commander \
	    --decks $(POD_DECKS) \
	    --agents random heuristic kg_heuristic llm_fusion \
	    --rounds $(POD_TOURNEY_ROUNDS) --max-turns $(POD_TURNS) --seed 7 \
	    --out $(ROOT)/tournament_pod

# ----- housekeeping -------------------------------------------------------
clean-runs:
	@echo "Removing runs/overnight_* (use with care)"
	rm -rf runs/overnight_*

# ========================================================================
# PHASE-RS ABLATION SUITE (NEW — May 2026)
# ========================================================================
# These targets run comprehensive evaluation on phase-rs engine.
# All produce JSONL traces + structured summary.json for analysis.
#
# Usage:
#   make phase-rs-ablation              # Standard 1v1 against AI difficulties
#   make phase-rs-commander             # Commander/Brawl pod format
#   make phase-rs-all-formats           # Cartesian sweep: format × difficulty × deck
#   make phase-rs-traces N=16           # Collect traces for JEPA training
#   make phase-rs-training              # Full pipeline: traces → JEPA → eval
# ========================================================================

PHASE_RS_ROOT ?= runs/phase_rs_$(TS)

# Phase-RS 1v1 ablations (Standard format, various picker × difficulty combos)
phase-rs-ablation:
	@echo "=== Phase-RS 1v1 Ablation (Standard) ==="
	@mkdir -p $(PHASE_RS_ROOT)/1v1_standard
	$(PY) scripts/phase_rs_rollout_sweep.py \
	    --pickers random prefer-nonpass heuristic agent:heuristic agent:world_model \
	    --difficulties VeryEasy Easy Medium Hard VeryHard \
	    --ai-decks "Red Deck Wins" "Azorius Control" "Green Stompy" \
	    --games-per-cell 5 \
	    --autostart \
	    --stream-timeout 30 \
	    --max-retries 2 \
	    --output-dir $(PHASE_RS_ROOT)/1v1_standard
	@echo "Output: $(PHASE_RS_ROOT)/1v1_standard/"

# Phase-RS Commander pod (multiplayer format)
phase-rs-commander:
	@echo "=== Phase-RS Commander Pod (4-player) ==="
	@mkdir -p $(PHASE_RS_ROOT)/commander_pod
	$(PY) scripts/phase_rs_rollout_sweep.py \
	    --pickers random heuristic agent:heuristic \
	    --difficulties Medium Hard \
	    --format commander \
	    --ai-decks $(POD_DECKS) \
	    --games-per-cell 3 \
	    --autostart \
	    --stream-timeout 45 \
	    --max-retries 2 \
	    --output-dir $(PHASE_RS_ROOT)/commander_pod
	@echo "Output: $(PHASE_RS_ROOT)/commander_pod/"

# Phase-RS all formats (cartesian sweep across Standard, Pioneer, Modern, Commander)
phase-rs-all-formats:
	@echo "=== Phase-RS All Formats (Comprehensive Sweep) ==="
	@mkdir -p $(PHASE_RS_ROOT)/all_formats
	@echo "[1/4] Standard format..."
	$(PY) scripts/phase_rs_rollout_sweep.py \
	    --pickers random heuristic agent:heuristic \
	    --difficulties Easy Medium Hard \
	    --format standard \
	    --ai-decks "Red Deck Wins" "Azorius Control" \
	    --games-per-cell 3 \
	    --autostart \
	    --output-dir $(PHASE_RS_ROOT)/all_formats/standard
	@echo "[2/4] Pioneer format..."
	$(PY) scripts/phase_rs_rollout_sweep.py \
	    --pickers random heuristic agent:heuristic \
	    --difficulties Easy Medium Hard \
	    --format pioneer \
	    --ai-decks "Red Deck Wins" "Azorius Control" \
	    --games-per-cell 3 \
	    --autostart \
	    --output-dir $(PHASE_RS_ROOT)/all_formats/pioneer
	@echo "[3/4] Modern format..."
	$(PY) scripts/phase_rs_rollout_sweep.py \
	    --pickers random heuristic agent:heuristic \
	    --difficulties Easy Medium Hard \
	    --format modern \
	    --ai-decks "Red Deck Wins" "Azorius Control" \
	    --games-per-cell 3 \
	    --autostart \
	    --output-dir $(PHASE_RS_ROOT)/all_formats/modern
	@echo "[4/4] Commander format..."
	$(PY) scripts/phase_rs_rollout_sweep.py \
	    --pickers random heuristic agent:heuristic \
	    --difficulties Easy Medium \
	    --format commander \
	    --games-per-cell 3 \
	    --autostart \
	    --output-dir $(PHASE_RS_ROOT)/all_formats/commander
	@echo "Output: $(PHASE_RS_ROOT)/all_formats/"

# Collect JSONL traces for training (Phase-RS traces → JEPA training data)
phase-rs-traces:
	@echo "=== Collecting Phase-RS Traces (N=$(N) games) ==="
	@mkdir -p $(PHASE_RS_ROOT)/traces
	$(PY) scripts/collect_phase_rs_traces.py \
	    --games $(N) \
	    --picker agent:heuristic \
	    --ai-difficulty Medium \
	    --autostart \
	    --output-dir $(PHASE_RS_ROOT)/traces
	@echo "Output: $(PHASE_RS_ROOT)/traces/"
	@echo "Next: make phase-rs-training"

# Full phase-rs training pipeline (traces → JEPA → evaluation)
phase-rs-training:
	@echo "=== Phase-RS Training Pipeline (Stage 4.1 → 5 → 6) ==="
	@mkdir -p $(PHASE_RS_ROOT)/training
	$(PY) -u scripts/train_pipeline.py \
	    --stage 1 --end-stage 7 \
	    --phase-rs-traces \
	    --num-games 64 \
	    --phase-rs-picker heuristic \
	    --phase-rs-difficulty Medium \
	    --jepa-epochs 40 \
	    --skip-dream \
	    > $(PHASE_RS_ROOT)/training/train.log 2>&1
	@echo "Output: $(PHASE_RS_ROOT)/training/"
	@echo "Checkpoint: checkpoints/jepa/jepa_final.pt"
