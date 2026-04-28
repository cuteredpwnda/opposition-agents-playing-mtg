@echo off
REM ============================================================================
REM run_overnight_benchmarks.bat
REM
REM Single-shot dispatcher that:
REM   1. Trains the world model in three configurations
REM        A_small   :  20 epochs, batch  8, no-KG  (fast baseline)
REM        B_medium  :  60 epochs, batch 16, no-KG  (default)
REM        C_kg      :  60 epochs, batch 16, KG ON  (full)
REM      Each writes its own checkpoints/jepa/jepa_final.pt and we copy
REM      it under runs\overnight_<TS>\wm\<name>\jepa_final.pt for benchmarking.
REM   2. Runs four 1v1 ablations (LLM / WM / KG / fusion) on Modern.
REM   3. Compares the three WM configs head-to-head.
REM   4. Runs an EDH 4-player pod ablation.
REM   5. Runs Swiss + pod tournaments.
REM
REM Per-stage logs land in runs\overnight_<TS>\<stage>\.  Bails on first
REM non-zero errorlevel.
REM
REM Usage:
REM   scripts\run_overnight_benchmarks.bat
REM   scripts\run_overnight_benchmarks.bat --skip-training
REM   scripts\run_overnight_benchmarks.bat --quick     (smaller params for smoke)
REM ============================================================================

setlocal ENABLEDELAYEDEXPANSION
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

REM Build a filesystem-safe timestamp YYYYMMDD_HHMMSS
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set _DT=%%I
set TS=%_DT:~0,8%_%_DT:~8,6%
set ROOT=runs\overnight_%TS%

if not exist runs mkdir runs
mkdir %ROOT%
mkdir %ROOT%\wm 2>nul

set PY=.\.venv\Scripts\python.exe
if not exist %PY% (
    echo ERROR: %PY% not found.  Activate or create the venv first.
    exit /b 1
)

set DECK_BURN=data\decks\modern\modern_mono_red_burn.txt
set DECK_CONTROL=data\decks\modern\modern_azorius_control.txt
set POD_DECKS=data\decks\edh\krenko-mob-boss_core.txt data\decks\edh\atraxa-praetors-voice_core.txt data\decks\edh\urza-lord-high-artificer_core.txt data\decks\edh\meren-of-clan-nel-toth_core.txt

set GAMES_PER_PAIRING=8
set MATCH_TURNS=100
set POD_GAMES=4
set POD_TURNS=150
set TOURNEY_ROUNDS=6
set POD_TOURNEY_ROUNDS=8

set TRAIN_GAMES_A=30
set TRAIN_EPOCHS_A=20
set TRAIN_BATCH_A=8

set TRAIN_GAMES_B=80
set TRAIN_EPOCHS_B=60
set TRAIN_BATCH_B=16

set TRAIN_GAMES_C=80
set TRAIN_EPOCHS_C=60
set TRAIN_BATCH_C=16

REM Quick mode shrinks every knob so you can dry-run the dispatcher.
echo %* | findstr /C:"--quick" >nul
if not errorlevel 1 (
    echo *** QUICK MODE: tiny parameters ***
    set GAMES_PER_PAIRING=2
    set MATCH_TURNS=20
    set POD_GAMES=2
    set POD_TURNS=20
    set TOURNEY_ROUNDS=2
    set POD_TOURNEY_ROUNDS=2
    set TRAIN_GAMES_A=4
    set TRAIN_EPOCHS_A=2
    set TRAIN_BATCH_A=2
    set TRAIN_GAMES_B=4
    set TRAIN_EPOCHS_B=2
    set TRAIN_BATCH_B=2
    set TRAIN_GAMES_C=4
    set TRAIN_EPOCHS_C=2
    set TRAIN_BATCH_C=2
)

set CKPT_A=%ROOT%\wm\A_small\jepa_final.pt
set CKPT_B=%ROOT%\wm\B_medium\jepa_final.pt
set CKPT_C=%ROOT%\wm\C_kg\jepa_final.pt

REM --------------------------------------------------------------------- TRAIN
echo %* | findstr /C:"--skip-training" >nul
if errorlevel 1 (
    echo === [1/6] WM training A (small, no-KG) ====================================
    mkdir %ROOT%\wm\A_small 2>nul
    %PY% -u scripts\train_pipeline.py --stage 1 --end-stage 5 ^
         --num-games %TRAIN_GAMES_A% --jepa-epochs %TRAIN_EPOCHS_A% ^
         --jepa-batch-size %TRAIN_BATCH_A% --no-kg --skip-eval --skip-dream ^
         > %ROOT%\wm\A_small\train.log 2>&1
    if errorlevel 1 ( echo TRAIN A FAILED, see %ROOT%\wm\A_small\train.log & exit /b 1 )
    if exist checkpoints\jepa\jepa_final.pt copy /Y checkpoints\jepa\jepa_final.pt %CKPT_A% >nul

    echo === [2/6] WM training B (medium, no-KG) ===================================
    mkdir %ROOT%\wm\B_medium 2>nul
    %PY% -u scripts\train_pipeline.py --stage 1 --end-stage 5 ^
         --num-games %TRAIN_GAMES_B% --jepa-epochs %TRAIN_EPOCHS_B% ^
         --jepa-batch-size %TRAIN_BATCH_B% --no-kg --skip-eval --skip-dream ^
         > %ROOT%\wm\B_medium\train.log 2>&1
    if errorlevel 1 ( echo TRAIN B FAILED, see %ROOT%\wm\B_medium\train.log & exit /b 1 )
    if exist checkpoints\jepa\jepa_final.pt copy /Y checkpoints\jepa\jepa_final.pt %CKPT_B% >nul

    echo === [3/6] WM training C (medium + KG) =====================================
    mkdir %ROOT%\wm\C_kg 2>nul
    %PY% -u scripts\train_pipeline.py --stage 1 --end-stage 5 ^
         --num-games %TRAIN_GAMES_C% --jepa-epochs %TRAIN_EPOCHS_C% ^
         --jepa-batch-size %TRAIN_BATCH_C% --skip-eval --skip-dream ^
         > %ROOT%\wm\C_kg\train.log 2>&1
    if errorlevel 1 ( echo TRAIN C FAILED, see %ROOT%\wm\C_kg\train.log & exit /b 1 )
    if exist checkpoints\jepa\jepa_final.pt copy /Y checkpoints\jepa\jepa_final.pt %CKPT_C% >nul
) else (
    echo === [1-3/6] Skipping training, using checkpoints\jepa\jepa_final.pt =======
    set CKPT_A=checkpoints\jepa\jepa_final.pt
    set CKPT_B=checkpoints\jepa\jepa_final.pt
    set CKPT_C=checkpoints\jepa\jepa_final.pt
)

REM ----------------------------------------------------------------- ABLATIONS
echo === [4/6] 1v1 ablations (LLM / WM / KG / fusion) on default WM ============
%PY% scripts\run_matchups.py --format standard ^
    --decks %DECK_BURN% %DECK_CONTROL% ^
    --agents llm heuristic ^
    --games %GAMES_PER_PAIRING% --max-turns %MATCH_TURNS% --seed 7 ^
    --out %ROOT%\ablation\A_llm_only
if errorlevel 1 exit /b 1

%PY% scripts\run_matchups.py --format standard ^
    --decks %DECK_BURN% %DECK_CONTROL% ^
    --agents world_model heuristic ^
    --games %GAMES_PER_PAIRING% --max-turns %MATCH_TURNS% --seed 7 ^
    --wm-checkpoint %CKPT_B% ^
    --out %ROOT%\ablation\B_wm_only
if errorlevel 1 exit /b 1

%PY% scripts\run_matchups.py --format standard ^
    --decks %DECK_BURN% %DECK_CONTROL% ^
    --agents kg_heuristic heuristic ^
    --games %GAMES_PER_PAIRING% --max-turns %MATCH_TURNS% --seed 7 ^
    --out %ROOT%\ablation\C_kg_only
if errorlevel 1 exit /b 1

%PY% scripts\run_matchups.py --format standard ^
    --decks %DECK_BURN% %DECK_CONTROL% ^
    --agents llm_fusion heuristic ^
    --games %GAMES_PER_PAIRING% --max-turns %MATCH_TURNS% --seed 7 ^
    --wm-checkpoint %CKPT_B% ^
    --out %ROOT%\ablation\D_fusion
if errorlevel 1 exit /b 1

REM ------------------------------------------------- WM CONFIG HEAD-TO-HEAD
echo === [5/6] World-model config head-to-head =================================
%PY% scripts\run_matchups.py --format standard ^
    --decks %DECK_BURN% %DECK_CONTROL% ^
    --agents world_model heuristic ^
    --games %GAMES_PER_PAIRING% --max-turns %MATCH_TURNS% --seed 7 ^
    --wm-checkpoint %CKPT_A% ^
    --out %ROOT%\wm_compare\A_small
if errorlevel 1 exit /b 1

%PY% scripts\run_matchups.py --format standard ^
    --decks %DECK_BURN% %DECK_CONTROL% ^
    --agents world_model heuristic ^
    --games %GAMES_PER_PAIRING% --max-turns %MATCH_TURNS% --seed 7 ^
    --wm-checkpoint %CKPT_B% ^
    --out %ROOT%\wm_compare\B_medium
if errorlevel 1 exit /b 1

%PY% scripts\run_matchups.py --format standard ^
    --decks %DECK_BURN% %DECK_CONTROL% ^
    --agents world_model heuristic ^
    --games %GAMES_PER_PAIRING% --max-turns %MATCH_TURNS% --seed 7 ^
    --wm-checkpoint %CKPT_C% ^
    --out %ROOT%\wm_compare\C_kg
if errorlevel 1 exit /b 1

REM ----------------------------------------------------------------- TOURNEYS
echo === [6/6] Pod ablation + tournaments ======================================
%PY% scripts\run_matchups.py --format commander --pod ^
    --decks %POD_DECKS% ^
    --agents random heuristic kg_heuristic llm_fusion ^
    --games %POD_GAMES% --max-turns %POD_TURNS% --seed 7 ^
    --wm-checkpoint %CKPT_B% ^
    --out %ROOT%\pod_ablation
if errorlevel 1 exit /b 1

%PY% scripts\run_tournament.py --mode swiss ^
    --format standard ^
    --decks %DECK_BURN% %DECK_CONTROL% ^
    --agents random heuristic kg_heuristic active_inference world_model llm llm_fusion ^
    --rounds %TOURNEY_ROUNDS% --max-turns %MATCH_TURNS% --seed 7 ^
    --out %ROOT%\tournament_swiss
if errorlevel 1 exit /b 1

%PY% scripts\run_tournament.py --mode pod ^
    --format commander ^
    --decks %POD_DECKS% ^
    --agents random heuristic kg_heuristic llm_fusion ^
    --rounds %POD_TOURNEY_ROUNDS% --max-turns %POD_TURNS% --seed 7 ^
    --out %ROOT%\tournament_pod
if errorlevel 1 exit /b 1

echo.
echo ============================================================================
echo  ALL BENCHMARKS COMPLETED.  Output under %ROOT%
echo  Read %ROOT%\wm_compare\*\summary.json to compare WM configurations.
echo ============================================================================
endlocal
