<#
.SYNOPSIS
  Weekend ablation campaign — Streams A (matchups), B (training + benchmark), C (LLM sweep).

.DESCRIPTION
  Orchestrates the long-running ablation suite that produces the data
  for the paper's results section. Designed to run unattended for ~24h.

  Layout under runs/weekend/:
    A1_matrix_1v1/     — round-robin among 7 agents on 2 standard decks
    A2_pod_edh/        — 4-player EDH pod with mixed agents
    A3_seeds/seed_*/   — 5-seed robustness on the top-4 1v1 agents
    B_train/{full,no_kg,no_kl,small}/ — 4 JEPA training variants
    B_eval/            — benchmark trained checkpoints vs baselines
    C1_llm_models/<m>/ — LLM-agent matchups across small Ollama models

  Each cell is independent: a failure in one does NOT abort the rest.
  Per-cell stdout+stderr is teed to a .log file alongside the artefacts.

.PARAMETER Stream
  Which stream(s) to run. One of: A, B, C, AB, AC, BC, ABC. Default ABC.

.PARAMETER SkipTrajectories
  Skip the pre-Stream-B trajectory regeneration (assumes data/trajectories/
  is already populated with a B2-aware self-play run).

.PARAMETER DryRun
  Print the commands without running them.

.EXAMPLE
  pwsh scripts/weekend_campaign.ps1
  pwsh scripts/weekend_campaign.ps1 -Stream A
  pwsh scripts/weekend_campaign.ps1 -Stream BC -SkipTrajectories
#>
[CmdletBinding()]
param(
    [ValidateSet("A","B","C","AB","AC","BC","ABC")]
    [string]$Stream = "ABC",
    [switch]$SkipTrajectories,
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$root      = Split-Path -Parent $PSScriptRoot
$python    = Join-Path $root ".venv\Scripts\python.exe"
$outRoot   = Join-Path $root "runs\weekend"
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

# -------- shared deck pools --------------------------------------------------
$standardDecks = @(
    "data/decks/modern/modern_mono_red_burn.txt",
    "data/decks/modern/modern_azorius_control.txt"
)
$edhDecks = @(
    "data/decks/edh/krenko-mob-boss_core.txt",
    "data/decks/edh/atraxa-praetors-voice_core.txt",
    "data/decks/edh/urza-lord-high-artificer_core.txt",
    "data/decks/edh/meren-of-clan-nel-toth_core.txt"
)

# -------- helpers ------------------------------------------------------------
function Invoke-Cell {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [string[]]$Args,
        [Parameter(Mandatory)] [string]$LogPath,
        [hashtable]$Env = @{}
    )
    $banner = "=" * 70
    Write-Host ""
    Write-Host $banner -ForegroundColor Cyan
    Write-Host "[$(Get-Date -Format o)] CELL: $Name" -ForegroundColor Cyan
    Write-Host "  cmd : $python $($Args -join ' ')" -ForegroundColor DarkGray
    Write-Host "  log : $LogPath" -ForegroundColor DarkGray
    Write-Host $banner -ForegroundColor Cyan

    if ($DryRun) { return 0 }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null

    $prevEnv = @{}
    foreach ($k in $Env.Keys) {
        $prevEnv[$k] = [Environment]::GetEnvironmentVariable($k, "Process")
        [Environment]::SetEnvironmentVariable($k, $Env[$k], "Process")
    }

    $sw = [Diagnostics.Stopwatch]::StartNew()
    try {
        & $python @Args 2>&1 | Tee-Object -FilePath $LogPath
        $code = $LASTEXITCODE
    } finally {
        foreach ($k in $prevEnv.Keys) {
            [Environment]::SetEnvironmentVariable($k, $prevEnv[$k], "Process")
        }
        $sw.Stop()
    }
    $color = if ($code -eq 0) { "Green" } else { "Yellow" }
    Write-Host "[$(Get-Date -Format o)] CELL DONE: $Name  exit=$code  elapsed=$([int]$sw.Elapsed.TotalSeconds)s" -ForegroundColor $color
    return $code
}

# -------- pre-flight: regenerate trajectories with B2 fix --------------------
if (($Stream -match "B") -and -not $SkipTrajectories) {
    $trajLog = Join-Path $outRoot "00_trajectories.log"
    Invoke-Cell -Name "regenerate trajectories (stage 4 only)" `
        -Args @("scripts/train_pipeline.py", "--stage", "4", "--end-stage", "4",
                "--num-games", "120", "--max-turns", "40") `
        -LogPath $trajLog | Out-Null
}

# =====================================================================
# STREAM A — matchups (no training)
# =====================================================================
if ($Stream -match "A") {

    # A1: 1v1 round robin, 7 agents x 2 decks x 8 games each pair
    $a1Out = Join-Path $outRoot "A1_matrix_1v1"
    $a1Args = @("scripts/run_matchups.py",
                "--format", "standard",
                "--decks") + $standardDecks + @(
                "--agents", "random", "heuristic", "kg_heuristic",
                            "world_model", "active_inference",
                            "llm_fusion", "ollama",
                "--games", "8",
                "--max-turns", "60",
                "--seed", "20260428",
                "--out", $a1Out)
    Invoke-Cell -Name "A1 matrix 1v1" -Args $a1Args `
        -LogPath (Join-Path $a1Out "run.log") | Out-Null

    # A2: 4-player EDH pod
    $a2Out = Join-Path $outRoot "A2_pod_edh"
    $a2Args = @("scripts/run_matchups.py",
                "--format", "commander", "--pod",
                "--decks") + $edhDecks + @(
                "--agents", "heuristic", "kg_heuristic",
                            "world_model", "llm_fusion",
                "--games", "6",
                "--max-turns", "120",
                "--seed", "20260428",
                "--out", $a2Out)
    Invoke-Cell -Name "A2 EDH pod" -Args $a2Args `
        -LogPath (Join-Path $a2Out "run.log") | Out-Null

    # A3: 5-seed robustness on top-4 1v1 agents
    $a3Seeds = @(11, 23, 47, 101, 20260428)
    foreach ($s in $a3Seeds) {
        $a3Out = Join-Path $outRoot "A3_seeds\seed_$s"
        $a3Args = @("scripts/run_matchups.py",
                    "--format", "standard",
                    "--decks") + $standardDecks + @(
                    "--agents", "heuristic", "kg_heuristic",
                                "world_model", "active_inference",
                    "--games", "6",
                    "--max-turns", "60",
                    "--seed", "$s",
                    "--out", $a3Out)
        Invoke-Cell -Name "A3 seed=$s" -Args $a3Args `
            -LogPath (Join-Path $a3Out "run.log") | Out-Null
    }
}

# =====================================================================
# STREAM B — JEPA training variants + benchmark
# =====================================================================
if ($Stream -match "B") {
    $variants = @(
        @{ name = "full";   args = @("--jepa-beta", "1.0", "--free-bits", "0.5") },
        @{ name = "no_kg";  args = @("--jepa-beta", "1.0", "--free-bits", "0.5", "--no-kg") },
        @{ name = "no_kl";  args = @("--jepa-beta", "0.0", "--free-bits", "0.0") },
        @{ name = "small";  args = @("--jepa-beta", "1.0", "--free-bits", "0.5", "--kg-embed-dim", "64") }
    )
    $checkpoints = @()
    foreach ($v in $variants) {
        $vOut  = Join-Path $outRoot ("B_train\" + $v.name)
        $ckpt  = Join-Path $vOut "model.pt"
        $checkpoints += $ckpt
        $argsList = @("scripts/train_stable_worldmodel.py",
                      "--trajectories", "data/trajectories",
                      "--epochs", "30",
                      "--batch-size", "64",
                      "--device", "cuda",
                      "--checkpoint", $ckpt,
                      "--output-dir", $vOut) + $v.args
        Invoke-Cell -Name "B train $($v.name)" -Args $argsList `
            -LogPath (Join-Path $vOut "train.log") | Out-Null
    }

    # B_eval: benchmark all four checkpoints vs baselines
    $bEvalOut = Join-Path $outRoot "B_eval"
    $bEvalArgs = @("scripts/benchmark_trained_agents.py",
                "--checkpoints") + $checkpoints + @(
                "--baselines", "heuristic", "random", "llm", "active_inference",
                "--decks") + $standardDecks + @(
                "--games", "8",
                "--max-turns", "60",
                "--seed", "20260428",
                "--out", $bEvalOut)
    Invoke-Cell -Name "B eval (benchmark trained checkpoints)" -Args $bEvalArgs `
        -LogPath (Join-Path $bEvalOut "run.log") | Out-Null
}

# =====================================================================
# STREAM C — LLM model sweep
# =====================================================================
if ($Stream -match "C") {
    $models = @("llama3.2:1b", "qwen2.5-coder:1.5b", "gemma3:4b")
    foreach ($m in $models) {
        $safeName = ($m -replace "[:/\\]", "_")
        $cOut = Join-Path $outRoot ("C1_llm_models\" + $safeName)
        $cArgs = @("scripts/run_matchups.py",
                    "--format", "standard",
                    "--decks") + $standardDecks + @(
                    "--agents", "ollama", "heuristic",
                    "--games", "4",
                    "--max-turns", "50",
                    "--seed", "20260428",
                    "--out", $cOut)
        Invoke-Cell -Name "C1 llm=$m" -Args $cArgs `
            -LogPath (Join-Path $cOut "run.log") `
            -Env @{ OLLAMA_MODEL = $m } | Out-Null
    }
}

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Green
Write-Host "[$(Get-Date -Format o)] WEEKEND CAMPAIGN COMPLETE" -ForegroundColor Green
Write-Host "Artefacts under: $outRoot" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Green
