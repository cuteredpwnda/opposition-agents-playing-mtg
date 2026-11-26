$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$paper = Join-Path $root "paper"
$logs = Join-Path $paper "build"
$compiler = (Get-Command latexmk -ErrorAction Stop).Source
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = (Get-Command python -ErrorAction Stop).Source
}
$validator = Join-Path $root "scripts\validate_paper_references.py"
& $python $validator --check-database
if ($LASTEXITCODE -ne 0) { throw "Paper citation source validation failed" }
New-Item -ItemType Directory -Force -Path $logs | Out-Null

Push-Location $paper
try {
    foreach ($document in @("fig_stack", "mtg_ontology", "opposition_agents_mtg", "agents_tech_report")) {
        $log = Join-Path $logs "$document.build.log"
        $errorLog = Join-Path $logs "$document.build.err"
        $process = Start-Process -FilePath $compiler -ArgumentList @(
            "-pdf", "-interaction=nonstopmode", "-halt-on-error", "$document.tex"
        ) -RedirectStandardOutput $log -RedirectStandardError $errorLog -Wait -PassThru
        if ($process.ExitCode -ne 0) {
            Get-Content -LiteralPath $log -Tail 60
            Get-Content -LiteralPath $errorLog -Tail 20
            throw "LaTeX build failed for $document; see $log"
        }
        Write-Output "Compiled: $(Join-Path $paper "$document.pdf")"
    }
} finally {
    Pop-Location
}
& $python $validator --check-build
if ($LASTEXITCODE -ne 0) { throw "Paper reference build validation failed" }
