# run.ps1 - Windows PowerShell version of run.sh
# Usage: .\run.ps1 <raw_dir> <out_dir>

param(
    [Parameter(Mandatory=$true)][string]$RawDir,
    [Parameter(Mandatory=$true)][string]$OutDir
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "============================================"
Write-Host "Flam Corpus Pipeline"
Write-Host "Date: $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ')"
Write-Host "Input:  $RawDir"
Write-Host "Output: $OutDir"
Write-Host "============================================"

# Step 0: Install dependencies
Write-Host ""
Write-Host "[Step 0] Installing dependencies..."
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
python -m pip install -q -r "$ScriptDir\requirements.txt" *>$null
$ErrorActionPreference = $prevEAP

# Step 1: Run cleaning pipeline
Write-Host ""
Write-Host "[Step 1] Running cleaning pipeline..."
$env:PYTHONPATH = $ScriptDir
python -m src.pipeline $RawDir $OutDir

# Step 2: Build eval set
Write-Host ""
Write-Host "[Step 2] Building eval set..."
python -m src.eval_builder $OutDir

# Step 3: Run contamination check
Write-Host ""
Write-Host "[Step 3] Checking for contamination..."
python -m src.contamination $OutDir

# Step 4: Generate sample
Write-Host ""
Write-Host "[Step 4] Generating 1000-row sample..."
python -m src.sample $OutDir

Write-Host ""
Write-Host "============================================"
Write-Host "Pipeline complete!"
Write-Host "============================================"
Write-Host ""
Write-Host "Output:"
Write-Host "  Clean corpus:     $OutDir\clean\"
Write-Host "  Ledger:           $OutDir\ledger.jsonl"
Write-Host "  Eval set:         $OutDir\eval.jsonl"
Write-Host "  Contamination:    $OutDir\contamination.jsonl"
Write-Host "  Manifest:         $OutDir\manifest.json"
Write-Host "  Sample:           $OutDir\sample_1000.jsonl"
