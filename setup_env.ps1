$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param([string]$Description, [scriptblock]$Command)
    Write-Host ">> $Description"
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Description (exit code $LASTEXITCODE)" }
}

Invoke-Checked "Check Python 3.11 is installed" { py -3.11 --version }

if (-not (Test-Path ".venv")) {
    Invoke-Checked "Create .venv with Python 3.11" { py -3.11 -m venv .venv }
} else {
    Write-Host ">> .venv already exists, reusing it"
}

$python = Join-Path (Get-Location) ".venv\Scripts\python.exe"

Invoke-Checked "Upgrade pip" { & $python -m pip install --upgrade pip }
Invoke-Checked "Install pinned requirements" { & $python -m pip install -r requirements-laptop.txt }
Invoke-Checked "pip check (no dependency conflicts)" { & $python -m pip check }
Invoke-Checked "Environment report" { & $python -m src.Project_Setup check }

Write-Host ""
Write-Host "Done. Activate the environment with:  .\.venv\Scripts\Activate.ps1"
