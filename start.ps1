$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$voicePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $voicePython)) {
    Write-Host 'The Python environment .venv was not found. Run install.cmd (or .\setup.ps1) first; see README.md.'
    exit 1
}
& $voicePython app.py --open
