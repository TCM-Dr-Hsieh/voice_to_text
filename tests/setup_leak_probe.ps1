# Used by tests/test_install_tools.py. ASCII only on purpose (the test reads the PROBE| lines).
#
# A PowerShell function RETURNS everything it writes to the pipeline. setup.ps1's Install-Python312 once let winget's
# output ride along with the path it returned, so $python became an array and the next step failed with
# "cannot recognize 'The `msstore` source requires ...'" -- only on a computer that had to install Python first.
#
# This probe loads the real functions out of setup.ps1 (through the parser; the installer itself is never run), gives
# them native commands that print noise, and reports what each function returns.
param([Parameter(Mandatory)][string]$Setup)
$ErrorActionPreference = 'Stop'
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Setup, [ref]$null, [ref]$errors)
if ($errors) { throw "setup.ps1 does not parse: $errors" }
$wanted = 'Write-Note', 'Write-Ok', 'Invoke-Checked', 'Invoke-Quiet', 'Ensure-Venv', 'Get-VenvHome', 'Install-Python312', 'Test-Python312'
foreach ($f in $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
    if ($f.Name -in $wanted) { . ([scriptblock]::Create($f.Extent.Text)) }
}
$Yes = [switch]$true
$Force = [switch]$false

function Report($name, $value) {
    $items = @($value)
    $type = if ($null -eq $value) { 'null' } else { $value.GetType().Name }
    Write-Host ('PROBE|{0}|{1}|{2}|{3}' -f $name, $items.Count, $type, ($items -join ' // '))
}

# Install-Python312, with a winget that prints (as the real one does) -- once succeeding, once "already installed" (exit 1).
function Find-Python312 { 'C:\fake\Python312\python.exe' }
function winget { 'The msstore source requires that you view the following agreements'; 'Successfully installed'; $global:LASTEXITCODE = 0 }
Report 'install-python' (Install-Python312)
function winget { 'Found an existing package already installed'; $global:LASTEXITCODE = 1 }
Report 'install-python-already-installed' (Install-Python312)

# Ensure-Venv, with a python whose `-m venv` prints something (a warning, a banner, ...).
$tmp = Join-Path ([IO.Path]::GetTempPath()) ('setup-probe-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    $fake = Join-Path $tmp 'fakepython.cmd'
    Set-Content -LiteralPath $fake -Encoding ASCII -Value @'
@echo off
echo some noisy output from python -m venv
echo and a second line
exit /b 0
'@
    Report 'ensure-venv' (Ensure-Venv (Join-Path $tmp '.venv') $fake $false)
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
