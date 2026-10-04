<#
.SYNOPSIS
  語音書寫 一鍵安裝：全新的 Windows 電腦也能用，不依賴 conda 等既有 Python 環境。

.DESCRIPTION
  依序完成：
    1. 預檢（作業系統、磁碟空間、NVIDIA 顯卡與驅動、網路）
    2. 找到 Python 3.12（找不到時，經你同意用 winget 安裝）
    3. 建立 UI 用的 .venv 並安裝套件
    4. 建立獨立的 ASR 用 .venv-asr（含 CUDA 版 PyTorch，約 4 GB）
    5. 從 Hugging Face 下載 Qwen3-ASR-1.7B 與 Qwen3-ForcedAligner-0.6B（約 6.1 GB，固定版本）
    6. 寫入 data\settings.json
    7. 健檢（含實際載入 ASR 模型辨識一次）
  可重複執行：已完成的步驟會略過，下載可續傳。記錄寫在 setup.log。

.PARAMETER BasePython   指定 Python 3.12 的 python.exe（預設自動尋找）
.PARAMETER ModelsDir    模型存放資料夾（預設 <專案>\models）
.PARAMETER LlmUrl       校稿 LLM 伺服器網址（例 http://192.168.1.10:8080/v1）
.PARAMETER LlmModel     校稿 LLM 模型名稱
.PARAMETER LlmKey       校稿 LLM API Key
.PARAMETER HfEndpoint   Hugging Face 鏡像網址（網路受限時使用）
.PARAMETER SkipAsr      不安裝 ASR 環境與模型（只裝 UI）
.PARAMETER SkipModels   不下載模型
.PARAMETER SkipDoctor   不做最後的健檢
.PARAMETER SkipSmoke    健檢時不實際載入 ASR 模型
.PARAMETER SkipGpuCheck 不檢查 NVIDIA 顯卡與驅動
.PARAMETER Dev          另外安裝開發用套件（pytest）
.PARAMETER Force        重建 .venv 與 .venv-asr
.PARAMETER Yes          不詢問，自動同意（例如用 winget 安裝 Python）
#>
[CmdletBinding()]
param(
    [string]$BasePython = '',
    [string]$ModelsDir = '',
    [string]$LlmUrl = '',
    [string]$LlmModel = '',
    [string]$LlmKey = '',
    [string]$HfEndpoint = '',
    [switch]$SkipAsr,
    [switch]$SkipModels,
    [switch]$SkipDoctor,
    [switch]$SkipSmoke,
    [switch]$SkipGpuCheck,
    [switch]$Dev,
    [switch]$Force,
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}
$TotalSteps = 7
$MinDriverMajor = 570            # CUDA 12.8 (the pinned PyTorch build) needs a Windows driver of about 570 or newer
$NeedDiskGb = 20

function Write-Step([int]$n, [string]$text) {
    Write-Host ''
    Write-Host "==> [$n/$TotalSteps] $text" -ForegroundColor Cyan
}
function Write-Ok([string]$text)   { Write-Host "    [OK]   $text" -ForegroundColor Green }
function Write-Note([string]$text) { Write-Host "    $text" }
function Write-Warn([string]$text) { Write-Host "    [注意] $text" -ForegroundColor Yellow }

function Invoke-Checked([string]$what, [string]$exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$what 失敗（結束碼 $LASTEXITCODE）。" }
}

function Test-Reachable([string]$url) {
    try {
        Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing -TimeoutSec 20 | Out-Null
        return $true
    } catch {
        # An HTTP error status (403/404/405) still proves the host is reachable.
        if ($_.Exception.Response) { return $true }
        return $false
    }
}

function Get-FreeGb([string]$path) {
    $drive = [System.IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $path).Path)
    if ($drive.StartsWith('\\')) { return [double]::MaxValue }      # network share: cannot tell, do not block
    return (New-Object System.IO.DriveInfo $drive).AvailableFreeSpace / 1GB
}

function Get-NvidiaSmi {
    $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $fallback = Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'
    if (Test-Path -LiteralPath $fallback) { return $fallback }
    return $null
}

function Invoke-Quiet([string]$exe, [string[]]$arguments) {
    # Run a native probe without letting stderr output turn into a terminating error (Windows PowerShell 5.1).
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $lines = @(& $exe @arguments 2>&1 | ForEach-Object { "$_" })
        return [pscustomobject]@{ Code = $LASTEXITCODE; Lines = $lines }
    } catch {
        return [pscustomobject]@{ Code = -1; Lines = @() }
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Test-Python312([string]$exe) {
    if ([string]::IsNullOrWhiteSpace($exe) -or -not (Test-Path -LiteralPath $exe)) { return $false }
    $r = Invoke-Quiet $exe @('-c', "import sys; print('%d.%d' % sys.version_info[:2]); print(sys.maxsize > 2**32)")
    return ($r.Code -eq 0 -and $r.Lines.Count -ge 2 -and $r.Lines[0] -eq '3.12' -and $r.Lines[1] -eq 'True')
}

function Find-Python312 {
    $candidates = New-Object System.Collections.Generic.List[string]
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $r = Invoke-Quiet 'py' @('-3.12', '-c', 'import sys; print(sys.executable)')
        if ($r.Code -eq 0 -and $r.Lines.Count -ge 1) { $candidates.Add($r.Lines[0]) }
    }
    foreach ($name in 'python', 'python3') {
        $c = Get-Command $name -ErrorAction SilentlyContinue
        if ($c -and $c.Source) { $candidates.Add($c.Source) }
    }
    $candidates.Add((Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'))
    $candidates.Add((Join-Path $env:ProgramFiles 'Python312\python.exe'))
    foreach ($c in $candidates) { if (Test-Python312 $c) { return $c } }
    return $null
}

function Install-Python312 {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw ("找不到 Python 3.12，這台電腦也沒有 winget。請先從 https://www.python.org/downloads/ 安裝 " +
               "Python 3.12（64 位元），再重新執行本腳本；或用 -BasePython 指定 python.exe。")
    }
    if (-not $Yes) {
        $answer = Read-Host '    找不到 Python 3.12。要用 winget 安裝（只安裝給目前使用者，不需系統管理員）嗎？ [Y/n]'
        if ($answer -match '^(n|no)$') { throw '已取消：需要 Python 3.12 才能繼續。' }
    }
    Write-Note '用 winget 安裝 Python 3.12 …'
    & winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
    # winget returns a non-zero code when it is already installed; what matters is whether we can find it now.
    $found = Find-Python312
    if (-not $found) {
        throw 'Python 3.12 安裝後仍找不到。請關閉這個視窗、重新開啟後再執行 setup.ps1。'
    }
    return $found
}

function Get-VenvHome([string]$venvDir) {
    $cfg = Join-Path $venvDir 'pyvenv.cfg'
    if (-not (Test-Path -LiteralPath $cfg)) { return $null }
    foreach ($line in Get-Content -LiteralPath $cfg) {
        if ($line -match '^\s*home\s*=\s*(.+?)\s*$') { return $Matches[1] }
    }
    return $null
}

function Ensure-Venv([string]$dir, [string]$python, [bool]$mustBeStandalone) {
    $venvPy = Join-Path $dir 'Scripts\python.exe'
    if (Test-Path -LiteralPath $dir) {
        $why = $null
        if ($Force) { $why = '指定了 -Force' }
        else {
            $works = $false
            if (Test-Path -LiteralPath $venvPy) { $works = ((Invoke-Quiet $venvPy @('-c', 'import sys')).Code -eq 0) }
            $cfg = Join-Path $dir 'pyvenv.cfg'
            $baseHome = Get-VenvHome $dir
            $wantHome = Split-Path -Parent $python
            if (-not $works) { $why = '環境已損毀，或它的基底 Python 已不存在' }
            elseif ($mustBeStandalone -and (Select-String -LiteralPath $cfg -Pattern 'include-system-site-packages\s*=\s*true' -Quiet)) {
                $why = '這是舊式環境（借用其他 Python 環境的套件），改建成獨立環境'
            }
            elseif ($baseHome -and ($baseHome.TrimEnd('\') -ne $wantHome.TrimEnd('\'))) {
                $why = "基底 Python 不同（$baseHome），改用 $wantHome"
            }
        }
        if ($why) {
            Write-Note "重建 $dir：$why"
            try { Remove-Item -LiteralPath $dir -Recurse -Force }
            catch { throw "無法刪除 $dir（可能有程式正在使用它，請先關閉語音書寫與終端機）：$($_.Exception.Message)" }
        } else {
            Write-Ok "沿用既有環境 $dir"
            return $venvPy
        }
    }
    Invoke-Checked "建立 $dir" $python @('-m', 'venv', $dir)
    return $venvPy
}

function Install-Pip([string]$py, [string[]]$pipArguments) {
    $all = @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-input', '--retries', '10', '--timeout', '60') + $pipArguments
    & $py @all
    if ($LASTEXITCODE -ne 0) {
        throw "pip 安裝失敗（結束碼 $LASTEXITCODE）。請確認網路後重新執行 setup.ps1，已下載的部分會續用。"
    }
}

$logPath = Join-Path $Root 'setup.log'
try { Start-Transcript -Path $logPath -Append | Out-Null } catch {}
$started = Get-Date
try {
    Write-Host '語音書寫 安裝程式' -ForegroundColor White
    Write-Host "專案資料夾：$Root"
    Write-Host "記錄檔：$logPath"

    # ------------------------------------------------------------------ 1. preflight
    Write-Step 1 '預檢'
    if ([Environment]::OSVersion.Platform -ne 'Win32NT' -or -not [Environment]::Is64BitOperatingSystem) {
        throw '需要 64 位元的 Windows。'
    }
    Write-Ok 'Windows 64 位元'
    $needGb = $(if ($SkipAsr) { 2 } else { $NeedDiskGb })
    $freeGb = Get-FreeGb $Root
    if ($freeGb -lt $needGb) { throw ("磁碟空間不足：可用 {0:N1} GB，至少需要 {1} GB。" -f $freeGb, $needGb) }
    Write-Ok ("磁碟可用空間 {0:N0} GB（需要約 {1} GB）" -f [Math]::Min($freeGb, 99999), $needGb)

    # PyTorch's package tree is deeply nested; a long project path can hit Windows' 260-character limit.
    $longPaths = $false
    try { $longPaths = ((Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -ErrorAction Stop).LongPathsEnabled -eq 1) } catch {}
    if ($Root.Length -gt 90 -and -not $longPaths) {
        Write-Warn ("專案路徑很長（{0} 個字元），而這台電腦沒有啟用 Windows 長路徑支援，安裝 PyTorch 時可能失敗。" -f $Root.Length)
        Write-Warn '建議把專案資料夾搬到較短的路徑（例如 C:\voice-writing）後再安裝。'
    }

    if (-not $SkipAsr) {
        if ($SkipGpuCheck) { Write-Warn '已略過 NVIDIA 檢查（-SkipGpuCheck）。沒有 GPU 時語音辨識會非常慢。' }
        else {
            $smi = Get-NvidiaSmi
            if (-not $smi) {
                throw ('找不到 NVIDIA 驅動程式（nvidia-smi）。本系統的語音辨識需要 NVIDIA 顯卡；請先安裝最新驅動 ' +
                       '（https://www.nvidia.com/Download/index.aspx）。若確定要在沒有 GPU 的電腦上安裝，可加 -SkipGpuCheck。')
            }
            $gpuLine = (& $smi --query-gpu=name,driver_version --format=csv,noheader | Select-Object -First 1)
            $parts = $gpuLine -split ',\s*'
            $driver = [version]($parts[1].Trim())
            if ($driver.Major -lt $MinDriverMajor) {
                throw ("NVIDIA 驅動版本 {0} 太舊（需要 {1} 以上才支援 CUDA 12.8）。請到 https://www.nvidia.com/Download/index.aspx 更新驅動。" -f $driver, $MinDriverMajor)
            }
            Write-Ok ("NVIDIA GPU：{0}，驅動 {1}" -f $parts[0].Trim(), $driver)
        }
    }

    $hosts = @(@('PyPI', 'https://pypi.org/simple/pip/'), @('PyPI 檔案', 'https://files.pythonhosted.org/'))
    if (-not $SkipAsr) { $hosts += , @('PyTorch', 'https://download.pytorch.org/whl/cu128/') }
    if (-not $SkipAsr -and -not $SkipModels) {
        $hf = $(if ($HfEndpoint) { $HfEndpoint } else { 'https://huggingface.co' })
        $hosts += , @('Hugging Face', $hf)
    }
    foreach ($h in $hosts) {
        if (Test-Reachable $h[1]) { Write-Ok "連得上 $($h[0])" }
        else { throw "連不到 $($h[0])（$($h[1])）。請檢查網路、VPN 或代理（可設定環境變數 HTTPS_PROXY）。" }
    }

    # ------------------------------------------------------------------ 2. python
    Write-Step 2 '尋找 Python 3.12'
    if ($BasePython) {
        if (-not (Test-Python312 $BasePython)) { throw "-BasePython 指定的不是 64 位元 Python 3.12：$BasePython" }
        $python = (Resolve-Path -LiteralPath $BasePython).Path
    } else {
        $python = Find-Python312
        if (-not $python) { $python = Install-Python312 }
    }
    Write-Ok "Python 3.12：$python"

    # ------------------------------------------------------------------ 3. UI venv
    Write-Step 3 '建立 UI 環境（.venv）'
    $uiPy = Ensure-Venv (Join-Path $Root '.venv') $python $false
    Install-Pip $uiPy @('--upgrade', 'pip')
    $reqFile = $(if ($Dev) { 'requirements-dev.txt' } else { 'requirements.txt' })
    Install-Pip $uiPy @('-r', (Join-Path $Root $reqFile), '-c', (Join-Path $Root 'constraints.txt'))
    Invoke-Checked '確認 UI 套件' $uiPy @('-c', "import nicegui, httpx, numpy, soundcard, av; print('UI packages OK')")
    Write-Ok '.venv 就緒'

    # ------------------------------------------------------------------ 4. ASR venv
    $asrPy = Join-Path $Root '.venv-asr\Scripts\python.exe'
    if ($SkipAsr) {
        Write-Step 4 '建立 ASR 環境（已略過 -SkipAsr）'
    } else {
        Write-Step 4 '建立獨立的 ASR 環境（.venv-asr，含 CUDA 版 PyTorch，下載約 4 GB，請耐心等候）'
        $asrPy = Ensure-Venv (Join-Path $Root '.venv-asr') $python $true
        Install-Pip $asrPy @('--upgrade', 'pip')
        Install-Pip $asrPy @('-r', (Join-Path $Root 'requirements-asr.txt'))
        $probe = "import torch, qwen_asr; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
        Invoke-Checked '確認 ASR 套件' $asrPy @('-c', $probe)
        if (-not $SkipGpuCheck) {
            & $asrPy -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)'
            if ($LASTEXITCODE -ne 0) {
                throw 'PyTorch 看不到 GPU（CUDA 不可用）。請更新 NVIDIA 驅動並重新開機後再執行 setup.ps1。'
            }
        }
        Write-Ok '.venv-asr 就緒'
    }

    # ------------------------------------------------------------------ 5. models
    if ($SkipAsr -or $SkipModels) {
        Write-Step 5 '下載模型（已略過）'
        Write-Note '之後可執行：.venv-asr\Scripts\python.exe tools\download_models.py --write-config'
    } else {
        Write-Step 5 '下載語音模型（Qwen3-ASR-1.7B 與 ForcedAligner-0.6B，共約 6.1 GB，可續傳）'
        $dlArgs = @((Join-Path $Root 'tools\download_models.py'), '--write-config')
        # 明確指定 -ModelsDir 時以它為準；否則才沿用 data\settings.json 裡已指到完整模型的資料夾。
        if ($ModelsDir) { $dlArgs += @('--dir', $ModelsDir) } else { $dlArgs += '--reuse-config' }
        if ($HfEndpoint) { $dlArgs += @('--endpoint', $HfEndpoint) }
        Invoke-Checked '下載模型' $asrPy $dlArgs
        Write-Ok '模型就緒'
    }

    # ------------------------------------------------------------------ 6. settings
    Write-Step 6 '寫入設定（data\settings.json）'
    $cfgArgs = @((Join-Path $Root 'tools\configure.py'))
    if ($LlmUrl) { $cfgArgs += @('--llm-url', $LlmUrl) }
    if ($LlmModel) { $cfgArgs += @('--llm-model', $LlmModel) }
    if ($PSBoundParameters.ContainsKey('LlmKey')) { $cfgArgs += @('--llm-key', $LlmKey) }
    Invoke-Checked '寫入設定' $uiPy $cfgArgs
    Write-Ok 'data\settings.json 就緒'

    # ------------------------------------------------------------------ 7. doctor
    $doctorFailed = $false
    if ($SkipDoctor) {
        Write-Step 7 '健檢（已略過）'
    } else {
        Write-Step 7 '健檢'
        $docArgs = @((Join-Path $Root 'tools\doctor.py'))
        if (-not $SkipSmoke -and -not $SkipAsr) { $docArgs += '--smoke' }
        if ($SkipAsr) { $docArgs += '--skip-asr' }
        & $uiPy @docArgs
        $doctorFailed = ($LASTEXITCODE -ne 0)
    }

    $minutes = [Math]::Round(((Get-Date) - $started).TotalMinutes, 1)
    Write-Host ''
    if ($doctorFailed) {
        Write-Host "安裝已完成，但健檢有失敗項目（見上方 [FAIL]）。處理後可重新執行 setup.ps1。（耗時 $minutes 分鐘）" -ForegroundColor Yellow
        exit 2
    }
    Write-Host "安裝完成！（耗時 $minutes 分鐘）" -ForegroundColor Green
    Write-Host '  啟動：    雙擊 start.cmd（或 .venv\Scripts\python.exe app.py --open）'
    Write-Host '  網址：    http://127.0.0.1:2020/'
    Write-Host '  LLM：     本腳本不安裝校稿用的 LLM 伺服器。請在「模型設定」填入你的 LLM 網址與模型，'
    Write-Host '            或用 -LlmUrl / -LlmModel 重新執行本腳本。'
    exit 0
} catch {
    Write-Host ''
    Write-Host "[失敗] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "詳細記錄：$logPath" -ForegroundColor Red
    exit 1
} finally {
    try { Stop-Transcript | Out-Null } catch {}
}
