"""Environment health check: environments, GPU, models, microphone, LLM server (and an optional ASR smoke test).

    .venv/Scripts/python.exe tools/doctor.py [--smoke] [--no-llm] [--skip-asr] [--verify-models]

Exit code 1 when something required is broken (FAIL). WARN items do not block using the app.
"""
import argparse
import asyncio
from dataclasses import replace
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from voice_app.config import ROOT, SETTINGS_PATH, Settings   # noqa: E402

ASR_PROBE = r'''
import json, sys
out = {"python": sys.version.split()[0]}
try:
    import torch
    out.update(torch=torch.__version__, cuda_build=torch.version.cuda, cuda=bool(torch.cuda.is_available()))
    if torch.cuda.is_available():
        out.update(gpu=torch.cuda.get_device_name(0), vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
except Exception as exc:
    out["torch_error"] = repr(exc)
try:
    import qwen_asr, transformers
    out.update(qwen_asr="ok", transformers=transformers.__version__)
except Exception as exc:
    out["qwen_error"] = repr(exc)
print(json.dumps(out))
'''

results: list[tuple[str, str, str]] = []


def report(status: str, name: str, detail: str = ''):
    results.append((status, name, detail))
    mark = {'OK': '[ OK ]', 'WARN': '[WARN]', 'FAIL': '[FAIL]'}[status]
    print(f'{mark} {name}' + (f' — {detail}' if detail else ''), flush=True)


def check_ui_environment():
    expected = ROOT / '.venv'
    inside = Path(sys.prefix).resolve() == expected.resolve()
    report('OK' if inside else 'WARN', f'UI Python {sys.version.split()[0]}',
           str(sys.executable) if inside else f'不是專案的 .venv（{sys.prefix}）；建議用 .venv\\Scripts\\python.exe 執行')
    if sys.version_info[:2] != (3, 12):
        report('WARN', 'Python 版本', '建議使用 3.12（套件鎖定檔以 3.12 為準）')
    for module in ('nicegui', 'httpx', 'numpy', 'soundcard', 'av'):
        try:
            imported = importlib.import_module(module)
            report('OK', f'套件 {module}', getattr(imported, '__version__', ''))
        except Exception as exc:
            report('FAIL', f'套件 {module}', f'{type(exc).__name__}: {exc}')


def check_asr_environment():
    from voice_app.asr import worker_python
    python = worker_python()
    if not python.is_file():
        report('FAIL', 'ASR 環境', f'找不到 {python}；請執行 install.cmd（或 setup.ps1）')
        return
    legacy = python.parent.parent / 'pyvenv.cfg'
    if legacy.exists() and 'include-system-site-packages = true' in legacy.read_text(encoding='utf-8', errors='ignore'):
        report('WARN', 'ASR 環境', '這是「疊加在其他環境上」的舊式 venv，依賴別的 Python 環境；建議重新執行 setup.ps1 -Force')
    try:
        completed = subprocess.run([str(python), '-c', ASR_PROBE], capture_output=True, text=True, timeout=180)
        info = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:
        report('FAIL', 'ASR 環境', f'無法執行 {python}：{exc}')
        return
    report('OK', f'ASR Python {info.get("python")}', str(python))
    if 'torch_error' in info:
        report('FAIL', 'PyTorch', info['torch_error'])
    else:
        report('OK', 'PyTorch', f'{info["torch"]}（CUDA 建置 {info["cuda_build"]}）')
        if info.get('cuda'):
            report('OK', 'GPU', f'{info["gpu"]}，顯存 {info["vram_gb"]} GB')
        else:
            report('FAIL', 'GPU', 'CUDA 不可用：請確認 NVIDIA 驅動（建議 ≥ 570）與 CUDA 版 PyTorch；CPU 太慢，不建議使用')
    if 'qwen_error' in info:
        report('FAIL', 'qwen-asr', info['qwen_error'])
    else:
        report('OK', 'qwen-asr', f'transformers {info.get("transformers")}')


def check_models(settings: Settings, deep: bool = False):
    from voice_app.asr import ASRError, validate_model_dir
    import modelcheck
    for label, value, aligner, repo in (('ASR 模型', settings.asr_model_dir, False, modelcheck.REPOS['asr']),
                                        ('ForcedAligner 模型', settings.aligner_model_dir, True,
                                         modelcheck.REPOS['aligner'])):
        try:
            folder = validate_model_dir(value, aligner=aligner)
        except ASRError as exc:
            report('FAIL', label, f'{exc}（可執行 install.cmd 或 tools\\download_models.py）')
            continue
        drift = modelcheck.problems(folder, repo, deep=deep)
        if drift:                                    # a different version may still work; it just was not verified
            report('WARN', label, f'{folder} 與本程式驗證的固定版本不同：{modelcheck.summarize(drift)}'
                                  '（可能是其他版本，相容性與辨識品質未經驗證）')
        else:
            report('OK', label, f'{folder}（與固定版本相符{"，已核對全部檔案的內容雜湊" if deep else ""}）')


def check_microphone():
    try:
        from voice_app.audio import devices
        found = devices('microphone')
        if found:
            report('OK', '麥克風', '、'.join(found.values())[:120])
        else:
            report('WARN', '麥克風', '找不到錄音裝置；請檢查 Windows 隱私權設定與裝置')
    except Exception as exc:
        report('WARN', '麥克風', f'無法列出裝置：{exc}')


async def check_llm(settings: Settings):
    from voice_app.api import ModelAPI
    editor = settings.editor
    try:
        async with ModelAPI(replace(settings, timeout_seconds=8)) as api:
            available = await api.models(editor)
        if editor.model not in available:
            report('WARN', f'校稿 LLM {editor.base_url}', f'連得上，但清單中沒有模型：{editor.model}')
        else:
            report('OK', f'校稿 LLM {editor.base_url}', f'連線成功（{len(available)} 個模型）')
    except Exception as exc:
        report('WARN', f'校稿 LLM {editor.base_url}',
               f'{type(exc).__name__}: {exc}（錄音前請確認伺服器已啟動，並在「模型設定」填入正確網址與模型）')


async def smoke(settings: Settings):
    """Load the worker and recognize a few seconds of silence: proves the GPU path end to end."""
    import numpy as np
    from voice_app.asr import local_asr
    try:
        started = time.monotonic()
        info = await local_asr.load(settings)
        loaded = time.monotonic() - started
        started = time.monotonic()
        await local_asr.recognize(settings, np.zeros(16000 * 4, dtype=np.float32))
        report('OK', 'ASR 實測', f'裝置 {info.get("device")}；載入 {loaded:.0f} 秒，4 秒音訊辨識 {time.monotonic() - started:.1f} 秒')
    except Exception as exc:
        report('FAIL', 'ASR 實測', f'{type(exc).__name__}: {exc}')
    finally:
        await local_asr.close()


def check_paths():
    data = SETTINGS_PATH.parent
    try:
        data.mkdir(parents=True, exist_ok=True)
        probe = data / '.write-test'
        probe.write_text('x', encoding='utf-8')
        probe.unlink()
        report('OK', '資料夾可寫入', str(data))
    except OSError as exc:
        report('FAIL', '資料夾', f'{data}：{exc}')


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--smoke', action='store_true', help='also load the ASR models and run one recognition')
    parser.add_argument('--no-llm', action='store_true', help='skip the LLM connectivity check')
    parser.add_argument('--skip-asr', action='store_true', help='skip the ASR environment and model checks')
    parser.add_argument('--verify-models', action='store_true',
                        help='also verify every file content hash against the pinned revision (reads ~6 GB)')
    args = parser.parse_args()
    print(f'專案：{ROOT}\n')
    check_ui_environment()
    try:
        settings = Settings.load(SETTINGS_PATH)
        report('OK', '設定檔', str(SETTINGS_PATH) if SETTINGS_PATH.exists() else '尚未建立，使用預設值')
    except Exception as exc:
        report('FAIL', '設定檔', f'{SETTINGS_PATH}：{exc}')
        return 1
    check_paths()
    if not args.skip_asr:
        check_asr_environment()
        check_models(settings, deep=args.verify_models)
    check_microphone()
    if not args.no_llm:
        await check_llm(settings)
    if args.smoke and not any(s == 'FAIL' for s, _, _ in results):
        await smoke(settings)
    fails = sum(1 for s, _, _ in results if s == 'FAIL')
    warns = sum(1 for s, _, _ in results if s == 'WARN')
    print(f'\n結果：{len(results) - fails - warns} 項正常，{warns} 項警告，{fails} 項失敗。')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
