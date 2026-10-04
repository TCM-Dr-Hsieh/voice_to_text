"""Download the two Qwen3 models the app needs from Hugging Face (pinned revisions, resumable).

    .venv-asr/Scripts/python.exe tools/download_models.py [--dir models] [--reuse-config] [--write-config]
                                                          [--check] [--verify] [--endpoint https://mirror.example]

Run it with the ASR venv's Python (it already contains huggingface_hub). Re-running is safe: a
folder that already passes the app's own validation is skipped, an interrupted download resumes.
Only the files the app needs are fetched (no READMEs). About 6.1 GB in total.

Every folder is compared with tools/model_manifest.json (the pinned revision): file sizes always, SHA-256 of the
weights with --verify. A folder we manage that differs is downloaded again; a reused folder that differs is kept
but reported (--check then exits 1), because you may have chosen another model version on purpose.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from voice_app.asr import ASRError, validate_model_dir                  # noqa: E402
from voice_app.config import ROOT, SETTINGS_PATH, Settings, resolve_path  # noqa: E402
import modelcheck                                                        # noqa: E402  (tools/ is sys.path[0])

IGNORE = ['*.md', '.gitattributes', 'examples/*', '*.png', '*.jpg']


@dataclass(frozen=True)
class Model:
    key: str                # 'asr' | 'aligner'
    repo: str
    revision: str           # pinned commit: the exact files this app was verified with
    folder: str             # default folder name under --dir
    size_gb: float
    aligner: bool


MODELS = {
    'asr': Model('asr', 'Qwen/Qwen3-ASR-1.7B', '7278e1e70fe206f11671096ffdd38061171dd6e5',
                 'Qwen3-ASR-1.7B', 4.4, False),
    'aligner': Model('aligner', 'Qwen/Qwen3-ForcedAligner-0.6B', 'c7cbfc2048c462b0d63a45797104fc9db3ad62b7',
                     'Qwen3-ForcedAligner-0.6B', 1.8, True),
}


def is_valid(folder: Path, model: Model, *, deep: bool = False) -> tuple[bool, str]:
    """Complete model (the app's own validation) that also matches the pinned revision."""
    try:
        validate_model_dir(str(folder), aligner=model.aligner)
    except ASRError as exc:
        return False, str(exc)
    drift = modelcheck.problems(folder, model.repo, deep=deep)
    return (False, '與固定版本不符：' + modelcheck.summarize(drift)) if drift else (True, '')


def display_path(folder: Path) -> str:
    """Store project-internal folders as relative paths so the project can be moved or copied."""
    try:
        return os.path.relpath(folder, ROOT) if folder.resolve().is_relative_to(ROOT) else str(folder)
    except ValueError:                                                   # different drive
        return str(folder)


def configured_folder(model: Model) -> Path | None:
    """The folder named in data/settings.json, if that file exists and the folder is a complete model
    (it may still differ from the pinned revision; the caller reports that)."""
    if not SETTINGS_PATH.exists():
        return None
    try:
        settings = Settings.load(SETTINGS_PATH)
    except Exception:
        return None
    folder = resolve_path(settings.aligner_model_dir if model.aligner else settings.asr_model_dir)
    try:
        validate_model_dir(str(folder), aligner=model.aligner)
    except ASRError:
        return None
    return folder


def free_gb(path: Path) -> float:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 2 ** 30


def download(model: Model, target: Path) -> None:
    from huggingface_hub import snapshot_download
    print(f'  下載 {model.repo}（約 {model.size_gb} GB）→ {target}', flush=True)
    snapshot_download(repo_id=model.repo, revision=model.revision, local_dir=str(target),
                      ignore_patterns=IGNORE, max_workers=4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', default='models', help='base folder for the downloads (default: <project>/models)')
    parser.add_argument('--reuse-config', action='store_true',
                        help='use folders already named in data/settings.json when they are complete models')
    parser.add_argument('--write-config', action='store_true', help='record the folders in data/settings.json')
    parser.add_argument('--check', action='store_true', help='only validate; download nothing')
    parser.add_argument('--verify', action='store_true',
                        help='also verify every file content hash (weights: SHA-256, others: git blob id) against the pinned revision '
                             '(reads ~6 GB)')
    parser.add_argument('--endpoint', default='', help='Hugging Face mirror endpoint (sets HF_ENDPOINT)')
    args = parser.parse_args()
    if args.endpoint:
        os.environ['HF_ENDPOINT'] = args.endpoint
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    base = resolve_path(args.dir)
    if args.check:
        args.reuse_config = True          # a check should look at the folders the app will actually use

    if args.verify:
        print('正在比對所有檔案的內容雜湊（權重 SHA-256、其餘 git blob；讀取約 6 GB，視磁碟速度需數秒到數分鐘）…', flush=True)
    chosen: dict[str, Path] = {}
    todo: list[Model] = []
    drifted: list[Model] = []
    for model in MODELS.values():
        existing = configured_folder(model) if args.reuse_config else None
        if existing is not None:
            drift = modelcheck.problems(existing, model.repo, deep=args.verify)
            if drift:
                drifted.append(model)
                print(f'[警告] {model.repo}：沿用設定中的 {existing}，但與本程式驗證的固定版本不同：'
                      f'{modelcheck.summarize(drift)}\n       仍會沿用；若要改回固定版本，請移除該設定或資料夾後重新執行。')
            else:
                print(f'[略過] {model.repo}：沿用設定中的完整模型 {existing}（與固定版本相符）')
            chosen[model.key] = existing
            continue
        target = base / model.folder
        ok, why = is_valid(target, model, deep=args.verify)
        if ok:
            print(f'[略過] {model.repo}：{target} 已存在且完整')
            chosen[model.key] = target
        else:
            todo.append(model)
            chosen[model.key] = target
            if target.exists():
                print(f'[待下載] {model.repo}：{target} 不完整（{why}）')
            else:
                print(f'[待下載] {model.repo}')

    if args.check:
        return 1 if todo or drifted else 0
    if todo:
        need = sum(m.size_gb for m in todo) * 1.15 + 1
        have = free_gb(base)
        print(f'需要約 {need:.1f} GB 磁碟空間，{base.drive or base} 可用 {have:.1f} GB')
        if have < need:
            print('磁碟空間不足。', file=sys.stderr)
            return 1
        base.mkdir(parents=True, exist_ok=True)
        for model in todo:
            try:
                download(model, chosen[model.key])
            except Exception as exc:
                print(f'下載失敗：{type(exc).__name__}: {exc}\n可重新執行以續傳；若在受限網路，可用 --endpoint 指定鏡像。',
                      file=sys.stderr)
                return 1
            ok, why = is_valid(chosen[model.key], model, deep=args.verify)
            if not ok:
                print(f'下載完成但驗證失敗：{why}', file=sys.stderr)
                return 1
            print(f'  ✓ {model.repo} 驗證通過')

    if args.write_config:
        settings = Settings.load(SETTINGS_PATH)
        settings.asr_model_dir = display_path(chosen['asr'])
        settings.aligner_model_dir = display_path(chosen['aligner'])
        settings.save(SETTINGS_PATH)
        print(f'已寫入 {SETTINGS_PATH.relative_to(ROOT)}：asr={settings.asr_model_dir}  aligner={settings.aligner_model_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
