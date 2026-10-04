"""Edit data/settings.json from the command line (used by setup.ps1; also handy for scripted installs).

    .venv/Scripts/python.exe tools/configure.py --llm-url http://host:8080/v1 --llm-model my-model [--llm-key KEY]

Creates the file with defaults when it does not exist yet; an existing file is only rewritten when something changes.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from voice_app.config import SETTINGS_PATH, Settings   # noqa: E402


def apply(settings: Settings, *, llm_url: str = '', llm_model: str = '', llm_key: str | None = None) -> list[str]:
    """Apply the given changes in place; returns a human-readable list of what changed."""
    changes: list[str] = []
    if llm_url:
        settings.editor.base_url = llm_url
        changes.append(f'校稿 LLM 網址 = {llm_url}')
    if llm_model:
        settings.editor.model = llm_model
        changes.append(f'校稿 LLM 模型 = {llm_model}')
    if llm_key is not None:
        settings.editor.api_key = llm_key
        changes.append('校稿 LLM API Key 已更新')
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--llm-url', default='')
    parser.add_argument('--llm-model', default='')
    parser.add_argument('--llm-key', default=None)
    args = parser.parse_args()
    existed = SETTINGS_PATH.exists()
    settings = Settings.load(SETTINGS_PATH)
    changes = apply(settings, llm_url=args.llm_url, llm_model=args.llm_model, llm_key=args.llm_key)
    if changes or not existed:
        settings.save(SETTINGS_PATH)
    print(f'{SETTINGS_PATH.name} ' + ('已更新：' + '；'.join(changes) if changes else
                                      ('已用預設值建立。' if not existed else '沒有變更。')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
