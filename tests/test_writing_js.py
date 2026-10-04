"""Run the browser-side note lifecycle smoke test with the Python suite."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_writing_insert_replace_and_rollback():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js 未安裝，略過瀏覽器端書寫邏輯測試')
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run([node, str(root / 'tests' / 'writing_smoke.js')],
                            cwd=root, capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_remote_capture_replays_unacknowledged_audio():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js 未安裝，略過瀏覽器端補傳邏輯測試')
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run([node, str(root / 'tests' / 'remote_capture_smoke.js')],
                            cwd=root, capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
