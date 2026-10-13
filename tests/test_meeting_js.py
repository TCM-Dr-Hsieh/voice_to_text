"""Run the browser-side meeting view smoke test with the Python suite."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_meeting_view_rows_labels_translation_and_scripts():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js 未安裝，略過瀏覽器端會議畫面測試')
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run([node, str(root / 'tests' / 'meeting_smoke.js')],
                            cwd=root, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
