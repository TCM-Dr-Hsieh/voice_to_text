from types import SimpleNamespace

from voice_app.text import join_texts
from voice_app.ui_render import revisable_split


def test_revisable_highlight_starts_after_ascii_boundary_separator():
    segments = [SimpleNamespace(corrected_tail=text) for text in ('前文', 'ABC', 'DEF', 'GHI', 'JKL')]
    corrected = join_texts(segment.corrected_tail for segment in segments)
    locked, revisable = revisable_split(corrected, segments)

    assert locked == '前文ABC DEF '
    assert revisable == 'GHI JKL'
    assert locked + revisable == corrected


def test_revisable_highlight_stays_after_gap():
    segments = [SimpleNamespace(corrected_tail=text) for text in ('前文', '後文')]
    corrected = '前文\n【音訊可能缺漏 4.0–8.0 秒】\n後文'
    locked, revisable = revisable_split(corrected, segments, rolling_start=1)
    assert locked.endswith('】\n')
    assert revisable == '後文'
