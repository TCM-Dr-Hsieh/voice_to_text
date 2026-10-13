from types import SimpleNamespace

import pytest

from voice_app.ui_meeting import read_translation_inputs


def inputs(language, lag):
    return SimpleNamespace(language=SimpleNamespace(value=language), lag=SimpleNamespace(value=lag))


def test_the_language_and_lag_boxes_are_read_and_checked():
    assert read_translation_inputs(inputs('  Traditional   Chinese ', 6)) == ('Traditional Chinese', 6)
    assert read_translation_inputs(inputs('日本語', 3.0)) == ('日本語', 3)
    assert read_translation_inputs(inputs('English', '12')) == ('English', 12)


@pytest.mark.parametrize('language, lag, message', [
    ('', 6, '語言'), (None, 6, '語言'), ('English', 2, '最小是 3'), ('English', 4.5, '整數'),
    ('English', None, '整數'), ('English', 'abc', '整數'), ('English', 0, '最小是 3'),
])
def test_unusable_language_or_lag_says_what_to_fix(language, lag, message):
    with pytest.raises(ValueError, match=message):
        read_translation_inputs(inputs(language, lag))
