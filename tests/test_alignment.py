import pytest
from voice_app.alignment import AlignmentError, map_alignment, select_uncommitted


def item(text, start, end):
    return {'text': text, 'start': start, 'end': end}


def test_mixed_language_offsets_and_punctuation_are_preserved():
    text = '用 API，做 C++。'
    units = map_alignment(text, [item('用', 0, .2), item('API', .2, .5),
                                item('做', .6, .8), item('C', .8, 1)], 4, 2)
    added, pending, selected, _ = select_uncommitted(text, units, -1)
    assert added == text and not pending
    assert selected[0].start == 4 and selected[-1].end == 5
    assert select_uncommitted(text, units, 4.5)[0] == '做 C++。'


@pytest.mark.parametrize('items', [[], [item('假', 0, 1)], [item('你', 0, 1)],
    [item('你', 0, 1), item('好', .5, .8)],
    [item('你好', -1, 1)], [item('你好', 0, 10)], [item('你好', 0, float('nan'))]])
def test_bad_alignment_cannot_silently_drop_words(items):
    with pytest.raises(AlignmentError):
        map_alignment('你好', items, 0, 2)


def test_out_of_range_alignment_reports_the_offending_item():
    with pytest.raises(AlignmentError, match=r'第 2 項.*音訊長度 2 秒'):
        map_alignment('你好', [item('你', 0, .2), item('好', .3, 3)], 0, 2)


def test_repeated_words_at_different_times_are_retained():
    units = map_alignment('好好', [item('好', 0, .4), item('好', .6, 1)], 3, 2)
    added, _, selected, _ = select_uncommitted('好好', units, 3.4, previous_unit=units[0])
    assert added == '好' and selected[0].start == 3.6


def test_small_alignment_drift_matches_same_word_not_next_repetition():
    old = map_alignment('好', [item('好', 0, .5)], 3, 2)[0]
    new = map_alignment('好好', [item('好', 0, .56), item('好', .6, 1)], 3, 2)
    assert select_uncommitted('好好', new, 3.5, previous_unit=old)[0] == '好'


def test_long_window_defers_right_context():
    text = '甲乙丙丁'
    units = map_alignment(text, [item(c, i, i + .8) for i, c in enumerate(text)], 8, 4)
    added, pending, selected, _ = select_uncommitted(text, units, 8.8, 10)
    assert added == '乙' and pending == '丙丁'
    assert selected[-1].end == 9.8
