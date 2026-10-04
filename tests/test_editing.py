from voice_app.editing import changes


def test_edit_offsets_can_reconstruct_original_and_result():
    old, new = '我們去散', '我們去散步。'
    edits = changes(old, new)
    rebuilt = old
    for edit in reversed(edits):
        assert old[edit['start']:edit['end']] == edit['original']
        assert new[edit['result_start']:edit['result_end']] == edit['replacement']
        rebuilt = rebuilt[:edit['start']] + edit['replacement'] + rebuilt[edit['end']:]
    assert rebuilt == new
