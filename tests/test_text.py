from voice_app.text import clean_editor, content_chars, join_text


def test_content_chars():
    assert content_chars('你好，AI 3.5！') == '你好AI35'
    assert content_chars('C++ / $5') == 'C++$5'


def test_editor_cleanup_and_join():
    assert clean_editor('<think>秘密推理</think>你好。') == '你好。'
    assert clean_editor('<think>unfinished') == ''
    assert join_text('test', 'API') == 'test API'
