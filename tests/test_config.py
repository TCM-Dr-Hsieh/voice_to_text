"""Settings boundary checks for the shared ASR/LLM vocabulary."""
import json

import pytest

from voice_app.config import MAX_VOCABULARY_CHARS, Settings


def test_vocabulary_accepts_25000_characters_and_survives_reload(tmp_path):
    vocabulary = '專' * MAX_VOCABULARY_CHARS
    path = tmp_path / 'settings.json'
    Settings(vocabulary=vocabulary).save(path)
    assert Settings.load(path).vocabulary == vocabulary


def test_vocabulary_rejects_25001_characters():
    with pytest.raises(ValueError, match='專有詞提示最多 25000 字元'):
        Settings(vocabulary='專' * (MAX_VOCABULARY_CHARS + 1)).validate()


def test_domain_terms_migrate_and_roundtrip(tmp_path):
    path = tmp_path / 'settings.json'
    Settings().save(path)
    import json
    data = json.loads(path.read_text(encoding='utf-8'))
    del data['editor']['domain_terms']
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    settings = Settings.load(path)
    assert settings.editor.domain_terms == ''
    settings.editor.domain_terms = '中醫：腎陽、夜尿\n西醫：enuresis'
    settings.save(path)
    assert Settings.load(path).editor.domain_terms == settings.editor.domain_terms


def test_domain_terms_reject_25001_characters():
    settings = Settings()
    settings.editor.domain_terms = '詞' * (MAX_VOCABULARY_CHARS + 1)
    with pytest.raises(ValueError, match='領域常見詞彙最多 25000 字元'):
        settings.validate()


def test_old_edit_limit_is_ignored_when_loading_and_saving(tmp_path):
    path = tmp_path / 'settings.json'
    Settings().save(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    data['max_edit_fraction'] = 0.5
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')

    settings = Settings.load(path)
    settings.save(path)
    assert 'max_edit_fraction' not in json.loads(path.read_text(encoding='utf-8'))


def test_window_must_exceed_overlap():
    Settings(window_seconds=5, overlap_seconds=0).validate()
    Settings(window_seconds=5, overlap_seconds=2).validate()
    Settings(window_seconds=7, overlap_seconds=4).validate()
    Settings(window_seconds=9, overlap_seconds=4).validate()
    Settings(window_seconds=9, overlap_seconds=6).validate()
    Settings(window_seconds=30, overlap_seconds=27).validate()
    with pytest.raises(ValueError, match='window_seconds 須介於 5–30'):
        Settings(window_seconds=4.99, overlap_seconds=0).validate()
    with pytest.raises(ValueError, match='至少保留 3 秒'):
        Settings(window_seconds=9, overlap_seconds=6.01).validate()
    with pytest.raises(ValueError, match='至少保留 3 秒'):
        Settings(window_seconds=5, overlap_seconds=-.01).validate()
