"""Settings boundary checks for the shared ASR/LLM vocabulary."""
import json

import pytest

from voice_app.config import MAX_VOCABULARY_CHARS, Settings, listen_host


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
    Settings(window_seconds=9, overlap_seconds=8).validate()               # (a stride of 1 s: allowed, not advised)
    Settings(window_seconds=30, overlap_seconds=29).validate()
    with pytest.raises(ValueError, match='window_seconds 須介於 5–30'):
        Settings(window_seconds=4.99, overlap_seconds=0).validate()
    with pytest.raises(ValueError, match='至少保留 1 秒.*建議 L−Z 大於 3 秒'):
        Settings(window_seconds=9, overlap_seconds=8.01).validate()
    with pytest.raises(ValueError, match='至少保留 1 秒'):
        Settings(window_seconds=5, overlap_seconds=4.5).validate()
    with pytest.raises(ValueError, match='至少保留 1 秒'):
        Settings(window_seconds=5, overlap_seconds=-.01).validate()


# --- meeting mode ---------------------------------------------------------------------------------------------------
def test_settings_saved_before_meeting_mode_load_and_the_translator_follows_the_proofreader(tmp_path):
    path = tmp_path / 'settings.json'
    data = {'editor': {'model': 'my-editor', 'base_url': 'http://10.1.2.3:8080/v1', 'api_key': 'secret', 'domain_terms': '手部衛生'},
            'window_seconds': 8, 'overlap_seconds': 2, 'unknown_old_key': 1}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    settings = Settings.load(path)
    assert settings.translator.model == 'my-editor' and settings.translator.base_url == 'http://10.1.2.3:8080/v1'
    assert settings.translator.api_key == 'secret' and settings.translator.domain_terms == ''       # a glossary is for one job only
    assert settings.meeting.target_language == 'English' and settings.meeting.translation_lag == 6


def test_meeting_and_translator_settings_roundtrip_and_ignore_unknown_keys(tmp_path):
    path = tmp_path / 'settings.json'
    settings = Settings()
    settings.translator.model, settings.translator.domain_terms = 'translator-x', '手部衛生 = hand hygiene'
    settings.meeting.target_language, settings.meeting.translation_lag, settings.meeting.text_fill2 = '日本語', 8, True
    settings.save(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    data['meeting']['removed_option'] = 1
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    again = Settings.load(path)
    assert again.translator.model == 'translator-x' and again.translator.domain_terms == '手部衛生 = hand hygiene'
    assert (again.meeting.target_language, again.meeting.translation_lag, again.meeting.text_fill2) == ('日本語', 8, True)


@pytest.mark.parametrize('field, value, message', [
    ('translation_lag', 2, '延遲語音數'), ('translation_lag', True, '延遲語音數'), ('translation_lag', 61, '延遲語音數'),
    ('target_language', '   ', '翻譯語言'), ('target_language', 'x' * 61, '翻譯語言'),
    ('unknown_percentile', 51, '百分位'), ('max_speakers', 1, '人數上限'), ('translation_batch', 0, '補翻'),
    ('autosave_seconds', 1, '自動存檔'), ('speaker_model_path', '  ', '聲紋模型'), ('translation_context', -1, '上下文'),
    ('background_rel_db', 0.5, '比常態小聲'), ('background_rel_db', float('nan'), '比常態小聲'), ('background_rel_db', 31, '比常態小聲'),
    ('background_max_cos', 0.05, '相似度上限'), ('background_max_cos', 0.95, '相似度上限'), ('background_filter', 'yes', '背景人聲過濾'),
    ('live_translation', 'yes', '即時暫定翻譯'), ('live_translation', 1, '即時暫定翻譯'),
])
def test_meeting_settings_are_checked(field, value, message):
    settings = Settings()
    setattr(settings.meeting, field, value)
    with pytest.raises(ValueError, match=message):
        settings.validate()


def test_the_language_is_tidied_and_the_translator_endpoint_is_checked():
    settings = Settings()
    settings.meeting.target_language = '  Traditional   Chinese '
    settings.validate()
    assert settings.meeting.target_language == 'Traditional Chinese'
    settings.translator.base_url = 'not a url'
    with pytest.raises(ValueError, match='Base URL'):
        settings.validate()


def test_background_settings_have_safe_defaults_and_survive_a_save(tmp_path):
    settings = Settings()
    assert settings.meeting.background_filter is True
    assert (settings.meeting.background_rel_db, settings.meeting.background_max_cos) == (6.0, 0.45)
    settings.meeting.background_filter, settings.meeting.background_rel_db, settings.meeting.background_max_cos = False, 8.5, 0.4
    settings.validate()
    path = tmp_path / 'settings.json'
    settings.save(path)
    again = Settings.load(path)
    assert (again.meeting.background_filter, again.meeting.background_rel_db, again.meeting.background_max_cos) == (False, 8.5, 0.4)


def test_the_page_listens_on_every_interface_unless_the_host_is_set(monkeypatch):
    monkeypatch.delenv('VOICE_APP_HOST', raising=False)
    assert listen_host() == '0.0.0.0'
    monkeypatch.setenv('VOICE_APP_HOST', '')
    assert listen_host() == '0.0.0.0'
    monkeypatch.setenv('VOICE_APP_HOST', ' 127.0.0.1 ')
    assert listen_host() == '127.0.0.1'


def test_every_meeting_setting_the_dialogs_use_is_a_field_of_the_meeting_settings():
    """The settings dialog reads and writes `draft.meeting.<name>`: a name that is not a field only shows when someone opens the dialog."""
    import re
    from dataclasses import fields
    from pathlib import Path
    from voice_app.config import MeetingSettings
    known = {f.name for f in fields(MeetingSettings)} | {'validate'}
    seen = {module.name: set(re.findall(r'(?:draft|settings)\.meeting\.(\w+)', module.read_text(encoding='utf-8')))
            for module in sorted(Path(__file__).resolve().parent.parent.glob('voice_app/ui*.py'))}
    assert len(seen['ui_settings.py']) > 10                                               # (the scan does see the dialog)
    for name, used in seen.items():
        assert not (used - known), (name, sorted(used - known))


def test_the_live_translation_switch_is_on_by_default_survives_a_save_and_is_missing_from_old_files(tmp_path):
    settings = Settings()
    assert settings.meeting.live_translation is True
    settings.meeting.live_translation = False
    settings.save(tmp_path / 'settings.json')
    assert Settings.load(tmp_path / 'settings.json').meeting.live_translation is False
    data = json.loads((tmp_path / 'settings.json').read_text(encoding='utf-8'))
    del data['meeting']['live_translation']                                             # a file saved before the switch existed
    (tmp_path / 'settings.json').write_text(json.dumps(data), encoding='utf-8')
    assert Settings.load(tmp_path / 'settings.json').meeting.live_translation is True
