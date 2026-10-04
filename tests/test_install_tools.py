"""Install-time pieces: project-relative model paths, configure.py, download_models.py and modelcheck.py."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

import voice_app.asr as asr_module
import voice_app.config as config_module
from voice_app.asr import ASRError, validate_model_dir, worker_python
from voice_app.config import DEFAULT_ALIGNER_DIR, DEFAULT_ASR_DIR, ROOT, Settings, resolve_path

sys.path.insert(0, str(ROOT / 'tools'))
import configure                    # noqa: E402
import download_models              # noqa: E402
import modelcheck                   # noqa: E402


def make_model(folder: Path, *, aligner: bool = False) -> Path:
    folder.mkdir(parents=True)
    config = {'model_type': 'qwen3_asr'}
    if aligner:
        config['timestamp_token_id'] = 1
    (folder / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    for name in ('preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json'):
        (folder / name).write_text('{}', encoding='utf-8')
    (folder / 'model.safetensors').write_bytes(b'weights')
    return folder


# --- project-relative paths ------------------------------------------------------------------
def test_defaults_are_project_relative_so_the_folder_can_be_copied_anywhere():
    settings = Settings()
    assert settings.asr_model_dir == DEFAULT_ASR_DIR
    assert settings.aligner_model_dir == DEFAULT_ALIGNER_DIR
    assert not Path(DEFAULT_ASR_DIR).is_absolute() and DEFAULT_ASR_DIR.startswith('models')
    assert resolve_path(DEFAULT_ASR_DIR) == ROOT / DEFAULT_ASR_DIR
    settings.validate()


def test_relative_paths_resolve_against_the_project_root_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, 'ROOT', tmp_path)
    monkeypatch.chdir(Path(__file__).parent)                         # a different cwd must not matter
    assert resolve_path('models/x') == tmp_path / 'models' / 'x'
    absolute = tmp_path / 'elsewhere'
    assert resolve_path(str(absolute)) == absolute
    assert resolve_path(' "models/x" ') == tmp_path / 'models' / 'x'  # stray quotes/spaces from pasted paths


def test_model_validation_and_worker_python_use_the_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, 'ROOT', tmp_path)
    monkeypatch.setattr(asr_module, 'ROOT', tmp_path)
    make_model(tmp_path / 'models' / 'asr')
    make_model(tmp_path / 'models' / 'aligner', aligner=True)
    monkeypatch.chdir(Path(__file__).parent)
    assert validate_model_dir('models/asr') == (tmp_path / 'models' / 'asr').resolve()
    assert validate_model_dir('models/aligner', aligner=True)
    with pytest.raises(ASRError, match='ForcedAligner'):
        validate_model_dir('models/aligner')                          # wrong model kind is reported clearly
    with pytest.raises(ASRError, match='不存在'):
        validate_model_dir('models/missing')
    assert worker_python() == tmp_path / '.venv-asr' / 'Scripts' / 'python.exe'


def test_settings_keep_a_legacy_absolute_model_folder(tmp_path):
    path = tmp_path / 'settings.json'
    data = {'asr_model_dir': r'D:\models\Qwen-Qwen3-ASR', 'aligner_model_dir': r'D:\models\aligner'}
    path.write_text(json.dumps(data), encoding='utf-8')
    settings = Settings.load(path)
    assert settings.asr_model_dir == data['asr_model_dir']
    assert settings.aligner_model_dir == data['aligner_model_dir']


# --- configure.py ------------------------------------------------------------------------------
def test_configure_applies_llm_settings_to_the_editor():
    settings = Settings()
    changes = configure.apply(settings, llm_url='http://10.0.0.5:8080/v1', llm_model='m1', llm_key='k')
    assert (settings.editor.base_url, settings.editor.model, settings.editor.api_key) == (
        'http://10.0.0.5:8080/v1', 'm1', 'k')
    assert len(changes) == 3
    assert configure.apply(Settings()) == []                           # no arguments: nothing changes
    assert configure.apply(Settings(), llm_key='') == ['校稿 LLM API Key 已更新']   # an empty key is a real value
    settings.validate()


def test_configure_main_creates_defaults_and_leaves_an_untouched_file_alone(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'data' / 'settings.json'
    monkeypatch.setattr(configure, 'SETTINGS_PATH', path)
    monkeypatch.setattr(sys, 'argv', ['configure.py'])
    assert configure.main() == 0
    assert Settings.load(path) == Settings()                           # created from defaults
    assert '已用預設值建立' in capsys.readouterr().out
    path.write_text('{"window_seconds": 8, "unknown_legacy_key": 1}', encoding='utf-8')
    before = path.read_text(encoding='utf-8')
    assert configure.main() == 0
    assert path.read_text(encoding='utf-8') == before                  # nothing to change: not rewritten
    monkeypatch.setattr(sys, 'argv', ['configure.py', '--llm-url', 'http://h:1/v1', '--llm-model', 'm'])
    assert configure.main() == 0
    reloaded = Settings.load(path)
    assert reloaded.window_seconds == 8 and reloaded.editor.base_url == 'http://h:1/v1' and reloaded.editor.model == 'm'


# --- modelcheck.py -----------------------------------------------------------------------------
def test_modelcheck_reports_missing_wrong_size_and_bad_hash(tmp_path):
    good, bad_blob = b'weights-data', b'config-data'
    manifest = {'models': {'x/y': {'revision': 'r', 'files': {
        'model.safetensors': {'size': len(good), 'sha256': hashlib.sha256(good).hexdigest(), 'git_sha1': None},
        'config.json': {'size': len(bad_blob),
                        'sha256': None,
                        'git_sha1': hashlib.sha1(b'blob %d\0' % len(bad_blob) + bad_blob).hexdigest()},
        'vocab.json': {'size': 3, 'sha256': None, 'git_sha1': 'x'}}}}}
    (tmp_path / 'model.safetensors').write_bytes(good)
    (tmp_path / 'config.json').write_bytes(bad_blob)
    assert modelcheck.problems(tmp_path, 'x/y', manifest=manifest) == ['缺少 vocab.json']
    (tmp_path / 'vocab.json').write_bytes(b'abcd')
    assert 'vocab.json 大小不符' in modelcheck.problems(tmp_path, 'x/y', manifest=manifest)[0]
    (tmp_path / 'vocab.json').write_bytes(b'abc')
    (tmp_path / 'model.safetensors').write_bytes(b'weights-DATA')       # same size, different content
    assert modelcheck.problems(tmp_path, 'x/y', manifest=manifest) == []   # sizes only: content is not read
    deep = modelcheck.problems(tmp_path, 'x/y', deep=True, manifest=manifest)
    assert 'model.safetensors 的 SHA-256 不符' in deep
    assert 'vocab.json 的內容雜湊（git）不符' in deep
    assert modelcheck.problems(tmp_path, 'unknown/repo', manifest=manifest)[0].startswith('無法讀取')


def test_shipped_manifest_matches_the_pinned_revisions_in_download_models():
    manifest = modelcheck.load_manifest()
    for model in download_models.MODELS.values():
        spec = manifest['models'][model.repo]
        assert spec['revision'] == model.revision
        assert 'config.json' in spec['files'] and any(n.endswith('.safetensors') for n in spec['files'])
        assert not any(n.endswith(('.md', '.png', '.jpg')) for n in spec['files'])     # ignored files stay out
    assert modelcheck.REPOS == {key: model.repo for key, model in download_models.MODELS.items()}


# --- download_models.py ------------------------------------------------------------------------
def test_display_path_is_relative_inside_the_project_and_absolute_outside(tmp_path, monkeypatch):
    monkeypatch.setattr(download_models, 'ROOT', tmp_path)
    inside = tmp_path / 'models' / 'asr'
    inside.mkdir(parents=True)
    assert Path(download_models.display_path(inside)) == Path('models') / 'asr'
    outside = tmp_path.parent / 'elsewhere'
    assert download_models.display_path(outside) == str(outside)


def test_configured_folder_reuses_only_complete_models_of_the_right_kind(tmp_path, monkeypatch):
    path = tmp_path / 'settings.json'
    monkeypatch.setattr(download_models, 'SETTINGS_PATH', path)
    asr, aligner = download_models.MODELS['asr'], download_models.MODELS['aligner']
    assert download_models.configured_folder(asr) is None              # no settings file yet
    make_model(tmp_path / 'my-asr')
    make_model(tmp_path / 'my-aligner', aligner=True)
    path.write_text(json.dumps({'asr_model_dir': str(tmp_path / 'my-asr'),
                                'aligner_model_dir': str(tmp_path / 'my-aligner')}), encoding='utf-8')
    assert download_models.configured_folder(asr) == tmp_path / 'my-asr'
    assert download_models.configured_folder(aligner) == tmp_path / 'my-aligner'
    path.write_text(json.dumps({'asr_model_dir': str(tmp_path / 'my-aligner'),      # swapped: wrong kind
                                'aligner_model_dir': str(tmp_path / 'missing')}), encoding='utf-8')
    assert download_models.configured_folder(asr) is None
    assert download_models.configured_folder(aligner) is None
    path.write_text('not json', encoding='utf-8')
    assert download_models.configured_folder(asr) is None              # unreadable settings never crash the installer


def test_check_mode_downloads_nothing_and_reports_missing_models(tmp_path, monkeypatch):
    monkeypatch.setattr(download_models, 'SETTINGS_PATH', tmp_path / 'settings.json')
    monkeypatch.setattr(download_models, 'download',
                        lambda *a, **k: pytest.fail('--check must never download'))
    monkeypatch.setattr(sys, 'argv', ['download_models.py', '--check', '--dir', str(tmp_path / 'models')])
    assert download_models.main() == 1
    assert not (tmp_path / 'models').exists()
