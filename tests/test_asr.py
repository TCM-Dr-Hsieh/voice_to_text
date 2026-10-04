import asyncio
import base64
from dataclasses import asdict
import json
import os

import numpy as np
import pytest

from voice_app.asr import ASRError, LocalASR, validate_model_dir
from voice_app.config import ModelConfig, Settings


@pytest.fixture
def model_dir(tmp_path):
    folder = tmp_path / 'model'
    folder.mkdir()
    for name in ('preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json'):
        (folder / name).write_text('{}')
    (folder / 'config.json').write_text('{"model_type":"qwen3_asr"}')
    (folder / 'model.safetensors').write_bytes(b'test-fixture')
    return folder


def test_model_folder_and_wrong_architecture(model_dir):
    assert validate_model_dir(str(model_dir)) == model_dir
    (model_dir / 'config.json').write_text('{"model_type":"qwen3"}')
    with pytest.raises(ASRError, match='不是 Qwen3-ASR'):
        validate_model_dir(str(model_dir))


def test_aligner_folder_errors_name_the_aligner(model_dir):
    with pytest.raises(ASRError, match='ForcedAligner 模型資料夾不存在'):
        validate_model_dir(str(model_dir / 'missing'), aligner=True)
    (model_dir / 'tokenizer_config.json').unlink()
    with pytest.raises(ASRError, match='ForcedAligner 模型資料夾缺少 tokenizer_config.json'):
        validate_model_dir(str(model_dir), aligner=True)
    (model_dir / 'tokenizer_config.json').write_text('{}')
    (model_dir / 'config.json').write_text('{"model_type":"qwen3_asr","timestamp_token_id":1}')
    (model_dir / 'model.safetensors').unlink()
    with pytest.raises(ASRError, match='ForcedAligner 模型資料夾缺少或有不合法的權重'):
        validate_model_dir(str(model_dir), aligner=True)


@pytest.mark.parametrize('weight', ['missing.safetensors', '../escape.safetensors'])
def test_shard_index_rejects_missing_or_external_weights(model_dir, weight):
    (model_dir.parent / 'escape.safetensors').write_bytes(b'outside')
    (model_dir / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'layer': weight}}))
    with pytest.raises(ASRError, match='權重'):
        validate_model_dir(str(model_dir))


def test_old_http_settings_migrate_without_changing_editor(tmp_path):
    data = asdict(Settings())
    data.pop('asr_model_dir')
    data.pop('asr_device')
    data['asr'] = {'model': 'old-asr', 'base_url': 'http://localhost:9999/v1', 'api_key': 'old-key'}
    data['asr_format'] = 'audio_url'
    data['editor']['model'] = 'custom-editor'
    data['overlap_seconds'] = 0.5
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(data))
    settings = Settings.load(path)
    assert settings.asr_model_dir == Settings().asr_model_dir
    assert settings.editor.model == 'custom-editor'
    assert settings.overlap_seconds == 0.5
    settings.save(path)
    saved = json.loads(path.read_text())
    assert 'asr' not in saved and 'asr_format' not in saved


def test_unknown_and_missing_settings_fields_keep_known_preferences(tmp_path):
    data = asdict(Settings())
    data['future_option'] = {'enabled': True}
    data['overlap_seconds'] = 0.5
    data.pop('window_seconds')
    data['editor'] = {'model': 'my-editor', 'future_editor_option': 'ignored'}
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    settings = Settings.load(path)
    assert settings.editor.model == 'my-editor'
    assert settings.editor.base_url == Settings().editor.base_url
    assert settings.overlap_seconds == 0.5
    assert settings.window_seconds == Settings().window_seconds
    settings.save(path)
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert 'future_option' not in saved
    assert 'future_editor_option' not in saved['editor']


def test_editor_temperature_migrates_and_roundtrips(tmp_path):
    data = asdict(Settings())
    del data['editor']['temperature']
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    settings = Settings.load(path)
    assert settings.editor.temperature == 0
    settings.editor.temperature = 0.65
    settings.save(path)
    assert Settings.load(path).editor.temperature == 0.65


@pytest.mark.parametrize('value', [-0.1, 2.1, float('nan'), float('inf'), '0.5', True])
def test_editor_temperature_rejects_invalid_values(value):
    with pytest.raises(ValueError, match='Temperature'):
        ModelConfig('model', temperature=value).validate()


@pytest.mark.asyncio
async def test_engine_reuses_worker_and_cancels_inflight(monkeypatch, tmp_path, model_dir):
    import voice_app.asr as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    python = tmp_path / '.venv-asr' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    python.parent.mkdir(parents=True)
    python.touch()
    engine = LocalASR()
    started = asyncio.Event()

    class FakeProcess:
        returncode = None
        reads = 0
        def __init__(self):
            self.stdin = self.stdout = self

        async def readline(self):
            self.reads += 1
            if self.reads == 1:
                return b'{"type":"ready","device":"cpu"}\n'
            if self.reads == 2:
                return '{"text":"辨識成功"}\n'.encode()
            started.set()
            await asyncio.Event().wait()

        def write(self, data):
            samples = np.frombuffer(base64.b64decode(json.loads(data)['audio']), dtype='<f4')
            assert len(samples) == 16000

        async def drain(self):
            pass

        def kill(self):
            self.returncode = -1

        async def wait(self):
            return self.returncode

    process = FakeProcess()
    launches = []
    async def spawn(*args, **kwargs):
        launches.append(args)
        assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        return process
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    aligner = model_dir.parent / 'aligner'
    import shutil
    shutil.copytree(model_dir, aligner)
    (aligner / 'config.json').write_text('{"model_type":"qwen3_asr","timestamp_token_id":1}')
    settings = Settings(asr_model_dir=str(model_dir), aligner_model_dir=str(aligner))
    await engine.load(settings)
    await engine.load(settings)
    assert await engine.transcribe(settings, np.zeros(16000)) == '辨識成功'
    assert len(launches) == 1
    task = asyncio.create_task(engine.transcribe(settings, np.zeros(16000)))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.returncode == -1
    assert engine.process is None
