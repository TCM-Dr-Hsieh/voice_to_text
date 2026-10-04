import asyncio
from datetime import datetime
import json
import time
import wave

import numpy as np
import pytest

from voice_app.audio import AudioSource, RATE, reserve_recording_path
from voice_app.config import Settings
from voice_app.session import Session


def test_recording_names_are_separate_by_source_and_never_overwrite(tmp_path):
    day = datetime(2026, 9, 26)
    first = reserve_recording_path('microphone', tmp_path, day)
    first.write_bytes(b'existing recording')
    second = reserve_recording_path('microphone', tmp_path, day)
    assert first.name == '麥克風-20260926-001.wav'
    assert second.name == '麥克風-20260926-002.wav'
    assert first.read_bytes() == b'existing recording'
    third_json = tmp_path / '麥克風-20260926-003.json'
    third_json.write_text('existing transcript', encoding='utf-8')
    assert reserve_recording_path('microphone', tmp_path, day).name == '麥克風-20260926-004.wav'
    assert reserve_recording_path('loopback', tmp_path, day).name == '電腦音訊-20260926-001.wav'
    assert third_json.read_text(encoding='utf-8') == 'existing transcript'


@pytest.mark.asyncio
async def test_recorded_wav_keeps_every_sample_once_including_silence(monkeypatch, tmp_path):
    import voice_app.audio as audio
    samples = np.concatenate((np.full(RATE * 6, .25), np.zeros(RATE * 2 // 5),
                              np.full(RATE * 7 - RATE * 2 // 5, -.25))).astype(np.float32)

    def fake_device(kind, device_id, stop):
        for block in np.array_split(samples, 130):
            yield block

    monkeypatch.setattr(audio, 'record_device', fake_device)
    path = reserve_recording_path('microphone', tmp_path, datetime(2026, 9, 26))
    source = AudioSource('microphone', 'fake', None, settings=Settings(), recording_path=path)
    source.start()
    events = []
    while (event := await source.next_chunk()) is not None:
        events.append(event)
    assert any(event.reason == 'window' for event in events)
    assert source.error == ''
    assert source.recorded_samples == len(samples)
    with wave.open(str(path), 'rb') as saved:
        assert (saved.getframerate(), saved.getnchannels(), saved.getsampwidth()) == (RATE, 1, 2)
        assert saved.getnframes() == len(samples)
        recorded = np.frombuffer(saved.readframes(len(samples)), dtype='<i2')
    np.testing.assert_array_equal(recorded, (samples * 32767).astype('<i2'))


class FakeAPI:
    def __init__(self, settings, fail=False):
        self.fail = fail

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def recognize(self, samples):
        if self.fail:
            raise RuntimeError('ASR failure')
        return {'text': '你好', 'items': [
            {'text': '你', 'start': .2, 'end': .4},
            {'text': '好', 'start': .4, 'end': .6},
        ]}

    async def revise_recent(self, context, segments, on_preview, **kwargs):
        from voice_app.api import RollingResult
        return RollingResult(['您好'])


@pytest.mark.asyncio
async def test_completed_recording_autosaves_paired_json_and_corrected_text(monkeypatch, tmp_path):
    import voice_app.audio as audio
    import voice_app.session as session_module
    blocks = [np.full(1600, .2, dtype=np.float32) for _ in range(10)]
    blocks += [np.zeros(1600, dtype=np.float32) for _ in range(9)]

    def fake_device(kind, device_id, stop):
        yield from blocks

    monkeypatch.setattr(audio, 'record_device', fake_device)
    monkeypatch.setattr(session_module, 'reserve_recording_path',
                        lambda kind: reserve_recording_path(kind, tmp_path, datetime(2026, 9, 26)))
    monkeypatch.setattr(session_module, 'ModelAPI', FakeAPI)
    session = Session()
    session.start(Settings(), 'microphone', 'fake', record_audio=True)
    await asyncio.wait_for(session.task, 5)
    assert session.recording_state == 'complete'
    assert session.recording_path.name == '麥克風-20260926-001.wav'
    record = json.loads(session.recording_path.with_suffix('.json').read_text(encoding='utf-8'))
    assert record['recording'] == {'file': session.recording_path.name, 'status': 'complete',
                                   'sample_rate': RATE, 'sample_count': 19 * 1600}
    assert record['raw'] == '你好' and record['segments'][0]['alignment'][0]['start'] == .2
    assert record['corrected'] == '您好'
    assert record['revisions'][0]['after'] == ['您好']


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [False, True])
async def test_cancel_or_asr_failure_keeps_partial_recording(monkeypatch, tmp_path, failure):
    import voice_app.audio as audio
    import voice_app.session as session_module

    def fake_device(kind, device_id, stop):
        yield np.full(1600, .2, dtype=np.float32)
        if failure:
            for _ in range(9):
                yield np.zeros(1600, dtype=np.float32)
        else:
            while not stop.is_set():
                time.sleep(.01)

    monkeypatch.setattr(audio, 'record_device', fake_device)
    monkeypatch.setattr(session_module, 'reserve_recording_path',
                        lambda kind: reserve_recording_path(kind, tmp_path, datetime(2026, 9, 26)))
    monkeypatch.setattr(session_module, 'ModelAPI',
                        lambda settings: FakeAPI(settings, fail=failure))
    session = Session()
    session.start(Settings(), 'microphone', 'fake', record_audio=True)
    if failure:
        await asyncio.wait_for(session.task, 5)
        assert session.failed_chunk is not None
    else:
        for _ in range(100):
            if session.source.recorded_samples:
                break
            await asyncio.sleep(.01)
        await session.cancel()
    assert session.recording_state == 'incomplete'
    record = json.loads(session.recording_path.with_suffix('.json').read_text(encoding='utf-8'))
    assert record['recording']['status'] == 'incomplete'
    assert record['recording']['sample_count'] >= 1600
    with wave.open(str(session.recording_path), 'rb') as saved:
        assert saved.getnframes() == record['recording']['sample_count']
    if failure:
        await session.cancel()


@pytest.mark.asyncio
async def test_audio_source_failure_marks_recording_incomplete(monkeypatch, tmp_path):
    import voice_app.audio as audio
    import voice_app.session as session_module

    def broken_device(kind, device_id, stop):
        yield np.full(RATE, .2, dtype=np.float32)
        raise OSError('device disconnected')

    monkeypatch.setattr(audio, 'record_device', broken_device)
    monkeypatch.setattr(session_module, 'reserve_recording_path',
                        lambda kind: reserve_recording_path(kind, tmp_path, datetime(2026, 9, 26)))
    monkeypatch.setattr(session_module, 'ModelAPI', FakeAPI)
    session = Session()
    session.start(Settings(), 'loopback', 'fake', record_audio=True)
    await asyncio.wait_for(session.task, 5)
    assert session.recording_path.name == '電腦音訊-20260926-001.wav'
    assert session.recording_state == 'incomplete'
    saved = json.loads(session.recording_path.with_suffix('.json').read_text(encoding='utf-8'))
    assert saved['recording']['status'] == 'incomplete'
    assert saved['recording']['sample_count'] == RATE
    assert 'device disconnected' in saved['warning']
