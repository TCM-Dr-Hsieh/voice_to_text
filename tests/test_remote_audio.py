import asyncio
from datetime import datetime
import json
import struct
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from voice_app.audio import Chunk, RATE, reserve_recording_path
from voice_app.config import Settings, source_allowed
from voice_app.remote_audio import RemoteAudioSource
from voice_app.session import Session


def test_remote_audio_resamples_and_records_continuous_wav(tmp_path):
    path = reserve_recording_path('remote_microphone', tmp_path, datetime(2026, 9, 26))
    source = RemoteAudioSource('remote_microphone', settings=Settings(), recording_path=path)
    source.start()
    source.connect(48000)
    samples = np.concatenate((np.full(48000, .2), np.zeros(48000))).astype('<f4')
    for block in np.array_split(samples, 100):
        source.receive(block.tobytes())
    source.finish()
    assert source.done.is_set() and source.error == ''
    assert source.recorded_samples == RATE * 2
    assert path.name == '遠端麥克風-20260926-001.wav'
    with wave.open(str(path), 'rb') as saved:
        assert (saved.getframerate(), saved.getnchannels(), saved.getnframes()) == (RATE, 1, RATE * 2)
    assert any(chunk.final for chunk in list(source.queue.queue) + source.emergency)


def test_remote_audio_rejects_invalid_samples(tmp_path):
    source = RemoteAudioSource('remote_system', settings=Settings())
    source.start()
    with pytest.raises(ValueError, match='取樣率'):
        source.connect(1000)
    source.connect(48000)
    with pytest.raises(ValueError, match='封包'):
        source.receive(b'bad')
    with pytest.raises(ValueError, match='數值'):
        source.receive(np.array([np.nan], dtype='<f4').tobytes())
    source.finish('連線中斷')
    assert source.done.is_set() and source.error == '連線中斷'


def test_remote_queue_overflow_keeps_final_audio_without_blocking():
    source = RemoteAudioSource('remote_microphone', settings=Settings())
    chunk = Chunk(1, 0, np.ones(160, dtype=np.float32))
    for _ in range(source.queue.maxsize):
        source.queue.put_nowait(chunk)
    source._put(Chunk(2, 0, np.ones(160, dtype=np.float32), final=False))
    assert not source.emergency and not source.stop.is_set()
    source._put(chunk)
    assert source.emergency == [chunk]
    assert source.stop.is_set() and '已停止錄音' in source.error


def test_shared_mode_restricts_host_audio_and_settings(monkeypatch):
    from voice_app.ui_settings import show_settings_dialog
    from voice_app.ui import writing_source_allowed

    monkeypatch.setenv('VOICE_APP_SHARED', '1')
    assert not source_allowed('microphone')
    assert not source_allowed('loopback')
    assert not writing_source_allowed('loopback')
    assert all(source_allowed(kind) for kind in ('remote_microphone', 'remote_system', 'file'))
    assert show_settings_dialog(None, None) is None
    monkeypatch.delenv('VOICE_APP_SHARED')
    assert source_allowed('microphone')
    assert writing_source_allowed('loopback')
    assert writing_source_allowed('file')


class FakeAPI:
    def __init__(self, settings):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def prepare_asr(self):
        pass

    async def recognize(self, samples):
        return {'text': '你好', 'items': [
            {'text': '你', 'start': .2, 'end': .35},
            {'text': '好', 'start': .35, 'end': .5},
        ]}

    async def revise_recent(self, context, segments, on_preview, **kwargs):
        from voice_app.api import RollingResult
        texts = [row['current'] for row in segments]
        return RollingResult(texts)


class FakeWebSocket:
    def __init__(self, kind, messages, *, hello_type='start'):
        self.kind = kind
        self.messages = iter(messages)
        self.hello_type = hello_type
        self.headers = {'origin': 'https://voice.example', 'host': 'voice.example'}
        self.sent = []
        self.accepted = False
        self.close_code = None

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        return json.dumps({'type': self.hello_type, 'kind': self.kind,
                           'session_id': 'test-session-1234',
                           'writing_context': {'before': '', 'after': ''}})

    async def receive(self):
        return next(self.messages)

    async def send_json(self, value):
        self.sent.append(value)

    async def close(self, code=1000):
        self.close_code = code


def audio_messages(samples, *, start_sequence=0):
    return [{'type': 'websocket.receive', 'bytes': struct.pack('<I', index) + block.tobytes()}
            for index, block in enumerate(samples, start_sequence)]


@pytest.mark.asyncio
@pytest.mark.parametrize('origin', [None, 'https://wrong.example'])
async def test_remote_socket_rejects_missing_or_foreign_origin(monkeypatch, origin):
    import voice_app.remote_audio as remote_module

    websocket = FakeWebSocket('remote_microphone', [])
    if origin is None:
        websocket.headers.pop('origin')
    else:
        websocket.headers['origin'] = origin
    monkeypatch.setitem(remote_module.REMOTE_PAGES, 'origin-token', object())
    await remote_module.remote_audio_socket(websocket, 'origin-token')
    assert not websocket.accepted
    assert websocket.close_code == 1008


@pytest.mark.asyncio
async def test_remote_stop_accepts_delayed_tail_ack(monkeypatch):
    import voice_app.remote_audio as remote_module
    import voice_app.session as session_module
    import voice_app.ui as page

    monkeypatch.setattr(remote_module, 'ModelAPI', FakeAPI)
    monkeypatch.setattr(session_module, 'ModelAPI', FakeAPI)
    session = Session()
    state = page.PageState(session, Settings(), remote_token='slow-stop',
                           controls=SimpleNamespace(record_switch=SimpleNamespace(value=False),
                                                    source=SimpleNamespace(value='remote_microphone')))
    monkeypatch.setitem(remote_module.REMOTE_PAGES, 'slow-stop', state)

    class DelayedStop(FakeWebSocket):
        def __init__(self):
            super().__init__('remote_microphone', [])
            self.started = None

        async def receive(self):
            if self.started is None:
                self.started = asyncio.get_running_loop().time()
                session.source.stop.set()
            if asyncio.get_running_loop().time() - self.started < 2.2:
                await asyncio.sleep(.05)
                return {'type': 'websocket.receive'}
            return {'type': 'websocket.receive', 'text': '{"type":"stop"}'}

    websocket = DelayedStop()
    await remote_module.remote_audio_socket(websocket, 'slow-stop')
    await asyncio.wait_for(session.task, 5)
    assert any(message['type'] == 'stop' for message in websocket.sent)
    assert not session.source.error


@pytest.mark.asyncio
async def test_remote_writing_start_keeps_cursor_context(monkeypatch):
    import voice_app.remote_audio as remote_module
    import voice_app.session as session_module
    import voice_app.ui as page

    monkeypatch.setattr(remote_module, 'ModelAPI', FakeAPI)
    monkeypatch.setattr(session_module, 'ModelAPI', FakeAPI)
    session = Session()
    state = page.PageState(session, Settings(), remote_token='writing-token', writing_mode=True,
                           controls=SimpleNamespace(record_switch=SimpleNamespace(value=False),
                                                    source=SimpleNamespace(value='remote_microphone')))
    monkeypatch.setitem(remote_module.REMOTE_PAGES, 'writing-token', state)

    class WritingSocket(FakeWebSocket):
        async def receive_text(self):
            return json.dumps({'type': 'start', 'kind': self.kind,
                               'session_id': 'test-session-1234',
                               'writing_context': {'before': '上文', 'after': '下文'}})

    websocket = WritingSocket('remote_microphone', [
        {'type': 'websocket.receive', 'text': '{"type":"stop"}'}])
    await remote_module.remote_audio_socket(websocket, 'writing-token')
    await asyncio.wait_for(session.task, 5)
    assert state.writing_pending
    assert (session.writing_context_before, session.writing_context_after) == ('上文', '下文')


@pytest.mark.asyncio
async def test_writing_disconnect_keeps_finalized_text_available_for_insert(monkeypatch):
    import voice_app.remote_audio as remote_module
    import voice_app.session as session_module
    import voice_app.ui as page

    monkeypatch.setattr(remote_module, 'ModelAPI', FakeAPI)
    monkeypatch.setattr(session_module, 'ModelAPI', FakeAPI)
    session = Session()
    state = page.PageState(session, Settings(), remote_token='writing-disconnect', writing_mode=True,
                           controls=SimpleNamespace(record_switch=SimpleNamespace(value=False),
                                                    source=SimpleNamespace(value='remote_microphone')))
    monkeypatch.setitem(remote_module.REMOTE_PAGES, 'writing-disconnect', state)

    class WritingSocket(FakeWebSocket):
        async def receive_text(self):
            return json.dumps({'type': 'start', 'kind': self.kind,
                               'session_id': 'test-session-1234',
                               'writing_context': {'before': '上文', 'after': '下文'}})

    samples = np.concatenate((np.full(48000, .2), np.zeros(48000))).astype('<f4')
    messages = [{'type': 'websocket.receive', 'text': json.dumps({'type': 'format', 'sample_rate': 48000})}]
    messages += audio_messages(np.array_split(samples, 40))
    messages.append({'type': 'websocket.disconnect'})
    await remote_module.remote_audio_socket(WritingSocket('remote_microphone', messages), 'writing-disconnect')
    assert session.active and not session.source.done.is_set()
    assert not page.writing_commit_ready(session)
    resumed = FakeWebSocket('remote_microphone', [
        {'type': 'websocket.receive', 'text': '{"type":"stop"}'}], hello_type='resume')
    await remote_module.remote_audio_socket(resumed, 'writing-disconnect')
    await asyncio.wait_for(session.task, 5)

    assert session.corrected == '你好'
    assert session.error == ''
    assert not session.source.error
    assert any(message['type'] == 'resumed' and message['next_sequence'] == 40
               for message in resumed.sent)
    assert not session.active and not session.processing
    assert state.writing_pending and page.writing_commit_ready(session)
    session.error = '辨識失敗'
    assert not page.writing_commit_ready(session)


@pytest.mark.asyncio
@pytest.mark.parametrize('disconnect', [False, True])
async def test_remote_socket_finalizes_recording_on_stop_or_disconnect(monkeypatch, tmp_path, disconnect):
    import voice_app.session as session_module
    import voice_app.ui as page
    import voice_app.remote_audio as remote_module
    from nicegui import app

    assert any(route.path == '/remote-audio/{token}' for route in app.routes)

    monkeypatch.setattr(remote_module, 'ModelAPI', FakeAPI)
    monkeypatch.setattr(remote_module, 'RECONNECT_SECONDS', .02)
    monkeypatch.setattr(session_module, 'ModelAPI', FakeAPI)
    monkeypatch.setattr(session_module, 'reserve_recording_path',
                        lambda kind: reserve_recording_path(kind, tmp_path, datetime(2026, 9, 26)))
    session = Session()
    state = page.PageState(session, Settings(), remote_token='test-token',
                           controls=SimpleNamespace(record_switch=SimpleNamespace(value=True),
                                                    source=SimpleNamespace(value='remote_system')))
    monkeypatch.setitem(remote_module.REMOTE_PAGES, 'test-token', state)
    samples = np.concatenate((np.full(48000, .2), np.zeros(48000))).astype('<f4')
    messages = [{'type': 'websocket.receive', 'text': json.dumps({'type': 'format', 'sample_rate': 48000})}]
    messages += audio_messages(np.array_split(samples, 40))
    messages.append({'type': 'websocket.disconnect'} if disconnect else
                    {'type': 'websocket.receive', 'text': '{"type":"stop"}'})
    websocket = FakeWebSocket('remote_system', messages)
    await remote_module.remote_audio_socket(websocket, 'test-token')
    if disconnect:
        assert not session.source.done.is_set()
    await asyncio.wait_for(session.task, 5)
    assert any(message['type'] == 'ready' for message in websocket.sent)
    assert session.recording_state == ('incomplete' if disconnect else 'complete')
    assert session.recording_path.name == '遠端電腦音訊-20260926-001.wav'
    record = json.loads(session.recording_path.with_suffix('.json').read_text(encoding='utf-8'))
    assert record['recording']['sample_count'] == RATE * 2
    assert record['recording']['status'] == session.recording_state
    assert session.raw == '你好'


@pytest.mark.asyncio
async def test_remote_reconnect_replays_only_missing_packets_into_same_recording(monkeypatch, tmp_path):
    import voice_app.remote_audio as remote_module
    import voice_app.session as session_module
    import voice_app.ui as page

    monkeypatch.setattr(remote_module, 'ModelAPI', FakeAPI)
    monkeypatch.setattr(session_module, 'ModelAPI', FakeAPI)
    monkeypatch.setattr(session_module, 'reserve_recording_path',
                        lambda kind: reserve_recording_path(kind, tmp_path, datetime(2026, 9, 26)))
    session = Session()
    state = page.PageState(session, Settings(), remote_token='resume-recording',
                           controls=SimpleNamespace(record_switch=SimpleNamespace(value=True),
                                                    source=SimpleNamespace(value='remote_microphone')))
    monkeypatch.setitem(remote_module.REMOTE_PAGES, 'resume-recording', state)
    samples = np.concatenate((np.full(48000, .2), np.zeros(48000))).astype('<f4')
    blocks = np.array_split(samples, 40)
    first = FakeWebSocket('remote_microphone', [
        {'type': 'websocket.receive', 'text': json.dumps({'type': 'format', 'sample_rate': 48000})},
        *audio_messages(blocks[:20]),
        {'type': 'websocket.disconnect'},
    ])
    await remote_module.remote_audio_socket(first, 'resume-recording')
    assert session.active and session.source.next_sequence == 20
    assert 0 < session.source.recorded_samples <= RATE
    assert not session.source.done.is_set()

    second = FakeWebSocket('remote_microphone', [
        *audio_messages(blocks[19:], start_sequence=19),  # Lost ACK: resend the last accepted packet.
        {'type': 'websocket.receive', 'text': '{"type":"stop"}'},
    ], hello_type='resume')
    await remote_module.remote_audio_socket(second, 'resume-recording')
    await asyncio.wait_for(session.task, 5)
    assert second.sent[0] == {'type': 'resumed', 'next_sequence': 20}
    assert session.source.next_sequence == 40
    assert session.recording_state == 'complete'
    assert session.recording_path.name == '遠端麥克風-20260926-001.wav'
    with wave.open(str(session.recording_path), 'rb') as saved:
        assert saved.getnframes() == RATE * 2
    assert session.raw == '你好'


@pytest.mark.asyncio
async def test_old_socket_cannot_suspend_new_owner(monkeypatch):
    import voice_app.remote_audio as remote_module

    monkeypatch.setattr(remote_module, 'RECONNECT_SECONDS', .02)
    source = RemoteAudioSource('remote_microphone', settings=Settings())
    source.session_id = 'test-session-1234'
    old = object()
    new = object()
    source.socket_owner = old
    source.socket_active = True
    source.suspend(old)
    assert source.reconnect_task is not None
    assert source.attach(source.session_id, new) == 0
    source.suspend(old)
    await asyncio.sleep(.05)
    assert source.socket_owner is new and source.socket_active and not source.done.is_set()
    source.finish()


@pytest.mark.asyncio
async def test_resume_takes_over_active_old_socket(monkeypatch):
    import voice_app.remote_audio as remote_module
    import voice_app.session as session_module
    import voice_app.ui as page

    monkeypatch.setattr(session_module, 'ModelAPI', FakeAPI)
    session = Session()
    state = page.PageState(session, Settings(), remote_token='busy-resume')
    monkeypatch.setitem(remote_module.REMOTE_PAGES, 'busy-resume', state)
    session.start(Settings(), 'remote_microphone')
    source = session.source
    source.session_id = 'test-session-1234'
    source.socket_active = True
    old_owner = object()
    source.socket_owner = old_owner
    websocket = FakeWebSocket('remote_microphone', [
        {'type': 'websocket.disconnect'}], hello_type='resume')
    await remote_module.remote_audio_socket(websocket, 'busy-resume')
    assert websocket.sent == [{'type': 'resumed', 'next_sequence': 0}]
    assert source.socket_owner is None and not source.socket_active
    assert source.reconnect_task is not None and not source.done.is_set()
    source.suspend(old_owner)
    assert source.reconnect_task is not None and not source.done.is_set()
    assert session.error == ''
    await session.cancel()


@pytest.mark.asyncio
async def test_old_socket_stop_cannot_finish_new_owner_after_takeover(monkeypatch):
    import voice_app.remote_audio as remote_module
    import voice_app.session as session_module
    import voice_app.ui as page

    monkeypatch.setattr(remote_module, 'ModelAPI', FakeAPI)
    monkeypatch.setattr(session_module, 'ModelAPI', FakeAPI)
    session = Session()
    state = page.PageState(session, Settings(), remote_token='takeover-stop',
                           controls=SimpleNamespace(record_switch=SimpleNamespace(value=False),
                                                    source=SimpleNamespace(value='remote_microphone')))
    monkeypatch.setitem(remote_module.REMOTE_PAGES, 'takeover-stop', state)

    class QueueSocket(FakeWebSocket):
        def __init__(self, hello_type):
            super().__init__('remote_microphone', [], hello_type=hello_type)
            self.incoming = asyncio.Queue()
            self.ready = asyncio.Event()

        async def receive(self):
            return await self.incoming.get()

        async def send_json(self, value):
            await super().send_json(value)
            if value['type'] in ('ready', 'resumed'):
                self.ready.set()

    old = QueueSocket('start')
    old.incoming.put_nowait({'type': 'websocket.receive',
                            'text': '{"type":"format","sample_rate":48000}'})
    old_task = asyncio.create_task(remote_module.remote_audio_socket(old, 'takeover-stop'))
    await asyncio.wait_for(old.ready.wait(), 5)
    new = QueueSocket('resume')
    new.incoming.put_nowait({'type': 'websocket.receive',
                            'text': '{"type":"format","sample_rate":48000}'})
    new.incoming.put_nowait(audio_messages([np.zeros(4800, dtype='<f4')])[0])
    new_task = asyncio.create_task(remote_module.remote_audio_socket(new, 'takeover-stop'))
    await asyncio.wait_for(new.ready.wait(), 5)

    async def received_packet():
        while session.source.next_sequence < 1:
            await asyncio.sleep(.01)

    await asyncio.wait_for(received_packet(), 5)
    old.incoming.put_nowait({'type': 'websocket.receive', 'text': '{"type":"stop"}'})
    await asyncio.wait_for(old_task, 5)
    assert session.source.socket_owner is new
    assert not session.source.done.is_set()
    new.incoming.put_nowait({'type': 'websocket.receive', 'text': '{"type":"stop"}'})
    await asyncio.wait_for(new_task, 5)
    await asyncio.wait_for(session.task, 5)
    assert session.source.next_sequence == 1
    assert session.source.error == ''


@pytest.mark.asyncio
async def test_remote_heartbeat_answers_ping_and_expires_half_open_socket(monkeypatch):
    import voice_app.remote_audio as remote_module
    import voice_app.session as session_module
    import voice_app.ui as page

    monkeypatch.setattr(remote_module, 'ModelAPI', FakeAPI)
    monkeypatch.setattr(session_module, 'ModelAPI', FakeAPI)
    monkeypatch.setattr(remote_module, 'STALE_SOCKET_SECONDS', .02)
    monkeypatch.setattr(remote_module, 'RECONNECT_SECONDS', .02)
    session = Session()
    state = page.PageState(session, Settings(), remote_token='half-open',
                           controls=SimpleNamespace(record_switch=SimpleNamespace(value=False),
                                                    source=SimpleNamespace(value='remote_microphone')))
    monkeypatch.setitem(remote_module.REMOTE_PAGES, 'half-open', state)

    class HalfOpenSocket(FakeWebSocket):
        async def receive(self):
            try:
                return next(self.messages)
            except StopIteration:
                await asyncio.sleep(60)  # No close event and no further input.

    websocket = HalfOpenSocket('remote_microphone', [
        {'type': 'websocket.receive', 'text': '{"type":"ping"}'},
    ])
    await remote_module.remote_audio_socket(websocket, 'half-open')
    assert any(message['type'] == 'pong' for message in websocket.sent)
    assert session.source.reconnect_task is not None
    await asyncio.wait_for(session.task, 5)
    assert '超過 30 秒' in session.warning


@pytest.mark.asyncio
async def test_startup_progress_allows_same_session_start_to_take_over(monkeypatch):
    import voice_app.remote_audio as remote_module
    import voice_app.session as session_module
    import voice_app.ui as page

    preparing = asyncio.Event()
    release = asyncio.Event()

    class SlowAPI(FakeAPI):
        async def prepare_asr(self):
            preparing.set()
            await release.wait()

    class QueueSocket(FakeWebSocket):
        def __init__(self):
            super().__init__('remote_microphone', [])
            self.incoming = asyncio.Queue()
            self.progress = asyncio.Event()
            self.ready = asyncio.Event()

        async def receive(self):
            return await self.incoming.get()

        async def send_json(self, value):
            await super().send_json(value)
            if value['type'] == 'preparing':
                self.progress.set()
            if value['type'] in ('ready', 'resumed'):
                self.ready.set()

    monkeypatch.setattr(remote_module, 'ModelAPI', SlowAPI)
    monkeypatch.setattr(session_module, 'ModelAPI', FakeAPI)
    monkeypatch.setattr(remote_module, 'PREPARING_INTERVAL_SECONDS', .01)
    session = Session()
    state = page.PageState(session, Settings(), remote_token='startup-takeover',
                           controls=SimpleNamespace(record_switch=SimpleNamespace(value=False),
                                                    source=SimpleNamespace(value='remote_microphone')))
    monkeypatch.setitem(remote_module.REMOTE_PAGES, 'startup-takeover', state)
    old = QueueSocket()
    old_task = asyncio.create_task(remote_module.remote_audio_socket(old, 'startup-takeover'))
    try:
        await asyncio.wait_for(preparing.wait(), 5)
        await asyncio.wait_for(old.progress.wait(), 5)
        assert state.loading and state.remote_start_id == 'test-session-1234'
        replacement = QueueSocket()
        new_task = asyncio.create_task(remote_module.remote_audio_socket(replacement, 'startup-takeover'))
        await asyncio.wait_for(replacement.progress.wait(), 5)
        release.set()
        await asyncio.wait_for(replacement.ready.wait(), 5)
        assert any(item['type'] == 'resumed' for item in replacement.sent)
        assert session.source.socket_owner is replacement
        replacement.incoming.put_nowait({'type': 'websocket.receive',
                                         'text': '{"type":"stop"}'})
        await asyncio.wait_for(new_task, 5)
        await asyncio.wait_for(old_task, 5)
        await asyncio.wait_for(session.task, 5)
        assert not state.loading and state.remote_start_id == ''
        assert session.source.error == ''
    finally:
        release.set()
        if not old_task.done():
            old_task.cancel()


@pytest.mark.asyncio
async def test_startup_timeout_releases_loading_state(monkeypatch):
    import voice_app.remote_audio as remote_module
    import voice_app.ui as page

    class StuckAPI(FakeAPI):
        async def prepare_asr(self):
            await asyncio.sleep(60)

    monkeypatch.setattr(remote_module, 'ModelAPI', StuckAPI)
    monkeypatch.setattr(remote_module, 'STARTUP_SECONDS', .02)
    monkeypatch.setattr(remote_module, 'PREPARING_INTERVAL_SECONDS', .01)
    session = Session()
    state = page.PageState(session, Settings(), remote_token='startup-timeout',
                           controls=SimpleNamespace(record_switch=SimpleNamespace(value=False),
                                                    source=SimpleNamespace(value='remote_microphone')))
    monkeypatch.setitem(remote_module.REMOTE_PAGES, 'startup-timeout', state)
    websocket = FakeWebSocket('remote_microphone', [])
    await asyncio.wait_for(remote_module.remote_audio_socket(websocket, 'startup-timeout'), 5)
    assert any(item['type'] == 'preparing' for item in websocket.sent)
    assert any(item['type'] == 'error' and '準備逾時' in item['message']
               for item in websocket.sent)
    assert not state.loading and state.remote_start_id == ''
    assert session.source is None


@pytest.mark.asyncio
async def test_remote_cancel_keeps_partial_wav_and_json(monkeypatch, tmp_path):
    import voice_app.session as session_module

    monkeypatch.setattr(session_module, 'ModelAPI', FakeAPI)
    monkeypatch.setattr(session_module, 'reserve_recording_path',
                        lambda kind: reserve_recording_path(kind, tmp_path, datetime(2026, 9, 26)))
    session = Session()
    session.start(Settings(), 'remote_microphone', record_audio=True)
    session.source.connect(48000)
    session.source.receive(np.full(4800, .2, dtype='<f4').tobytes())
    await session.cancel()
    with wave.open(str(session.recording_path), 'rb') as saved:
        assert saved.getnframes() == session.source.recorded_samples
    record = json.loads(session.recording_path.with_suffix('.json').read_text(encoding='utf-8'))
    assert record['recording']['status'] == 'incomplete'
    assert record['recording']['sample_count'] > 0
