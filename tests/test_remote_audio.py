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
from voice_app.meeting import Meeting
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


GO = {'type': 'websocket.receive', 'text': '{"type":"go"}'}


class FakeWebSocket:
    """A browser as the server meets it: it says nothing but 'go' (when asked) until the server has started the recording, then
    sends `messages`. `confirms=False` is a browser that never answers the question."""

    def __init__(self, kind, messages, *, hello_type='start', confirms=True):
        self.kind = kind
        self.messages = iter(messages)
        self.hello_type = hello_type
        self.confirms = confirms
        self.owed_go = False
        self.released = False
        self.headers = {'origin': 'https://voice.example', 'host': 'voice.example'}
        self.sent = []
        self.accepted = False
        self.close_code = None

    def take_go(self):
        if self.owed_go:
            self.owed_go = False
            return GO

    async def quiet(self):
        """A connection with nothing to say, until it is asked to confirm."""
        for _ in range(12000):
            if (go := self.take_go()) is not None:
                return go
            await asyncio.sleep(.005)
        raise RuntimeError('a quiet socket was never asked anything')

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        return json.dumps({'type': self.hello_type, 'kind': self.kind,
                           'session_id': 'test-session-1234',
                           'writing_context': {'before': '', 'after': ''}})

    async def receive(self):
        while not self.released:
            if (go := self.take_go()) is not None:
                return go
            await asyncio.sleep(.005)
        return next(self.messages)

    async def send_json(self, value):
        self.sent.append(value)
        if value['type'] == 'prepared' and self.confirms:
            self.owed_go = True
        if value['type'] in ('ready', 'resumed'):
            self.released = True

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
            if session.source is None:                 # still loading the models: this browser says nothing but 'go'
                return await self.quiet()
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
async def test_while_a_remote_recording_runs_the_article_follows_it_live(monkeypatch):
    """The socket handler IS the recording loop and holds `state.loading` for as long as it runs: the page must show the live text and
    insert it when the run is over, not wait for the socket to close."""
    import voice_app.remote_audio as remote_module
    import voice_app.session as session_module
    import voice_app.ui as page

    monkeypatch.setattr(remote_module, 'ModelAPI', FakeAPI)
    monkeypatch.setattr(session_module, 'ModelAPI', FakeAPI)
    session = Session()
    state = page.PageState(session, Settings(), remote_token='writing-live', writing_mode=True,
                           controls=SimpleNamespace(record_switch=SimpleNamespace(value=False),
                                                    source=SimpleNamespace(value='remote_microphone')))
    monkeypatch.setitem(remote_module.REMOTE_PAGES, 'writing-live', state)
    gate = asyncio.Event()

    class HeldSocket(FakeWebSocket):
        async def receive(self):
            message = await super().receive()
            if '"stop"' in (message.get('text') or ''):
                await gate.wait()                                  # the speaker is still talking
            return message

    websocket = HeldSocket('remote_microphone', [{'type': 'websocket.receive', 'text': '{"type":"stop"}'}])
    handler = asyncio.create_task(remote_module.remote_audio_socket(websocket, 'writing-live'))
    for _ in range(2000):
        if session.active and state.writing_pending:
            break
        await asyncio.sleep(.005)
    assert session.active and state.writing_pending
    assert state.loading                                           # (the handler holds it while it records)
    assert page.writing_in_play(state, session) and not page.writing_commit_ready(session)
    gate.set()
    await asyncio.wait_for(handler, 10)
    await asyncio.wait_for(session.task, 10)
    assert not state.loading and page.writing_in_play(state, session) and page.writing_commit_ready(session)


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
    await session.start(Settings(), 'remote_microphone')
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
            if value['type'] == 'prepared':
                self.incoming.put_nowait(GO)
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
            if (go := self.take_go()) is not None:
                return go
            if not self.released:
                return await self.quiet()
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
            if value['type'] == 'prepared':
                self.incoming.put_nowait(GO)
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
    await session.start(Settings(), 'remote_microphone', record_audio=True)
    session.source.connect(48000)
    session.source.receive(np.full(4800, .2, dtype='<f4').tobytes())
    await session.cancel()
    with wave.open(str(session.recording_path), 'rb') as saved:
        assert saved.getnframes() == session.source.recorded_samples
    record = json.loads(session.recording_path.with_suffix('.json').read_text(encoding='utf-8'))
    assert record['recording']['status'] == 'incomplete'
    assert record['recording']['sample_count'] > 0

@pytest.mark.asyncio
async def test_remote_start_in_meeting_mode_needs_no_cursor_and_starts_a_meeting(monkeypatch, tmp_path):
    import voice_app.remote_audio as remote_module
    import voice_app.session as session_module
    import voice_app.ui as page

    monkeypatch.setattr(remote_module, 'ModelAPI', FakeAPI)
    monkeypatch.setattr(session_module, 'ModelAPI', FakeAPI)
    session = Session()
    made = []

    async def prepare_meeting():
        meeting = Meeting(Settings(), lambda samples: None, path=tmp_path / 'meeting.json')
        made.append(meeting)
        return meeting

    state = page.PageState(session, Settings(), remote_token='meeting-token', writing_mode=False, meeting_mode=True,
                           prepare_meeting=prepare_meeting,
                           controls=SimpleNamespace(record_switch=SimpleNamespace(value=False),
                                                    source=SimpleNamespace(value='remote_microphone')))
    monkeypatch.setitem(remote_module.REMOTE_PAGES, 'meeting-token', state)
    websocket = FakeWebSocket('remote_microphone', [{'type': 'websocket.receive', 'text': '{"type":"stop"}'}])
    try:
        await remote_module.remote_audio_socket(websocket, 'meeting-token')
        await asyncio.wait_for(session.task, 5)
        assert made and session.meeting is made[0] and made[0].ended        # started with no article anchor, finished with the session
        assert not state.writing_pending and session.error == ''
    finally:
        for meeting in made:
            await meeting.close()


class LoadingModels:
    """A remote start that is still loading the models (until `release`), with a previous meeting on the session. `made` collects
    the meetings the page prepared for the start."""

    def __init__(self, monkeypatch, tmp_path, token):
        import voice_app.remote_audio as remote_module
        import voice_app.session as session_module
        import voice_app.ui as page
        from voice_app.meeting import Meeting, reserve_meeting_path
        from tests.speaker_fakes import FakeEmbedder, bases

        self.release = asyncio.Event()
        self.models_fail = False
        outer = self

        class SlowAPI(FakeAPI):
            async def prepare_asr(self):
                await outer.release.wait()
                if outer.models_fail:
                    raise RuntimeError('ASR 載入失敗（測試）')

        monkeypatch.setattr(remote_module, 'ModelAPI', SlowAPI)
        monkeypatch.setattr(session_module, 'ModelAPI', FakeAPI)
        monkeypatch.setattr(remote_module, 'PREPARING_INTERVAL_SECONDS', .01)
        monkeypatch.setattr(remote_module, 'GO_SECONDS', .4)
        self.module = remote_module
        self.token = token
        self.session = Session()
        self.previous = Meeting(Settings(), FakeEmbedder(bases(2)), path=tmp_path / 'previous.json')
        self.session.meeting = self.previous
        self.made = []

        async def prepare_meeting():
            self.made.append(Meeting(Settings(), FakeEmbedder(bases(2)), path=reserve_meeting_path(tmp_path)))
            return self.made[-1]

        self.state = page.PageState(self.session, Settings(), remote_token=token, meeting_mode=True,
                                    prepare_meeting=prepare_meeting,
                                    controls=SimpleNamespace(record_switch=SimpleNamespace(value=False),
                                                             source=SimpleNamespace(value='remote_microphone')))
        monkeypatch.setitem(remote_module.REMOTE_PAGES, token, self.state)

    def connect(self, websocket):
        return asyncio.create_task(self.module.remote_audio_socket(websocket, self.token))

    def assert_nothing_leaked(self):
        """Every socket of the start has ended with it: no handler, no carrier task is left waiting."""
        assert not [task for task in asyncio.all_tasks()
                    if task is not asyncio.current_task() and not task.done()
                    and any(name in repr(task.get_coro()) for name in ('_carry', 'remote_audio_socket'))]

    def assert_nothing_started(self):
        assert self.session.meeting is self.previous and not self.previous._retired and not self.previous._closed
        assert self.session.source is None and not self.session.active
        assert all(meeting._closed and not meeting.path.exists() for meeting in self.made)     # what was prepared is thrown away
        assert not self.state.loading and self.state.remote_start_id == '' and self.state.remote_attempt is None


ERROR = {'type': 'websocket.receive', 'text': json.dumps({'type': 'error', 'message': '遠端音訊分享已中斷。'})}
DISCONNECT = {'type': 'websocket.disconnect'}


class ScriptedBrowser(FakeWebSocket):
    """A browser that says what it is told to (`speak`), and answers the question 'are you still there?' (`prepared`) as
    `on_prepared` says: 'go', 'error', 'disconnect' or 'silence'."""

    def __init__(self, on_prepared='go'):
        super().__init__('remote_microphone', [], confirms=False)
        self.on_prepared = on_prepared
        self.incoming = asyncio.Queue()
        self.preparing, self.prepared, self.ready = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def receive(self):
        return await self.incoming.get()

    def speak(self, message):
        self.incoming.put_nowait(message)

    async def send_json(self, value):
        await super().send_json(value)
        if value['type'] == 'preparing':
            self.preparing.set()
        if value['type'] == 'prepared':
            self.prepared.set()
            answers = {'go': GO, 'disconnect': DISCONNECT, 'error': ERROR}
            if self.on_prepared in answers:
                self.speak(answers[self.on_prepared])
        if value['type'] in ('ready', 'resumed'):
            self.ready.set()


@pytest.mark.asyncio
async def test_a_browser_that_gives_up_while_the_models_load_starts_nothing(monkeypatch, tmp_path):
    start = LoadingModels(monkeypatch, tmp_path, 'gives-up')
    websocket = ScriptedBrowser()
    task = start.connect(websocket)
    await asyncio.wait_for(websocket.preparing.wait(), 5)
    websocket.speak(ERROR)
    await asyncio.wait_for(task, 5)                                  # the models are still loading: nobody waits for them
    assert not start.release.is_set()
    assert any(item['type'] == 'error' and '分享已中斷' in item['message'] for item in websocket.sent)
    start.assert_nothing_started()
    start.release.set()
    await start.previous.close()


@pytest.mark.asyncio
async def test_a_socket_that_closed_during_the_wait_and_never_came_back_starts_nothing(monkeypatch, tmp_path):
    start = LoadingModels(monkeypatch, tmp_path, 'gone')
    websocket = ScriptedBrowser()
    task = start.connect(websocket)
    await asyncio.wait_for(websocket.preparing.wait(), 5)
    websocket.speak(DISCONNECT)                                      # the browser stopped sharing: no error could be sent
    await asyncio.sleep(.05)
    start.release.set()                                              # the models finish loading; nobody is there to confirm
    await asyncio.wait_for(task, 5)
    start.assert_nothing_started()
    assert '沒有回應' in start.session.error
    assert any(item['type'] == 'error' and '沒有回應' in item['message'] for item in websocket.sent)
    await start.previous.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('answer', ['error', 'silence', 'disconnect'])
async def test_a_browser_that_does_not_confirm_once_the_models_are_loaded_starts_nothing(monkeypatch, tmp_path, answer):
    """It failed (or went away, or never answers) at the very moment the models finished: nothing may start."""
    start = LoadingModels(monkeypatch, tmp_path, 'no-confirmation')
    websocket = ScriptedBrowser(on_prepared=answer)
    task = start.connect(websocket)
    await asyncio.wait_for(websocket.preparing.wait(), 5)
    start.release.set()
    await asyncio.wait_for(task, 5)
    assert websocket.prepared.is_set() and not websocket.ready.is_set()
    start.assert_nothing_started()
    if answer == 'silence':
        # a page that does not know 'prepared' (an older one) would take a plain close for a lost connection and reconnect, over
        # and over: it is told with an error, which stops it
        assert any(item['type'] == 'error' and '沒有回應' in item['message'] for item in websocket.sent)
    await start.previous.close()


@pytest.mark.asyncio
async def test_a_socket_that_closed_during_the_wait_is_replaced_in_time_and_the_start_goes_on(monkeypatch, tmp_path):
    start = LoadingModels(monkeypatch, tmp_path, 'replaced')
    old, replacement = ScriptedBrowser(on_prepared='silence'), ScriptedBrowser()
    old_task = start.connect(old)
    await asyncio.wait_for(old.preparing.wait(), 5)
    old.speak(DISCONNECT)                                              # a network blip during the long wait
    await asyncio.sleep(.05)
    new_task = start.connect(replacement)                              # the browser reconnects, as its script does
    await asyncio.wait_for(replacement.preparing.wait(), 5)
    start.release.set()
    await asyncio.wait_for(replacement.ready.wait(), 5)                # ... and it is the replacement that confirms
    assert start.session.source is not None and start.session.source.socket_owner is replacement
    assert start.session.meeting is start.made[0] and start.previous._retired
    replacement.speak({'type': 'websocket.receive', 'text': '{"type":"stop"}'})
    await asyncio.wait_for(new_task, 5)
    await asyncio.wait_for(old_task, 5)
    await asyncio.wait_for(start.session.task, 5)
    await start.session.meeting.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('how', ['error', 'closes'])
async def test_a_replacement_that_died_does_not_count_as_somebody_still_there(monkeypatch, tmp_path, how):
    """The first socket is gone; a replacement joined, and then it too ended (the browser gave up, or its connection closed)."""
    start = LoadingModels(monkeypatch, tmp_path, 'replacement-died')
    old, replacement = ScriptedBrowser(on_prepared='silence'), ScriptedBrowser(on_prepared='silence')
    old_task = start.connect(old)
    await asyncio.wait_for(old.preparing.wait(), 5)
    old.speak(DISCONNECT)
    await asyncio.sleep(.05)
    new_task = start.connect(replacement)
    await asyncio.wait_for(replacement.preparing.wait(), 5)
    replacement.speak(ERROR if how == 'error' else DISCONNECT)
    await asyncio.sleep(.05)
    start.release.set()                                                # the models finish loading
    await asyncio.wait_for(old_task, 5)
    await asyncio.wait_for(new_task, 5)
    start.assert_nothing_started()
    await start.previous.close()


@pytest.mark.asyncio
async def test_remote_meeting_start_works_with_the_page_state_the_app_really_builds(monkeypatch, tmp_path):
    """The page never turns `writing_mode` off: in meeting mode the browser simply sends no article anchor."""
    import voice_app.remote_audio as remote_module
    import voice_app.session as session_module
    import voice_app.ui as page

    monkeypatch.setattr(remote_module, 'ModelAPI', FakeAPI)
    monkeypatch.setattr(session_module, 'ModelAPI', FakeAPI)
    session = Session()
    made = []

    async def prepare_meeting():
        meeting = Meeting(Settings(), lambda samples: None, path=tmp_path / 'meeting.json')
        made.append(meeting)
        return meeting

    state = page.PageState(session, Settings(), remote_token='real-state', meeting_mode=True, prepare_meeting=prepare_meeting,
                           controls=SimpleNamespace(record_switch=SimpleNamespace(value=False),
                                                    source=SimpleNamespace(value='remote_microphone')))
    assert state.writing_mode is True                                            # as built by the page
    monkeypatch.setitem(remote_module.REMOTE_PAGES, 'real-state', state)

    class NoAnchor(FakeWebSocket):
        async def receive_text(self):
            return json.dumps({'type': 'start', 'kind': self.kind, 'session_id': 'test-session-1234', 'writing_context': None})

    websocket = NoAnchor('remote_microphone', [{'type': 'websocket.receive', 'text': '{"type":"stop"}'}])
    try:
        await remote_module.remote_audio_socket(websocket, 'real-state')
        await asyncio.wait_for(session.task, 5)
        assert made and session.meeting is made[0] and made[0].ended
        assert not state.writing_pending and session.error == ''
        assert not any(item['type'] == 'error' for item in websocket.sent)
    finally:
        for meeting in made:
            await meeting.close()


@pytest.mark.asyncio
async def test_a_replacement_that_stays_silent_ends_with_the_start_when_the_confirmation_times_out(monkeypatch, tmp_path):
    start = LoadingModels(monkeypatch, tmp_path, 'silent-replacement')
    old, replacement = ScriptedBrowser(on_prepared='silence'), ScriptedBrowser(on_prepared='silence')
    old_task = start.connect(old)
    await asyncio.wait_for(old.preparing.wait(), 5)
    old.speak(DISCONNECT)
    await asyncio.sleep(.05)
    new_task = start.connect(replacement)
    await asyncio.wait_for(replacement.preparing.wait(), 5)
    start.release.set()                                                # the models load; nobody confirms
    await asyncio.wait_for(asyncio.gather(old_task, new_task), 5)      # both sockets end by themselves (the limit is .4 s)
    for websocket in (old, replacement):
        assert any(item['type'] == 'error' and '沒有回應' in item['message'] for item in websocket.sent)
    start.assert_nothing_started()
    start.assert_nothing_leaked()
    await start.previous.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('ending', ['models fail', 'handler cancelled'])
async def test_every_socket_of_a_start_ends_with_it_whatever_ends_it(monkeypatch, tmp_path, ending):
    start = LoadingModels(monkeypatch, tmp_path, 'ends-together')
    first, replacement = ScriptedBrowser(), ScriptedBrowser()
    first_task = start.connect(first)
    await asyncio.wait_for(first.preparing.wait(), 5)
    new_task = start.connect(replacement)                              # a second socket of the same start, both alive
    await asyncio.wait_for(replacement.preparing.wait(), 5)
    if ending == 'models fail':
        start.models_fail = True
        start.release.set()
    else:
        first_task.cancel()
    await asyncio.wait_for(new_task, 5)                                # the replacement is not left waiting
    if ending == 'models fail':
        await asyncio.wait_for(first_task, 5)
        assert any(item['type'] == 'error' and 'ASR 載入失敗' in item['message'] for item in replacement.sent)
    else:
        assert any(item['type'] == 'error' for item in replacement.sent)
        await asyncio.gather(first_task, return_exceptions=True)
    start.assert_nothing_started()
    start.assert_nothing_leaked()
    start.release.set()
    await start.previous.close()


@pytest.mark.asyncio
async def test_a_start_cancelled_after_the_browser_was_asked_to_confirm_throws_its_meeting_away(monkeypatch, tmp_path):
    """The meeting is made before the question is asked: cancelling while waiting for the answer must not leave it behind."""
    start = LoadingModels(monkeypatch, tmp_path, 'cancel-after-prepared')
    browser = ScriptedBrowser(on_prepared='silence')
    task = start.connect(browser)
    await asyncio.wait_for(browser.preparing.wait(), 5)
    start.release.set()
    await asyncio.wait_for(browser.prepared.wait(), 5)
    assert len(start.made) == 1 and not start.made[0]._closed and start.made[0].path.exists()
    task.cancel()                                                      # the page goes away while the answer is awaited
    await asyncio.gather(task, return_exceptions=True)
    meeting = start.made[0]
    assert meeting._closed and not meeting.path.exists()
    assert meeting._editor_api.client.is_closed and meeting._translator_api.client.is_closed
    start.assert_nothing_started()
    start.assert_nothing_leaked()
    await start.previous.close()


@pytest.mark.asyncio
async def test_a_second_cancel_while_the_unused_meeting_is_being_thrown_away_cuts_nothing_short(monkeypatch, tmp_path):
    """The cleanup of the handler's last step runs to its end however many times it is cancelled meanwhile: the page state is
    released, the meeting is gone, and the socket is closed."""
    from voice_app.meeting import Meeting

    real_discard = Meeting.discard

    async def slow_discard(self):
        await asyncio.sleep(.3)
        await real_discard(self)

    monkeypatch.setattr(Meeting, 'discard', slow_discard)
    start = LoadingModels(monkeypatch, tmp_path, 'second-cancel')
    browser = ScriptedBrowser(on_prepared='silence')
    task = start.connect(browser)
    await asyncio.wait_for(browser.preparing.wait(), 5)
    start.release.set()
    await asyncio.wait_for(browser.prepared.wait(), 5)
    meeting = start.made[0]
    task.cancel()                                                      # the page goes away ...
    await asyncio.sleep(.1)                                            # ... the meeting is being thrown away ...
    assert not meeting._closed and not task.done()
    task.cancel()                                                      # ... and the page goes away again
    await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()                                            # the cancellation is passed on, but only at the end
    assert meeting._closed and not meeting.path.exists()
    assert meeting._editor_api.client.is_closed and meeting._translator_api.client.is_closed
    assert browser.close_code is not None                              # the socket was closed after all
    start.assert_nothing_started()
    start.assert_nothing_leaked()
    await start.previous.close()


@pytest.mark.asyncio
async def test_a_cleanup_that_fails_is_reported_and_does_not_stop_the_others(caplog):
    import voice_app.remote_audio as remote_module

    done = []

    async def fails():
        await asyncio.sleep(.01)
        raise PermissionError('檔案被占用（測試）')

    async def slow_but_fine():
        await asyncio.sleep(.1)
        done.append('closed')

    with caplog.at_level('ERROR'):
        with pytest.raises(PermissionError, match='檔案被占用'):
            await remote_module._finish(fails(), slow_but_fine())
    assert done == ['closed']                                          # the other cleanup was not cut short by the failure
    assert any('清理失敗' in record.getMessage() and record.exc_info for record in caplog.records)


@pytest.mark.asyncio
async def test_a_failed_cleanup_is_logged_when_a_cancellation_is_on_its_way(caplog):
    import voice_app.remote_audio as remote_module

    done = []

    async def fails():
        await asyncio.sleep(.05)
        raise PermissionError('檔案被占用（測試）')

    async def slow_but_fine():
        await asyncio.sleep(.15)
        done.append('closed')

    task = asyncio.create_task(remote_module._finish(fails(), slow_but_fine()))
    await asyncio.sleep(.02)
    with caplog.at_level('ERROR'):
        task.cancel()                                                  # the caller is cancelled while the cleanups run
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()                                            # the cancellation goes first ...
    assert done == ['closed']
    assert any('清理失敗' in record.getMessage() and '檔案被占用' in str(record.exc_info[1]) for record in caplog.records)   # ... the failure is on record


@pytest.mark.asyncio
async def test_a_meeting_that_cannot_be_thrown_away_is_reported_and_the_socket_is_still_closed(monkeypatch, tmp_path, caplog):
    from voice_app.meeting import Meeting

    async def cannot_discard(self):
        raise PermissionError('預留檔案無法刪除（測試）')

    monkeypatch.setattr(Meeting, 'discard', cannot_discard)
    start = LoadingModels(monkeypatch, tmp_path, 'discard-fails')
    browser = ScriptedBrowser(on_prepared='silence')                   # never confirms: the start ends and its meeting must go
    task = start.connect(browser)
    await asyncio.wait_for(browser.preparing.wait(), 5)
    start.release.set()
    with caplog.at_level('ERROR'):
        with pytest.raises(PermissionError, match='預留檔案無法刪除'):
            await asyncio.wait_for(task, 5)
    assert browser.close_code is not None                              # still closed
    assert not start.state.loading and start.state.remote_start_id == '' and start.state.remote_attempt is None
    assert any('清理失敗' in record.getMessage() for record in caplog.records)
    start.session.meeting = start.previous
    await start.previous.close()
    for meeting in start.made:                                         # the fake discard left it open: close it for real
        await meeting.close()


@pytest.mark.asyncio
async def test_a_cancellation_that_was_already_on_its_way_is_not_replaced_by_a_cleanup_failure(caplog):
    """The caller entered its `finally` with a CancelledError: when a cleanup there fails, the task must still end cancelled."""
    import voice_app.remote_audio as remote_module

    async def fails():
        await asyncio.sleep(.01)
        raise PermissionError('檔案被占用（測試）')

    async def caller():
        try:
            await asyncio.sleep(10)
        finally:
            await remote_module._finish(fails())

    task = asyncio.create_task(caller())
    await asyncio.sleep(.02)
    with caplog.at_level('ERROR'):
        task.cancel()                                                  # once, before the cleanup starts
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
    assert any('清理失敗' in record.getMessage() and '檔案被占用' in str(record.exc_info[1]) for record in caplog.records)


@pytest.mark.asyncio
async def test_a_handler_cancelled_while_waiting_for_the_answer_stays_cancelled_when_the_meeting_cannot_be_discarded(
        monkeypatch, tmp_path, caplog):
    from voice_app.meeting import Meeting

    async def cannot_discard(self):
        await asyncio.sleep(.05)
        raise PermissionError('預留檔案無法刪除（測試）')

    monkeypatch.setattr(Meeting, 'discard', cannot_discard)
    start = LoadingModels(monkeypatch, tmp_path, 'cancelled-and-failing')
    browser = ScriptedBrowser(on_prepared='silence')
    task = start.connect(browser)
    await asyncio.wait_for(browser.preparing.wait(), 5)
    start.release.set()
    await asyncio.wait_for(browser.prepared.wait(), 5)
    with caplog.at_level('ERROR'):
        task.cancel()                                                  # one cancellation only, while the answer is awaited
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()                                            # not replaced by the PermissionError
    assert any('清理失敗' in record.getMessage() and '預留檔案無法刪除' in str(record.exc_info[1]) for record in caplog.records)
    assert browser.close_code is not None
    assert not start.state.loading and start.state.remote_attempt is None
    start.session.meeting = start.previous
    await start.previous.close()
    for meeting in start.made:
        await meeting.close()
