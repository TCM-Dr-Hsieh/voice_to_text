"""The replay clip of a meeting: cut out of a recording that is still being written, only for the meeting whose token is given."""
import io
import math
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI

import voice_app.replay as replay
from voice_app.replay import ReplayError, blocked_reason, clip_wav, register_replay_routes

RATE = 16000


class Answer:
    def __init__(self, response):
        self.status_code, self.content = response.status_code, response.body
        self.headers = {name.lower(): value for name, value in response.headers.items()}

    @property
    def text(self):
        return self.content.decode('utf-8')


class Route:
    """The route called as a function: no web server and no threads. `get` answers the way the test client of the web framework does."""

    def __init__(self, app):
        self.routes = {route.path: route.endpoint for route in app.routes if hasattr(route, 'endpoint')}

    def get(self, path, params=None, headers=None):
        params = params or {}
        request = SimpleNamespace(headers={name.lower(): value for name, value in (headers or {}).items()})
        return Answer(self.routes[path](request, start=float(params.get('start', 0.0)), token=params.get('token', '')))


def live_wav(path: Path, seconds: float, *, rate: int = RATE, extra_bytes: int = b''):
    """A recording as it is while it is written: a header whose sizes are still those of an empty file, then the samples. The sample at
    index i is i % 30000, so the start of a clip tells where it was cut."""
    count = int(seconds * rate)
    header = (b'RIFF' + struct.pack('<I', 36) + b'WAVE' + b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 1, rate, rate * 2, 2, 16)
              + b'data' + struct.pack('<I', 0))
    samples = (np.arange(count) % 30000).astype('<i2')
    path.write_bytes(header + samples.tobytes() + extra_bytes)
    return path


def read_clip(body: bytes):
    with wave.open(io.BytesIO(body)) as clip:
        return clip.getframerate(), clip.getnchannels(), clip.getsampwidth(), np.frombuffer(clip.readframes(clip.getnframes()), dtype='<i2')


def test_the_clip_starts_at_the_requested_second_and_lasts_30_seconds(tmp_path):
    path = live_wav(tmp_path / 'a.wav', 100)
    rate, channels, width, samples = read_clip(clip_wav(path, 42))
    assert (rate, channels, width) == (RATE, 1, 2) and len(samples) == 30 * RATE
    assert samples[0] == (42 * RATE) % 30000 and samples[-1] == (72 * RATE - 1) % 30000


def test_a_clip_that_reaches_past_what_is_recorded_is_shorter_and_not_padded(tmp_path):
    path = live_wav(tmp_path / 'a.wav', 50)
    _, _, _, samples = read_clip(clip_wav(path, 40))
    assert len(samples) == 10 * RATE and samples[0] == (40 * RATE) % 30000


def test_a_start_between_two_seconds_is_cut_where_it_says(tmp_path):
    path = live_wav(tmp_path / 'a.wav', 60)
    _, _, _, samples = read_clip(clip_wav(path, 12.25))
    assert samples[0] == int(12.25 * RATE) % 30000


def test_a_half_written_sample_at_the_end_is_ignored(tmp_path):
    path = live_wav(tmp_path / 'a.wav', 12, extra_bytes=b'\x7f')
    _, _, _, samples = read_clip(clip_wav(path, 10))
    assert len(samples) == 2 * RATE


def test_a_recording_with_a_patched_header_and_other_chunks_is_cut_the_same(tmp_path):
    path = tmp_path / 'closed.wav'
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes((np.arange(20 * RATE) % 30000).astype('<i2').tobytes())
    _, _, _, samples = read_clip(clip_wav(path, 5))
    assert len(samples) == 15 * RATE and samples[0] == (5 * RATE) % 30000


@pytest.mark.parametrize('start', [20.0, 21, 1e9])
def test_a_start_past_what_is_recorded_is_refused(tmp_path, start):
    path = live_wav(tmp_path / 'a.wav', 20)
    with pytest.raises(ReplayError, match='還沒有錄到') as caught:
        clip_wav(path, start)
    assert caught.value.status == 404


@pytest.mark.parametrize('start', [-1, -0.001, math.nan, math.inf, -math.inf])
def test_a_start_that_is_not_a_time_is_refused(tmp_path, start):
    path = live_wav(tmp_path / 'a.wav', 20)
    with pytest.raises(ReplayError) as caught:
        clip_wav(path, start)
    assert caught.value.status == 400


def test_a_missing_empty_or_foreign_file_is_refused_with_a_reason(tmp_path):
    with pytest.raises(ReplayError, match='還不存在'):
        clip_wav(tmp_path / 'none.wav', 0)
    empty = tmp_path / 'empty.wav'
    empty.write_bytes(b'')
    with pytest.raises(ReplayError, match='還沒有可用的內容'):
        clip_wav(empty, 0)
    junk = tmp_path / 'junk.wav'
    junk.write_bytes(b'not a wave file at all')
    with pytest.raises(ReplayError, match='還沒有可用的內容'):
        clip_wav(junk, 0)
    stereo = tmp_path / 'stereo.wav'
    stereo.write_bytes(b'RIFF' + struct.pack('<I', 36) + b'WAVE' + b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 2, RATE, RATE * 4, 4, 16)
                       + b'data' + struct.pack('<I', 0) + bytes(64))
    with pytest.raises(ReplayError, match='單聲道') as caught:
        clip_wav(stereo, 0)
    assert caught.value.status == 500


class VirtualWav:
    """A recording of `seconds` that is not on any disk: the 44 bytes of the header, then samples (the sample at index i is i % 30000).
    It counts what is read from it, which is what a clip costs."""

    def __init__(self, seconds: float):
        self.header = (b'RIFF' + struct.pack('<I', 36) + b'WAVE' + b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 1, RATE, RATE * 2, 2, 16)
                       + b'data' + struct.pack('<I', 0))
        self.size = 44 + 2 * int(seconds * RATE)
        self.position = 0
        self.read_bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.size + offset if whence == 2 else self.position + offset)

    def tell(self):
        return self.position

    def read(self, count=-1):
        end = self.size if count < 0 else min(self.size, self.position + count)
        out = self.header[self.position:min(end, 44)] if self.position < 44 else b''
        if end > 44:
            first, last = max(self.position, 44) - 44, end - 44
            assert first % 2 == 0 and last % 2 == 0, 'the samples are read whole'
            out += (np.arange(first // 2, last // 2) % 30000).astype('<i2').tobytes()
        self.read_bytes += len(out)
        self.position = end
        return out


def test_a_recording_of_two_hours_is_cut_from_any_place_in_it_reading_only_the_clip(monkeypatch):
    opened = []

    def virtual_open(path, mode='rb'):
        opened.append(VirtualWav(7200))                                  # two hours of 16-bit mono: 230 MB, nowhere on a disk
        return opened[-1]

    monkeypatch.setattr(replay, 'open', virtual_open, raising=False)
    for start, wanted in ((0.0, 30 * RATE), (5400.5, 30 * RATE), (7199.0, RATE)):
        _, _, _, samples = read_clip(clip_wav(Path('two-hours.wav'), start))
        assert len(samples) == wanted and samples[0] == int(start * RATE) % 30000 and samples[-1] == (int(start * RATE) + wanted - 1) % 30000
        assert opened[-1].read_bytes <= 2 * wanted + 44 + 64              # the clip and a little header: not the file
    assert opened[0].size > 230_000_000


def test_the_sources_that_capture_what_the_computer_plays_cannot_be_replayed_and_say_why():
    here = Path('a.wav')
    assert blocked_reason('microphone', here) == '' and blocked_reason('remote_microphone', here) == ''
    assert '同一個輸出' in blocked_reason('loopback', here) and blocked_reason('remote_system', here) == blocked_reason('loopback', here)
    assert '沒有錄音' in blocked_reason('microphone', None) and '沒有錄音' in blocked_reason('file', None)


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, 'MEETINGS', {})
    app = FastAPI()
    register_replay_routes(app)
    meeting = SimpleNamespace(replay_token='secret-token-1', audio_path=live_wav(tmp_path / 'm.wav', 60), replay_blocked='')
    replay.register(meeting)
    return Route(app), meeting


def test_the_route_answers_with_the_clip_of_the_meeting_whose_token_it_is_given(served):
    client, meeting = served
    response = client.get('/meeting-replay.wav', params={'start': 10, 'token': meeting.replay_token})
    assert response.status_code == 200 and response.headers['content-type'] == 'audio/wav'
    assert response.headers['cache-control'] == 'no-store'
    _, _, _, samples = read_clip(response.content)
    assert len(samples) == 30 * RATE and samples[0] == (10 * RATE) % 30000


@pytest.mark.parametrize('token', ['', 'wrong', 'secret-token-', 'secret-token-1 ', 'SECRET-TOKEN-1', '密碼'])
def test_the_route_refuses_any_other_token(served, token):
    client, _ = served
    response = client.get('/meeting-replay.wav', params={'start': 0, 'token': token})
    assert response.status_code == 409 and '重新整理' in response.text and response.headers['cache-control'] == 'no-store'
    assert client.get('/meeting-replay.wav', params={'start': 0}).status_code == 409


def test_a_meeting_that_has_ended_for_the_page_is_not_served_any_more(served):
    client, meeting = served
    replay.unregister(meeting)
    assert client.get('/meeting-replay.wav', params={'start': 0, 'token': meeting.replay_token}).status_code == 409
    replay.unregister(meeting)                                                  # (twice is harmless)


def test_the_route_tells_why_a_meeting_cannot_be_replayed_and_why_a_time_cannot_be_served(served):
    client, meeting = served
    meeting.replay_blocked = blocked_reason('loopback', meeting.audio_path)
    blocked = client.get('/meeting-replay.wav', params={'start': 0, 'token': meeting.replay_token})
    assert blocked.status_code == 409 and '同一個輸出' in blocked.text
    meeting.replay_blocked = ''
    assert client.get('/meeting-replay.wav', params={'start': 600, 'token': meeting.replay_token}).status_code == 404
    assert client.get('/meeting-replay.wav', params={'start': -3, 'token': meeting.replay_token}).status_code == 400


def test_the_route_refuses_a_request_the_browser_calls_cross_site(served):
    client, meeting = served
    response = client.get('/meeting-replay.wav', params={'start': 0, 'token': meeting.replay_token}, headers={'Sec-Fetch-Site': 'cross-site'})
    assert response.status_code == 403
    assert client.get('/meeting-replay.wav', params={'start': 0, 'token': meeting.replay_token}, headers={'Sec-Fetch-Site': 'same-origin'}).status_code == 200


def test_two_meetings_open_at_once_each_get_their_own_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, 'MEETINGS', {})
    app = FastAPI()
    register_replay_routes(app)
    first = SimpleNamespace(replay_token='token-of-first', audio_path=live_wav(tmp_path / '1.wav', 40), replay_blocked='')
    second = SimpleNamespace(replay_token='token-of-second', audio_path=live_wav(tmp_path / '2.wav', 10), replay_blocked='')
    replay.register(first)
    replay.register(second)
    client = Route(app)
    assert len(read_clip(client.get('/meeting-replay.wav', params={'start': 0, 'token': 'token-of-first'}).content)[3]) == 30 * RATE
    assert len(read_clip(client.get('/meeting-replay.wav', params={'start': 0, 'token': 'token-of-second'}).content)[3]) == 10 * RATE


# ---------------------------------------------------------------------- the meeting: token, audio, what the page is told

import asyncio

from voice_app.audio import Chunk
from voice_app.config import Settings
import voice_app.meeting as meeting_module
from voice_app.meeting import Meeting
from voice_app.session import Session
from voice_app.speaker.render import Run
from tests.speaker_fakes import FakeEmbedder, bases


def meeting_of(kind: str, recording: Path | None, folder: Path):
    """A meeting attached to a session of this kind, as the session starts one. Its saving does nothing here: it is done in a worker thread
    that an `asyncio.wait_for` cannot stop (the saving waits for it even when it is cancelled), and these tests are about the replay."""
    folder.mkdir(exist_ok=True)
    meeting = Meeting(Settings(), FakeEmbedder(bases(1)), path=folder / 'meeting.json')

    async def no_saving():
        pass
    meeting.save = no_saving
    session = Session()
    session.kind = kind
    session.recording_path = recording
    session.meeting = meeting
    return meeting, session


@pytest.mark.asyncio
async def test_a_meeting_with_a_recording_registers_for_the_replay_and_leaves_when_its_page_lets_go(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, 'MEETINGS', {})
    meeting, session = meeting_of('microphone', live_wav(tmp_path / 'm.wav', 40), tmp_path / 'meeting')
    assert replay.MEETINGS == {} and meeting.replay_blocked == replay.NO_RECORDING           # (not before it is attached)
    meeting.attach(session)
    try:
        assert replay.MEETINGS == {meeting.replay_token: meeting} and meeting.replay_blocked == ''
        assert len(meeting.replay_token) >= 20
        labels = meeting._labels()
        assert labels['replay'] == {'token': meeting.replay_token, 'blocked': ''}
        app = FastAPI()
        register_replay_routes(app)
        response = Route(app).get('/meeting-replay.wav', params={'start': 5, 'token': meeting.replay_token})
        assert response.status_code == 200 and len(read_clip(response.content)[3]) == 30 * RATE
    finally:
        await meeting.close()
    assert replay.MEETINGS == {}                                                              # closed: its audio is not asked for any more


@pytest.mark.asyncio
async def test_the_next_run_of_the_session_ends_the_replay_of_the_meeting_before_it(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, 'MEETINGS', {})
    meeting, session = meeting_of('microphone', live_wav(tmp_path / 'm.wav', 40), tmp_path / 'meeting')
    meeting.attach(session)
    meeting.retire()
    assert replay.MEETINGS == {}                                                              # at once, not when its closing is done
    await asyncio.gather(*list(meeting_module._CLOSING))                                      # (its closing runs as a task of its own)
    assert meeting.replay_token not in replay.MEETINGS and meeting._closed


@pytest.mark.asyncio
@pytest.mark.parametrize('kind, recording, why', [('loopback', True, '同一個輸出'), ('remote_system', True, '同一個輸出'),
                                                 ('microphone', False, '沒有錄音'), ('file', False, '沒有錄音')])
async def test_a_meeting_whose_audio_cannot_be_replayed_says_why_to_the_page_and_to_the_route(tmp_path, monkeypatch, kind, recording, why):
    monkeypatch.setattr(replay, 'MEETINGS', {})
    meeting, session = meeting_of(kind, live_wav(tmp_path / 'm.wav', 40) if recording else None, tmp_path / 'meeting')
    meeting.attach(session)
    try:
        assert why in meeting.replay_blocked and meeting._labels()['replay'] == {'token': meeting.replay_token, 'blocked': meeting.replay_blocked}
        app = FastAPI()
        register_replay_routes(app)
        response = Route(app).get('/meeting-replay.wav', params={'start': 0, 'token': meeting.replay_token})
        assert response.status_code == 409 and why in response.text
    finally:
        await meeting.close()


@pytest.mark.asyncio
async def test_every_row_tells_the_page_where_its_audio_starts(tmp_path):
    meeting, _ = meeting_of('microphone', None, tmp_path / 'meeting')
    meeting._segments = [SimpleNamespace(index=1, start=0.0, end=9.0, owned_start=0.0), SimpleNamespace(index=2, start=3.0, end=12.0, owned_start=9.0),
                         SimpleNamespace(index=3, start=6.0, end=15.0, owned_start=12.004),
                         SimpleNamespace(index=4, start=9.0, end=18.0, owned_start=None)]       # (a segment made without it: its window)
    meeting.speaker.runs = lambda segment: [Run(0, 'voice', f'text {segment.index}')]
    try:
        rows = [meeting._row(index) for index in (1, 2, 3, 4)]
        assert [row['t'] for row in rows] == [0.0, 9.0, 12.0, 9.0]                              # (rounded to the hundredth)
        assert all(isinstance(row['t'], float) for row in rows)
    finally:
        await meeting.close()


@pytest.mark.asyncio
async def test_two_meetings_on_one_server_have_tokens_of_their_own(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, 'MEETINGS', {})
    first, a = meeting_of('microphone', live_wav(tmp_path / '1.wav', 40), tmp_path / 'one')
    second, b = meeting_of('microphone', live_wav(tmp_path / '2.wav', 40), tmp_path / 'two')
    first.attach(a)
    second.attach(b)
    try:
        assert first.replay_token != second.replay_token and len(replay.MEETINGS) == 2
    finally:
        await first.close()
        await second.close()


def test_the_page_of_the_app_serves_the_clip_route_and_loads_the_player_script():
    import voice_app.ui as page
    from nicegui import app
    paths = {getattr(route, 'path', '') for route in app.routes}
    assert '/meeting-replay.wav' in paths and '/replay.js' in paths
    assert '/replay.js?v=' in Path(page.__file__).read_text(encoding='utf-8')            # (the head of every page loads it)


ASGI_CHILD = """
import asyncio, json, sys
from pathlib import Path
from types import SimpleNamespace
import httpx
from fastapi import FastAPI
import voice_app.replay as replay
from tests.test_replay import RATE, live_wav, read_clip


async def main():
    app = FastAPI()
    replay.register_replay_routes(app)
    meeting = SimpleNamespace(replay_token='token-through-asgi', audio_path=live_wav(Path(sys.argv[1]) / 'a.wav', 60), replay_blocked='')
    replay.register(meeting)
    seen = {}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        async def ask(**kwargs):
            return (await client.get('/meeting-replay.wav', **kwargs))
        ok = await ask(params={'start': '10.5', 'token': meeting.replay_token})
        samples = read_clip(ok.content)[3]
        seen['ok'] = [ok.status_code, ok.headers['content-type'], ok.headers['cache-control'], len(samples), int(samples[0]) == int(10.5 * RATE) % 30000]
        seen['soon'] = (await ask(params={'start': 'soon', 'token': meeting.replay_token})).status_code
        seen['wrong token'] = (await ask(params={'start': 0, 'token': 'wrong'})).status_code
        seen['no token'] = (await ask()).status_code
        seen['negative'] = (await ask(params={'start': -1, 'token': meeting.replay_token})).status_code
        seen['past the end'] = (await ask(params={'start': 1e9, 'token': meeting.replay_token})).status_code
        seen['cross-site'] = (await ask(params={'start': 0, 'token': meeting.replay_token}, headers={'Sec-Fetch-Site': 'cross-site'})).status_code
    print(json.dumps(seen))


asyncio.run(main())
"""


def test_through_the_web_framework_the_query_the_request_and_the_thread_pool_work_and_a_bad_query_is_refused(tmp_path):
    """The route as the web framework serves it (the query read and checked by it, the request handed to the endpoint, the endpoint run
    in its thread pool). The endpoint runs in a worker thread that nothing in this process could stop, so this runs in a process of
    its own: if it does not answer, the process is killed and the test fails, instead of holding the whole suite."""
    import subprocess
    import sys
    root = Path(__file__).resolve().parent.parent
    try:
        run = subprocess.run([sys.executable, '-c', ASGI_CHILD, str(tmp_path)], cwd=root, capture_output=True, text=True, encoding='utf-8',
                             timeout=90)
    except subprocess.TimeoutExpired:
        pytest.fail('the route did not answer within 90 s (the worker thread of the web framework is stuck): its process was killed')
    assert run.returncode == 0, run.stderr[-2000:]
    import json
    assert json.loads(run.stdout.strip().splitlines()[-1]) == {
        'ok': [200, 'audio/wav', 'no-store', 30 * RATE, True], 'soon': 422, 'wrong token': 409, 'no token': 409, 'negative': 400,
        'past the end': 404, 'cross-site': 403}
