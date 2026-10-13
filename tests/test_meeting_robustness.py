"""Regressions found in review: a second meeting on the same session, overlapping saves, names after an automatic merge, and
translation segments that failed."""
import asyncio
import json
import queue
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import voice_app.audio as audio
import voice_app.session as session_module
from voice_app.audio import Chunk, RATE, reserve_recording_path
from voice_app.config import MeetingSettings, Settings
from voice_app.meeting import Meeting, reserve_meeting_path
from voice_app.session import Session, StartCancelled
from voice_app.speaker import default_name
from voice_app.speaker.service import SpeakerService
from voice_app.translation import Translated
from tests.speaker_fakes import FakeEmbedder, bases, build, make_turns, window_of
from tests.test_meeting import ReplayAPI, fake_meeting, window_results
from tests.test_speaker_service import feed_meeting
from tests.test_translation import FakeLLM, drain, make


class ScriptedSource:
    """Stands in for AudioSource: hands the session the chunks it is given, then ends."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.stop, self.cancelled, self.done = threading.Event(), threading.Event(), threading.Event()
        self.error, self.level, self.recorded_samples = '', 0.0, 0
        self.queue = queue.Queue()
        self.thread = SimpleNamespace(is_alive=lambda: False)

    def start(self):
        pass

    async def wait_open(self):
        pass

    async def next_chunk(self):
        if self.chunks:
            return self.chunks.pop(0)
        self.done.set()
        return None

    async def close(self):
        pass


class ReplayContext(ReplayAPI):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def scripted_run(monkeypatch, pattern, seed):
    """The chunks and the recognition of one fake meeting, wired in as the session's audio source and model client."""
    audio, tokens, _ = build(make_turns(pattern, seed=seed))
    count = window_of(tokens[-1].start) + 2
    results = window_results(tokens, count)
    chunks = [Chunk(k, 3.0 * k, audio[int(3.0 * k * RATE):int((3.0 * k + 6.0) * RATE)], True, 'window') for k in range(count)]
    monkeypatch.setattr(session_module, 'AudioSource', lambda *a, **k: ScriptedSource(chunks))
    monkeypatch.setattr(session_module, 'ModelAPI', lambda settings: ReplayContext(results))
    return count


@pytest.mark.asyncio
@pytest.mark.parametrize('second_pattern', [[1, 0] * 4, [1, 0] * 16], ids=['shorter second meeting', 'longer second meeting'])
async def test_a_second_meeting_on_the_same_session_leaves_the_first_ones_file_alone(tmp_path, monkeypatch, second_pattern):
    session = Session()
    first = Meeting(Settings(), FakeEmbedder(bases(2)), path=tmp_path / 'first.json')
    first.settings.meeting.autosave_seconds = 0.03
    scripted_run(monkeypatch, [0, 1] * 10, seed=4)
    await session.start(Settings(), 'file', path=Path('x.wav'), meeting=first)
    await asyncio.wait_for(session.task, 20)
    before = json.loads(first.path.read_text(encoding='utf-8'))
    assert first.ended and len(before['segments']) == len(session.segments) > 5

    second = Meeting(Settings(), FakeEmbedder(bases(2)), path=tmp_path / 'second.json')
    scripted_run(monkeypatch, second_pattern, seed=9)
    await session.start(Settings(), 'file', path=Path('y.wav'), meeting=second)
    await asyncio.wait_for(session.task, 20)
    assert len(session.segments) != len(before['segments'])                      # a different meeting

    first._mark({1})                                                              # whatever makes the first one want to save again
    await asyncio.sleep(0.3)
    after = json.loads(first.path.read_text(encoding='utf-8'))
    assert len(after['segments']) == len(before['segments'])                      # still its own segments
    assert len(first.doc()['segments']) == len(before['segments'])
    assert not first.translator.running
    assert len(json.loads(second.path.read_text(encoding='utf-8'))['segments']) == len(session.segments)
    await second.close()


OPENING_FAILS = ['construct', 'start', 'busy device', 'corrupt file', 'device that cannot be opened']


def failing_audio(monkeypatch, tmp_path, case):
    """Every way the audio can fail to open, as the session's start meets it: the source cannot be built, its `start()` raises, or
    (the real ones) the thread it starts finds the device in use, the file undecodable, the device unopenable. The arguments
    for `session.start` come back."""
    if case in ('construct', 'start'):
        class Broken:
            def __init__(self, *args, **kwargs):
                if case == 'construct':
                    raise OSError('找不到錄音裝置')

            def start(self):
                raise OSError('錄音裝置無法開啟')

            async def close(self):
                pass

        monkeypatch.setattr(session_module, 'AudioSource', Broken)
        return 'microphone', 'device-1', None
    monkeypatch.setattr(session_module, 'AudioSource', audio.AudioSource)         # the real one, whatever an earlier step put there
    if case == 'busy device':
        held = threading.Lock()                                                    # another window is recording
        held.acquire()
        monkeypatch.setattr(audio, 'HARDWARE_LOCK', held)
    elif case == 'corrupt file':
        path = tmp_path / 'broken.wav'
        path.write_bytes(b'this is not audio at all' * 50)
        return 'file', '', path
    else:
        def refuses(kind, device_id, stop):
            raise OSError('裝置無法開啟')
            yield

        monkeypatch.setattr(audio, 'record_device', refuses)
    return 'microphone', 'device-1', None


def reserved_recordings(monkeypatch, directory):
    monkeypatch.setattr(session_module, 'reserve_recording_path',
                        lambda kind: reserve_recording_path(kind, directory))


@pytest.mark.asyncio
@pytest.mark.parametrize('case', OPENING_FAILS)
async def test_audio_that_cannot_open_leaves_neither_a_meeting_nor_files_behind(tmp_path, monkeypatch, case):
    recordings = tmp_path / 'recordings'
    reserved_recordings(monkeypatch, recordings)
    kind, device, path = failing_audio(monkeypatch, tmp_path, case)
    session = Session()
    meeting = Meeting(Settings(), FakeEmbedder(bases(2)), path=reserve_meeting_path(tmp_path))
    assert meeting.path.exists()                                                   # reserved the moment the meeting is made
    with pytest.raises((OSError, RuntimeError)):
        await session.start(Settings(), kind, device, path, record_audio=kind != 'file', meeting=meeting)
    assert session.meeting is None and not session.active and session.recording_path is None and not session._starting
    assert not list(recordings.glob('*'))                                          # the reserved recording file is given back too
    await meeting.discard()
    assert not meeting.path.exists() and not meeting.path.with_suffix('.txt').exists()
    assert meeting._editor_api.client.is_closed and meeting._translator_api.client.is_closed
    await meeting.discard()                                                        # twice is harmless
    assert list(tmp_path.glob('*.json')) == []


@pytest.mark.asyncio
async def test_the_reason_the_thread_gave_is_what_the_failed_start_says(tmp_path, monkeypatch):
    kind, device, path = failing_audio(monkeypatch, tmp_path, 'busy device')
    with pytest.raises(RuntimeError, match='已有其他視窗正在錄音'):
        await Session().start(Settings(), kind, device, path)
    kind, device, path = failing_audio(monkeypatch, tmp_path, 'corrupt file')
    with pytest.raises(RuntimeError, match='音訊來源錯誤'):
        await Session().start(Settings(), kind, device, path)


@pytest.mark.asyncio
@pytest.mark.parametrize('case', OPENING_FAILS)
async def test_a_start_that_fails_leaves_the_previous_meeting_running_as_it_was(tmp_path, monkeypatch, case):
    session = Session()
    first = Meeting(Settings(), FakeEmbedder(bases(2)), path=tmp_path / 'first.json')
    scripted_run(monkeypatch, [0, 1] * 10, seed=4)
    await session.start(Settings(), 'file', path=Path('x.wav'), meeting=first)
    await asyncio.wait_for(session.task, 20)
    kept = len(session.segments)
    before = first.path.read_text(encoding='utf-8')
    assert first.ended and kept > 5

    kind, device, path = failing_audio(monkeypatch, tmp_path, case)
    second = Meeting(Settings(), FakeEmbedder(bases(2)), path=reserve_meeting_path(tmp_path))
    with pytest.raises((OSError, RuntimeError)):
        await session.start(Settings(), kind, device, path, meeting=second)
    assert session.meeting is first and len(session.segments) == kept             # still showing the finished meeting
    assert not first._retired and not first._closed
    await second.discard()
    await asyncio.sleep(0.1)
    assert first.path.read_text(encoding='utf-8') == before and not second.path.exists()
    first.translator.start('English', 3)                                           # and it still works
    assert first.translator.running
    await first.close()


def slow_device(monkeypatch, delay, blocks=3):
    """A device that takes `delay` seconds to open, then gives a few blocks and waits to be stopped."""
    def device(kind, device_id, stop):
        time.sleep(delay)
        for _ in range(blocks):
            yield np.full(1600, .1, dtype=np.float32)
        while not stop.is_set():
            time.sleep(.01)

    monkeypatch.setattr(audio, 'record_device', device)
    monkeypatch.setattr(session_module, 'ModelAPI', lambda settings: ReplayContext([]))


@pytest.mark.asyncio
async def test_a_device_that_is_slow_to_open_is_waited_for_and_then_the_meeting_replaces_the_old_one(tmp_path, monkeypatch):
    slow_device(monkeypatch, 0.4)
    session = Session()
    meeting = Meeting(Settings(), FakeEmbedder(bases(2)), path=reserve_meeting_path(tmp_path))
    started = time.monotonic()
    await session.start(Settings(), 'microphone', 'device-1', meeting=meeting)
    assert time.monotonic() - started >= 0.35                                      # it did not commit before the device was open
    assert session.meeting is meeting and session.active and session.source.opened.is_set()
    await session.cancel()
    await meeting.close()


@pytest.mark.asyncio
async def test_a_device_that_fails_only_after_a_long_wait_is_still_a_failed_start(tmp_path, monkeypatch):
    """No time limit decides a slow source is a working one: the start waits until it has opened or failed."""
    def late_failure(kind, device_id, stop):
        time.sleep(0.5)
        raise OSError('裝置在很久之後才失敗')
        yield

    monkeypatch.setattr(audio, 'record_device', late_failure)
    session = Session()
    meeting = Meeting(Settings(), FakeEmbedder(bases(2)), path=reserve_meeting_path(tmp_path))
    with pytest.raises(RuntimeError, match='很久之後'):
        await session.start(Settings(), 'microphone', 'device-1', meeting=meeting)
    assert session.meeting is None and not session.active and session.source is None
    await meeting.discard()
    assert not meeting.path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('record', [False, True], ids=['no recording', 'with recording'])
async def test_cancel_stops_a_source_that_is_still_opening_and_nothing_starts_afterwards(tmp_path, monkeypatch, record):
    recordings = tmp_path / 'recordings'
    reserved_recordings(monkeypatch, recordings)
    slow_device(monkeypatch, 0.8)
    session = Session()
    meeting = Meeting(Settings(), FakeEmbedder(bases(2)), path=reserve_meeting_path(tmp_path))
    status = session.status
    opening = asyncio.create_task(session.start(Settings(), 'microphone', 'device-1', record_audio=record, meeting=meeting))
    await asyncio.sleep(0.15)
    assert session.starting and session._opening is not None and session.status == '正在開啟音訊來源'
    started = time.monotonic()
    await session.cancel()                                                         # the user, or the page going away
    assert not session.starting and time.monotonic() - started < 2.5
    with pytest.raises(StartCancelled):
        await opening
    await asyncio.sleep(1.0)                                                       # past the moment the device would have opened
    assert session.meeting is None and not session.active and session.source is None and session.task is None
    assert session.status == status
    assert not list(recordings.glob('*'))
    await meeting.discard()
    assert not meeting.path.exists()


@pytest.mark.asyncio
async def test_a_cancel_while_opening_leaves_the_previous_meeting_alone(tmp_path, monkeypatch):
    session = Session()
    first = Meeting(Settings(), FakeEmbedder(bases(2)), path=tmp_path / 'first.json')
    scripted_run(monkeypatch, [0, 1] * 10, seed=4)
    await session.start(Settings(), 'file', path=Path('x.wav'), meeting=first)
    await asyncio.wait_for(session.task, 20)
    kept = len(session.segments)

    monkeypatch.setattr(session_module, 'AudioSource', audio.AudioSource)
    slow_device(monkeypatch, 0.6)
    second = Meeting(Settings(), FakeEmbedder(bases(2)), path=reserve_meeting_path(tmp_path))
    opening = asyncio.create_task(session.start(Settings(), 'microphone', 'device-1', meeting=second))
    await asyncio.sleep(0.1)
    await session.cancel()
    with pytest.raises(StartCancelled):
        await opening
    assert session.meeting is first and len(session.segments) == kept and not first._retired
    await second.discard()
    await first.close()


@pytest.mark.asyncio
async def test_a_file_without_any_sound_is_a_meeting_that_ends_not_a_failed_start(tmp_path, monkeypatch):
    path = tmp_path / 'silent.wav'
    with wave.open(str(path), 'wb') as empty:
        empty.setnchannels(1)
        empty.setsampwidth(2)
        empty.setframerate(RATE)
    monkeypatch.setattr(session_module, 'ModelAPI', lambda settings: ReplayContext([]))
    session = Session()
    meeting = Meeting(Settings(), FakeEmbedder(bases(2)), path=reserve_meeting_path(tmp_path))
    await session.start(Settings(), 'file', path=path, meeting=meeting)
    await asyncio.wait_for(session.task, 5)
    assert session.meeting is meeting and meeting.ended and not session.segments
    await meeting.close()


@pytest.mark.asyncio
async def test_a_second_start_while_the_first_is_still_opening_is_refused(tmp_path, monkeypatch):
    slow_device(monkeypatch, 0.3)
    session = Session()
    first = asyncio.create_task(session.start(Settings(), 'microphone', 'device-1'))
    await asyncio.sleep(0.05)
    with pytest.raises(ValueError, match='請先完成'):
        await session.start(Settings(), 'microphone', 'device-1')
    await first
    assert session.active
    await session.cancel()


@pytest.mark.asyncio
async def test_discarding_a_meeting_that_did_start_saves_it_like_closing(tmp_path):
    meeting, session, play, _ = fake_meeting(tmp_path, [0, 1] * 8)
    await play(0, 8)
    await meeting.discard()
    assert meeting._closed and json.loads(meeting.path.read_text(encoding='utf-8'))['segments']


@pytest.mark.asyncio
async def test_a_remote_start_that_fails_does_not_leave_an_empty_meeting(tmp_path, monkeypatch):
    import voice_app.remote_audio as remote_module
    import voice_app.ui as page
    from tests.test_remote_audio import FakeAPI, FakeWebSocket

    reserved_recordings(monkeypatch, tmp_path / 'recordings')
    monkeypatch.setattr(remote_module, 'ModelAPI', FakeAPI)
    monkeypatch.setattr(session_module, 'ModelAPI', FakeAPI)

    def broken(*args, **kwargs):
        raise OSError('遠端音訊來源無法建立')

    monkeypatch.setattr(session_module, 'RemoteAudioSource', broken)
    session = Session()
    made = []

    async def prepare_meeting():
        made.append(Meeting(Settings(), FakeEmbedder(bases(2)), path=reserve_meeting_path(tmp_path)))
        return made[0]

    state = page.PageState(session, Settings(), remote_token='remote-fails', meeting_mode=True, prepare_meeting=prepare_meeting,
                           controls=SimpleNamespace(record_switch=SimpleNamespace(value=True),
                                                    source=SimpleNamespace(value='remote_system')))
    monkeypatch.setitem(remote_module.REMOTE_PAGES, 'remote-fails', state)
    class QuietSocket(FakeWebSocket):                                                # connected, and says nothing but 'go'
        async def receive(self):
            return await self.quiet()

    websocket = QuietSocket('remote_system', [])
    await remote_module.remote_audio_socket(websocket, 'remote-fails')
    assert made and websocket.sent[-1]['type'] == 'error'
    assert session.meeting is None and not session.active and not state.loading
    assert '無法建立' in session.error
    assert made[0]._closed and not made[0].path.exists()
    assert not list((tmp_path / 'recordings').glob('*'))


@pytest.mark.asyncio
async def test_saves_never_overlap_and_the_last_snapshot_is_the_one_that_stays(tmp_path, monkeypatch):
    meeting, session, play, _ = fake_meeting(tmp_path, [0, 1] * 14)
    state = {'now': 0, 'most': 0}
    written = []

    def slow_store(files):
        state['now'] += 1
        state['most'] = max(state['most'], state['now'])
        time.sleep(0.06)
        written.append(len(json.loads(files[0][1])['segments']))
        state['now'] -= 1

    monkeypatch.setattr(Meeting, '_store', staticmethod(slow_store))
    await play(0, 6)
    under_way = asyncio.create_task(meeting.save())
    await asyncio.sleep(0.02)                                                     # its write is running in a thread
    await play(6, 14)
    under_way.cancel()                                                            # what closing does to a save in progress
    await meeting.close()
    assert state['most'] == 1
    assert written[-1] == len(session.segments)


@pytest.mark.asyncio
async def test_many_saves_at_once_do_not_trip_over_one_another_on_disk(tmp_path):
    meeting, session, play, _ = fake_meeting(tmp_path, [0, 1] * 8)
    try:
        await play(0, 10)
        await asyncio.gather(*(meeting.save() for _ in range(8)))
        assert meeting.save_error == ''
        assert json.loads(meeting.path.read_text(encoding='utf-8'))['segments']
        assert not list(tmp_path.glob('*.tmp'))                                   # no half-written leftovers
    finally:
        await meeting.close()


@pytest.mark.asyncio
async def test_a_name_survives_the_automatic_merge_of_its_group_into_a_bigger_one():
    segments = []
    service = SpeakerService(settings=MeetingSettings(), embed=FakeEmbedder(bases(3)), llm=lambda **kw: asyncio.sleep(0),
                             segments=lambda: segments)
    await feed_meeting(service, segments, make_turns([0, 1, 2] * 14, seed=4))
    tracker = service.tracker
    assert tracker.group_count == 3
    big, small = sorted(tracker.group_ids, key=lambda g: -tracker._voices[g].seconds)[:2]
    service.set_name(small, '陳教授')
    # the two voices turn out to be one: make their centres the same and let the minute's housekeeping notice
    tracker._voices[small].sum = tracker._voices[big].sum.copy()
    tracker._voices[small].centroid = tracker._voices[big].centroid.copy()
    tracker._rebuild()
    changed = tracker._auto_merge()
    assert tracker.resolve(small) == big and changed
    service._after_tracker(changed)
    assert service.name_of(big) == '陳教授'
    assert small not in service.names
    assert any(e.get('event') == 'merge_auto' for e in service.name_events)


@pytest.mark.asyncio
async def test_a_segment_that_failed_to_translate_is_translated_when_translation_is_started_again(monkeypatch):
    real = asyncio.sleep

    async def quick(seconds, *a):
        await real(0)

    monkeypatch.setattr('voice_app.translation.asyncio.sleep', quick)
    meeting, llm, translator, _ = make(count=8, lag=3)
    llm.fail = 100                                                                 # the server is down for a while
    for _ in range(14):
        await translator._step()
    assert translator.failed() and all(translator.tr_state(i) == 'error' for i in translator.failed())
    failed = list(translator.failed())
    llm.fail = 0                                                                   # it is back
    translator.running = False
    translator.start('English', 3)                                                 # the user presses "start" again
    await translator.stop()
    translator.running = True
    await drain(translator)
    assert translator.failed() == [] and all(translator.tr_state(i) == 'ok' for i in failed)
    assert '失敗' not in translator.status and translator.error == ''


@pytest.mark.asyncio
async def test_failed_segments_can_be_retried_on_their_own_and_the_status_says_how_many(monkeypatch):
    real = asyncio.sleep

    async def quick(seconds, *a):
        await real(0)

    monkeypatch.setattr('voice_app.translation.asyncio.sleep', quick)
    meeting, llm, translator, changes = make(count=8, lag=3)
    llm.fail = 100
    for _ in range(14):
        await translator._step()
    meeting.ended = True
    await drain(translator, 80)
    assert translator.failed() and '失敗' in translator.status                     # not a plain "done"
    llm.fail = 0
    count = translator.retry_failed()
    assert count > 0 and translator.failed() == [] and any(c for c in changes)
    await drain(translator)
    assert translator.status == '翻譯完成' and translator.stats['segments'] >= count


@pytest.mark.asyncio
async def test_after_a_merge_a_translation_split_by_speaker_is_shown_and_exported_under_the_speaker_that_remains(tmp_path):
    meeting, session, play, count = fake_meeting(tmp_path, [0, 1] * 14)
    try:
        await play(0, count)
        speaker, translator = meeting.speaker, meeting.translator
        keep, drop = speaker.tracker.group_ids[:2]
        both = [segment for segment in meeting._segments if {r.gid for r in speaker.translation_runs(segment)} == {keep, drop}]
        assert both
        translator.language = 'English'
        for segment in both:                                                # translated while they were two people
            runs = speaker.translation_runs(segment)
            translator.results[segment.index] = Translated(segment.index, tuple(r.text for r in runs),
                                                           [f'T{k}' for k in range(len(runs))], tuple(r.gid for r in runs))
        meeting.drain()
        speaker.merge(drop, keep)
        fewer = [segment for segment in both if len(speaker.translation_runs(segment)) != len(translator.results[segment.index].texts)]
        assert fewer                                                        # the runs of one person ran together: not one per old text
        rows = {row['i']: row for row in meeting.drain()['rows']}
        for segment in fewer:
            assert {run['g'] for run in rows[segment.index]['tr']['runs']} == {keep}      # not the colour and name of the one merged away
        text = meeting.export_text()
        assert default_name(drop) not in text and default_name(keep) in text
    finally:
        await meeting.close()
