import asyncio
import json
import queue
import threading
import numpy as np
import pytest

from voice_app.api import ModelError, RollingResult
from voice_app.audio import AudioSource, Chunk, RATE, wav_bytes
from voice_app.config import Settings
from voice_app.session import Gap, Session, before_context
from voice_app.text import join_texts


def recognition(text, times=None):
    letters = [c for c in text if c.isalnum()]
    times = times or [(i * .2, i * .2 + .15) for i in range(len(letters))]
    return {'text': text, 'items': [{'text': c, 'start': a, 'end': b} for c, (a, b) in zip(letters, times)]}


class API:
    def __init__(self, results):
        self.results = iter(results)
        self.corrections = []

    async def recognize(self, _):
        return next(self.results)

    async def revise_recent(self, context, segments, on_preview, **kwargs):
        self.corrections.append((context, segments))
        texts = [row['current'] for row in segments]
        on_preview(join_texts(texts))
        return RollingResult(texts)


@pytest.mark.asyncio
async def test_each_window_is_corrected_when_asr_arrives():
    api = API([recognition('天氣很好。'), recognition('我們去散步。')])
    session = Session()
    await session.process_chunk(api, Chunk(0, 0, np.zeros(RATE * 4), True, 'window'))
    assert session.corrected == '天氣很好。'
    await session.process_chunk(api, Chunk(1, 4, np.zeros(RATE * 4), True, 'stop'))
    assert session.corrected == '天氣很好。我們去散步。'
    assert api.corrections[1][0] == ''
    assert api.corrections[1][1][0]['current'] == '天氣很好。'
    assert 'api_key' not in session.export_json()


@pytest.mark.asyncio
async def test_writing_context_surrounds_only_current_voice_window():
    class ContextAPI(API):
        async def revise_recent(self, context, segments, on_preview, **kwargs):
            self.corrections.append((context, kwargs['context_after'],
                                     [row['added'] for row in segments]))
            return RollingResult([row['current'] for row in segments])

    api = ContextAPI([recognition(item) for item in ('甲', '乙', '丙', '丁')])
    session = Session()
    session.writing_context_before = '前文：'
    session.writing_context_after = '，後文。'
    for index in range(4):
        await session.process_chunk(api, Chunk(index, index * 4, np.zeros(RATE * 4), True, 'pause'))
    assert api.corrections[0][:2] == ('前文：', '，後文。')
    assert api.corrections[3][0] == '前文：甲'
    assert api.corrections[3][1] == '，後文。'
    assert api.corrections[3][2] == ['乙', '丙', '丁']


def test_long_writing_keeps_article_and_recent_locked_voice_within_budget():
    assert before_context('文' * 100, '聲' * 100, 100) == '文' * 40 + '聲' * 60
    assert before_context('開頭', '聲' * 100, 100) == '開頭' + '聲' * 98
    assert before_context('文' * 100, '短', 100) == '文' * 99 + '短'
    assert before_context('', '聲' * 120, 100) == '聲' * 100


@pytest.mark.asyncio
async def test_left_overlap_is_owned_by_previous_window():
    api = API([recognition('甲乙丙', [(1, 1.5), (9, 9.5), (10.5, 11)]),
               recognition('丙丁', [(0, .5), (3, 3.5)])])
    session = Session()
    await session.process_chunk(api, Chunk(0, 0, np.zeros(RATE * 12), True, 'window'))
    assert session.corrected == '甲乙丙'
    await session.process_chunk(api, Chunk(1, 11, np.zeros(RATE * 5), True, 'stop'))
    assert session.corrected == session.raw == '甲乙丙丁'
    assert session.segments[1].raw == '丙丁'
    assert session.segments[1].added == '丁'
    assert session.committed_end == 16


@pytest.mark.asyncio
async def test_alignment_failure_does_not_change_committed_state():
    api = API([recognition('原稿'), {'text': '文字', 'items': []}])
    session = Session()
    await session.process_chunk(api, Chunk(0, 0, np.zeros(RATE * 4)))
    with pytest.raises(RuntimeError):
        await session.process_chunk(api, Chunk(1, 4, np.zeros(RATE * 4)))
    assert session.corrected == '原稿' and len(session.segments) == 1


@pytest.mark.asyncio
async def test_cancel_during_correction_keeps_previous_commit():
    entered = asyncio.Event()
    class SlowAPI(API):
        async def revise_recent(self, *args, **kwargs):
            entered.set()
            await asyncio.Future()
    session = Session()
    await session.process_chunk(API([recognition('原稿')]), Chunk(0, 0, np.zeros(RATE * 4)))
    api = SlowAPI([recognition('新句')])
    task = asyncio.create_task(session.process_chunk(api, Chunk(1, 4, np.zeros(RATE * 4))))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert session.corrected == '原稿' and len(session.segments) == 1


@pytest.mark.asyncio
async def test_valid_edit_is_applied_to_session_without_pending_review():
    class EditAPI(API):
        async def revise_recent(self, context, segments, on_preview, **kwargs):
            return RollingResult(['修改'])
    session = Session()
    await session.process_chunk(EditAPI([recognition('原稿')]), Chunk(0, 0, np.zeros(RATE * 4)))
    assert session.corrected == '修改'
    assert session.revisions[0].after == ['修改']
    assert session.revisions[0].after == ['修改']


@pytest.mark.asyncio
async def test_latest_three_segments_revise_together_then_oldest_locks():
    class RollingAPI(API):
        async def revise_recent(self, context, segments, on_preview, **kwargs):
            self.corrections.append((context, segments))
            texts = [row['current'] for row in segments]
            if [row['index'] for row in segments] == [1, 2, 3]:
                texts[0] = '甲改'
            return RollingResult(texts)

    api = RollingAPI([recognition(item) for item in ('甲', '乙', '丙', '丁')])
    session = Session()
    for index in range(4):
        await session.process_chunk(api, Chunk(index, index * 4, np.zeros(RATE * 4), True, 'pause'))
    assert session.corrected == '甲改乙丙丁'
    assert [row['index'] for row in api.corrections[2][1]] == [1, 2, 3]
    assert [row['index'] for row in api.corrections[3][1]] == [2, 3, 4]
    assert api.corrections[3][0] == '甲改'
    assert session.revisions[0].before == ['甲', '乙', '丙']
    assert session.revisions[0].after == ['甲改', '乙', '丙']
    exported = json.loads(session.export_json())
    assert exported['revisable_segments'] == []  # process_chunk alone does not activate a session.
    assert exported['revisions'][0]['after'] == ['甲改', '乙', '丙']


@pytest.mark.asyncio
async def test_failed_three_segment_revision_preserves_current_text():
    class FailedAPI(API):
        async def revise_recent(self, context, segments, on_preview, **kwargs):
            texts = [row['current'] for row in segments]
            return RollingResult(texts, '校稿失敗')

    session = Session()
    await session.process_chunk(FailedAPI([recognition('原稿')]),
                                Chunk(0, 0, np.zeros(RATE * 4)))
    assert session.corrected == '原稿' and not session.revisions
    assert session.warning == '校稿失敗'


@pytest.mark.asyncio
async def test_retry_preserves_failed_audio(monkeypatch, tmp_path):
    import voice_app.session as module
    class FailingAPI(API):
        attempts = 0
        def __init__(self, settings):
            super().__init__([recognition('測試')])
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def recognize(self, samples):
            FailingAPI.attempts += 1
            if FailingAPI.attempts == 1:
                raise ModelError('temporary')
            return await super().recognize(samples)
    monkeypatch.setattr(module, 'ModelAPI', FailingAPI)
    path = tmp_path / 'speech.wav'
    path.write_bytes(wav_bytes(np.full(RATE * 2, .1)))
    session = Session()
    session.start(Settings(), 'file', path=path)
    await session.task
    assert session.failed_chunk is not None and not session.segments
    session.retry()
    await session.task
    assert session.corrected == '測試' and not session.active
    assert len(session.segments) == 1


@pytest.mark.asyncio
async def test_repeated_final_alignment_failure_can_be_skipped_and_later_audio_continues(monkeypatch):
    import voice_app.session as module

    class AlignmentAPI(API):
        calls = []
        def __init__(self, settings):
            super().__init__([])
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def recognize(self, samples):
            number = round(float(samples[0]))
            if number == 2:
                return {'text': '失敗原稿', 'items': [
                    {'text': '失敗原稿', 'start': 0, 'end': 9}]}
            return recognition('前文' if number == 1 else '後文')
        async def revise_recent(self, context, segments, on_preview, **kwargs):
            self.calls.append([row['index'] for row in segments])
            texts = [row['current'] for row in segments]
            return RollingResult(texts)

    class Source:
        def __init__(self):
            self.chunks = [Chunk(i - 1, (i - 1) * 4,
                                 np.full(RATE * 4, i, dtype=np.float32), True, 'pause')
                           for i in (1, 2, 3)]
            self.queue = queue.Queue()
            self.error = ''
        async def next_chunk(self):
            return self.chunks.pop(0) if self.chunks else None

    monkeypatch.setattr(module, 'ModelAPI', AlignmentAPI)
    session = Session()
    session.source = Source()
    session.kind = 'file'
    session.active = session.processing = True
    await session._run()
    assert session.failed_alignment and session.failed_chunk.index == 1
    assert session.raw == '前文'
    session.retry()
    await session.task
    assert session.failed_alignment and session.failed_chunk.index == 1
    session.skip_failed_chunk()
    await session.task

    assert not session.active and not session.failed_chunk
    assert session.raw == '前文後文'
    assert '【音訊可能缺漏 4.0–8.0 秒】' in session.corrected
    assert '【音訊可能缺漏 4.0–8.0 秒】' in session.display_raw()
    assert AlignmentAPI.calls == [[1], [2]]  # The LLM never edits across a gap.
    exported = json.loads(session.export_json())
    assert exported['gaps'][0]['raw'] == '失敗原稿'
    assert exported['gaps'][0]['alignment_items'][0]['end'] == 9
    assert exported['gaps'][0]['possible_start'] == 4
    assert exported['gaps'][0]['possible_end'] == 8
    assert exported['segments'][1]['added'] == '後文'
    assert '可能缺漏' in exported['status']
    assert session.writing_text() == '前文\n後文'
    assert '【音訊可能缺漏 4.0–8.0 秒】' in session.corrected


def test_gap_only_does_not_replace_selected_writing_text():
    session = Session()
    session.gaps = [Gap(0, 0, 4, 0, 4, 0, '未採用原稿', [], '時間超界')]
    session._refresh_transcripts()
    assert session.corrected == '【音訊可能缺漏 0.0–4.0 秒】'
    assert session.writing_text() == ''  # writing.js commits empty text by rollback.


def test_skip_is_restricted_to_paused_final_alignment_errors():
    session = Session()
    session.active = True
    session.source = object()
    session.failed_chunk = Chunk(0, 0, np.zeros(RATE), True, 'stop')
    with pytest.raises(ValueError, match='沒有可略過'):
        session.skip_failed_chunk()


@pytest.mark.asyncio
async def test_live_skip_only_drains_received_audio_and_marks_recording_incomplete(monkeypatch, tmp_path):
    import voice_app.session as module

    class LiveAPI(API):
        def __init__(self, settings):
            super().__init__([])
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def recognize(self, samples):
            return ({'text': '失敗', 'items': [{'text': '失敗', 'start': 0, 'end': 9}]}
                    if samples[0] == 1 else recognition('已收到'))

    class Source:
        def __init__(self):
            self.chunks = [Chunk(0, 0, np.ones(RATE * 4), True, 'pause'),
                           Chunk(1, 4, np.full(RATE * 4, 2), True, 'stop')]
            self.queue = queue.Queue()
            self.stop = threading.Event()
            self.done = threading.Event()
            self.done.set()
            self.error = ''
            self.recorded_samples = RATE * 8
        async def next_chunk(self):
            return self.chunks.pop(0) if self.chunks else None

    monkeypatch.setattr(module, 'ModelAPI', LiveAPI)
    session = Session()
    session.source = Source()
    session.kind = 'microphone'
    session.recording_path = tmp_path / 'recording.wav'
    session.recording_state = 'recording'
    session.active = session.processing = True
    await session._run()
    assert session.source.stop.is_set()
    assert session.recording_state == 'incomplete'
    session.skip_failed_chunk()
    await session.task
    assert session.raw == '已收到'
    assert session.recording_state == 'incomplete'
    saved = json.loads(session.recording_path.with_suffix('.json').read_text(encoding='utf-8'))
    assert len(saved['gaps']) == 1
    assert saved['recording']['status'] == 'incomplete'


@pytest.mark.asyncio
async def test_skipped_window_protects_next_rolling_edit_start(monkeypatch):
    import voice_app.session as module

    class WindowAPI(API):
        protected = []
        def __init__(self, settings):
            super().__init__([])
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def recognize(self, samples):
            return ({'text': '錯誤視窗', 'items': [
                {'text': '錯誤視窗', 'start': 0, 'end': 13}]}
                if samples[0] == 1 else recognition('後文', [(3.1, 3.2), (3.3, 3.4)]))
        async def revise_recent(self, context, segments, on_preview, **kwargs):
            self.protected.append(kwargs['protect_start'])
            texts = [row['current'] for row in segments]
            return RollingResult(texts)

    class Source:
        def __init__(self):
            self.chunks = [Chunk(0, 0, np.ones(RATE * 12), True, 'window', 10),
                           Chunk(1, 9, np.full(RATE * 5, 2), True, 'stop')]
            self.queue = queue.Queue()
            self.error = ''
        async def next_chunk(self):
            return self.chunks.pop(0) if self.chunks else None

    monkeypatch.setattr(module, 'ModelAPI', WindowAPI)
    session = Session()
    session.source = Source()
    session.kind = 'file'
    session.active = session.processing = True
    await session._run()
    session.skip_failed_chunk()
    await session.task
    assert session.gaps[0].reason == 'window'
    assert WindowAPI.protected == [True]


def test_settings_roundtrip_and_validation(tmp_path):
    settings = Settings()
    settings.editor.base_url += '，'
    path = tmp_path / 'settings.json'
    settings.save(path)
    assert Settings.load(path) == settings
    assert settings.editor.base_url.endswith('/v1')
    settings.overlap_seconds = 10
    with pytest.raises(ValueError):
        settings.validate()


@pytest.mark.asyncio
async def test_cancel_file_producer_under_backpressure(tmp_path):
    path = tmp_path / 'long.wav'
    path.write_bytes(wav_bytes(np.full(RATE * 20, .1)))
    source = AudioSource('file', '', path, settings=Settings(window_seconds=5, overlap_seconds=2))
    source.queue = queue.Queue(maxsize=2)
    source.start()
    for _ in range(300):
        if source.queue.full(): break
        await asyncio.sleep(.01)
    assert source.queue.full()
    await source.close()
    assert not source.thread.is_alive()
