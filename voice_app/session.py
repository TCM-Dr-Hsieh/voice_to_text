import asyncio
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import time

from .api import ModelAPI, RollingResult
from .audio import AudioSource, Chunk, RATE, reserve_recording_path
from .remote_audio import RemoteAudioSource
from .alignment import AlignmentError, map_alignment, select_new_window
from .config import Settings
from .editing import changes
from .text import join_texts


@dataclass
class Segment:
    index: int
    start: float
    end: float
    raw: str
    added: str
    corrected_tail: str
    elapsed: float
    warning: str
    reason: str = ''
    alignment: list = field(default_factory=list)
    edits: list = field(default_factory=list)


@dataclass
class Revision:
    index: int
    segments: list[int]
    before: list[str]
    after: list[str]
    edits: list
    reason: str = ''


@dataclass
class Gap:
    chunk_index: int
    audio_start: float
    audio_end: float
    possible_start: float
    possible_end: float
    after_segment: int
    raw: str
    alignment_items: list
    error: str
    reason: str = ''
    raw_source: str = '定稿'

    @property
    def marker(self) -> str:
        return f'\n【音訊可能缺漏 {self.possible_start:.1f}–{self.possible_end:.1f} 秒】\n'


def before_context(article_before: str, frozen_voice: str, limit: int) -> str:
    """Keep nearby article context while favoring the most recent locked speech."""
    if not article_before:
        return frozen_voice[-limit:]
    article_count = min(len(article_before), max(1, limit * 2 // 5))
    voice_count = min(len(frozen_voice), limit - article_count)
    article_count = min(len(article_before), limit - voice_count)
    return article_before[-article_count:] + (frozen_voice[-voice_count:] if voice_count else '')


class Session:
    def __init__(self):
        self.settings = Settings()
        self.source: AudioSource | RemoteAudioSource | None = None
        self.task: asyncio.Task | None = None
        self.failed_chunk: Chunk | None = None
        self.failed_alignment = False
        self.failed_recognition: dict | None = None
        self.raw = self.corrected = self.preview = ''
        self.segments: list[Segment] = []
        self.revisions: list[Revision] = []
        self.gaps: list[Gap] = []
        self.rolling_start = 0
        self.status = '準備就緒'
        self.error = self.warning = self.created = self.kind = ''
        self.active = self.processing = False
        self.owned_file: Path | None = None
        self.recording_path: Path | None = None
        self.recording_state = ''
        self.recording_save_error = ''
        self.committed_end = -1.0
        self.writing_context_before = self.writing_context_after = ''

    def start(self, settings: Settings, kind: str, device_id='', path=None, *, record_audio=False,
              writing_context=None):
        if self.active:
            raise ValueError('請先完成或取消目前工作。')
        settings.validate()
        recording_path = reserve_recording_path(kind) if record_audio and kind in (
            'microphone', 'loopback', 'remote_microphone', 'remote_system') else None
        self.settings = deepcopy(settings)
        self.writing_context_before, self.writing_context_after = writing_context or ('', '')
        self.raw = self.corrected = self.preview = ''
        self.error = self.warning = ''
        self.segments = []
        self.revisions = []
        self.gaps = []
        self.rolling_start = 0
        self.failed_chunk = None
        self.failed_alignment = False
        self.failed_recognition = None
        self.committed_end = -1.0
        self.created = datetime.now().astimezone().isoformat()
        self.kind = kind
        self.recording_path = recording_path
        self.recording_state = 'recording' if recording_path else ''
        self.recording_save_error = ''
        self.source = (RemoteAudioSource(kind, settings=settings, recording_path=recording_path)
                       if kind in ('remote_microphone', 'remote_system') else
                       AudioSource(kind, device_id, path, settings=settings, recording_path=recording_path))
        self.active = self.processing = True
        self.status = '讀取音訊檔' if kind == 'file' else '錄音中，等待語音'
        try:
            self.source.start()
        except Exception:
            self.active = self.processing = False
            self.recording_path = None
            self.recording_state = ''
            if recording_path:
                recording_path.unlink(missing_ok=True)
            raise
        self.task = asyncio.create_task(self._run())

    def request_stop(self):
        if self.active and self.source:
            self.source.stop.set()
            self.status = '正在完成剩餘音訊並定稿'

    def retry(self):
        if self.failed_chunk is not None and not self.processing:
            self.processing = True
            self.error = ''
            self.task = asyncio.create_task(self._run())

    def can_skip_failed_chunk(self) -> bool:
        return bool(self.active and not self.processing and
                    self.failed_alignment and
                    self.failed_chunk is not None and self.failed_chunk.final and
                    self.source is not None)

    def skip_failed_chunk(self):
        chunk = self.failed_chunk
        if not self.can_skip_failed_chunk():
            raise ValueError('目前沒有可略過的定稿失敗片段。')
        end = chunk.start + chunk.duration
        possible_start = max(chunk.start, self.committed_end)
        possible_end = min(end, chunk.commit_until if chunk.commit_until is not None else end)
        recognized = self.failed_recognition or {}
        self.gaps.append(Gap(chunk.index, chunk.start, end, possible_start,
                             max(possible_start, possible_end), len(self.segments),
                             recognized.get('text', ''), recognized.get('items', []),
                             self.error, chunk.reason, '定稿'))
        self.committed_end = end
        self.rolling_start = len(self.segments)
        self.failed_chunk = None
        self.failed_alignment = False
        self.failed_recognition = None
        self.preview = ''
        self.error = ''
        self.status = ('已標記可能缺漏的音訊，繼續處理後續片段' if self.kind == 'file' else
                       '已標記可能缺漏的音訊，處理已收到的剩餘片段')
        if self.recording_path:
            self.recording_state = 'incomplete'
        self._refresh_transcripts()
        self._save_recording_json()
        self.processing = True
        self.task = asyncio.create_task(self._run())

    def _compose_text(self, field: str, through: int | None = None) -> str:
        count = len(self.segments) if through is None else through
        parts = []
        for boundary in range(count + 1):
            if boundary:
                parts.append(getattr(self.segments[boundary - 1], field))
            parts.extend(gap.marker for gap in self.gaps if gap.after_segment == boundary)
        return join_texts(parts).strip('\n')

    def display_raw(self) -> str:
        return self._compose_text('added')

    def writing_text(self) -> str:
        """Return aligned speech for article insertion, separating speech across gaps."""
        boundaries = {gap.after_segment for gap in self.gaps}
        groups, current = [], []
        for index, segment in enumerate(self.segments):
            if index in boundaries:
                text = join_texts(current)
                if text:
                    groups.append(text)
                current = []
            current.append(segment.corrected_tail)
        text = join_texts(current)
        if text:
            groups.append(text)
        return '\n'.join(groups)

    def _refresh_transcripts(self):
        self.raw = join_texts(segment.added for segment in self.segments)
        self.corrected = self._compose_text('corrected_tail')

    async def _run(self):
        try:
            async with ModelAPI(self.settings) as api:
                while True:
                    chunk = self.failed_chunk or await self.source.next_chunk()
                    if chunk is None:
                        break
                    self.failed_chunk = chunk
                    await self.process_chunk(api, chunk)
                    self.failed_chunk = None
                    self.failed_alignment = False
                    self.failed_recognition = None
            self.active = False
            self.status = (f'處理完成，{len(self.gaps)} 段音訊可能缺漏'
                           if self.gaps else '處理完成，所有音訊已定稿')
            if self.source.error:
                self.warning = self.source.error
                self.status = '已結束，請查看提示'
            if self.gaps:
                self.warning = ' '.join(filter(None, (self.warning,
                    f'{len(self.gaps)} 段音訊未能安全定稿並已略過；請核對語音插入紀錄中的缺漏標記。')))
            if self.recording_path:
                self.recording_state = 'incomplete' if self.source.error or self.recording_state == 'incomplete' else 'complete'
            self._remove_owned_file()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__
            self.failed_alignment = isinstance(exc, AlignmentError)
            if self.failed_alignment and self.failed_chunk and self.failed_chunk.final:
                self.status = ('處理暫停：可重試或略過此段並繼續' if self.kind == 'file' else
                               '處理暫停：可重試或略過此段並完成已收到音訊')
            else:
                self.status = '處理暫停：可重試失敗片段或取消'
            if self.recording_path:
                self.recording_state = 'incomplete'
            if self.kind != 'file' and self.source:
                self.source.stop.set()
        finally:
            self.processing = False
            self.preview = ''
            if self.recording_path and self.recording_state != 'recording':
                if self.source and not self.source.done.is_set():
                    await asyncio.to_thread(self.source.done.wait, 3)
                self._save_recording_json()

    async def process_chunk(self, api, chunk):
        started = time.monotonic()
        end = chunk.start + chunk.duration
        self.status = f'辨識／對齊音訊視窗 · {chunk.start:.1f}–{end:.1f} 秒'
        self.failed_recognition = None
        recognized = await api.recognize(chunk.samples)
        self.failed_recognition = recognized
        raw = recognized['text']
        units = map_alignment(raw, recognized['items'], chunk.start, chunk.duration)
        added, _ = select_new_window(raw, units, self.committed_end)
        warning = ''
        candidate = Segment(len(self.segments) + 1, chunk.start, end, raw, added, added,
                            time.monotonic() - started, warning, chunk.reason,
                            [asdict(unit) for unit in units])
        recent_start = max(self.rolling_start, len(self.segments) - 2)
        recent = self.segments[recent_start:] + [candidate]
        before = [segment.corrected_tail for segment in recent]
        result = RollingResult(before)
        if any(segment.added for segment in recent):
            self.status = '綜合最新三段校稿，較早內容保持固定'
            frozen = self._compose_text('corrected_tail', recent_start)
            protect_start = bool(
                (recent_start and self.segments[recent_start - 1].reason == 'window') or
                any(gap.after_segment == recent_start and gap.reason == 'window'
                    for gap in self.gaps))
            result = await api.revise_recent(
                before_context(self.writing_context_before, frozen, self.settings.context_chars),
                [{'index': segment.index, 'start': segment.start, 'end': segment.end,
                  'reason': segment.reason, 'added': segment.added,
                  'current': segment.corrected_tail, 'raw': segment.raw} for segment in recent],
                on_preview=lambda value: setattr(self, 'preview', value),
                protect_start=protect_start,
                protect_end=chunk.reason == 'window',
                context_after=self.writing_context_after[:self.settings.context_chars])
        warning = ' '.join(filter(None, (warning, result.warning)))
        candidate.warning = warning
        candidate.elapsed = time.monotonic() - started
        # No awaits below: commit the new window, rolling edit, watermark and audit record together.
        self.segments.append(candidate)
        if result.texts != before:
            for segment, revised in zip(recent, result.texts):
                segment.corrected_tail = revised
                segment.edits = changes(segment.added, revised)
            self.revisions.append(Revision(len(self.revisions) + 1,
                                           [segment.index for segment in recent], before,
                                           result.texts, result.edits,
                                           '依後續 ASR 補齊接縫' if result.seam_repaired else ''))
        self._refresh_transcripts()
        self.committed_end = end
        self.preview = ''
        self.warning = warning
        self.status = '視窗已校稿，繼續錄音' if chunk.reason != 'stop' else '剩餘音訊已校稿'

    async def cancel(self):
        was_active = self.active
        if self.source:
            self.source.cancelled.set()
            self.source.stop.set()
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        if self.source:
            await self.source.close()
        self.active = self.processing = False
        self.failed_chunk = None
        self.failed_alignment = False
        self.failed_recognition = None
        self.preview = ''
        if was_active:
            self.status = '已取消，保留已完成的校稿；目前音訊不會寫入文章'
            if self.recording_path:
                self.recording_state = 'incomplete'
                self._save_recording_json()
        self._remove_owned_file()

    def _save_recording_json(self):
        if not self.recording_path:
            return
        path = self.recording_path.with_suffix('.json')
        temporary = path.with_suffix('.json.tmp')
        try:
            temporary.write_text(self.export_json(), encoding='utf-8')
            temporary.replace(path)
            self.recording_save_error = ''
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            self.recording_save_error = f'錄音已保留，但完整紀錄 JSON 儲存失敗：{exc}'

    def _remove_owned_file(self):
        if self.owned_file and (not self.source or not self.source.thread.is_alive()):
            self.owned_file.unlink(missing_ok=True)
            self.owned_file = None

    def export_json(self):
        recording = ({'file': self.recording_path.name, 'status': self.recording_state,
                      'sample_rate': RATE,
                      'sample_count': self.source.recorded_samples if self.source else 0}
                     if self.recording_path else None)
        return json.dumps({'created': self.created, 'source': self.kind,
                           'raw': self.raw, 'corrected': self.corrected,
                           'status': self.status, 'warning': self.warning, 'error': self.error,
                           'recording': recording,
                           'segments': [asdict(s) for s in self.segments],
                           'gaps': [asdict(gap) for gap in self.gaps],
                           'revisable_segments': [s.index for s in self.segments[
                               max(self.rolling_start, len(self.segments) - 2):]] if self.active else [],
                           'revisions': [asdict(item) for item in self.revisions]},
                          ensure_ascii=False, indent=2)
