"""Meeting mode: one meeting's speakers, live translation, the rows the screen shows, the autosave and the bilingual export.

A meeting rides on a `Session` (the speech recognition and proofreading are unchanged): the session tells it about every committed
segment, the meeting feeds the speaker marking, wakes the translator, and remembers which rows of the two-column view changed so
the page sends only those. Everything the user would hate to lose is written to `data/meetings/` while the meeting runs: the
browser may disconnect, the page may be reloaded, the meeting goes on and the file is current to within `autosave_seconds`.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path

from .api import ModelAPI
from .config import ROOT, Settings
from .speaker import default_name
from .speaker.palette import SPEAKER_COLORS
from .speaker.service import SpeakerService
from .translation import Translator

MEETINGS_DIR = ROOT / 'data' / 'meetings'
REVISABLE = 3                      # the newest segments the proofreader may still change
_CLOSING: set = set()           # the tasks closing retired meetings (kept alive until they are done)


def reserve_meeting_path(directory: Path = MEETINGS_DIR, now: datetime | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    day = (now or datetime.now()).strftime('%Y%m%d')
    number = 1
    while True:
        path = directory / f'會議-{day}-{number:03d}.json'
        try:
            with path.open('xb'):
                return path
        except FileExistsError:
            number += 1


class Meeting:
    def __init__(self, settings: Settings, embed, *, path: Path | None = None):
        self.settings = settings                      # a copy made when the meeting started
        self.session = None
        self._segments: list = []                     # the segments of THIS meeting's run: a new run of the session starts a new list
        self._source_kind = ''
        self._recording: dict | None = None
        self._retired = self._closed = self._started = False
        self._save_lock = asyncio.Lock()
        self.created = datetime.now().astimezone().isoformat()
        self.path = path or reserve_meeting_path()
        self.ended = False
        self.events: list[dict] = []
        self.dirty: set[int] = set()
        self.reset = True                             # the next update tells the page to start from an empty view
        self.names_dirty = True
        self._unsaved = True
        self._save_error = ''
        self._editor_api = ModelAPI(settings)
        self._translator_api = ModelAPI(settings)
        self.speaker = SpeakerService(settings=settings.meeting, embed=embed, llm=self._editor_call,
                                      segments=lambda: self._segments,
                                      emit=self.emit, on_change=self._speaker_changed, audio_t=self._audio_t)
        self.translator = Translator(self, self._translator_call, settings.translator, settings.meeting,
                                     emit=self.emit, on_change=self._translator_changed)
        self._autosave: asyncio.Task | None = None

    # ------------------------------------------------------------------ plumbing
    def emit(self, type: str, **fields):
        self.events.append({'t': self._audio_t(), 'type': type, **fields})
        del self.events[:-2000]

    def _audio_t(self) -> float | None:
        return round(self._segments[-1].end, 1) if self._segments else None

    async def _editor_call(self, *, system, user, max_tokens, validator, retries, retry_note):
        return await self._editor_api.chat_json(self.settings.editor, system, user, max_tokens=max_tokens, validator=validator,
                                                retries=retries, retry_note=retry_note)

    async def _translator_call(self, model, *, system, user, max_tokens, validator, retries, retry_note):
        return await self._translator_api.chat_json(model, system, user, max_tokens=max_tokens, validator=validator,
                                                    retries=retries, retry_note=retry_note)

    def attach(self, session):
        self._started = True
        self.session = session
        self._segments = session.segments
        self._source_kind = session.kind
        self.speaker.start()
        self._autosave = asyncio.create_task(self._autosave_loop(), name='meeting-autosave')
        self.emit('meeting_started', source=session.kind, file=self.path.name)

    # ------------------------------------------------------------------ what the session and the translator ask
    def segment_count(self) -> int:
        return len(self._segments)

    def segment(self, index: int):
        rows = self._segments
        return rows[index - 1] if 1 <= index <= len(rows) else None

    def segment_indexes(self) -> list[int]:
        return list(range(1, self.segment_count() + 1))

    async def after_commit(self, chunk, segment, selected, added: str, revised: list[int]):
        """Called by the session after it committed a segment (and revised the recent ones)."""
        self._mark(set(revised) | {segment.index - REVISABLE})
        await self.speaker.feed_segment(chunk.start, chunk.samples, segment, selected, added)
        self.translator.wake()

    async def end(self):
        """The session has no more audio: decide the last sentences, then let the translator finish the last segments."""
        if self.ended:
            return
        await self.speaker.finish()
        self.ended = True
        self._mark({i for i in range(max(1, self.segment_count() - REVISABLE), self.segment_count() + 1)})
        self.emit('meeting_ended', segments=self.segment_count())
        self.translator.wake()
        await self.save()

    def retire(self):
        """The session is about to start another run: this meeting is over for good. It keeps its own segments, stops its tasks and
        writes its file one last time, so nothing it does afterwards can touch the next meeting (or be touched by it)."""
        if self._retired:
            return
        self._retired = True
        self._recording = self._recording_info()
        self.session = None
        try:
            task = asyncio.get_running_loop().create_task(self.close(), name='meeting-close')
        except RuntimeError:                                # no loop: nothing is running either
            return
        _CLOSING.add(task)
        task.add_done_callback(_CLOSING.discard)

    def _recording_info(self) -> dict | None:
        session = self.session
        if session is not None and session.meeting is self and session.recording_path:
            return {'file': session.recording_path.name, 'status': session.recording_state}
        return self._recording

    async def close(self):
        """The page is gone or the meeting was cancelled: stop every task, save what there is. Closing twice is harmless."""
        if self._closed:
            return
        self._closed = True
        if self._autosave is not None:
            self._autosave.cancel()
            await asyncio.gather(self._autosave, return_exceptions=True)
            self._autosave = None
        await self.translator.stop()
        await self.speaker.stop()
        try:
            await self.save()
        finally:
            await self._editor_api.__aexit__()
            await self._translator_api.__aexit__()

    async def discard(self):
        """A meeting whose audio never opened (no device, unreadable file): let go of its connections and take back the empty file
        reserved for it, so nothing of it is left. A meeting that did start is closed and saved like any other."""
        if self._started:
            await self.close()
            return
        if self._closed:
            return
        self._closed = True
        try:
            await self.translator.stop()
            await self.speaker.stop()
            if self.path.exists() and self.path.stat().st_size == 0:      # still the reserved, never-written file
                self.path.unlink(missing_ok=True)
        finally:
            await self._editor_api.__aexit__()
            await self._translator_api.__aexit__()

    # ------------------------------------------------------------------ rows for the screen
    def _mark(self, indexes: set[int]):
        self.dirty |= {i for i in indexes if i >= 1}
        self._unsaved = True

    def _speaker_changed(self, changed: set[int]):
        if changed:
            self._mark(changed)
            self.translator.requeue(changed)
        else:                                          # a rename: the labels change, and the translator has a new name to translate
            self.names_dirty = True
            self._unsaved = True
            self.translator.wake()

    def _translator_changed(self, changed: set[int]):
        if changed:
            self._mark(changed)
        else:                                          # the names were translated
            self.names_dirty = True
            self._unsaved = True

    def _labels(self) -> dict:
        speaker, translator = self.speaker, self.translator
        gids = speaker.tracker.group_ids
        return {'marking': speaker.active, 'names': {str(g): speaker.name_of(g) for g in gids}, 'unknown': '不明', 'voice': '語音',
                'background': '背景', 'palette': len(SPEAKER_COLORS),
                'tnames': {str(g): translator.label_for(g) for g in gids},
                'tunknown': translator.names.unknown, 'tvoice': translator.names.voice}

    def _translation_gids(self, segment, result) -> list:
        """The speaker of each translated text: the runs as they are now when they still match the texts one to one, else the ones the
        translation was made with, followed to whoever they have been merged into since (a group merged away has no name, no colour)."""
        now = self.speaker.translation_runs(segment)
        if len(now) == len(result.texts):
            return [r.gid for r in now]
        return [self.speaker.tracker.resolve(g) for g in result.gids]

    def _row(self, index: int) -> dict | None:
        segment = self.segment(index)
        if segment is None:
            return None
        count = self.segment_count()
        revisable = not self.ended and index > count - REVISABLE
        runs = self.speaker.runs(segment)
        row = {'i': index, 'rev': revisable, 'runs': [{'g': r.gid, 't': r.text, 's': r.source} for r in runs]}
        result = self.translator.visible(index)
        if result is not None:
            gids = self._translation_gids(segment, result)
            state = self.translator.tr_state(index)
            row['tr'] = {'st': state, 'runs': [{'g': g, 't': t} for g, t in zip(gids, result.texts)],
                         'err': result.error[:120] if result.error else ''}
        return row

    def drain(self) -> dict | None:
        """What changed since the last call, for the page; None when nothing did."""
        rows = sorted(self.dirty)
        self.dirty = set()
        if not rows and not self.names_dirty and not self.reset:
            return None
        update = {'rows': [row for row in (self._row(i) for i in rows) if row is not None], 'labels': self._labels()}
        self.names_dirty = False
        if self.reset:
            update['reset'] = True
            self.reset = False
        return update

    # ------------------------------------------------------------------ saving
    async def _autosave_loop(self):
        while True:
            await asyncio.sleep(self.settings.meeting.autosave_seconds)
            if self._unsaved:
                await self.save()

    def doc(self) -> dict:
        segments = []
        for segment in self._segments:
            runs = self.speaker.runs(segment)
            segments.append({'index': segment.index, 'start': segment.start, 'end': segment.end,
                             'runs': [{'speaker': default_name(r.gid) if r.gid is not None else None,
                                       'name': '背景' if r.background else self.speaker.name_of(r.gid), 'source': r.source,
                                       'background': r.background, 'text': r.text} for r in runs]})
        return {'created': self.created, 'source': self._source_kind, 'ended': self.ended,
                'recording': self._recording_info(), 'speakers': self.speaker.map_doc(), 'segments': segments,
                'translation': self.translator.doc(), 'events': self.events[-500:]}

    async def save(self):
        """Save the meeting file and its bilingual text next to each other. One save at a time (the autosave, the end and the close
        may all want to), and the content is put together inside the turn, so a later save always holds the newer content. It is put
        together here, in the event loop, where nothing else changes it; only the writing happens in a thread. A failure (a full
        disk) is shown, never raised."""
        async with self._save_lock:
            self._unsaved = False
            try:
                files = ((self.path, json.dumps(self.doc(), ensure_ascii=False, indent=1), 'utf-8'),
                         (self.path.with_suffix('.txt'), self.export_text(), 'utf-8-sig'))
                writing = asyncio.ensure_future(asyncio.to_thread(self._store, files))
                try:
                    await asyncio.shield(writing)
                except asyncio.CancelledError:
                    await asyncio.wait({writing})        # a thread cannot be cancelled: do not let the next save start before it is done
                    if not writing.cancelled():
                        writing.exception()
                    raise
                self._save_error = ''
            except Exception as exc:
                self._unsaved = True
                self._save_error = f'會議存檔失敗：{exc}'

    @staticmethod
    def _store(files):
        for path, text, encoding in files:             # atomic: a crash never leaves half a file
            temporary = path.with_name(f'{path.name}.{uuid.uuid4().hex[:8]}.tmp')
            try:
                temporary.write_text(text, encoding=encoding)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)

    @property
    def save_error(self) -> str:
        return self._save_error

    def export_text(self) -> str:
        """The bilingual text: each segment as `語音#n 名稱: 原文`, followed by its translation when there is one."""
        speaker, translator = self.speaker, self.translator
        started = datetime.fromisoformat(self.created)
        lines = [f'會議紀錄　{started:%Y-%m-%d %H:%M}']
        who = [f'{default_name(s["gid"])} = {s["label"]}' if s['name'] else s['default'] for s in speaker.speakers()]
        if who:
            lines.append('說話者：' + '、'.join(who))
        if translator.results:
            names = '、'.join(f'{translator.label_for(s["gid"])}' for s in speaker.speakers() if s['name'])
            lines.append(f'翻譯語言：{translator.language}' + (f'（{names}）' if names else ''))
        lines.append('')
        for segment in self._segments:
            runs = speaker.runs(segment)
            if not runs:
                continue
            for k, run in enumerate(runs):
                prefix = f'語音#{segment.index} ' if k == 0 else ' ' * (len(f'語音#{segment.index}') + 1)
                who = '[背景]' if run.background else f'{speaker.name_of(run.gid)}{"*" if run.source in ("text", "text2") else ""}'
                lines.append(f'{prefix}{who}: {run.text}')
            result = translator.visible(segment.index)
            if result is not None and result.texts and not result.error:
                gids = self._translation_gids(segment, result)
                for k, (gid, text) in enumerate(zip(gids, result.texts)):
                    prefix = f'{translator.names.voice}#{segment.index} ' if k == 0 else ' ' * (len(f'{translator.names.voice}#{segment.index}') + 1)
                    lines.append(f'{prefix}{translator.label_for(gid)}: {text}')
            lines.append('')
        return '\n'.join(lines).rstrip('\n') + '\n'
