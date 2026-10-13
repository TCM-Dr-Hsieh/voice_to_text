"""One meeting's speaker marking: the tracker, the names the user gives, the text fill, and the views the transcript and the UI read.

The transcript pipeline only calls `feed_segment` (after each committed window), `runs` (to print a segment with its speakers) and
`finish`. Nothing here may stop or slow the transcript: an error turns the marking off for the rest of the meeting and is logged.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Awaitable, Callable

from ..api import ModelError
from ..config import MeetingSettings
from . import TEXT_SOURCES, default_name, fill, tuning
from .render import Run, project, segment_runs, speaker_runs
from .tracker import Params, SpeakerTracker
from .units import Token, UnitParams, ends_sentence

FINAL_PASS_SECONDS = 60.0         # the last fill pass at the end of a meeting may not hold it open longer than this
MAX_NAME = 40
LLMCall = Callable[..., Awaitable[object]]     # llm(system=, user=, max_tokens=, validator=, retries=, retry_note=) -> validated value


def window_tokens(index: int, selected, added: str, added_conv: str) -> list[Token]:
    """The timed pieces of one segment's new text, as character ranges of its Traditional-script form `added_conv`."""
    if not selected:
        return []
    origin = selected[0].begin
    starts = [u.begin - origin for u in selected] + [len(added)]
    cuts = starts if len(added_conv) == len(added) else project(added, added_conv, starts)
    cuts[0], cuts[-1] = 0, len(added_conv)
    return [Token(index, cuts[k], cuts[k + 1], u.start, u.end,
                  ends_sentence(added[u.begin - origin:u.finish - origin])) for k, u in enumerate(selected)]


def build_params(meeting: MeetingSettings) -> Params:
    """The tracker's numbers for a meeting: the defaults, with the ones the user changed (the advanced pop-up and the boxes of the tab)."""
    values = tuning.effective(meeting)
    units = UnitParams(**{key: float(values[key]) for key in tuning.UNIT_KEYS})
    own = {key: int(value) if tuning.BY_KEY[key].integer else float(value) for key, value in values.items() if key not in tuning.UNIT_KEYS}
    params = Params(**own, units=units)
    return params if meeting.background_filter else replace(params, background_max_cos=0.0)


class SpeakerService:
    def __init__(self, *, settings: MeetingSettings, embed, llm: LLMCall, segments: Callable[[], list],
                 emit: Callable[..., object] | None = None, on_change: Callable[[set], None] | None = None,
                 audio_t: Callable[[], float | None] = lambda: None):
        self.settings = settings
        self.tracker = SpeakerTracker(embed, build_params(settings))
        self.tuning = tuning.of_params(self.tracker.params)       # every number this meeting really runs with, for its file
        self.tuning_changed = tuning.differing(self.tuning)       # and the ones among them that are not the default
        self.names: dict[int, str] = {}
        self.name_events: list[dict] = []
        self.failed = ''
        self.unavailable = ''             # why the marking cannot run at all (e.g. the voiceprint model is missing)
        self.fill_stats = {'asked': 0, 'accepted': 0, 'rejected': 0, 'failed_calls': 0}
        self.fill2_stats = {'asked': 0, 'accepted': 0, 'rejected': 0, 'failed_calls': 0, 'accepted_agreeing': 0, 'accepted_differing': 0}
        self._llm = llm
        self._runs_cache: dict[int, tuple] = {}
        self._segments = segments
        self._emit_event = emit or (lambda *a, **k: None)
        self._on_change = on_change or (lambda changed: None)
        self._audio_t = audio_t
        self._finished = False
        self._wake = asyncio.Event()
        self._settle_lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------ plumbing
    def emit(self, type: str, **fields):
        try:
            self._emit_event(type, **fields)
        except Exception:                     # logging must never break the transcript
            pass

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name='speaker-fill')

    async def stop(self):
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def _fail(self, exc: BaseException, where: str):
        self.failed = f'{type(exc).__name__}: {exc}'
        self.emit('speaker_error', where=where, error=self.failed)
        self._on_change(set())

    # ------------------------------------------------------------------ names
    def name_of(self, gid: int | None) -> str:
        """What a speaker is called: the name the user typed, else Speaker-001 and so on. None (unknown) is 不明."""
        if gid is None:
            return '不明'
        return self.names.get(gid) or default_name(gid)

    def custom_name(self, gid: int) -> str:
        return self.names.get(gid, '')

    def set_name(self, gid: int, name: str) -> set[int]:
        """The user names a speaker (an empty name goes back to Speaker-00N). Returns the segments whose text changes."""
        gid = self.tracker.resolve(gid)
        if gid not in self.tracker.group_ids:
            raise ValueError('找不到這位說話者。')
        name = ' '.join(name.split())[:MAX_NAME]
        if name:
            self.names[gid] = name
        else:
            self.names.pop(gid, None)
        self.name_events.append({'t': self._audio_t(), 'event': 'name', 'speaker': default_name(gid), 'name': name})
        self.emit('speaker_named', speaker=default_name(gid), name=name)
        self._on_change(set())
        return set()

    def merge(self, drop: int, keep: int) -> set[int]:
        """The user says two speakers are one person: `drop` becomes part of `keep` (its name is kept when `keep` has none)."""
        drop, keep = self.tracker.resolve(drop), self.tracker.resolve(keep)
        changed = self.tracker.merge(drop, keep)
        self._carry_name(drop, keep, '')
        self.name_events.append({'t': self._audio_t(), 'event': 'merge', 'speaker': default_name(drop), 'into': default_name(keep)})
        self.emit('speaker_merged', speaker=default_name(drop), into=default_name(keep))
        self._on_change(changed)
        return changed

    # ------------------------------------------------------------------ views
    @property
    def active(self) -> bool:
        return not (self.failed or self.unavailable)

    @property
    def status(self) -> str:
        if self.unavailable:
            return f'說話者標記未啟用：{self.unavailable}'
        if self.failed:
            return f'說話者標記已停止（{self.failed}）'
        count = self.tracker.group_count
        if count == 0:
            return '說話者標記：聽取聲音中（約需 10–60 秒才會出現第一位）'
        stats = self.tracker.stats()
        share = f'，不明 {stats["unknown"] * 100 // max(1, stats["decided"])}%' if stats['decided'] else ''
        background = f'，背景 {stats["background"]} 句' if stats['background'] else ''
        return f'說話者標記：已分出 {count} 位說話者{share}{background}'

    def segment(self, index: int):
        rows = self._segments()
        return rows[index - 1] if 1 <= index <= len(rows) else None

    def runs(self, segment) -> list[Run]:
        """The segment as runs of one speaker, for the screen; one run with no speaker while marking is off or has nothing yet."""
        text = segment.corrected_tail
        labels = self.tracker.labels_for(segment.index) if self.active else None
        if not labels:
            return [Run(None, '', text.strip())] if text.strip() else []
        added = segment.added_conv or segment.added
        key = (text, added, labels)
        cached = self._runs_cache.get(segment.index)
        if cached is not None and cached[0] == key:
            return cached[1]
        runs = segment_runs(added, text, labels)
        self._runs_cache[segment.index] = (key, runs)
        return runs

    def translation_runs(self, segment) -> list[Run]:
        """What the translator is given: background talk is left out (shown grey on the screen, never translated)."""
        return speaker_runs([run for run in self.runs(segment) if not run.background])

    def has_background(self, segment) -> bool:
        return segment is not None and any(run.background for run in self.runs(segment))

    def set_background(self, index: int, run_index: int, text: str, flag: bool) -> set[int]:
        """The user says the sentences of run `run_index` of segment `index` are (flag) or are not (not flag) background talk. `text` is
        the run as the user saw it: when the transcript has changed since, nothing is touched."""
        segment = self.segment(index)
        runs = self.runs(segment) if segment is not None and self.active else []
        if not 0 <= run_index < len(runs) or runs[run_index].text != text:
            raise ValueError('這一句剛剛被更新了，請再點一次。')
        first, end = runs[run_index].span
        sentences = self.tracker.sentences_of(index, first, end)
        if not sentences:
            raise ValueError('這一句還沒有聲音資料可以標記。')
        changed = self.tracker.set_background(sentences, bool(flag))
        self.emit('speaker_background', segment=index, background=bool(flag), sentences=len(sentences))
        self._on_change(changed)
        return changed

    def speakers(self) -> list[dict]:
        """What the speaker window shows: per speaker its name, speech time and share, and two example sentences."""
        out = []
        for info in self.tracker.speaker_info():
            gid = info['gid']
            examples = [self._sentence_text(i) for i in info['examples']]
            out.append({'gid': gid, 'key': fill.speaker_key(gid), 'default': default_name(gid), 'name': self.custom_name(gid),
                        'label': self.name_of(gid), 'seconds': info['seconds'], 'share': info['share'],
                        'sentences': info['sentences'], 'examples': [e for e in examples if e],
                        'merged': [self.name_of(g) for g in info['merged']], 'merged_default': [default_name(g) for g in info['merged']]})
        return out

    def labels_doc(self) -> dict:
        """Per segment, one entry per timed piece: [c0, c1, speaker group, source] over the segment's Traditional-script text."""
        return {str(seg): [list(item) for item in labels] for seg, labels in sorted(self.tracker.all_labels().items())}

    def map_doc(self) -> dict:
        """No voiceprints: only who was heard, the names the user gave, per-minute counts and the text fill statistics."""
        return {'status': self.status, 'failed': self.failed,
                'speakers': [{'speaker': default_name(s['gid']), 'name': s['name'], 'seconds': round(s['seconds'], 1),
                              'share': round(s['share'], 3), 'sentences': s['sentences'],
                              'merged': s['merged_default']}
                             for s in self.speakers()],
                'name_events': self.name_events, 'minutes': self.tracker.minutes, 'stats': self.tracker.stats(),
                'fill': self.fill_stats, 'fill2': self.fill2_stats,
                'settings': {'unknown_percentile': self.settings.unknown_percentile, 'max_speakers': self.settings.max_speakers,
                             'text_fill': self.settings.text_fill, 'text_fill2': self.settings.text_fill2,
                             'background_filter': self.settings.background_filter, 'background_rel_db': self.settings.background_rel_db,
                             'background_max_cos': self.settings.background_max_cos,
                             'tuning': dict(self.tuning), 'tuning_changed': dict(self.tuning_changed)}}

    # ------------------------------------------------------------------ feeding (called by the transcript pipeline)
    async def feed_segment(self, chunk_start: float, samples, segment, selected, added: str):
        """Called after a segment was committed: `selected` are the aligned units of the segment's new text (`added`)."""
        if not self.active:
            return
        try:
            tokens = window_tokens(segment.index, selected, added, segment.added_conv or added)
        except Exception as exc:
            self._fail(exc, 'tokens')
            return
        try:
            changed = await asyncio.to_thread(self._process, chunk_start, samples, tokens)
        except Exception as exc:
            self._fail(exc, 'feed')
            return
        self._after_tracker(changed)

    def _process(self, chunk_start, samples, tokens):
        self.tracker.add_audio(chunk_start, samples)
        return self.tracker.add_tokens(tokens)

    def _after_tracker(self, changed: set[int]):
        renamed = False
        for event in self.tracker.events:
            event = dict(event)
            kind = event.pop('event')
            if kind == 'speaker_merged':               # the user's own merge: `merge()` has logged it and moved the name
                continue
            if kind == 'speaker_merged_auto':          # two groups turned out to be one person: the name goes with the group
                renamed |= self._carry_name(event['merged'], event['into'], 'merge_auto')
            self.emit(kind, **event)
        self.tracker.events.clear()
        if changed:
            self._on_change(changed)
        elif renamed:
            self._on_change(set())
        self._wake.set()

    def _carry_name(self, drop: int, keep: int, how: str) -> bool:
        """The name of a group that was joined into another is kept when the other has none. True when a name was involved."""
        name = self.names.pop(drop, '')
        if name and keep not in self.names:
            self.names[keep] = name
        if name and how:
            self.name_events.append({'t': self._audio_t(), 'event': how, 'speaker': default_name(drop),
                                     'into': default_name(keep), 'name': name})
        return bool(name)

    # ------------------------------------------------------------------ the text fill, in the background
    async def _loop(self):
        while True:
            await self._wake.wait()
            self._wake.clear()
            try:
                await self.settle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:              # a failing fill step must not end the loop
                self.emit('speaker_error', where='fill', error=f'{type(exc).__name__}: {exc}')

    async def settle(self, *, final: bool = False):
        """One round of text fill (when there are at least two speakers there is something to choose between)."""
        async with self._settle_lock:
            if not self.active or not self.settings.text_fill or self.tracker.group_count < 2:
                return
            await self._fill_stage(final, second=False)
            if self.settings.text_fill2:
                await self._fill_stage(final, second=True)

    def _pieces(self, seg_index: int) -> dict[int, str]:
        """Global token index -> the piece of the segment's CURRENT text that belongs to it."""
        segment = self.segment(seg_index)
        indexes = self.tracker.segment_token_indexes(seg_index)
        added = segment.added_conv or segment.added
        starts = [self.tracker.tokens[i].c0 for i in indexes] + [len(added)]
        cuts = project(added, segment.corrected_tail, starts)
        cuts[0], cuts[-1] = 0, len(segment.corrected_tail)
        return {i: segment.corrected_tail[cuts[k]:cuts[k + 1]] for k, i in enumerate(indexes)}

    def _sentence_text(self, index: int) -> str:
        tracker = self.tracker
        first, last = tracker.sentence_range(index)
        cache: dict[int, dict[int, str]] = {}
        parts = []
        for i in range(first, last + 1):
            seg = tracker.tokens[i].seg
            if seg not in cache:
                cache[seg] = self._pieces(seg)
            parts.append(cache[seg][i])
        return ''.join(parts).strip()

    async def _fill_stage(self, final: bool, *, second: bool):
        tracker = self.tracker
        wanted = tracker.fill2_candidates() if second else tracker.fill_candidates(final)
        due = tracker.fill2_due if second else tracker.fill_due
        if not wanted or not (final or due(wanted)):
            return
        agent, stats = ('speaker_fill2', self.fill2_stats) if second else ('speaker_fill', self.fill_stats)
        mark = tracker.mark_asked2 if second else tracker.mark_asked
        apply = tracker.apply_fill2 if second else tracker.apply_fill
        system = fill.PROMPT2 if second else fill.PROMPT
        for ask in fill.chunks(wanted):
            if self.failed:
                return
            mark(ask)                                 # asked once: a failed call leaves its sentences unknown rather than retrying forever
            ids = tracker.group_ids
            valid = {fill.speaker_key(g): g for g in ids}
            speakers = [{'id': fill.speaker_key(g), 'name': self.name_of(g)} for g in ids]
            shown = fill.window(ask, tracker.stats()['decided'])
            rows, previous_end = [], None
            for i in shown:
                sentence = tracker.sentences[i]
                if i > 0 and previous_end is None:
                    previous_end = tracker.sentences[i - 1].t1
                gid = tracker.resolve(sentence.gid)
                if gid is None or (not second and sentence.source != 'voice'):
                    who = '?'
                else:
                    who = fill.speaker_key(gid) + ('*' if second and sentence.source in TEXT_SOURCES else '')
                rows.append(fill.row(i, sentence.t0, previous_end, who, self._sentence_text(i)))
                previous_end = sentence.t1
            user = fill.request_payload(ask, speakers, rows)
            try:
                answers = await self._llm(system=system, user=user, max_tokens=min(3000, 60 * len(ask) + 300), retries=1,
                                          retry_note=fill.RETRY_NOTE,
                                          validator=lambda text, ask=ask, valid=valid: fill.parse_answers(text, ask, valid))
            except asyncio.CancelledError:
                mark(ask, False)                      # cancelled (the meeting is ending): the final pass asks these again
                raise
            except ModelError as exc:
                stats['failed_calls'] += 1
                self.emit(f'{agent}_failed', error=str(exc), sentences=len(ask))
                continue
            taken = [i for i, gid in answers.items() if apply(i, gid)]
            stats['asked'] += len(ask)
            stats['accepted'] += len(taken)
            stats['rejected'] += len(answers) - len(taken)
            extra = {}
            if second:
                differing = sum(1 for i in taken if tracker.resolve(tracker.sentences[i].said) != tracker.resolve(answers[i]))
                stats['accepted_differing'] += differing
                stats['accepted_agreeing'] += len(taken) - differing
                extra = {'agreeing': len(taken) - differing, 'differing': differing}
            self.emit(agent, asked=len(ask), answered=len(answers), accepted=len(taken), **extra)
            if taken:
                self._on_change({tracker.tokens[i].seg for s in taken for i in range(*self._token_span(s))})

    def _token_span(self, sentence: int) -> tuple[int, int]:
        first, last = self.tracker.sentence_range(sentence)
        return first, last + 1

    # ------------------------------------------------------------------ the end of the meeting
    async def finish(self):
        await self.stop()
        if not self.active or self._finished:
            return
        self._finished = True
        try:
            changed = await asyncio.to_thread(self.tracker.finish)
            self._after_tracker(changed)
            await asyncio.wait_for(self.settle(final=True), FINAL_PASS_SECONDS)
        except asyncio.TimeoutError:
            self.emit('speaker_error', where='final_pass', error=f'最後一輪補標超過 {FINAL_PASS_SECONDS:g} 秒，已略過。')
        except Exception as exc:
            self._fail(exc, 'finish')
        self.emit('speaker_finished', **self.tracker.stats(), fill=self.fill_stats, fill2=self.fill2_stats)
