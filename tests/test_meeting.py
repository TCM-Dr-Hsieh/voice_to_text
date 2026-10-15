import asyncio
import json

import numpy as np
import pytest

from voice_app.api import RollingResult
from voice_app.audio import Chunk, RATE
from voice_app.config import Settings
from voice_app.meeting import Meeting, clock, reserve_meeting_path, segment_span
from voice_app.session import Session
from voice_app.translation import Draft, Translated
from tests.speaker_fakes import FakeEmbedder, bases, build, make_turns, window_of


class ReplayAPI:
    """The part of ModelAPI a session uses: recognition replayed from a script, and a proofreader that changes nothing."""

    def __init__(self, results):
        self.results = iter(results)
        self.revisions = []

    async def recognize(self, samples):
        return next(self.results)

    async def revise_recent(self, context, segments, on_preview, **kwargs):
        self.revisions.append(segments)
        return RollingResult([row['current'] for row in segments])


def window_results(tokens, count):
    """What the ASR would say for each 6-second window of the fake meeting: one character per token, with the aligner's times."""
    out = []
    for k in range(count):
        inside = [t for t in tokens if 3.0 * k <= t.start and t.end <= 3.0 * k + 6.0 + 1e-9]       # whole tokens only: the aligner never exceeds the audio
        text = ''.join('字' + ('。' if t.sentence_end else '') for t in inside)
        out.append({'text': text, 'items': [{'text': '字', 'start': t.start - 3.0 * k, 'end': t.end - 3.0 * k} for t in inside]})
    return out


def fake_meeting(tmp_path, pattern, speakers=2, seed=4, embed=None, **turn_kw):
    audio, tokens, truth = build(make_turns(pattern, seed=seed, **turn_kw))
    count = window_of(tokens[-1].start) + 2
    results = window_results(tokens, count)
    meeting = Meeting(Settings(), embed or FakeEmbedder(bases(speakers)), path=tmp_path / 'meeting.json')
    session = Session()
    session.kind = 'file'
    session.meeting = meeting
    meeting.attach(session)
    api = ReplayAPI(results)

    async def play(first, last):
        for k in range(first, min(last, count)):
            samples = audio[int(3.0 * k * RATE):int((3.0 * k + 6.0) * RATE)]
            await session.process_chunk(api, Chunk(k, 3.0 * k, samples, True, 'window'))
    return meeting, session, play, count


@pytest.mark.asyncio
async def test_the_page_is_told_about_changed_rows_only_and_the_newest_three_are_marked_revisable(tmp_path):
    meeting, session, play, count = fake_meeting(tmp_path, [0, 1] * 14)
    try:
        await play(0, 30)
        first = meeting.drain()
        assert first['reset'] is True and first['labels']['marking'] is True
        assert [row['i'] for row in first['rows']] == list(range(1, len(session.segments) + 1))
        assert [row['i'] for row in first['rows'] if row['rev']] == [len(session.segments) - 2, len(session.segments) - 1, len(session.segments)]
        assert meeting.drain() is None                                         # nothing changed: nothing is sent
        assert set(first['labels']['names']) == {str(g) for g in meeting.speaker.tracker.group_ids} and len(first['labels']['names']) == 2
        before = len(session.segments)
        await play(30, 31)
        update = meeting.drain()
        assert 'reset' not in update
        rows = {row['i']: row for row in update['rows']}
        new = len(session.segments)
        assert new == before + 1 and new in rows and not rows[new - 3]['rev'] and rows[new]['rev']     # the one that just locked is sent again
        assert all(run['t'] for row in rows.values() for run in row['runs'])
    finally:
        await meeting.close()


@pytest.mark.asyncio
async def test_a_rename_changes_the_labels_but_sends_no_rows_and_a_merge_resends_them(tmp_path):
    meeting, session, play, _ = fake_meeting(tmp_path, [0, 1] * 14)
    try:
        await play(0, 30)
        meeting.drain()
        gids = meeting.speaker.tracker.group_ids
        meeting.speaker.set_name(gids[0], '王先生')
        update = meeting.drain()
        assert update['rows'] == [] and update['labels']['names'][str(gids[0])] == '王先生'
        meeting.speaker.merge(gids[1], gids[0])
        update = meeting.drain()
        assert len(update['rows']) == len(session.segments) and len(update['labels']['names']) == 1
    finally:
        await meeting.close()




@pytest.mark.asyncio
async def test_when_the_marking_stops_the_page_is_told_with_the_labels_alone_and_must_redraw_its_rows_from_them(tmp_path):
    meeting, session, play, _ = fake_meeting(tmp_path, [0, 1] * 14)
    try:
        await play(0, 12)
        meeting.drain()
        meeting.speaker._fail(RuntimeError('onnx exploded'), 'feed')
        update = meeting.drain()
        assert update['rows'] == [] and update['labels']['marking'] is False        # the rows the page has stay as they were
    finally:
        await meeting.close()


@pytest.mark.asyncio
async def test_ending_the_meeting_decides_the_last_sentences_unlocks_the_rows_and_writes_both_files(tmp_path):
    meeting, session, play, count = fake_meeting(tmp_path, [0, 1] * 14)
    try:
        await play(0, count)
        meeting.drain()
        await meeting.end()
        assert meeting.ended and meeting.speaker.tracker.finished
        update = meeting.drain()
        assert update and not any(row['rev'] for row in update['rows'])
        doc = json.loads(meeting.path.read_text(encoding='utf-8'))
        assert doc['ended'] is True and len(doc['segments']) == len(session.segments) and doc['speakers']['stats']['groups'] == 2
        text = meeting.path.with_suffix('.txt').read_text(encoding='utf-8-sig')
        assert '語音#1(00:00-00:06) ' in text and 'Speaker-001' in text and 'Speaker-002' in text
        assert meeting.save_error == ''
        await meeting.end()                                                          # a second call changes nothing
    finally:
        await meeting.close()


@pytest.mark.asyncio
async def test_the_bilingual_text_puts_each_translation_under_its_segment_with_translated_names(tmp_path):
    meeting, session, play, _ = fake_meeting(tmp_path, [0, 1] * 14)
    try:
        await play(0, 12)
        gids = meeting.speaker.tracker.group_ids
        meeting.speaker.set_name(gids[0], '王先生')
        translator = meeting.translator
        translator.language = 'English'
        translator.names.language, translator.names.names = 'English', {gids[0]: 'Mr. Wang'}
        segment = session.segments[0]
        runs = meeting.speaker.translation_runs(segment)
        translator.results[1] = Translated(1, tuple(r.text for r in runs), [f'Hello {k}' for k in range(len(runs))], tuple(r.gid for r in runs))
        text = meeting.export_text()
        lines = text.splitlines()
        assert lines[0].startswith('會議紀錄') and '王先生' in lines[1] and 'Speaker-002' in lines[1]
        assert lines[2].startswith('翻譯語言：English')
        first = next(i for i, line in enumerate(lines) if line.startswith('語音#1('))
        assert lines[first].startswith('語音#1(00:00-00:06) ') and ': ' in lines[first]
        assert any(line.startswith('Voice#1 ') for line in lines[first:first + 4])
        assert 'Mr. Wang' in text or gids[0] not in [r.gid for r in runs]
        assert 'Voice#2 ' not in text                                        # untranslated segments have only the source line
    finally:
        await meeting.close()


@pytest.mark.asyncio
async def test_a_draft_translation_goes_to_the_page_as_one_text_and_into_the_bilingual_text_until_the_finished_one_replaces_it(tmp_path):
    meeting, session, play, _ = fake_meeting(tmp_path, [0, 1] * 14)
    try:
        await play(0, 12)
        index = len(session.segments)
        runs = meeting.speaker.translation_runs(session.segments[index - 1])
        translator = meeting.translator
        translator.language = 'English'
        translator.drafts[index] = Draft('Hello there', tuple(r.text for r in runs))
        meeting.drain()
        meeting._mark({index})
        row = next(r for r in meeting.drain()['rows'] if r['i'] == index)
        assert row['tr'] == {'st': 'draft', 'runs': [{'g': None, 't': 'Hello there'}], 'err': ''}
        text = meeting.export_text()
        assert f'Voice#{index} Hello there' in text and '翻譯語言：English' in text
        assert translator.doc()['segments'][str(index)] == {'texts': ['Hello there'], 'error': '', 'draft': True}
        translator.results[index] = Translated(index, tuple(r.text for r in runs), [f'T{k}' for k in range(len(runs))], tuple(r.gid for r in runs))
        meeting._mark({index})
        row = next(r for r in meeting.drain()['rows'] if r['i'] == index)
        assert row['tr']['st'] == 'ok' and row['tr']['runs'][0]['t'] == 'T0'
        assert 'Hello there' not in meeting.export_text()
    finally:
        await meeting.close()


def test_a_time_is_minutes_and_seconds_and_has_the_hours_from_an_hour_on():
    assert [clock(x) for x in (0, 9, 121, 3599)] == ['00:00', '00:09', '02:01', '59:59']
    assert [clock(x) for x in (3600, 3723)] == ['1:00:00', '1:02:03']
    from types import SimpleNamespace as S
    assert segment_span(S(owned_start=121.0, start=118.0, end=124.0)) == '02:01-02:04'          # (the window is 118-124, the segment owns 121-124)
    assert segment_span(S(owned_start=57.0, start=54.0, end=61.5)) == '00:57-01:02'              # whole seconds that hold it
    assert segment_span(S(owned_start=0.0, start=0.0, end=9.0)) == '00:00-00:09'
    assert segment_span(S(owned_start=3599.2, start=3596.0, end=3602.0)) == '0:59:59-1:00:02'
    assert segment_span(S(owned_start=None, start=30.0, end=39.0)) == '00:30-00:39'              # a segment made without it: its window
    assert segment_span(S(owned_start=12.0, start=9.0, end=12.0)) == '00:12-00:12'


@pytest.mark.asyncio
async def test_the_bilingual_text_says_where_in_the_audio_each_segment_is_but_the_page_does_not(tmp_path):
    meeting, session, play, count = fake_meeting(tmp_path, [0, 1] * 14)
    try:
        await play(0, count)
        await meeting.end()
        import re
        lines = meeting.export_text().splitlines()
        heads = [re.match(r'語音#(\d+)\((\d\d:\d\d)-(\d\d:\d\d)\) ', line) for line in lines if line.startswith('語音#')]
        assert heads and all(heads)
        seen = [(int(m[1]), m[2], m[3]) for m in heads]
        assert seen[0] == (1, '00:00', '00:06')
        for (_, _, before_end), (_, start, _) in zip(seen, seen[1:]):
            assert start == before_end                                                    # each segment starts where the one before it ended
        assert [segment.owned_start for segment in session.segments[:3]] == [0.0, 6.0, 9.0]    # (windows of 6 s every 3 s)
        update = meeting.drain()
        assert update and update['rows']
        assert [row['t'] for row in update['rows']] == [round(segment.owned_start, 2) for segment in session.segments]      # (what the page replays from)
        assert 'span' not in json.dumps(update) and 'owned_start' not in json.dumps(update)   # (the start is sent as `t`; no field of the exported text's)
        many = next(index for index in range(1, len(session.segments) + 1) if len(meeting.speaker.runs(session.segments[index - 1])) > 1)
        at = next(i for i, line in enumerate(lines) if line.startswith(f'語音#{many}('))
        head = lines[at].split(' ')[0]
        assert lines[at + 1].startswith(' ' * (len(head) + 1)) and '(' not in lines[at + 1].split(':')[0]       # only the first line has the time
    finally:
        await meeting.close()


@pytest.mark.asyncio
async def test_a_meeting_starts_from_traditional_text_so_a_proofreading_failure_leaves_no_simplified_text(tmp_path):
    meeting = Meeting(Settings(), FakeEmbedder(bases(1)), path=tmp_path / 'm.json')
    session = Session()
    session.meeting = meeting
    meeting.attach(session)
    try:
        api = ReplayAPI([{'text': '头发里面。', 'items': [{'text': c, 'start': i * .3, 'end': i * .3 + .2} for i, c in enumerate('头发里面')]}])
        await session.process_chunk(api, Chunk(0, 0, np.zeros(RATE * 4, dtype=np.float32), True, 'window'))
        segment = session.segments[0]
        assert segment.added == '头发里面。' and segment.added_conv == '頭髮裡面。' and segment.corrected_tail == '頭髮裡面。'
        assert api.revisions[0][0]['added'] == '头发里面。' and api.revisions[0][0]['current'] == '頭髮裡面。'     # the model still sees the ASR text
    finally:
        await meeting.close()


@pytest.mark.asyncio
async def test_cancelling_a_meeting_keeps_the_transcript_and_finishes_the_meeting(tmp_path):
    meeting, session, play, _ = fake_meeting(tmp_path, [0, 1] * 14)
    try:
        await play(0, 12)
        session.active = True
        session.source = None
        await session.cancel()
        assert meeting.ended and len(session.segments) > 0
        assert meeting.path.exists()
    finally:
        await meeting.close()


@pytest.mark.asyncio
async def test_a_full_disk_is_reported_and_retried_never_raised(tmp_path, monkeypatch):
    meeting, session, play, _ = fake_meeting(tmp_path, [0, 1] * 6)
    try:
        await play(0, 6)
        monkeypatch.setattr(Meeting, '_store', staticmethod(lambda files: (_ for _ in ()).throw(OSError('disk full'))))
        await meeting.save()
        assert 'disk full' in meeting.save_error and meeting._unsaved is True
        monkeypatch.undo()
        await meeting.save()
        assert meeting.save_error == '' and meeting.path.read_text(encoding='utf-8').startswith('{')
    finally:
        await meeting.close()


@pytest.mark.asyncio
async def test_the_meeting_saves_itself_while_it_runs(tmp_path):
    meeting, session, play, _ = fake_meeting(tmp_path, [0, 1] * 6)
    try:
        meeting.settings.meeting.autosave_seconds = 0.02
        meeting._autosave.cancel()
        meeting._autosave = asyncio.create_task(meeting._autosave_loop())
        await play(0, 6)
        for _ in range(200):
            await asyncio.sleep(0.02)
            if meeting.path.exists() and meeting.path.stat().st_size > 10:
                break
        assert json.loads(meeting.path.read_text(encoding='utf-8'))['segments']
    finally:
        await meeting.close()


def test_a_meeting_file_never_takes_the_name_of_an_earlier_one(tmp_path):
    first = reserve_meeting_path(tmp_path)
    second = reserve_meeting_path(tmp_path)
    assert first != second and first.name.startswith('會議-') and first.suffix == '.json'
    first.write_text('{}')
    assert reserve_meeting_path(tmp_path) not in (first, second)
