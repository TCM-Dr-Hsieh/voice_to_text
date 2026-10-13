import asyncio
import json
from dataclasses import replace

import numpy as np
import pytest

from voice_app.alignment import TimedText
from voice_app.api import ModelError
from voice_app.config import MeetingSettings
from voice_app.session import Segment
from voice_app.speaker.fill import clock, parse_answers
from voice_app.speaker.render import Run, project, segment_runs, speaker_runs
from voice_app.speaker.service import SpeakerService, window_tokens
from tests.speaker_fakes import FakeEmbedder, RATE, bases, build, make_turns, window_of


def selected_units(tokens, window):
    """The aligned units one window adds, as the session's `select_new_window` returns them, and the text they stand for."""
    added, units = '', []
    for token in tokens:
        piece = '字' + ('。' if token.sentence_end else '')
        units.append(TimedText(piece, token.start, token.end, len(added), len(added) + len(piece)))
        added += piece
    return added, units


async def feed_meeting(service, segments, turns, *, count_windows=None):
    audio, tokens, truth = build(turns)
    by_window = {}
    for token in tokens:
        by_window.setdefault(window_of(token.start), []).append(token)
    for k in range(count_windows or int(np.ceil(len(audio) / RATE / 3.0))):
        added, units = selected_units(by_window.get(k, []), k)
        segment = Segment(k + 1, 3.0 * k, 3.0 * k + 6, added, added, added, 0.0, '', 'window', [], [], added_conv=added)
        segments.append(segment)
        if not units:
            continue
        # the tokens carry segment numbers of their own in the fakes: renumber them like the session does
        await service.feed_segment(3.0 * k, audio[int(3.0 * k * RATE):int((3.0 * k + 6.0) * RATE)], segment, units, added)
    return truth


def make_service(llm=None, embed=None, **settings):
    segments = []
    service = SpeakerService(settings=MeetingSettings(**settings), embed=embed or FakeEmbedder(bases(3)),
                             llm=llm or (lambda **kw: asyncio.sleep(0)), segments=lambda: segments)
    return service, segments


def test_window_tokens_follow_the_converted_text_when_it_has_another_length():
    units = [TimedText('头', 0.0, 0.2, 0, 1), TimedText('发。', 0.2, 0.4, 1, 3)]
    same = window_tokens(1, units, '头发。', '頭髮。')
    assert [(t.c0, t.c1, t.sentence_end) for t in same] == [(0, 1, False), (1, 3, True)]
    longer = window_tokens(1, units, '头发。', '頭髮。。')
    assert longer[0].c0 == 0 and longer[-1].c1 == 4 and longer[0].c1 <= longer[1].c0 + 1
    assert window_tokens(1, [], '', '') == []


def test_runs_are_cut_where_the_speaker_changes_and_follow_the_proofread_text():
    labels = ((0, 4, 0, 'voice'), (4, 8, 1, 'voice'))
    runs = segment_runs('甲乙丙丁戊己庚辛', '甲乙丙丁，戊己庚辛', labels)
    assert [(r.gid, r.text) for r in runs] == [(0, '甲乙丙丁'), (1, '，戊己庚辛')]     # text the proofreader inserted joins the run after it
    unknown = segment_runs('甲乙', '甲乙', ((0, 1, None, ''), (1, 2, None, '')))
    assert [(r.gid, r.source, r.text) for r in unknown] == [(None, '', '甲乙')]
    assert segment_runs('甲', '', ((0, 1, 0, 'voice'),)) == []
    assert project('abcdef', 'abXYcdef', [0, 2, 6]) == [0, 2, 8]               # a boundary stays before what was inserted at it
    merged = speaker_runs([Run(0, 'voice', '你好，'), Run(0, 'text', '我是'), Run(1, 'voice', '好')])
    assert [(r.gid, r.text, r.source) for r in merged] == [(0, '你好，我是', ''), (1, '好', 'voice')]


@pytest.mark.asyncio
async def test_speakers_are_found_named_and_shown_on_the_proofread_text():
    service, segments = make_service()
    turns = make_turns([0, 1, 2] * 14, seed=4)
    truth = await feed_meeting(service, segments, turns)
    await service.finish()
    assert service.tracker.group_count == 3
    assert '已分出 3 位說話者' in service.status
    runs = [r for s in segments for r in service.runs(s)]
    assert {r.gid for r in runs if r.gid is not None} == set(service.tracker.group_ids)
    assert service.name_of(0) == 'Speaker-001' and service.name_of(None) == '不明'
    service.set_name(0, '  王   先生  ')
    assert service.name_of(0) == '王 先生'
    service.set_name(0, '')
    assert service.name_of(0) == 'Speaker-001'
    service.set_name(1, 'x' * 100)
    assert len(service.name_of(1)) == 40
    with pytest.raises(ValueError):
        service.set_name(99, '誰')
    info = {s['gid']: s for s in service.speakers()}
    assert set(info) == {0, 1, 2} and info[1]['name'] == 'x' * 40 and info[0]['key'] == 'S1'
    assert all(s['sentences'] > 0 and s['examples'] for s in info.values())
    json.dumps(service.map_doc(), ensure_ascii=False)                       # no voiceprints, nothing that cannot be saved
    doc = service.labels_doc()
    assert doc and all(len(item) == 4 for items in doc.values() for item in items)


@pytest.mark.asyncio
async def test_joining_two_speakers_keeps_a_name_and_changes_the_displayed_label():
    service, segments = make_service()
    await feed_meeting(service, segments, make_turns([0, 1, 2] * 14, seed=4))
    await service.finish()
    service.set_name(2, '陳教授')
    changed = service.merge(2, 1)
    assert changed and service.name_of(1) == '陳教授'
    assert service.tracker.group_count == 2
    shown = {r.gid for s in segments for r in service.runs(s)}
    assert 2 not in shown
    assert service.name_events[-1]['event'] == 'merge'


@pytest.mark.asyncio
async def test_the_text_fill_names_speakers_the_voice_could_not_decide():
    # the speaker -1 is silent in the audio: no voiceprint, so those sentences stay unknown for the voice
    calls = []

    async def llm(**kw):
        calls.append(kw)
        ask = json.loads(kw['user'])['ask_ids']
        reply = json.dumps({'answers': [{'id': i, 'speaker': 'S1', 'basis': '其他'} for i in ask]})
        return kw['validator'](reply)

    pattern = [0, 1, -1, 0, 1, 1, -1, 0, 1, 0] * 4
    service, segments = make_service(llm=llm, embed=FakeEmbedder(bases(2)), text_fill=True, text_fill2=True)
    # the second pass normally waits a minute after the first one's answer; here both run in the final pass
    service.tracker.params = replace(service.tracker.params, fill2_refits=0, fill2_batch=1)
    await feed_meeting(service, segments, make_turns(pattern, seed=5))
    await service.finish()
    stats, stats2 = service.fill_stats, service.fill2_stats
    assert calls and stats['asked'] > 0
    assert stats['accepted'] == 0                        # no voice evidence to agree with: the first pass takes nothing
    assert stats2['accepted'] > 0                        # the second pass takes the named speaker
    text_labels = [item[3] for items in service.labels_doc().values() for item in items]
    assert 'text2' in text_labels
    sent = json.loads(calls[0]['user'])
    assert [s['id'] for s in sent['speakers']] == ['S1', 'S2'] and sent['dialogue'] and 'ask_ids' in sent
    assert all('S1' in r['who'] or 'S2' in r['who'] or r['who'] == '?' for r in sent['dialogue'])


@pytest.mark.asyncio
async def test_a_text_fill_that_fails_leaves_the_sentences_unknown_and_does_not_retry_them():
    async def llm(**kw):
        raise ModelError('server down')

    service, segments = make_service(llm=llm, embed=FakeEmbedder(bases(2)))
    await feed_meeting(service, segments, make_turns([0, 1, -1, 0, 1, 1, -1, 0, 1, 0] * 4, seed=5))
    await service.finish()
    assert service.fill_stats['failed_calls'] >= 1 and service.fill_stats['accepted'] == 0
    assert not service.failed                           # a model server's trouble never turns the marking off
    asked = service.fill_stats['failed_calls']
    await service.settle(final=True)
    assert service.fill_stats['failed_calls'] == asked


@pytest.mark.asyncio
async def test_an_error_in_the_marking_stops_it_but_never_the_transcript():
    def broken(samples):
        raise RuntimeError('onnx exploded')

    service, segments = make_service(embed=broken)
    await feed_meeting(service, segments, make_turns([0, 1] * 6))     # must not raise
    assert 'RuntimeError' in service.failed and '已停止' in service.status
    runs = service.runs(segments[1])
    assert runs and runs[0].gid is None
    await service.finish()                                              # still harmless


@pytest.mark.asyncio
async def test_without_a_voiceprint_model_the_meeting_has_no_speakers_at_all():
    service, segments = make_service()
    service.unavailable = '找不到說話者聲紋模型'
    await feed_meeting(service, segments, make_turns([0, 1] * 6))
    assert service.tracker.group_count == 0 and not service.active
    assert '未啟用' in service.status and service.runs(segments[1])[0].gid is None


def test_fill_answers_are_read_leniently_but_must_cover_every_question():
    valid = {'S1': 0, 'S2': 1}
    reply = '```json\n{"answers":[{"id":4,"speaker":"s2*"},{"id":5,"speaker":"不明"},{"id":6,"speaker":"S9"},],}\n```'
    assert parse_answers(reply, [4, 5, 6], valid) == {4: 1}              # a star, other case and a trailing comma are fine
    with pytest.raises(ModelError):
        parse_answers('{"answers":[{"id":4,"speaker":"S1"}]}', [4, 5], valid)
    with pytest.raises(ModelError):
        parse_answers('not json', [4], valid)
    assert clock(75) == '01:15' and clock(3725) == '1:02:05'
