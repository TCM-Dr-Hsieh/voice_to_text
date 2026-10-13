import asyncio
import json
from types import SimpleNamespace

import pytest

import voice_app.translation as translation
from voice_app.api import ModelError
from voice_app.config import MeetingSettings, ModelConfig
from voice_app.speaker import default_name
from voice_app.speaker.render import Run
from voice_app.translation import ROLLING_SEGMENTS, Translator, fit_runs, parse_names, parse_rolling, parse_translation


class FakeSpeaker:
    def __init__(self, names=None):
        self.names = names or {}
        self.tracker = SimpleNamespace(group_ids=[0, 1])

    def translation_runs(self, segment):
        return [run for run in segment.runs if not run.background]

    def has_background(self, segment):
        return any(run.background for run in segment.runs)

    def name_of(self, gid):
        return '不明' if gid is None else self.names.get(gid) or default_name(gid)

    def custom_name(self, gid):
        return self.names.get(gid, '')


class FakeMeeting:
    """Segments with one run each (speaker 0 for odd numbers, 1 for even) and a speaker table the test can change."""

    def __init__(self, count, names=None):
        self.speaker = FakeSpeaker(names)
        self.segments = [SimpleNamespace(index=i, runs=[Run(i % 2, 'voice', f'原文{i}')]) for i in range(1, count + 1)]
        self.ended = False

    def segment_count(self):
        return len(self.segments)

    def segment(self, index):
        return self.segments[index - 1] if 1 <= index <= len(self.segments) else None

    def segment_indexes(self):
        return list(range(1, len(self.segments) + 1))


class FakeLLM:
    """Stands in for ModelAPI.chat_json: records every request, answers like a model that translates "原文N" into "T-N"."""

    def __init__(self):
        self.calls = []
        self.mode = 'good'
        self.fail = 0
        self.rolling_fail = 0                                    # how many of the next rolling (newest three segments) calls fail
        self.rolling_reply = None                                # a function (payload) -> {i: text} to say something else than "D-N"

    async def __call__(self, model, *, system, user, max_tokens, validator, retries, retry_note):
        payload = json.loads(user)
        self.calls.append({'system': system, 'user': payload, 'max_tokens': max_tokens})
        if self.fail:
            self.fail -= 1
            raise ModelError('server down')
        if '三段滾動' in system:                                  # the newest three segments: answers a draft "D-N" for the text "原文N"
            if self.rolling_fail:
                self.rolling_fail -= 1
                raise ModelError('server down')
            said = self.rolling_reply(payload) if self.rolling_reply else {
                item['i']: item['source'].replace('原文', 'D-') for item in payload['translate']}
            reply = {'segments': [{'i': i, 'text': text} for i, text in said.items()]}
            return validator(json.dumps(reply, ensure_ascii=False))
        if 'names' in payload:                                   # the names call
            reply = {'names': {k: f'EN-{v}' for k, v in payload['names'].items()}, 'voice': 'Voice', 'unknown': 'Unknown'}
            return validator(json.dumps(reply, ensure_ascii=False))
        segments = []
        for item in payload['translate']:
            runs = [{'n': r['n'], 'text': r['text'].replace('原文', 'T-')} for r in item['runs']]
            if self.mode == 'joined' and len(runs) > 1:
                runs = [{'n': 1, 'text': ' '.join(r['text'] for r in runs)}]
            segments.append({'i': item['i'], 'runs': runs})
        return validator(json.dumps({'segments': segments}, ensure_ascii=False))


def make(count=12, *, lag=3, batch=5, context=4, names=None, rolling=False):
    """The translator of a fake meeting. `rolling=False`: only the translation that follows the speaker runs, `lag` segments behind."""
    meeting = FakeMeeting(count, names)
    llm = FakeLLM()
    settings = MeetingSettings(translation_lag=lag, translation_batch=batch, translation_context=context)
    changes = []
    translator = Translator(meeting, llm, ModelConfig('m'), settings, on_change=lambda c: changes.append(set(c)))
    translator.language, translator.lag = 'English', lag
    translator.running = True
    translator.rolling = rolling
    return meeting, llm, translator, changes


async def drain(translator, limit=50):
    for _ in range(limit):
        if not await translator._step():
            return
    raise AssertionError('the translator never settled')


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    real = asyncio.sleep

    async def quick(seconds, *a):
        await real(0)

    monkeypatch.setattr(translation.asyncio, 'sleep', quick)


def test_replies_are_read_in_the_shapes_models_really_produce():
    assert parse_translation('{"segments":[{"i":1,"runs":["a","b"],"speaker":"x"}]}', [1]) == {1: ['a', 'b']}
    numbered = '```json\n{"segments":[{"i":2,"runs":[{"n":2,"text":"second"},{"n":1,"text":"first"},],}]}\n```'
    assert parse_translation(numbered, [2]) == {2: ['first', 'second']}
    with pytest.raises(ModelError, match='缺少'):
        parse_translation('{"segments":[{"i":1,"runs":["a"]}]}', [1, 2])
    with pytest.raises(ModelError):
        parse_translation('Sure! Here you go', [1])
    assert fit_runs(['all'], 2) == ['all', ''] and fit_runs(['a', 'b', 'c'], 2) == ['a', 'b c']
    assert parse_names('{"names":{"S1":"Mr. Wang"},"voice":"Voice","unknown":"Unknown"}', ['S1'])['names'] == {'S1': 'Mr. Wang'}
    with pytest.raises(ModelError):
        parse_names('{"names":{},"voice":"V","unknown":"U"}', ['S1'])


@pytest.mark.asyncio
async def test_the_translation_stays_the_lag_behind_the_transcript_until_the_meeting_ends():
    meeting, llm, translator, _ = make(count=10, lag=6)
    await drain(translator)
    assert sorted(translator.results) == [1, 2, 3, 4]                 # 10 - 6
    meeting.segments.append(SimpleNamespace(index=11, runs=[Run(0, 'voice', '原文11')]))
    await drain(translator)
    assert sorted(translator.results) == [1, 2, 3, 4, 5]               # one new segment, one more translated
    meeting.ended = True
    await drain(translator)
    assert sorted(translator.results) == list(range(1, 12))
    assert translator.results[7].texts == ['T-7'] and translator.status == '翻譯完成'
    assert translator.tr_state(7) == 'ok'


@pytest.mark.asyncio
async def test_catching_up_goes_in_batches_and_live_translation_one_segment_at_a_time():
    meeting, llm, translator, _ = make(count=14, lag=3, batch=4)
    await drain(translator)                                            # 11 to translate, 4 per call
    sizes = [len(c['user']['translate']) for c in llm.calls if 'translate' in c['user']]
    assert sizes == [4, 4, 3] and sorted(translator.results) == list(range(1, 12))
    meeting.segments.append(SimpleNamespace(index=15, runs=[Run(1, 'voice', '原文15')]))
    llm.calls.clear()
    await drain(translator)
    assert [len(c['user']['translate']) for c in llm.calls] == [1]


@pytest.mark.asyncio
async def test_the_model_sees_the_translated_past_and_the_untranslated_future():
    meeting, llm, translator, _ = make(count=20, lag=3, batch=1, context=4)
    await drain(translator)
    last = [c for c in llm.calls if 'translate' in c['user']][-1]['user']
    target = last['translate'][0]['i']
    assert target == 17
    assert [x['i'] for x in last['already_translated']] == [13, 14, 15, 16]          # `context` segments, already translated
    assert [x['i'] for x in last['read_ahead']] == [18, 19, 20]                       # the lag: the text that follows
    assert last['already_translated'][0]['runs'][0]['text'] == 'T-13'
    assert last['read_ahead'][0]['runs'][0] == {'n': 1, 'speaker': default_name(0), 'text': '原文18'}
    first = [c for c in llm.calls if 'translate' in c['user']][0]['user']
    assert first['already_translated'] == []
    assert 'English' in llm.calls[0]['system']


@pytest.mark.asyncio
async def test_names_are_translated_first_and_a_rename_translates_only_the_names():
    meeting, llm, translator, changes = make(count=10, lag=3, names={0: '王先生'})
    await drain(translator)
    assert 'names' in llm.calls[0]['user'] and llm.calls[0]['user']['names'] == {'S1': '王先生'}
    assert translator.label_for(0) == 'EN-王先生' and translator.label_for(1) == default_name(1) and translator.label_for(None) == 'Unknown'
    done = dict(translator.results)
    calls = len(llm.calls)
    meeting.speaker.names[1] = '陳主任'
    await drain(translator)
    assert len(llm.calls) == calls + 1 and 'names' in llm.calls[-1]['user']       # one call, for the names
    assert translator.label_for(1) == 'EN-陳主任'
    assert translator.results == done                                              # not one segment was translated again


@pytest.mark.asyncio
async def test_a_segment_whose_speaker_runs_changed_is_translated_again():
    meeting, llm, translator, _ = make(count=8, lag=3)
    await drain(translator)
    assert translator.tr_state(2) == 'ok'
    meeting.segments[1].runs = [Run(0, 'voice', '原文2甲'), Run(1, 'voice', '原文2乙')]      # a later label split it in two
    assert translator.tr_state(2) == 'stale'
    translator.requeue({2})                                                              # what the meeting does when marks change
    await drain(translator)
    assert translator.tr_state(2) == 'ok' and translator.results[2].texts == ['T-2甲', 'T-2乙']
    assert translator.results[2].gids == (0, 1) and translator.stats['retranslated'] == 1


@pytest.mark.asyncio
async def test_a_changed_label_with_the_same_runs_does_not_cost_a_translation():
    meeting, llm, translator, _ = make(count=8, lag=3)
    await drain(translator)
    calls = len(llm.calls)
    meeting.segments[1].runs = [Run(0, 'voice', '原文2')]                                # the same words, another speaker
    assert translator.tr_state(2) == 'ok'
    await drain(translator)
    assert len(llm.calls) == calls


@pytest.mark.asyncio
async def test_runs_the_model_joined_are_asked_again_alone_and_then_fitted():
    meeting, llm, translator, _ = make(count=8, lag=3, batch=5)
    meeting.segments[2].runs = [Run(0, 'voice', '原文3甲'), Run(1, 'voice', '原文3乙')]
    llm.mode = 'joined'
    await drain(translator)
    assert translator.results[3].texts == ['T-3甲 T-3乙', '']                           # the text stays on the first run
    assert translator.stats['fitted'] == 1 and not translator.results[3].error
    assert all(translator.results[i].texts == [f'T-{i}'] for i in (1, 2, 4, 5))         # the others were not thrown away
    assert any(len(c['user'].get('translate', [])) == 1 and c['user']['translate'][0]['i'] == 3 for c in llm.calls)


@pytest.mark.asyncio
async def test_a_failing_server_is_retried_and_one_hopeless_segment_does_not_hold_back_the_rest():
    meeting, llm, translator, _ = make(count=6, lag=3, batch=3)
    llm.fail = 2
    await drain(translator)
    assert sorted(translator.results) == [1, 2, 3] and translator.error == ''
    llm.fail = 100                                                                      # now it is down for good
    meeting.segments.append(SimpleNamespace(index=7, runs=[Run(0, 'voice', '原文7')]))
    for _ in range(12):
        await translator._step()
    assert translator.results[4].error and translator.tr_state(4) == 'error'            # given up after three tries, shown as failed
    assert translator.stats['failed_calls'] >= 3


@pytest.mark.asyncio
async def test_starting_checks_its_inputs_and_a_new_language_starts_over():
    meeting, llm, translator, _ = make(count=6, lag=3)
    translator.running = False
    with pytest.raises(ValueError, match='語言'):
        translator.start('   ', 6)
    with pytest.raises(ValueError, match='延遲'):
        translator.start('English', 2)
    translator.start('English', 5)
    assert translator.running and translator.lag == 5
    await translator.stop()
    translator.running = True
    await drain(translator)
    assert translator.results
    translator.start('日本語', 4)
    assert translator.results == {} and translator.language == '日本語'
    await translator.stop()
    assert translator.status == '翻譯已停止' and not translator.running


@pytest.mark.asyncio
async def test_the_loop_translates_in_the_background_and_wakes_up_for_new_segments():
    meeting, llm, translator, changes = make(count=6, lag=3)
    translator.running = False
    translator.start('English', 3)
    for _ in range(100):
        await asyncio.sleep(0)
        if len(translator.results) == 3:
            break
    assert sorted(translator.results) == [1, 2, 3]
    meeting.segments.append(SimpleNamespace(index=7, runs=[Run(0, 'voice', '原文7')]))
    translator.wake()
    for _ in range(100):
        await asyncio.sleep(0)
        if 4 in translator.results:
            break
    assert 4 in translator.results and any(4 in c for c in changes)
    await translator.stop()
    assert translator.doc()['segments']['1']['texts'] == ['T-1'] and translator.doc()['language'] == 'English'


@pytest.mark.asyncio
async def test_a_names_call_that_fails_does_not_hold_up_the_segments_and_is_tried_again_later(monkeypatch):
    meeting, llm, translator, _ = make(count=10, lag=3, names={0: '王先生'})
    real_call = llm.__call__

    async def names_fail(model, **kw):
        if 'names' in json.loads(kw['user']):
            llm.calls.append({'system': kw['system'], 'user': json.loads(kw['user']), 'max_tokens': kw['max_tokens']})
            raise ModelError('names reply was not JSON')
        return await real_call(model, **kw)

    translator._call = names_fail
    await drain(translator)
    assert sorted(translator.results) == list(range(1, 8))                       # the segments were translated regardless
    assert translator.label_for(0) == default_name(0)                            # with the default name for the time being
    assert translator.names_error.startswith('名稱翻譯失敗') and translator.error == ''   # shown on its own, not wiped by the next segment
    tries = sum('names' in c['user'] for c in llm.calls)
    assert tries == 1                                                            # one failure, then a pause, not a retry storm
    translator._names_retry_after = 0.0                                          # the pause is over
    translator._call = llm
    await drain(translator)
    assert translator.label_for(0) == 'EN-王先生' and translator.names_error == ''


@pytest.mark.asyncio
async def test_a_failed_segment_whose_runs_changed_is_tried_afresh_not_left_listed_as_failed():
    meeting, llm, translator, _ = make(count=8, lag=3)
    llm.fail = 100                                                                 # the server is down
    for _ in range(14):
        await translator._step()
    failed = translator.failed()
    assert len(failed) >= 2
    whole, part = failed[0], failed[1]
    llm.fail = 0
    meeting.segments[whole - 1].runs = [Run(None, 'background', f'原文{whole}')]                      # all of it was marked background talk
    meeting.segments[part - 1].runs = [Run(None, 'background', f'原文{part}甲'), Run(0, 'voice', f'原文{part}乙')]   # part of it was
    translator.requeue({whole, part})
    assert whole not in translator.failed() and part not in translator.failed()   # no failure, no retry button, for what is not what failed
    meeting.ended = True
    await drain(translator, 80)
    assert translator.results[whole].texts == [] and not translator.results[whole].error      # nothing left to translate: done, no call
    assert not translator.results[part].error and len(translator.results[part].texts) == 1    # only what is not background was translated
    assert translator.tr_state(whole) == 'ok' and translator.tr_state(part) == 'ok'
    assert not any(f'原文{part}甲' in json.dumps(call['user'], ensure_ascii=False) for call in llm.calls[-3:])


@pytest.mark.asyncio
async def test_a_failure_for_text_that_changed_during_the_call_is_not_held_against_the_new_text():
    meeting, llm, translator, _ = make(count=8, lag=3)
    translator._failures[1] = 2                                                    # two failures already: the next one would give it up
    translator._single.add(1)                                                      # (and it is asked alone)

    class MarkedWhileAsked(FakeLLM):
        async def __call__(self, model, **kwargs):
            if 'translate' not in json.loads(kwargs['user']):                      # (the names call is answered as usual)
                return await super().__call__(model, **kwargs)
            meeting.segments[0].runs = [Run(None, 'background', '原文1')]            # the user marks it background while the call is out
            raise translation.ModelError('server down')

    translator._call = MarkedWhileAsked()
    await translator._step()                                                       # (the names first)
    await translator._step()
    assert translator.stats['failed_calls'] == 1                                   # the segment call did fail
    assert 1 not in translator.failed() and 1 not in translator.results           # not given up: not the case that failed
    assert 1 not in translator._failures and 1 not in translator._single
    translator._call = FakeLLM()
    meeting.ended = True
    await drain(translator, 80)
    assert translator.results[1].texts == [] and not translator.results[1].error   # nothing is left to translate there


@pytest.mark.asyncio
async def test_an_old_error_message_goes_when_what_failed_turns_out_to_be_background_talk():
    meeting, llm, translator, _ = make(count=6, lag=3)
    llm.fail = 100                                                                 # the server is down
    for _ in range(14):
        await translator._step()
    assert translator.failed() and translator.error.startswith('翻譯失敗') and translator._failures
    for segment in meeting.segments:                                               # all of it is marked background talk
        segment.runs = [Run(None, 'background', f'原文{segment.index}')]
    translator.requeue(set(range(1, 7)))
    assert translator.failed() == [] and translator._failures == {}               # no failure is left over, counted or given up
    meeting.ended = True
    await drain(translator, 80)                                                    # nothing to translate: not one call to the (down) server
    assert all(r.texts == [] and not r.error for r in translator.results.values()) and len(translator.results) == 6
    assert translator.error == '' and translator.status == '翻譯完成'
    assert translator._batch == translator.settings.translation_batch


def test_failures_counted_for_a_segment_that_is_not_given_up_yet_are_forgotten_when_its_runs_change():
    meeting, llm, translator, _ = make(count=6, lag=3)
    translator._failures.update({1: 2, 2: 1})                                      # two segments mid-failure: no result yet
    translator._single.add(1)
    translator._trouble.update({1: ('原文1',), 2: ('原文2',)})                       # (their runs as they were when they went wrong)
    meeting.segments[0].runs = [Run(None, 'background', '原文1')]                    # segment 1 is marked background talk
    translator.requeue({1})
    assert translator._failures == {2: 1} and 1 not in translator._single          # only the one that changed starts afresh
    assert list(translator._trouble) == [2]


def test_a_marking_update_that_changes_nothing_the_translator_is_given_does_not_reset_the_failures():
    meeting, llm, translator, _ = make(count=6, lag=3)
    translator._failures.update({2: 2})                                            # one more failure and it is given up
    translator._single.add(2)
    translator._trouble[2] = ('原文2',)
    meeting.segments[1].runs = [Run(1, 'text', '原文2')]                             # the same words, the same speaker: only HOW it was decided changed
    translator.requeue({2})
    assert translator._failures == {2: 2} and 2 in translator._single and translator._trouble == {2: ('原文2',)}
    meeting.segments[1].runs = [Run(1, 'voice', '原文2（改）')]                       # now the words are not the same
    translator.requeue({2})
    assert translator._failures == {} and not translator._single and translator._trouble == {}




@pytest.mark.asyncio
async def test_new_input_never_inherits_old_failures_whether_or_not_anyone_announced_the_change():
    meeting, llm, translator, _ = make(count=6, lag=3)
    await translator._step()                                                       # (the names)
    translator._cursor = 2
    translator._failures[2], translator._trouble[2] = 2, ('原文2',)                  # two failures on the old words: one more and it is given up
    translator._single.add(2)
    meeting.segments[1].runs = [Run(1, 'voice', '整段新的文字')]                      # the words changed and nobody called requeue (the translator was only woken)
    llm.fail = 1
    await translator._step()
    assert translator.stats['failed_calls'] == 1
    assert translator._failures == {2: 1} and translator._trouble == {2: ('整段新的文字',)}   # the first failure of the new words, not the third
    assert 2 not in translator.failed() and 2 not in translator.results

    translator.results[3] = translation.Translated(3, ('原文3',), [], (), error='server down')    # given up on the old words ...
    meeting.segments[2].runs = [Run(0, 'voice', '另一段新的文字')]                    # ... which then changed, again without a word
    assert translator.failed() == [3]
    llm.fail = 0
    await translator._step()
    assert 3 not in translator.failed()                                            # not listed as failed for words that never failed
    meeting.ended = True
    await drain(translator, 80)
    assert translator.results[3].texts and not translator.results[3].error and not translator.failed()



# ---------------------------------------------------------------------- the newest segments, translated as they come

def rolling_calls(llm):
    return [c['user'] for c in llm.calls if '三段滾動' in c['system']]


def run_calls(llm):
    """The calls that translate by speaker runs (the finished translation)."""
    return [c['user'] for c in llm.calls if 'translate' in c['user'] and c['user']['translate'] and 'runs' in c['user']['translate'][0]]


def segments_of(calls):
    return [item['i'] for call in calls for item in call['translate']]


def add_segment(meeting, text=None):
    i = len(meeting.segments) + 1
    meeting.segments.append(SimpleNamespace(index=i, runs=[Run(i % 2, 'voice', text or f'原文{i}')]))


def test_the_rolling_translation_is_as_long_as_the_proofreader_revises_and_the_least_lag():
    from voice_app.config import MIN_TRANSLATION_LAG
    from voice_app.meeting import REVISABLE
    assert ROLLING_SEGMENTS == REVISABLE == MIN_TRANSLATION_LAG == 3


def test_the_rolling_reply_is_read_and_must_have_a_translation_for_every_segment():
    assert parse_rolling('{"segments":[{"i":8,"text":" a "},{"i":9,"text":"b"}]}', [8, 9]) == {8: 'a', 9: 'b'}
    assert parse_rolling('```json\n{"segments":[{"i":9,"text":"b",},],}\n```', [9]) == {9: 'b'}
    for reply, message in [('{"segments":[{"i":8,"text":"a"}]}', '缺少'), ('{"segments":[{"i":8,"text":"a"},{"i":9,"text":" "}]}', '空的'),
                           ('{"segments":[{"i":8}]}', '無效'), ('here you go', '無效')]:
        with pytest.raises(ModelError, match=message):
            parse_rolling(reply, [8, 9])


@pytest.mark.asyncio
async def test_the_newest_three_segments_are_translated_at_once_while_the_finished_translation_waits_for_the_lag():
    meeting, llm, translator, _ = make(count=10, lag=6, rolling=True)
    await drain(translator)
    first = rolling_calls(llm)[0]
    assert [item['i'] for item in first['translate']] == [8, 9, 10] and all(item['current'] == '' for item in first['translate'])
    assert first['translate'][0]['source'] == '原文8' and first['glossary'] == ''
    assert [translator.draft_text(i) for i in (8, 9, 10)] == ['D-8', 'D-9', 'D-10']
    assert sorted(translator.results) == [1, 2, 3, 4]                                  # the finished ones: 10 - lag
    assert [translator.tr_state(i) for i in (2, 6, 9)] == ['ok', '', 'draft']          # segment 6 has neither yet
    assert translator.stats['rolling'] == 1


@pytest.mark.asyncio
async def test_every_new_segment_translates_the_last_three_again_with_the_drafts_they_have():
    meeting, llm, translator, _ = make(count=10, lag=6, rolling=True)
    await drain(translator)
    add_segment(meeting)
    await drain(translator)
    second = rolling_calls(llm)[1]
    assert [(item['i'], item['current']) for item in second['translate']] == [(9, 'D-9'), (10, 'D-10'), (11, '')]
    llm.rolling_reply = lambda payload: {item['i']: (item['current'] + '+' if item['current'] else item['source'].replace('原文', 'D-'))
                                         for item in payload['translate']}
    add_segment(meeting)
    await drain(translator)
    assert [translator.draft_text(i) for i in (10, 11, 12)] == ['D-10+', 'D-11+', 'D-12']      # the model mended the older two
    assert translator.draft_text(9) == 'D-9'                                                    # (9 is out of the window: it stays)


@pytest.mark.asyncio
async def test_the_model_is_given_the_finished_translations_the_newest_five_with_their_source_and_the_rest_without():
    meeting, llm, translator, _ = make(count=14, lag=3, context=10, rolling=True)
    await drain(translator)
    for _ in range(2):
        add_segment(meeting)
        await drain(translator)
    call = rolling_calls(llm)[-1]
    first = call['translate'][0]['i']
    context = call['already_translated']
    assert [item['i'] for item in context] == list(range(max(1, first - 10), first))
    for item in context:
        assert ('source' in item) == (first - item['i'] <= 5), item['i']
        if 'source' in item:
            assert item['translation'] in (item['source'].replace('原文', 'T-'), item['source'].replace('原文', 'D-'))


@pytest.mark.asyncio
async def test_a_segment_with_one_speaker_whose_draft_was_made_from_its_text_needs_no_second_translation():
    meeting, llm, translator, changes = make(count=5, lag=3, rolling=True)
    await drain(translator)
    for _ in range(3):
        add_segment(meeting)
        await drain(translator)
    asked = segments_of(run_calls(llm))
    assert translator.results[3].texts == ['D-3'] and translator.results[3].sources == ('原文3',) and translator.tr_state(3) == 'ok'
    assert 3 not in asked and 4 not in asked and translator.stats['adopted'] >= 2
    assert translator.results[3].gids == (1,)                                        # segment 3 was spoken by speaker 1 (odd)
    assert any(3 in change for change in changes)                                    # the screen was told


@pytest.mark.asyncio
async def test_a_segment_with_several_speakers_is_translated_by_runs():
    meeting, llm, translator, _ = make(count=5, lag=3, rolling=True)
    await drain(translator)
    meeting.segments[2].runs = [Run(0, 'voice', '甲'), Run(1, 'voice', '乙')]            # the speakers of segment 3 turned out to be two
    add_segment(meeting)
    await drain(translator)
    assert 3 in segments_of(run_calls(llm)) and translator.results[3].texts == ['甲', '乙']


@pytest.mark.asyncio
async def test_a_segment_whose_text_changed_since_its_draft_is_translated_by_runs():
    meeting, llm, translator, _ = make(count=5, lag=3, rolling=True)
    await drain(translator)
    meeting.segments[2].runs = [Run(1, 'voice', '原文3改')]                              # (the draft was made from 原文3)
    add_segment(meeting)
    await drain(translator)
    assert 3 in segments_of(run_calls(llm)) and translator.results[3].texts == ['T-3改']


@pytest.mark.asyncio
async def test_a_failing_rolling_translation_waits_and_never_holds_the_finished_translation_back():
    meeting, llm, translator, _ = make(count=10, lag=6, rolling=True)
    llm.rolling_fail = 2
    await drain(translator)
    assert translator.drafts == {} and translator.error.startswith('翻譯失敗') and sorted(translator.results) == [1, 2, 3, 4]
    assert not translator._rolling_due()                                              # it waits (2 s after the first failure), no hot loop
    assert translator._rolling_retry_after > 0 and translator.stats['failed_calls'] == 1
    translator._rolling_retry_after = 0.0
    await drain(translator)                                                           # the second failure
    assert translator._rolling_failures == 2 and not translator._rolling_due() and translator.error.startswith('翻譯失敗')
    translator._rolling_retry_after = 0.0
    await drain(translator)
    assert [translator.draft_text(i) for i in (8, 9, 10)] == ['D-8', 'D-9', 'D-10'] and translator.error == ''
    assert translator._rolling_failures == 0


@pytest.mark.asyncio
async def test_the_rolling_and_the_finished_translation_take_turns_while_one_is_behind():
    meeting, llm, translator, _ = make(count=20, lag=3, batch=2, rolling=True)
    await translator._step()                                                          # (the names)
    kinds = []
    for _ in range(6):
        before = len(llm.calls)
        await translator._step()
        kinds += ['R' if '三段滾動' in c['system'] else 'B' for c in llm.calls[before:]]
        add_segment(meeting)
    assert kinds == ['R', 'B', 'R', 'B', 'R', 'B']


@pytest.mark.asyncio
async def test_background_talk_is_never_shown_translated_as_a_draft_either():
    meeting, llm, translator, _ = make(count=10, lag=6, rolling=True)
    await drain(translator)
    assert translator.draft_text(9) == 'D-9'
    meeting.segments[8].runs = [Run(None, 'background', '原文9')]                       # the user marks segment 9 as background talk
    assert translator.draft_text(9) is None and translator.tr_state(9) == ''
    add_segment(meeting)
    await drain(translator)
    assert [item['i'] for item in rolling_calls(llm)[-1]['translate']] == [10, 11]    # its text is not translated
    assert translator.draft_text(9) is None
    meeting.segments[9].runs = [Run(None, 'background', '原文10甲'), Run(0, 'voice', '原文10乙')]       # part of segment 10 is
    await drain(translator)
    items = {item['i']: item for item in rolling_calls(llm)[-1]['translate']}
    assert items[10] == {'i': 10, 'source': '原文10乙', 'current': ''}                  # the old draft held the background text: not shown
    assert translator.draft_text(10) == 'D-10乙'


@pytest.mark.asyncio
async def test_once_the_meeting_has_ended_the_rest_is_finished_by_runs_or_from_the_drafts_and_nothing_new_is_rolled():
    meeting, llm, translator, _ = make(count=8, lag=6, rolling=True)
    await drain(translator)
    rolled = len(rolling_calls(llm))
    meeting.ended = True
    meeting.segments[6].runs = [Run(0, 'voice', '甲'), Run(1, 'voice', '乙')]
    await drain(translator)
    assert len(rolling_calls(llm)) == rolled and sorted(translator.results) == list(range(1, 9))
    asked = segments_of(run_calls(llm))
    assert 7 in asked and 6 not in asked and 8 not in asked                           # 7 has two speakers; 6 and 8 take their drafts
    assert translator.results[6].texts == ['D-6'] and translator.status == '翻譯完成'


@pytest.mark.asyncio
async def test_a_new_language_starts_without_the_old_drafts_and_the_file_marks_what_is_only_a_draft():
    meeting, llm, translator, _ = make(count=10, lag=6, rolling=True)
    await drain(translator)
    doc = translator.doc()['segments']
    assert doc['2'] == {'texts': ['T-2'], 'error': ''} and doc['9'] == {'texts': ['D-9'], 'error': '', 'draft': True}
    translator.start('Japanese', 6)
    await translator.stop()
    translator.running = True
    assert translator.drafts == {} and translator.results == {}
    await drain(translator)
    assert translator.draft_text(10) == 'D-10'


@pytest.mark.asyncio
async def test_a_segment_whose_finished_translation_failed_still_shows_its_draft_and_stays_listed_as_failed():
    meeting, llm, translator, _ = make(count=10, lag=6, rolling=True)
    await drain(translator)
    translator.results[9] = translation.Translated(9, ('原文9',), [], (), error='server down')
    assert translator.tr_state(9) == 'draft' and translator.failed() == [9]
    assert translator.doc()['segments']['9'] == {'texts': ['D-9'], 'error': 'server down', 'draft': True}
    translator.results[2] = translation.Translated(2, ('原文2',), [], (), error='server down')
    assert translator.tr_state(2) == 'error'                                          # nothing else to show for it


@pytest.mark.asyncio
async def test_the_old_way_is_still_there_when_the_rolling_translation_is_off():
    meeting, llm, translator, _ = make(count=10, lag=6, rolling=False)
    await drain(translator)
    assert rolling_calls(llm) == [] and translator.drafts == {} and sorted(translator.results) == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_the_failure_of_the_newest_segments_ends_with_the_meeting():
    meeting, llm, translator, _ = make(count=8, lag=3, rolling=True)
    llm.rolling_fail = 1
    await drain(translator)
    assert translator._rolling_failures == 1 and translator.error.startswith('翻譯失敗')
    meeting.ended = True                                                              # nothing is rolled any more
    await drain(translator)
    assert translator._rolling_failures == 0 and translator.error == '' and translator.status == '翻譯完成'


@pytest.mark.asyncio
async def test_the_failure_of_the_newest_segments_ends_when_they_turn_out_to_be_background_talk():
    meeting, llm, translator, _ = make(count=10, lag=6, rolling=True)
    llm.rolling_fail = 1
    await drain(translator)
    assert translator._rolling_failures == 1 and translator.error.startswith('翻譯失敗')
    for i in (8, 9, 10):
        meeting.segments[i - 1].runs = [Run(None, 'background', f'原文{i}')]            # there is nothing left to translate there
    await drain(translator)
    assert translator._rolling_failures == 0 and translator.error == '' and not translator._rolling_due()


@pytest.mark.asyncio
async def test_the_failure_of_the_newest_segments_ends_when_the_rolling_translation_is_switched_off():
    meeting, llm, translator, _ = make(count=10, lag=6, rolling=True)
    llm.rolling_fail = 1
    await drain(translator)
    translator.rolling = False
    await drain(translator)
    assert translator._rolling_failures == 0 and translator.error == ''


def test_the_error_of_the_newest_segments_and_the_error_of_the_others_do_not_hide_each_other():
    meeting, llm, translator, _ = make(count=4, rolling=True)
    translator._rolling_error = 'newest failed'
    assert translator.error == 'newest failed'
    translator.error = 'finished failed'
    assert translator.error == 'finished failed'                                       # the finished translation's goes first
    translator.error = ''
    assert translator.error == 'newest failed'                                         # and mending it does not mend the other
    translator._rolling_error = ''
    assert translator.error == ''
