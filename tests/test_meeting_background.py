"""Background talk in a meeting: grey on the screen, never sent to the translator, marked in the files, and the user can change it."""
import pytest

from tests.speaker_fakes import bases
from tests.test_meeting import fake_meeting
from tests.test_speaker_background import WHISPER, WhisperEmbedder
from tests.test_translation import FakeLLM, drain


SHORT = [0, 1, 0] * 10                  # about a minute and a half: enough for the groups and the usual level


async def played(tmp_path):
    """A meeting of two people with a whisper nobody knows in the middle, played to its end."""
    vectors = bases(3)
    embed = WhisperEmbedder({0: vectors[0], 1: vectors[1], 2: vectors[2]}, bases(4, seed=99)[3])
    meeting, session, play, count = fake_meeting(tmp_path, SHORT + [WHISPER, 0, 1] * 4, embed=embed, lo=1.6, hi=2.6)
    await play(0, count)
    await meeting.end()
    return meeting, session


def background_runs(meeting):
    return [(segment.index, k, run) for segment in meeting._segments for k, run in enumerate(meeting.speaker.runs(segment)) if run.background]


@pytest.mark.asyncio
async def test_background_talk_is_sent_to_the_page_as_such(tmp_path):
    meeting, session = await played(tmp_path)
    try:
        found = background_runs(meeting)
        assert found
        update = meeting.drain()
        assert update['labels']['background'] == '背景'
        sent = {row['i']: row for row in update['rows']}
        for index, k, run in found:
            assert sent[index]['runs'][k] == {'g': None, 't': run.text, 's': 'background'}
        assert meeting.speaker.status.endswith(f'，背景 {meeting.speaker.tracker.stats()["background"]} 句') or '背景' in meeting.speaker.status
    finally:
        await meeting.close()


@pytest.mark.asyncio
async def test_background_talk_never_reaches_the_translator(tmp_path):
    meeting, session = await played(tmp_path)
    try:
        found = background_runs(meeting)
        assert found
        translator = meeting.translator
        llm = FakeLLM()
        translator._call = llm
        translator.language, translator.lag, translator.running = 'English', 3, True
        await drain(translator, 200)
        asked = {item['i']: item for call in llm.calls if 'translate' in call['user'] for item in call['user']['translate']}
        shown = {segment.index: list(meeting.speaker.runs(segment)) for segment in meeting._segments}
        assert asked and not translator.failed()
        mixed = [i for i, runs in shown.items() if any(r.background for r in runs) and any(not r.background for r in runs)]
        fully = [i for i, runs in shown.items() if runs and all(r.background for r in runs)]
        assert mixed or fully
        for i in mixed:                                       # only the runs that are not background were put to the model
            assert i in asked and len(asked[i]['runs']) < len(shown[i])
            assert [r['text'] for r in asked[i]['runs']] == [r.text for r in meeting.speaker.translation_runs(meeting.segment(i))]
        for i in fully:                                       # a segment of nothing else: done, with no call at all
            assert i not in asked and translator.results[i].texts == [] and not translator.results[i].error
    finally:
        await meeting.close()


@pytest.mark.asyncio
async def test_the_files_mark_background_talk_and_translate_none_of_it(tmp_path):
    meeting, session = await played(tmp_path)
    try:
        found = background_runs(meeting)
        text = meeting.export_text()
        for index, k, run in found:
            assert f'[背景]: {run.text}' in text
        doc = meeting.doc()
        marked = [r for segment in doc['segments'] for r in segment['runs'] if r['background']]
        assert len(marked) == len(found) and all(r['speaker'] is None and r['name'] == '背景' for r in marked)
        assert not any(r['background'] for segment in doc['segments'] for r in segment['runs'] if r['speaker'])
    finally:
        await meeting.close()


@pytest.mark.asyncio
async def test_the_user_can_unmark_background_talk_and_the_translator_then_takes_it_up(tmp_path):
    meeting, session = await played(tmp_path)
    try:
        index, k, run = background_runs(meeting)[0]
        translator = meeting.translator
        llm = FakeLLM()
        translator._call = llm
        translator.language, translator.lag, translator.running = 'English', 3, True
        await drain(translator, 200)
        before = len(meeting.speaker.translation_runs(meeting.segment(index)))
        meeting.drain()
        with pytest.raises(ValueError, match='再點一次'):
            meeting.speaker.set_background(index, k, run.text + '？', False)         # the text moved on since the click
        changed = meeting.speaker.set_background(index, k, run.text, False)
        assert index in changed and not any(r.background for r in meeting.speaker.runs(meeting.segment(index)) if r.text == run.text)
        assert index in meeting.dirty                                                 # the page will be sent the row again
        assert len(meeting.speaker.translation_runs(meeting.segment(index))) > before   # it now belongs to the translation
        await drain(translator, 200)
        assert translator.tr_state(index) == 'ok'
        again = meeting.speaker.set_background(index, k, run.text, True)              # and back
        assert index in again and background_runs(meeting)
    finally:
        await meeting.close()


@pytest.mark.asyncio
async def test_a_translation_made_before_the_sentence_was_marked_background_is_never_shown_again(tmp_path):
    meeting, session = await played(tmp_path)
    try:
        translator = meeting.translator
        translator._call = FakeLLM()
        translator.language, translator.lag, translator.running = 'English', 3, True
        await drain(translator, 200)
        index, k, run = next((segment.index, 0, meeting.speaker.runs(segment)[0]) for segment in meeting._segments
                             if len(meeting.speaker.runs(segment)) == 1 and not meeting.speaker.runs(segment)[0].background)
        exported = lambda: [line for line in meeting.export_text().splitlines() if line.startswith(f'Voice#{index} ')]
        assert meeting._row(index)['tr']['st'] == 'ok' and exported() and translator.visible(index) is not None
        translator.running = False                                                   # the translation is stopped: it will not be redone
        meeting.speaker.set_background(index, k, run.text, True)                     # "this was background talk"
        assert 'tr' not in meeting._row(index)                                       # not on the screen ...
        assert not exported()                                                        # ... not in the exported text ...
        assert str(index) not in meeting.doc()['translation']['segments']            # ... not in the saved file ...
        assert translator.tr_state(index) == '' and translator.visible(index) is None   # ... and not used any more (not even as context)
        translator.running = True                                                    # when it runs again, the segment has nothing to translate
        await drain(translator, 200)
        assert translator.results[index].texts == [] and meeting._row(index)['tr']['runs'] == []
        meeting.speaker.set_background(index, k, run.text, False)                    # and the user takes it back: it is translated again
        await drain(translator, 200)
        assert meeting._row(index)['tr']['st'] == 'ok' and meeting._row(index)['tr']['runs']
    finally:
        await meeting.close()
