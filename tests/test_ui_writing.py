"""The article page and a run that is only being started: until the audio is really coming in, `Session` still holds the LAST run, a
finished one, and the page must not take it for the new run (it would insert the last text a second time and stop showing the new one)."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import voice_app.session as session_module
import voice_app.ui as page
from voice_app.config import Settings
from voice_app.session import Session
from voice_app.ui import CLOSE_REPLAY, cancel_run, cancel_start, close_replay, launch_run, show_meeting_mode
from tests.test_meeting_robustness import scripted_run


@pytest.mark.parametrize('pending, page_starting, starting, expected', [
    (False, False, False, False),         # nothing waits for the article
    (True, False, False, True),           # a run is going on (or has just finished) and the article waits for it
    (True, True, False, False),           # the page is loading the models / opening the audio: that is not the run yet
    (True, False, True, False),
    (True, True, True, False),
])
def test_the_article_only_waits_for_a_run_that_has_started(pending, page_starting, starting, expected):
    state = page.PageState(SimpleNamespace(), Settings(), writing_pending=pending, writing_starting=page_starting)
    assert page.writing_in_play(state, SimpleNamespace(starting=starting)) is expected


def test_a_remote_recording_that_holds_the_loading_flag_is_still_followed():
    state = page.PageState(SimpleNamespace(), Settings(), writing_pending=True, loading=True)    # (the socket handler holds it while it records)
    assert page.writing_in_play(state, SimpleNamespace(starting=False))


@pytest.mark.asyncio
async def test_a_second_run_whose_audio_takes_a_while_to_open_does_not_insert_the_first_run_again(monkeypatch):
    session = Session()
    state = page.PageState(session, Settings())
    scripted_run(monkeypatch, [0, 1] * 4, seed=4)
    await session.start(Settings(), 'file', path=Path('x.wav'))
    state.writing_pending = True                                      # what the page does when the run starts
    await asyncio.wait_for(session.task, 20)
    assert page.writing_in_play(state, session) and page.writing_commit_ready(session)      # the first run is over: insert its text
    assert session.writing_text()
    state.writing_pending = False                                     # ... which the page did

    scripted_run(monkeypatch, [1, 0] * 4, seed=9)
    scripted_source = session_module.AudioSource
    gate = asyncio.Event()

    def gated(*args, **kwargs):                                       # the second run's audio opens when the test says
        source = scripted_source(*args, **kwargs)

        async def wait_open():
            await gate.wait()
        source.wait_open = wait_open
        return source

    monkeypatch.setattr(session_module, 'AudioSource', gated)
    state.writing_pending, state.loading, state.writing_starting = True, True, True    # the page starts the second run: the article is locked
    starting = asyncio.create_task(session.start(Settings(), 'file', path=Path('y.wav')))
    for _ in range(50):
        await asyncio.sleep(0)
        if session.starting:
            break
    assert session.starting and not session.active
    assert session.source.done.is_set() and session.error == ''       # to the old check this is a finished run waiting to be inserted ...
    assert not page.writing_in_play(state, session)                   # ... and the page must not see it so
    assert not page.writing_commit_ready(session)

    gate.set()
    await starting
    state.loading = state.writing_starting = False
    assert not session.starting and page.writing_in_play(state, session)      # now the page follows the new run
    await asyncio.wait_for(session.task, 20)
    assert page.writing_commit_ready(session) and session.writing_text()      # and inserts its text once it is over


class SlowModels:
    """The models, loading for as long as the test wants (`release()` ends it)."""
    release_event = None

    def __init__(self, settings):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def prepare_asr(self):
        await SlowModels.release_event.wait()


def rollbacks(codes):
    return [code for code in codes if code != CLOSE_REPLAY]


def starting_page(monkeypatch):
    """A page about to start a run: the article is locked (`writing_pending`), the audio source is a scripted one that is counted."""
    scripted_run(monkeypatch, [0, 1] * 4, seed=4)
    opened = []
    scripted_source = session_module.AudioSource

    def counted(*args, **kwargs):
        opened.append(args)
        return scripted_source(*args, **kwargs)

    monkeypatch.setattr(session_module, 'AudioSource', counted)
    SlowModels.release_event = asyncio.Event()
    monkeypatch.setattr(page, 'ModelAPI', SlowModels)
    rolled_back = []

    async def run_javascript(code, **kwargs):
        rolled_back.append(code)

    monkeypatch.setattr(page.ui, 'run_javascript', run_javascript)
    session = Session()
    state = page.PageState(session, Settings(), writing_pending=True, controls=SimpleNamespace(
        device=SimpleNamespace(value='dev'), record_switch=SimpleNamespace(value=False), file_label=SimpleNamespace(set_text=lambda text: None)))
    return state, session, opened, rolled_back


async def loading(state, session):
    run = asyncio.create_task(launch_run(state, session, 'file', None))
    for _ in range(100):
        await asyncio.sleep(0)
        if state.writing_starting:
            break
    assert state.writing_starting and state.loading and not session.starting
    return run


@pytest.mark.asyncio
async def test_without_a_cancel_the_run_starts_once_the_models_are_loaded(monkeypatch):
    state, session, opened, _ = starting_page(monkeypatch)
    run = await loading(state, session)
    assert not opened and not session.active
    SlowModels.release_event.set()
    await run
    assert len(opened) == 1 and session.source is not None and not state.loading and not state.writing_starting
    await asyncio.wait_for(session.task, 20)


@pytest.mark.asyncio
async def test_cancelling_while_the_models_load_starts_nothing_when_they_are_ready(monkeypatch):
    state, session, opened, _ = starting_page(monkeypatch)
    run = await loading(state, session)
    assert cancel_start(state, session)                               # the cancel button of a run that is still loading its models
    state.writing_pending = False                                     # (the page gives the article back)
    await session.cancel()
    assert not session.active and not opened
    SlowModels.release_event.set()                                    # only now are the models ready
    await run
    assert not opened and session.source is None and not session.active and session.task is None   # no audio was opened, nothing records
    assert session.error == '' and session.status == '已取消，音訊來源未啟動'
    assert not state.loading and not state.writing_starting and not state.writing_pending


@pytest.mark.asyncio
async def test_a_cancel_after_the_audio_is_opening_is_left_to_the_session(monkeypatch):
    state, session, opened, _ = starting_page(monkeypatch)
    SlowModels.release_event.set()
    scripted_source = session_module.AudioSource

    def gated(*args, **kwargs):
        source = scripted_source(*args, **kwargs)

        async def wait_open():                                        # a device that never opens: only a cancel ends the wait
            while not source.cancelled.is_set():
                await asyncio.sleep(.01)
        source.wait_open = wait_open
        return source

    monkeypatch.setattr(session_module, 'AudioSource', gated)
    run = asyncio.create_task(launch_run(state, session, 'file', None))
    for _ in range(100):
        await asyncio.sleep(0)
        if session.starting:
            break
    assert session.starting
    assert not cancel_start(state, session) and not state.start_cancelled      # `Session.cancel()` ends this one
    await session.cancel()
    await run
    assert not session.active and session.status == '已取消，音訊來源未啟動'


def cancel_aware(monkeypatch, *, endless=False):
    """The scripted source, as a device that is slow to open (only a cancel ends the wait) or, with `endless`, a recording that goes on
    until it is stopped or cancelled."""
    scripted_source = session_module.AudioSource

    def build(*args, **kwargs):
        source = scripted_source(*args, **kwargs)

        async def wait_open():
            while not endless and not source.cancelled.is_set():
                await asyncio.sleep(.01)

        async def next_chunk():
            while not (source.stop.is_set() or source.cancelled.is_set()):
                await asyncio.sleep(.01)
            source.done.set()
            return None
        source.wait_open = wait_open
        if endless:
            source.next_chunk = next_chunk
        return source

    monkeypatch.setattr(session_module, 'AudioSource', build)


@pytest.mark.asyncio
async def test_a_cancel_while_the_audio_opens_reaches_the_source_even_when_the_article_cannot_be_given_back(monkeypatch):
    state, session, opened, _ = starting_page(monkeypatch)
    SlowModels.release_event.set()
    cancel_aware(monkeypatch)
    seen_by_the_page = []

    async def stuck_rollback(code, **kwargs):                         # the browser does not answer (it is gone, or slow)
        if code == CLOSE_REPLAY:
            return
        for _ in range(100):
            if opened and opened[0] is not None and session._start_cancelled:
                seen_by_the_page.append('the source was told to stop')
                break
            await asyncio.sleep(.01)
        raise TimeoutError('the browser did not answer')

    monkeypatch.setattr(page.ui, 'run_javascript', stuck_rollback)
    run = asyncio.create_task(launch_run(state, session, 'file', None))
    for _ in range(100):
        await asyncio.sleep(0)
        if session.starting:
            break
    assert session.starting
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(cancel_run(state, session), 10)      # (the page's handler: the failure is still reported)
    await asyncio.wait_for(run, 10)
    assert seen_by_the_page                                           # the source was told while the page was still waiting for the browser
    assert not session.active and session.source is None and session.status == '已取消，音訊來源未啟動'
    assert not state.writing_pending and not state.loading


@pytest.mark.asyncio
async def test_a_cancel_of_a_recording_stops_it_even_when_the_article_cannot_be_given_back(monkeypatch):
    state, session, opened, _ = starting_page(monkeypatch)
    SlowModels.release_event.set()
    cancel_aware(monkeypatch, endless=True)

    async def failing_rollback(code, **kwargs):
        if code == CLOSE_REPLAY:
            return
        raise ConnectionError('the page went away')

    monkeypatch.setattr(page.ui, 'run_javascript', failing_rollback)
    await launch_run(state, session, 'file', None)
    assert session.active and not session.task.done()
    with pytest.raises(ConnectionError):
        await asyncio.wait_for(cancel_run(state, session), 10)
    assert not session.active and session.task.done()


@pytest.mark.asyncio
async def test_a_cancel_gives_the_article_back_once(monkeypatch):
    state, session, opened, rolled_back = starting_page(monkeypatch)
    SlowModels.release_event.set()
    cancel_aware(monkeypatch, endless=True)
    await launch_run(state, session, 'file', None)
    await cancel_run(state, session)
    assert rollbacks(rolled_back) == ['window.writing.rollback()'] and not state.writing_pending and not session.active
    await cancel_run(state, session)                                  # a second press has nothing left to give back
    assert rollbacks(rolled_back) == ['window.writing.rollback()']


def counting(monkeypatch, *, close_fails=False):
    """Endless, cancel-aware sources that count how often each is closed; and a meeting that counts how often it is ended."""
    cancel_aware(monkeypatch, endless=True)
    build = session_module.AudioSource
    closed, sources = [], []

    def counted(*args, **kwargs):
        source = build(*args, **kwargs)

        async def close():
            closed.append(source)
            await asyncio.sleep(.05)                                   # (closing takes a moment: a second cancel arrives meanwhile)
            if close_fails:
                raise RuntimeError('the device would not close')
        source.close = close
        sources.append(source)
        return source

    monkeypatch.setattr(session_module, 'AudioSource', counted)
    ended = []

    async def end():
        ended.append(1)
        await asyncio.sleep(.05)

    return closed, ended, SimpleNamespace(end=end, emit=lambda *a, **k: None, retire=lambda: None), sources


async def running_page(monkeypatch, **kwargs):
    state, session, opened, rolled_back = starting_page(monkeypatch)
    SlowModels.release_event.set()
    closed, ended, meeting, sources = counting(monkeypatch, **kwargs)
    await launch_run(state, session, 'file', None)
    session.meeting = meeting
    assert session.active
    return state, session, closed, ended, rolled_back, sources


@pytest.mark.asyncio
async def test_two_cancels_at_once_close_the_run_once(monkeypatch):
    state, session, closed, ended, _, _ = await running_page(monkeypatch)
    await asyncio.gather(session.cancel(), session.cancel(), session.cancel())      # a double click, the page going away, the shutdown
    assert len(closed) == 1 and len(ended) == 1 and not session.active


@pytest.mark.asyncio
async def test_two_presses_of_the_cancel_button_at_once_close_the_run_once_and_give_the_article_back_once(monkeypatch):
    state, session, closed, ended, rolled_back, _ = await running_page(monkeypatch)
    await asyncio.gather(cancel_run(state, session), cancel_run(state, session))
    assert len(closed) == 1 and len(ended) == 1 and rollbacks(rolled_back) == ['window.writing.rollback()'] and not session.active


@pytest.mark.asyncio
async def test_a_failure_of_the_cancel_reaches_every_waiter_and_a_later_cancel_runs_again(monkeypatch):
    state, session, closed, ended, _, sources = await running_page(monkeypatch, close_fails=True)
    results = await asyncio.gather(session.cancel(), session.cancel(), return_exceptions=True)
    assert [type(r) for r in results] == [RuntimeError, RuntimeError] and len(closed) == 1
    with pytest.raises(RuntimeError):
        await session.cancel()                                                      # not at once: it is a cancel of its own
    assert len(closed) == 2 and len(sources) == 1                                   # (the same source: the session has not started another)


@pytest.mark.asyncio
async def test_a_cancel_after_another_has_finished_cancels_the_next_run(monkeypatch):
    state, session, closed, ended, _, sources = await running_page(monkeypatch)
    await session.cancel()
    state.writing_pending = True
    await launch_run(state, session, 'file', None)                                 # the next run
    assert session.active and len(sources) == 2
    await session.cancel()
    assert closed == sources and not session.active


@pytest.mark.asyncio
async def test_a_run_that_starts_closes_the_replay_window_first_so_its_sound_is_not_recorded_into_it(monkeypatch):
    state, session, opened, told = starting_page(monkeypatch)
    SlowModels.release_event.set()
    await launch_run(state, session, 'file', None)
    assert told[0] == CLOSE_REPLAY and not any(code == CLOSE_REPLAY for code in told[1:])        # before the audio is opened, once
    await asyncio.wait_for(session.task, 20)


@pytest.mark.asyncio
async def test_a_page_that_does_not_answer_the_close_does_not_stop_a_run_from_starting(monkeypatch):
    state, session, opened, told = starting_page(monkeypatch)
    SlowModels.release_event.set()

    async def silent(code, **kwargs):
        raise TimeoutError('the browser did not answer')

    monkeypatch.setattr(page.ui, 'run_javascript', silent)
    await close_replay()                                                                          # (harmless by itself)
    await launch_run(state, session, 'file', None)
    assert len(opened) == 1 and session.error == ''                                              # the run started all the same
    await asyncio.wait_for(session.task, 20)


def meeting_mode_page(monkeypatch, *, on):
    told = []

    def run_javascript(code, **kwargs):
        told.append(code)

        class Sent:
            def __await__(self):
                yield from []
        return Sent()

    monkeypatch.setattr(page.ui, 'run_javascript', run_javascript)
    shown = {}
    widget = lambda name: SimpleNamespace(set_visibility=lambda value: shown.__setitem__(name, value), set_text=lambda text: None,
                                          set_value=lambda value: shown.__setitem__(name + ' value', value))
    controls = SimpleNamespace(meeting_button=widget('button'), record_switch=widget('record'))
    state = page.PageState(SimpleNamespace(), Settings(), meeting_mode=on, meeting_widgets=SimpleNamespace(card=widget('meeting')))
    return state, controls, widget('article'), told, shown


@pytest.mark.asyncio
async def test_leaving_the_meeting_mode_closes_the_replay_window_and_entering_it_does_not(monkeypatch):
    state, controls, article, told, shown = meeting_mode_page(monkeypatch, on=False)           # the mode has just been switched off
    await show_meeting_mode(state, controls, article)
    assert CLOSE_REPLAY in told and any('setMode(true' in code for code in told)
    assert shown['article'] is True and shown['meeting'] is False
    state, controls, article, told, shown = meeting_mode_page(monkeypatch, on=True)
    await show_meeting_mode(state, controls, article)
    assert CLOSE_REPLAY not in told and 'window.writing.setMode(false)' in told
    assert shown['article'] is False and shown['meeting'] is True and shown['record value'] is True
