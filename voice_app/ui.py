"""One-page voice writing UI. The article stays in the browser."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from secrets import token_urlsafe
from types import SimpleNamespace
from uuid import uuid4

from nicegui import app, ui

from .api import ModelAPI
from .asr import local_asr
from .audio import devices
from .config import ROOT, Settings, resolve_path, shared_mode, source_allowed
from .meeting import Meeting
from .remote_audio import REMOTE_PAGES, RemoteAudioSource
from .session import Session, StartCancelled
from .speaker.palette import palette_css
from .speaker.embedder import load_embedder
from .ui_controls import build_source_controls
from .ui_settings import show_settings_dialog
from .ui_meeting import (build_meeting_panel, export_filename, open_speakers_dialog, read_translation_inputs)
from .ui_render import revisable_split


SESSIONS: set[Session] = set()
app.add_static_file(local_file=ROOT / 'voice_app' / 'opencc.full.js',
                    url_path='/opencc.full.js', max_cache_age=31536000)
app.add_static_file(local_file=ROOT / 'voice_app' / 'script_converter.js',
                    url_path='/script_converter.js', max_cache_age=0)
app.add_static_file(local_file=ROOT / 'voice_app' / 'writing.js',
                    url_path='/writing.js', max_cache_age=0)
app.add_static_file(local_file=ROOT / 'voice_app' / 'meeting.js',
                    url_path='/meeting.js', max_cache_age=0)
ui.add_head_html('<script src="/opencc.full.js?v=1.4.2"></script>'
                 '<script src="/script_converter.js?v=3"></script>'
                 '<script src="/writing.js?v=7"></script>'
                 '<script src="/meeting.js?v=1"></script>', shared=True)


def writing_commit_ready(session: Session) -> bool:
    return bool(not session.starting and not session.active and not session.processing and session.source
                and session.source.done.is_set() and not session.error)


def writing_in_play(state, session: Session) -> bool:
    """Whether the article is waiting for the text of the run in `session`. Not while that run is only being started (the models load,
    the audio opens): `session` is then still the last run, a finished one, and its text must neither be shown nor inserted again.
    (`state.loading` says nothing of this: a remote recording holds it for as long as it runs.)"""
    return bool(state.writing_pending and not state.writing_starting and not session.starting)


def cancel_start(state, session: Session) -> bool:
    """The user cancels a run that this page is still preparing (the models load, a meeting is made): when the preparation is done it
    opens no audio. False when the run is further along (the audio is opening, or it records): `Session.cancel()` ends that one."""
    if state.writing_starting and not session.starting:
        state.start_cancelled = True
        session.status = '已取消；模型載入完成後不會開始錄音'
        return True
    return False


async def cancel_run(state, session: Session) -> None:
    """The user cancels: a run still loading its models opens no audio (`cancel_start`), the article is given back, and the session stops
    its source. The session is cancelled at once and alongside: a browser that does not answer (it is gone, or slow) must not be able to
    keep the audio running; a failure to give the article back is still reported, after the session has been cancelled."""
    cancel_start(state, session)
    cancelling = asyncio.ensure_future(session.cancel())
    try:
        if state.writing_pending:
            state.writing_pending = False
            state.last_writing_render = None
            await ui.run_javascript('window.writing.rollback()')
    finally:
        await cancelling


async def launch_run(state, session: Session, kind: str, writing_context) -> None:
    """Load the models, make the meeting of a meeting-mode run, and start the session. Whatever goes wrong or is cancelled on the way
    is told on the page and leaves the last run as it was."""
    controls = state.controls
    state.loading = state.writing_starting = True
    state.start_cancelled = False
    meeting = None
    try:
        state.settings.validate()
        session.status = '載入本機 ASR 與 ForcedAligner，準備辨識'
        async with ModelAPI(state.settings) as api:
            await api.prepare_asr()
        if state.start_cancelled:
            raise StartCancelled('已取消，音訊來源未啟動。')
        if state.disconnected:
            session.status = '頁面已離線，未啟動音訊'
            if not state.meeting_mode:
                state.writing_pending = False
                await ui.run_javascript('window.writing.rollback()')
            return
        path = state.uploaded_path if kind == 'file' else None
        meeting = await state.prepare_meeting() if state.meeting_mode else None
        if state.start_cancelled:
            raise StartCancelled('已取消，音訊來源未啟動。')
        await session.start(state.settings, kind, controls.device.value, path,
                            record_audio=bool(controls.record_switch.value) and kind != 'file',
                            writing_context=writing_context, meeting=meeting)
        if path:
            session.owned_file = path
            state.uploaded_path = None
            controls.file_label.set_text('音訊檔已交由背景處理')
    except Exception as exc:
        if meeting is not None and session.meeting is not meeting:
            await meeting.discard()                # the audio never opened: no empty meeting is left behind
        if isinstance(exc, StartCancelled):        # the user cancelled: nothing went wrong
            session.status = '已取消，音訊來源未啟動'
        else:
            session.error = f'無法開始：{exc}'
            session.status = '請檢查模型與音訊來源'
        if not state.meeting_mode and state.writing_pending:       # (a cancel has already taken the article back)
            state.writing_pending = False
            await ui.run_javascript('window.writing.rollback()')
    finally:
        state.loading = state.writing_starting = state.start_cancelled = False


def writing_source_allowed(kind: str) -> bool:
    return kind in ('microphone', 'loopback', 'remote_microphone',
                    'remote_system', 'file') and source_allowed(kind)


@dataclass
class PageState:
    session: Session
    settings: Settings
    settings_error: str = ''
    uploaded_path: Path | None = None
    loading: bool = False
    writing_starting: bool = False     # this page is starting a run (models, audio): `session` still holds the last run until it has begun
    start_cancelled: bool = False      # the user cancelled that run while its models were still loading: it must not open the audio
    remote_start_id: str = ''
    remote_attempt: object = None      # the remote start that is waiting for the models and for the browser's confirmation
    disconnected: bool = False
    remote_token: str = ''
    controls: SimpleNamespace | None = None
    writing_mode: bool = True
    writing_pending: bool = False
    writing_history: list[dict] = field(default_factory=list)
    last_writing_render: tuple | None = None
    script_mode: str = 'original'
    meeting_mode: bool = False
    prepare_meeting: object = None
    meeting_widgets: SimpleNamespace | None = None


CSS = '''
body {background:#f3f5f7;color:#203044;font-family:"Segoe UI","Microsoft JhengHei",sans-serif;}
.nicegui-content {padding:0;}
.shell {max-width:1280px;margin:auto;padding:32px 28px 48px;width:100%;}
.q-card {border:1px solid #e0e6ed;border-radius:16px;box-shadow:0 3px 14px #273b5110;}
.eyebrow {font-size:11px;letter-spacing:2px;color:#60758c;font-weight:700;}
.muted {color:#738295;}
.metric {background:#eaf0f7;border-radius:8px;padding:6px 12px;color:#526c87;font-size:12px;}
.notice {white-space:pre-wrap;overflow-wrap:anywhere;}
.writing-toolbar {display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px;}
.writing-toolbar button,.writing-confirm button {border:1px solid #bdcbd8;background:white;border-radius:8px;padding:6px 12px;cursor:pointer;color:#285f91;}
.writing-file {color:#60758c;font-size:13px;margin-left:auto;}
.writing-editor,.writing-preview {width:100%;min-height:55vh;border:1px solid #cad5df;border-radius:10px;padding:20px;background:white;color:#203044;font:18px/1.85 "Segoe UI","Microsoft JhengHei",sans-serif;white-space:pre-wrap;overflow-wrap:anywhere;}
.writing-editor {resize:vertical;}
.writing-preview {overflow:auto;max-height:70vh;}
.writing-revisable {background:#fff1c8;}
.writing-confirm {border:1px solid #bdcbd8;border-radius:12px;padding:24px;}
.writing-confirm button {margin-right:8px;}
.m-head {display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:center;width:100%;}
.m-scroll {width:100%;height:62vh;overflow:auto;border:1px solid #cad5df;border-radius:10px;background:white;font:16px/1.7 "Segoe UI","Microsoft JhengHei",sans-serif;}
.m-row {display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid #eef2f6;}
.m-left,.m-right {padding:6px 14px;overflow-wrap:anywhere;min-width:0;}
.m-right {border-left:1px solid #eef2f6;background:#fbfcfd;}
.m-line {border-left:3px solid transparent;padding-left:6px;}
.m-n {color:#8a9bb0;font-size:12px;margin-right:8px;}
.m-who {font-weight:600;color:#285f91;}
.m-unknown .m-who {color:#8a9bb0;font-weight:400;}
.m-bg {color:#8a9bb0;}
.m-bg .m-who {color:#8a9bb0;font-weight:400;}
.m-act .m-who, .m-who.m-act {cursor:pointer;text-decoration:underline dotted;}
.m-rev {background:#fff1c8;}
.m-stale {opacity:.65;}
.m-error {color:#b3261e;}
#meeting-host:empty {min-height:40vh;border:1px dashed #cad5df;border-radius:10px;display:flex;align-items:center;justify-content:center;}
#meeting-host:empty::before {content:'按「開始轉錄」後，語音（標示說話者）與翻譯會顯示在這裡。';color:#738295;padding:16px;text-align:center;}
@media (max-width:900px) {.m-head,.m-row {grid-template-columns:1fr;} .m-right {border-left:none;border-top:1px dashed #eef2f6;}}
'''


def build_page():
    ui.add_css(CSS + palette_css())
    ui.colors(primary='#285f91', secondary='#5f7c96', accent='#c58d23',
              positive='#2e7d64')
    session = Session()
    SESSIONS.add(session)
    try:
        settings = Settings.load()
        settings_error = ''
    except Exception as exc:
        settings = Settings()
        settings_error = f'設定檔無法讀取，暫用預設值：{exc}'
    state = PageState(session, settings, settings_error, remote_token=token_urlsafe(24))
    REMOTE_PAGES[state.remote_token] = state
    client = ui.context.client

    async def cleanup():
        REMOTE_PAGES.pop(state.remote_token, None)
        await session.cancel()
        if session.meeting is not None:
            await session.meeting.close()
        if state.uploaded_path:
            state.uploaded_path.unlink(missing_ok=True)
        SESSIONS.discard(session)

    def on_disconnect():
        state.disconnected = True
        if not isinstance(session.source, RemoteAudioSource) and session.meeting is None:
            session.request_stop()          # a meeting goes on while the page reconnects; its file is saved as it runs

    def on_connect():
        state.disconnected = False

    def mark_background(event):
        """A click on a speaker label in the transcript: "this is background talk" / "this is not"."""
        meeting = session.meeting
        args = event.args[0] if isinstance(event.args, (list, tuple)) and event.args else event.args
        if meeting is None or not isinstance(args, dict):
            return
        try:
            meeting.speaker.set_background(int(args['i']), int(args['r']), str(args['t']), bool(args['to']))
        except (ValueError, KeyError, TypeError) as exc:
            ui.notify(str(exc) or '無法標記這一句。', type='warning')

    ui.on('meeting_background', mark_background)
    client.on_disconnect(on_disconnect)
    client.on_connect(on_connect)
    client.on_delete(cleanup)

    async def cancel_current():
        await cancel_run(state, session)

    async def change_script(mode: str):
        if state.meeting_mode:
            mode = 'original' if state.script_mode == mode else mode
            await ui.run_javascript(f'window.scriptConverter.setMode({json.dumps(mode)}); window.meeting.setScript({json.dumps(mode)})')
            state.script_mode = mode
            controls.simplified_button.set_text('✓ 轉簡體' if mode == 'simplified' else '轉簡體')
            controls.traditional_button.set_text('✓ 轉繁體' if mode == 'traditional' else '轉繁體')
            return
        if state.writing_pending:
            ui.notify('請先完成或取消目前語音，再轉換文章字形。', type='warning')
            return
        try:
            result = await ui.run_javascript(
                f'window.writing.setScript({json.dumps(mode)})', timeout=10)
            if result.get('status') == 'edited':
                ui.notify('轉換後文章已有新修改，不能用舊快照還原。', type='warning')
                return
            if result.get('status') == 'unavailable':
                ui.notify('書寫編輯區尚未準備好。', type='warning')
                return
            state.script_mode = result['mode']
            controls.simplified_button.set_text(
                '✓ 轉簡體' if state.script_mode == 'simplified' else '轉簡體')
            controls.traditional_button.set_text(
                '✓ 轉繁體' if state.script_mode == 'traditional' else '轉繁體')
        except Exception as exc:
            ui.notify(f'繁簡轉換失敗：{exc}', type='negative')

    async def prepare_meeting() -> Meeting:
        """A new meeting for the session about to start. Without the voiceprint model the meeting still runs, without speakers."""
        settings = deepcopy(state.settings)
        note = ''
        try:
            embed = await asyncio.to_thread(load_embedder, resolve_path(settings.meeting.speaker_model_path))
        except Exception as exc:
            embed, note = (lambda samples: None), str(exc)
        meeting = Meeting(settings, embed)
        if note:
            meeting.speaker.unavailable = note
            meeting.emit('speaker_unavailable', reason=note)
        return meeting

    state.prepare_meeting = prepare_meeting

    async def apply_meeting_mode():
        widgets = state.meeting_widgets
        article_card.set_visibility(not state.meeting_mode)
        widgets.card.set_visibility(state.meeting_mode)
        controls.meeting_button.set_text('✓ 會議模式' if state.meeting_mode else '會議模式')
        if state.meeting_mode:
            controls.record_switch.set_value(True)
            await ui.run_javascript('window.writing.setMode(false)')
        else:
            await ui.run_javascript(f'window.writing.setMode(true, {state.settings.context_chars})')

    async def toggle_meeting():
        if session.active or state.loading or state.writing_pending:
            ui.notify('請先完成或取消目前的轉錄，再切換會議模式。', type='warning')
            return
        state.meeting_mode = not state.meeting_mode
        await apply_meeting_mode()

    async def start_translation():
        meeting = session.meeting
        if meeting is None:
            ui.notify('請先按「開始轉錄」；會議開始後才能翻譯。', type='warning')
            return
        try:
            language, lag = read_translation_inputs(state.meeting_widgets)
            meeting.translator.start(language, lag)
        except ValueError as exc:
            ui.notify(str(exc), type='warning')
            return
        state.settings.meeting.target_language, state.settings.meeting.translation_lag = language, lag
        try:
            state.settings.save()                  # remembered for the next meeting
        except Exception as exc:
            ui.notify(f'翻譯已開始，但設定無法儲存：{exc}', type='warning')

    async def stop_translation():
        if session.meeting is not None:
            await session.meeting.translator.stop()

    def retry_translation():
        meeting = session.meeting
        count = meeting.translator.retry_failed() if meeting is not None else 0
        ui.notify(f'已重新排入 {count} 段翻譯失敗的段落。' if count else '沒有翻譯失敗的段落。', type='info' if count else 'warning')

    def export_meeting():
        meeting = session.meeting
        if meeting is None:
            ui.notify('還沒有會議內容可以匯出。', type='warning')
            return
        ui.download.content(meeting.export_text().encode('utf-8-sig'), export_filename(meeting), media_type='text/plain')

    def translation_settings():
        if session.active or state.loading:
            ui.notify('轉錄進行中不能修改模型設定。', type='warning')
            return
        show_settings_dialog(state, controls.stream_switch, tab='translator')

    async def start():
        if session.active or state.loading:
            return
        kind = controls.source.value
        if not writing_source_allowed(kind):
            ui.notify('此音訊來源無法使用。', type='negative')
            return
        if kind == 'file' and state.uploaded_path is None:
            ui.notify('請先選擇音訊檔。', type='warning')
            return
        if kind not in ('file', 'remote_microphone', 'remote_system') and not controls.device.value:
            ui.notify('請選擇錄音裝置。', type='warning')
            return
        writing_context = None
        if not state.meeting_mode:
            try:
                anchor = await ui.run_javascript(
                    f'window.writing.begin({state.settings.context_chars})', timeout=5)
                writing_context = (anchor['before'], anchor['after'])
                state.writing_pending = True
            except Exception as exc:
                ui.notify(f'無法固定文章插入位置：{exc}', type='negative')
                return
        await launch_run(state, session, kind, writing_context)

    async def refresh_devices():
        kind = controls.source.value
        if not writing_source_allowed(kind):
            controls.source.set_value('remote_microphone' if shared_mode() else 'microphone')
            return
        is_file = kind == 'file'
        is_remote = kind in ('remote_microphone', 'remote_system')
        controls.record_switch.set_visibility(not is_file)
        controls.refresh_button.set_visibility(not is_file and not is_remote)
        controls.device_note.set_visibility(not is_file)
        controls.file_label.set_visibility(is_file)
        controls.start_button.set_visibility(not is_remote)
        controls.remote_mic_start.set_visibility(kind == 'remote_microphone')
        controls.remote_system_start.set_visibility(kind == 'remote_system')
        if is_file or is_remote:
            controls.device.set_visibility(False)
            controls.upload.set_visibility(is_file)
            if is_remote:
                controls.device_note.set_text(
                    '請允許瀏覽器使用麥克風，待狀態顯示「錄音中」再開始說話。'
                    if kind == 'remote_microphone' else
                    '請選擇要分享的分頁或螢幕，勾選分享音訊。')
            return
        controls.device.set_visibility(True)
        controls.upload.set_visibility(False)
        try:
            options = await asyncio.to_thread(devices, kind)
            selected = controls.device.value
            controls.device.set_options(
                options, value=selected if selected in options else next(iter(options), None))
            controls.device_note.set_text(
                '錄製此電腦的播放裝置' if kind == 'loopback' else '錄製此電腦的麥克風')
        except Exception as exc:
            controls.device_note.set_text(f'無法列出裝置：{exc}')
            controls.device.set_options({}, value=None)

    async def on_upload(event):
        if session.active or state.loading:
            ui.notify('目前工作尚未結束。', type='warning')
            return
        directory = ROOT / 'data' / 'uploads'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (uuid4().hex + Path(event.file.name).suffix)
        try:
            await event.file.save(path)
            if state.uploaded_path:
                state.uploaded_path.unlink(missing_ok=True)
            state.uploaded_path = path
            controls.file_label.set_text(f'已選擇：{event.file.name}')
        except Exception as exc:
            path.unlink(missing_ok=True)
            ui.notify(f'檔案讀取失敗：{exc}', type='negative')

    def download_json():
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = (session.recording_path.with_suffix('.json').name
                    if session.recording_path else f'voice-writing-{stamp}.json')
        ui.download.content(session.export_json().encode('utf-8-sig'),
                            filename, media_type='application/json')

    def download_recording():
        if not session.recording_path or not session.source or not session.source.done.is_set():
            ui.notify('錄音尚未完成寫入。', type='warning')
            return
        ui.download.file(session.recording_path, filename=session.recording_path.name,
                         media_type='audio/wav')

    with ui.column().classes('shell gap-6'):
        controls = build_source_controls(
            state, start, refresh_devices, on_upload, change_script, cancel_current, toggle_meeting)
        state.controls = controls
        state.meeting_widgets = build_meeting_panel(
            state, start_translation, stop_translation, lambda: open_speakers_dialog(state), export_meeting, translation_settings, retry_translation)
        state.meeting_widgets.card.set_visibility(False)
        with ui.card().classes('w-full p-5 gap-3') as article_card:
            ui.label('文章').classes('text-lg font-semibold')
            ui.label('將游標放在插入位置；音訊處理時文章暫停編輯，黃色為最新兩段可修訂文字。').classes('muted text-sm')
            ui.html('<div id="writing-host"></div>', sanitize=False).classes('w-full')
            with ui.row().classes('w-full gap-2'):
                history_download = ui.button('匯出本次語音紀錄 JSON',
                                             on_click=download_json).props('outline')
                recording_download = ui.button('下載錄音 WAV',
                                               on_click=download_recording).props('outline')
            with ui.expansion('本頁語音插入紀錄', icon='history').classes('w-full'):
                writing_history_view = ui.column().classes('w-full gap-2')
        ui.label('文章只保留於本頁；關閉前請儲存 TXT。語音文字會交給設定的校稿服務。').classes('muted text-xs')

    async def initialize_writing():
        await ui.run_javascript(
            f'window.writing.setMode(true, {state.settings.context_chars})')

    def update_meeting_panel(meeting, busy):
        widgets = state.meeting_widgets
        has = meeting is not None
        translating = has and meeting.translator.running
        widgets.speakers_button.set_enabled(has)
        widgets.export_button.set_enabled(has)
        widgets.start_button.set_enabled(has and not translating)
        widgets.stop_button.set_enabled(translating)
        widgets.language.set_enabled(not translating)
        widgets.lag.set_enabled(not translating)
        failed = len(meeting.translator.failed()) if has else 0
        widgets.retry_button.set_visibility(failed > 0)
        widgets.retry_button.set_text(f'重試失敗段落（{failed}）')
        widgets.retry_button.set_enabled(translating)
        if not has:
            widgets.speaker_status.set_text('按「開始轉錄」開始會議；會議內容會自動存到 data/meetings/。')
            widgets.translation_status.set_text('')
            widgets.notice.set_visibility(False)
            return
        update = meeting.drain()
        if update:
            ui.run_javascript('window.meeting.update(' + json.dumps(update, ensure_ascii=False) + ')')
        widgets.speaker_status.set_text(meeting.speaker.status)
        widgets.translation_status.set_text(
            meeting.translator.status if meeting.translator.running or meeting.translator.results else '尚未開始翻譯')
        problems = '\n'.join(filter(None, (meeting.save_error, meeting.translator.error if meeting.translator.running else '',
                                                meeting.translator.names_error if meeting.translator.running else '')))
        widgets.notice.set_text(problems)
        widgets.notice.set_visibility(bool(problems))

    def tick():
        busy = session.active or state.loading
        for control in (controls.source, controls.device, controls.refresh_button,
                        controls.upload, controls.settings_button,
                        controls.stream_switch, controls.record_switch,
                        controls.start_button, controls.remote_mic_start,
                        controls.remote_system_start):
            if control is not None:
                control.set_enabled(not busy)
        controls.simplified_button.set_enabled(not state.writing_pending)
        controls.traditional_button.set_enabled(not state.writing_pending)
        controls.stop_button.set_enabled(
            session.processing and session.source is not None
            and not session.source.stop.is_set())
        controls.cancel_button.set_enabled(session.active or session.starting or state.writing_starting or state.writing_pending)
        controls.retry_button.set_visibility(
            session.failed_chunk is not None and not session.processing)
        controls.skip_button.set_visibility(session.can_skip_failed_chunk())
        controls.skip_button.set_text(
            '略過此段並繼續' if session.kind == 'file' else '略過此段並處理已收音訊')
        stride = state.settings.window_seconds - state.settings.overlap_seconds
        controls.summary.set_text(
            f'視窗 {state.settings.window_seconds:g} 秒 · 左側參考 {state.settings.overlap_seconds:g} 秒 · 每 {stride:g} 秒產生新辨識')
        controls.status.set_text(session.status)
        latest_elapsed = (f' · 最新處理 {session.segments[-1].elapsed:.1f} 秒'
                          if session.segments else '')
        controls.metrics.set_text(
            f'{len(session.segments)} 段完成 · '
            f'{session.source.queue.qsize() if session.source else 0} 段待處理'
            f'{latest_elapsed}')
        controls.level.set_value(session.source.level if session.source else 0)
        error = '\n'.join(filter(None, (session.error, session.recording_save_error)))
        controls.error.set_text(error)
        controls.error.set_visibility(bool(error))
        controls.warning.set_text(session.warning)
        controls.warning.set_visibility(bool(session.warning))
        gap_notice = (f'{len(session.gaps)} 段音訊可能缺漏；請核對本頁語音插入紀錄。'
                      if session.gaps else '')
        controls.gap_notice.set_text(gap_notice)
        controls.gap_notice.set_visibility(bool(gap_notice))
        if session.recording_path:
            label = {'recording': '錄音中', 'complete': '完整',
                     'incomplete': '未完成'}[session.recording_state]
            controls.recording_note.set_text(
                f'錄音存檔：{session.recording_path.name} · {label}；同名 JSON 自動保存')
        else:
            controls.recording_note.set_text('')
        history_download.set_visibility(bool(session.segments or session.gaps))
        recording_download.set_visibility(bool(session.recording_path))
        recording_download.set_enabled(bool(
            session.source and session.source.done.is_set()
            and session.recording_path and session.recording_path.is_file()))
        meeting = session.meeting
        controls.meeting_button.set_enabled(not busy and not state.writing_pending)
        if state.meeting_mode:
            update_meeting_panel(meeting, busy)
        if not writing_in_play(state, session):
            return
        locked, revisable = revisable_split(
            session.corrected, session.segments, session.rolling_start)
        snapshot = (locked, revisable)
        if snapshot != state.last_writing_render:
            state.last_writing_render = snapshot
            ui.run_javascript('window.writing.render(' + ','.join(
                json.dumps(value, ensure_ascii=False) for value in snapshot) + ')')
        if writing_commit_ready(session):
            state.writing_pending = False
            state.last_writing_render = None
            writing_text = session.writing_text()
            ui.run_javascript(
                f'window.writing.commit({json.dumps(writing_text, ensure_ascii=False)})')
            record = json.loads(session.export_json())
            state.writing_history.append(record)
            if session.gaps:
                ui.notify('部分音訊未能安全定稿；已插入成功定稿的文字，請核對紀錄。',
                          type='warning')
            if session.source.error:
                ui.notify('音訊來源未完整結束；請核對是否遺漏。', type='warning')
            with writing_history_view:
                with ui.expansion(
                    f'第 {len(state.writing_history)} 次語音 · {record["created"]}',
                    icon='mic').classes('w-full'):
                    ui.label('ASR 原稿：' + record['raw']).classes('notice')
                    ui.label('校稿結果：' + record['corrected']).classes('notice')
                    if record['warning']:
                        ui.label('提示：' + record['warning']).classes('notice text-amber-800')
                    for gap in record['gaps']:
                        ui.label(
                            f'可能缺漏 {gap["possible_start"]:.1f}–{gap["possible_end"]:.1f} 秒 · '
                            f'{gap["raw_source"]} ASR：{gap["raw"] or "（空白）"} · '
                            f'原因：{gap["error"]}').classes('notice text-amber-800')
                    for revision in record['revisions']:
                        ui.label(
                            f'回改 {revision["index"]}：'
                            + ' / '.join(revision['before']) + ' → '
                            + ' / '.join(revision['after'])).classes('notice text-sm')

    tick()
    ui.timer(0.15, tick)
    ui.timer(0.1, refresh_devices, once=True)
    ui.timer(0.1, initialize_writing, once=True)


async def shutdown():
    try:
        await asyncio.gather(*(session.cancel() for session in list(SESSIONS)))
    finally:
        await local_asr.close()


app.on_shutdown(shutdown)
