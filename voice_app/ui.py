"""One-page voice writing UI. The article stays in the browser."""

import asyncio
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
from .config import ROOT, Settings, shared_mode, source_allowed
from .remote_audio import REMOTE_PAGES, RemoteAudioSource
from .session import Session
from .ui_controls import build_source_controls
from .ui_render import revisable_split


SESSIONS: set[Session] = set()
app.add_static_file(local_file=ROOT / 'voice_app' / 'opencc.full.js',
                    url_path='/opencc.full.js', max_cache_age=31536000)
app.add_static_file(local_file=ROOT / 'voice_app' / 'script_converter.js',
                    url_path='/script_converter.js', max_cache_age=0)
app.add_static_file(local_file=ROOT / 'voice_app' / 'writing.js',
                    url_path='/writing.js', max_cache_age=0)
ui.add_head_html('<script src="/opencc.full.js?v=1.4.2"></script>'
                 '<script src="/script_converter.js?v=3"></script>'
                 '<script src="/writing.js?v=7"></script>', shared=True)


def writing_commit_ready(session: Session) -> bool:
    return bool(not session.active and not session.processing and session.source
                and session.source.done.is_set() and not session.error)


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
    remote_start_id: str = ''
    disconnected: bool = False
    remote_token: str = ''
    controls: SimpleNamespace | None = None
    writing_mode: bool = True
    writing_pending: bool = False
    writing_history: list[dict] = field(default_factory=list)
    last_writing_render: tuple | None = None
    script_mode: str = 'original'


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
'''


def build_page():
    ui.add_css(CSS)
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
        if state.uploaded_path:
            state.uploaded_path.unlink(missing_ok=True)
        SESSIONS.discard(session)

    def on_disconnect():
        state.disconnected = True
        if not isinstance(session.source, RemoteAudioSource):
            session.request_stop()

    def on_connect():
        state.disconnected = False

    client.on_disconnect(on_disconnect)
    client.on_connect(on_connect)
    client.on_delete(cleanup)

    async def cancel_current():
        if state.writing_pending:
            state.writing_pending = False
            state.last_writing_render = None
            await ui.run_javascript('window.writing.rollback()')
        await session.cancel()

    async def change_script(mode: str):
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
        try:
            anchor = await ui.run_javascript(
                f'window.writing.begin({state.settings.context_chars})', timeout=5)
            writing_context = (anchor['before'], anchor['after'])
            state.writing_pending = True
        except Exception as exc:
            ui.notify(f'無法固定文章插入位置：{exc}', type='negative')
            return
        state.loading = True
        try:
            state.settings.validate()
            session.status = '載入本機 ASR 與 ForcedAligner，準備辨識'
            async with ModelAPI(state.settings) as api:
                await api.prepare_asr()
            if state.disconnected:
                session.status = '頁面已離線，未啟動音訊'
                state.writing_pending = False
                await ui.run_javascript('window.writing.rollback()')
                return
            path = state.uploaded_path if kind == 'file' else None
            session.start(state.settings, kind, controls.device.value, path,
                          record_audio=bool(controls.record_switch.value) and kind != 'file',
                          writing_context=writing_context)
            if path:
                session.owned_file = path
                state.uploaded_path = None
                controls.file_label.set_text('音訊檔已交由背景處理')
        except Exception as exc:
            session.error = f'無法開始：{exc}'
            session.status = '請檢查模型與音訊來源'
            state.writing_pending = False
            await ui.run_javascript('window.writing.rollback()')
        finally:
            state.loading = False

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
            state, start, refresh_devices, on_upload, change_script, cancel_current)
        state.controls = controls
        with ui.card().classes('w-full p-5 gap-3'):
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
        controls.cancel_button.set_enabled(session.active or state.writing_pending)
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
        if not state.writing_pending:
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
