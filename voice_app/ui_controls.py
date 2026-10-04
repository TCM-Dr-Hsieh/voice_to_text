"""Source selection and live recording controls."""
from types import SimpleNamespace

from nicegui import ui

from .config import shared_mode
from .ui_settings import show_settings_dialog

def build_source_controls(state, start, refresh_devices, on_upload, change_script, cancel):
    session = state.session
    with ui.row().classes('w-full items-center justify-between'):
        with ui.column().classes('gap-1'):
            ui.label('LOCAL VOICE WORKSPACE').classes('eyebrow')
            ui.label('語音書寫').classes('text-3xl font-semibold')
            ui.label('錄音或加入音訊檔，在文章游標位置寫入校稿文字。').classes('muted')
        settings_button = None
        if not shared_mode():
            settings_button = ui.button('模型設定', icon='tune',
                                        on_click=lambda: show_settings_dialog(state, stream_switch)).props('outline')
    if state.settings_error:
        ui.label(state.settings_error).classes('text-red-700 notice')
    with ui.card().classes('w-full p-5 gap-4'):
        with ui.row().classes('w-full items-center gap-4'):
            sources = {'remote_microphone': '遠端麥克風', 'remote_system': '遠端電腦音訊',
                       'file': '音訊檔'}
            if not shared_mode():
                sources = {'microphone': '麥克風', 'loopback': '電腦播放聲音', **sources}
            source = ui.toggle(sources, value='remote_microphone' if shared_mode() else 'microphone',
                               on_change=refresh_devices)
            async def to_simplified():
                await change_script('simplified')
            async def to_traditional():
                await change_script('traditional')
            simplified_button = ui.button('轉簡體', on_click=to_simplified).props('outline').tooltip('再按一次可還原轉換前原文；若已編輯則保護新內容')
            traditional_button = ui.button('轉繁體', on_click=to_traditional).props('outline').tooltip('再按一次可還原轉換前原文；若已編輯則保護新內容')
            stream_switch = ui.switch('校稿串流預覽', value=state.settings.streaming)
            record_switch = ui.switch('儲存錄音', value=False).tooltip('錄音來源從開始擷取到停止，保存連續 WAV 與同名完整紀錄 JSON')
            def change_stream(e):
                if not session.active:
                    state.settings.streaming = bool(e.value)
            stream_switch.on_value_change(change_stream)
        with ui.row().classes('w-full items-center'):
            device = ui.select({}, label='錄音裝置').classes('grow min-w-64')
            refresh_button = ui.button(icon='refresh', on_click=refresh_devices).props('flat round').tooltip('重新整理裝置')
        device_note = ui.label('錄製此電腦的麥克風').classes('muted text-xs')
        upload = ui.upload(label='選擇音訊檔 · WAV / MP3 / M4A / FLAC / OGG', auto_upload=True,
                           max_files=1, max_file_size=1024 ** 3, on_upload=on_upload,
                           on_rejected=lambda: ui.notify('檔案超過 1 GB 或格式不符。', type='warning')).props('accept="audio/*,.m4a,.flac,.ogg,.wav,.mp3"').classes('w-full')
        upload.set_visibility(False)
        file_label = ui.label('').classes('muted text-sm')
        with ui.row().classes('w-full items-center gap-3'):
            start_button = ui.button('開始轉錄', icon='mic', on_click=start)
            remote_mic_start = ui.button('開始轉錄', icon='mic')
            remote_mic_start.on('click', js_handler=f'() => window.remoteAudio.start("remote_microphone", "{state.remote_token}")')
            remote_mic_start.set_visibility(False)
            remote_system_start = ui.button('開始轉錄', icon='mic')
            remote_system_start.on('click', js_handler=f'() => window.remoteAudio.start("remote_system", "{state.remote_token}")')
            remote_system_start.set_visibility(False)
            stop_button = ui.button('停止並完成', icon='stop', on_click=session.request_stop).props('outline')
            stop_button.on('click', js_handler='() => window.remoteAudio?.stop()')
            retry_button = ui.button('重試失敗片段', icon='replay', on_click=session.retry).props('outline')
            def skip_failed_chunk():
                try:
                    session.skip_failed_chunk()
                except ValueError as exc:
                    ui.notify(str(exc), type='warning')
            skip_button = ui.button('略過此段並繼續', icon='skip_next',
                                    on_click=skip_failed_chunk).props('outline color=amber-9')
            cancel_button = ui.button('取消待處理', on_click=cancel).props('flat color=grey-7')
            summary = ui.label('').classes('metric')
        level = ui.linear_progress(value=0, show_value=False).classes('w-full').props('rounded size=4px')
        with ui.row().classes('w-full justify-between items-center'):
            status = ui.label('準備就緒').classes('font-medium')
            metrics = ui.label('').classes('muted text-xs')
        recording_note = ui.label('').classes('muted text-sm')
        error = ui.label('').classes('text-red-700 notice text-sm')
        warning = ui.label('').classes('text-amber-800 notice text-sm')
        gap_notice = ui.label('').classes('text-amber-900 notice text-sm font-semibold')
    return SimpleNamespace(
        settings_button=settings_button, source=source,
        simplified_button=simplified_button, traditional_button=traditional_button,
        stream_switch=stream_switch, record_switch=record_switch,
        device=device, refresh_button=refresh_button, device_note=device_note,
        upload=upload, file_label=file_label, start_button=start_button,
        remote_mic_start=remote_mic_start, remote_system_start=remote_system_start,
        stop_button=stop_button, retry_button=retry_button, skip_button=skip_button,
        cancel_button=cancel_button, summary=summary, level=level,
        status=status, metrics=metrics, recording_note=recording_note,
        error=error, warning=warning, gap_notice=gap_notice,
    )
