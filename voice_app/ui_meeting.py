"""The meeting panel (transcript | translation), the speakers window and the translation controls."""
from types import SimpleNamespace

from nicegui import ui

from .config import MIN_TRANSLATION_LAG
from .speaker.palette import color_of


def build_meeting_panel(state, on_start_translation, on_stop_translation, on_speakers, on_export, on_translation_settings,
                        on_retry_translation):
    """The card that replaces the article while meeting mode is on. Returns the widgets the page's timer updates."""
    settings = state.settings.meeting
    with ui.card().classes('w-full p-5 gap-3') as card:
        with ui.element('div').classes('m-head'):
            with ui.row().classes('items-center gap-2'):
                ui.label('會議即時語音').classes('text-lg font-semibold')
                speakers_button = ui.button('說話者', icon='groups', on_click=on_speakers).props('outline dense')
                export_button = ui.button('匯出雙語對照 TXT', icon='download', on_click=on_export).props('outline dense')
            with ui.row().classes('items-center gap-2'):
                ui.label('會議即時翻譯').classes('text-lg font-semibold')
                language = ui.input('語言', value=settings.target_language).props('dense outlined').classes('w-40')
                lag = ui.number('延遲語音數', value=settings.translation_lag, min=MIN_TRANSLATION_LAG, max=60, step=1,
                                format='%d').props('dense outlined').classes('w-32').tooltip(
                    f'翻譯會比逐字稿慢這麼多段（最小 {MIN_TRANSLATION_LAG}）：延遲越多，翻譯越能參考後文')
                start_button = ui.button('開始翻譯', icon='translate', on_click=on_start_translation)
                stop_button = ui.button('停止翻譯', icon='stop_circle', on_click=on_stop_translation).props('outline')
                retry_button = ui.button('重試失敗段落', icon='refresh', on_click=on_retry_translation).props('outline color=amber-9')
                retry_button.set_visibility(False)
                settings_button = ui.button(icon='tune', on_click=on_translation_settings).props('flat round dense').tooltip('翻譯模型設定')
        ui.html('<div id="meeting-host"></div>', sanitize=False).classes('w-full')
        with ui.element('div').classes('m-head'):
            speaker_status = ui.label('').classes('muted text-xs')
            translation_status = ui.label('').classes('muted text-xs')
        notice = ui.label('').classes('text-amber-800 notice text-sm')
        notice.set_visibility(False)
    return SimpleNamespace(card=card, speakers_button=speakers_button, export_button=export_button, language=language, lag=lag,
                           start_button=start_button, stop_button=stop_button, retry_button=retry_button, settings_button=settings_button,
                           speaker_status=speaker_status, translation_status=translation_status, notice=notice)


def read_translation_inputs(widgets) -> tuple[str, int]:
    """The language and the lag as typed; raises ValueError with the sentence to show when one is not usable."""
    language = ' '.join(str(widgets.language.value or '').split())
    if not language:
        raise ValueError('請輸入要翻譯成的語言，例如 English、日本語。')
    try:
        value = float(widgets.lag.value)
    except (TypeError, ValueError):
        raise ValueError('延遲語音數須為整數。') from None
    if value != int(value) or int(value) < MIN_TRANSLATION_LAG:
        raise ValueError(f'延遲語音數須為 {MIN_TRANSLATION_LAG} 以上的整數（最新 3 段仍可能被校稿修改，所以最小是 {MIN_TRANSLATION_LAG}）。')
    return language, int(value)


def export_filename(meeting) -> str:
    return f'{meeting.path.stem}.txt'


def open_speakers_dialog(state):
    """Name the speakers and join two voice groups that are one person."""
    meeting = state.session.meeting
    if meeting is None:
        ui.notify('開始會議轉錄後才有說話者。', type='warning')
        return
    speaker = meeting.speaker
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-4xl p-6 gap-3'):
        ui.label('說話者').classes('text-2xl font-semibold')
        ui.label('聲音分群會自動出現。替每位說話者輸入名字，左欄會立刻改用這個名字，翻譯欄的名字也會重新翻譯。'
                 '同一個人被分成兩群時，選「合併到」把這一群併進另一群（無法復原）。').classes('muted text-sm')
        body = ui.column().classes('w-full gap-3')

        def rename(gid, widget):
            try:
                speaker.set_name(gid, widget.value or '')
            except ValueError as exc:
                ui.notify(str(exc), type='warning')

        def merge(gid, target):
            if target.value is None:
                ui.notify('請先選擇要合併到哪一位。', type='warning')
                return
            try:
                speaker.merge(gid, int(target.value))
            except ValueError as exc:
                ui.notify(str(exc), type='warning')
                return
            ui.notify('已合併。', type='positive')
            refresh()

        def refresh():
            body.clear()
            people = speaker.speakers()
            with body:
                if not people:
                    ui.label('還沒有偵測到說話者（開始說話後約 10–60 秒會出現）。').classes('muted')
                for person in people:
                    others = {p['gid']: p['label'] for p in people if p['gid'] != person['gid']}
                    with ui.card().classes('w-full p-3 gap-1').props('flat bordered'):
                        with ui.row().classes('w-full items-center gap-3'):
                            ui.element('span').style(f'display:inline-block;width:14px;height:14px;border-radius:3px;background:{color_of(person["gid"])}')
                            ui.label(person['default']).classes('font-semibold w-28')
                            field = ui.input('名字', value=person['name'], placeholder=person['default']).props('dense outlined clearable').classes('grow')
                            field.on('blur', lambda e, g=person['gid'], w=field: rename(g, w))
                            field.on('keydown.enter', lambda e, g=person['gid'], w=field: rename(g, w))
                            ui.label(f'{person["seconds"] / 60:.1f} 分鐘 · {person["share"]:.0%} · {person["sentences"]} 句').classes('muted text-sm')
                        if person['merged']:
                            ui.label('已合併：' + '、'.join(person['merged'])).classes('muted text-xs')
                        for example in person['examples']:
                            ui.label('「' + example[:60] + '」').classes('muted text-xs')
                        if others:
                            with ui.row().classes('items-center gap-2'):
                                target = ui.select(others, label='合併到', value=None).props('dense outlined').classes('w-48')
                                ui.button('合併', on_click=lambda e, g=person['gid'], t=target: merge(g, t)).props('outline dense')

        refresh()
        with ui.row().classes('w-full justify-end'):
            ui.button('重新整理', on_click=refresh).props('flat')
            ui.button('關閉', on_click=dialog.close)
    dialog.open()
