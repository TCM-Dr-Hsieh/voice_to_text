"""The pop-up with every tunable number of speaker marking, opened from the meeting tab of the settings dialog.

Each number is shown with its value (the default beside it), what it does and what raising or lowering it does, grouped by stage of the
pipeline. "套用" checks the values the way saving the settings does and hands them back to the settings dialog (the four numbers that
have a box of their own in the meeting tab are written into those boxes); "取消" throws the changes away. Nothing is saved until the
settings dialog's own save, and the numbers take effect for the NEXT meeting: a meeting that is running keeps the numbers it began with.
"""
from nicegui import ui

from .config import MeetingSettings
from .speaker import tuning


def tuning_summary(advanced: dict, top_fields: dict) -> str:
    changed = len(advanced) + sum(1 for key, field in top_fields.items() if field.value is not None and field.value != tuning.DEFAULTS[key])
    return f'目前有 {changed} 項進階參數不是預設值。' if changed else '目前全部使用預設值。'


def open_tuning_dialog(state: dict, top_fields: dict, note):
    """`state['advanced']` holds the overrides of the settings dialog being edited, `top_fields` the boxes of the four numbers that have
    their own (`key -> ui.number`), `note` the label that says how many numbers were changed."""
    current = dict(tuning.DEFAULTS)
    current.update(state['advanced'])
    for key, field in top_fields.items():
        current[key] = field.value if field.value is not None else tuning.DEFAULTS[key]
    inputs: dict = {}
    dialog = ui.dialog().props('persistent')
    with dialog, ui.card().classes('w-[1180px] max-w-full gap-2').style('max-height:92vh'):
        ui.label('說話者標記的進階參數').classes('text-xl font-semibold')
        ui.label('★ 是最重要的參數。每個數字旁標著預設值（程式現在用的值）；沒有改動的不會寫進設定檔。多數參數互相牽連，建議一次只改一兩個，'
                 '改完用一場會議或錄音檢查。預設值是在少數家人或朋友模擬看診的錄音上調的，樣本非常少，僅供參考，不一定適合你的場合。'
                 '修改只對「下一場」會議有效，進行中的會議維持開始時的數字。').classes('muted text-sm')
        with ui.scroll_area().classes('w-full').style('height:62vh'):
            for stage in tuning.STAGES:
                ui.label(stage).classes('text-base font-semibold mt-3')
                with ui.grid(columns='230px 150px 1.2fr 1fr 1fr').classes('w-full gap-x-3 gap-y-2 items-start'):
                    for head in ('參數', '數值（預設）', '功能', '調高', '調低'):
                        ui.label(head).classes('muted text-sm font-medium')
                    for knob in (k for k in tuning.KNOBS if k.stage == stage):
                        ui.label(('★ ' if knob.star else '') + knob.label).classes('font-medium' + (' text-amber-800' if knob.star else ''))
                        with ui.column().classes('gap-0'):
                            inputs[knob.key] = ui.number(value=current[knob.key], min=knob.low, max=knob.high, step=knob.step,
                                                         format='%g').props('dense outlined').classes('w-32').mark(f'tuning-{knob.key}')
                            ui.label(f'預設 {knob.default:g}（{knob.low:g}–{knob.high:g}）').classes('muted text-xs')
                        with ui.column().classes('gap-1'):
                            ui.label(knob.what).classes('text-sm')
                            if knob.key in tuning.MEASURED:
                                ui.label(tuning.MEASURED[knob.key]).classes('muted text-xs')
                        ui.label(knob.higher).classes('text-sm')
                        ui.label(knob.lower).classes('text-sm')
        error = ui.label('').classes('text-red-700 text-sm')

        def reset():
            for key, field in inputs.items():
                field.set_value(tuning.DEFAULTS[key])
            error.set_text('')

        def apply():
            try:
                missing = [tuning.BY_KEY[key].label for key, field in inputs.items() if field.value is None]
                if missing:
                    raise ValueError('這些欄位不可留空：' + '、'.join(missing))
                values = {key: field.value for key, field in inputs.items()}
                for key in tuning.TOP_LEVEL:                                  # (a number box can hand over 12.0 for 12)
                    if tuning.BY_KEY[key].integer and float(values[key]).is_integer():
                        values[key] = int(values[key])
                candidate = MeetingSettings(advanced={key: value for key, value in values.items() if key not in tuning.TOP_LEVEL},
                                            **{key: values[key] for key in tuning.TOP_LEVEL})
                candidate.validate()                                         # the same check saving the settings makes
            except (ValueError, TypeError) as exc:
                error.set_text(str(exc))
                return
            state['advanced'] = dict(candidate.advanced)
            for key, field in top_fields.items():
                field.set_value(getattr(candidate, key))
            note.set_text(tuning_summary(state['advanced'], top_fields))
            dialog.close()

        with ui.row().classes('w-full justify-between'):
            ui.button('全部還原成預設值', on_click=reset).props('flat').mark('tuning-reset')
            with ui.row().classes('gap-2'):
                ui.button('取消', on_click=dialog.close).props('flat').mark('tuning-cancel')
                ui.button('套用', on_click=apply).mark('tuning-apply')
    dialog.open()
