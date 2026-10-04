"""Model and processing settings dialog for one transcription page."""
from copy import deepcopy

from nicegui import ui

from .api import ModelAPI
from .asr import local_asr, validate_model_dir
from .config import MAX_VOCABULARY_CHARS, shared_mode


def show_settings_dialog(state, stream_switch):
    if shared_mode():
        return
    if state.session.active or state.loading:
        return
    draft = deepcopy(state.settings)
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-3xl p-6 gap-4'):
        ui.label('模型與轉錄設定').classes('text-2xl font-semibold')
        ui.label('語音辨識從本機資料夾載入，文字校稿連接 OpenAI 相容服務。').classes('muted text-sm')
        with ui.tabs().classes('w-full') as tabs:
            asr_tab = ui.tab('語音辨識')
            editor_tab = ui.tab('文字校稿')
            audio_tab = ui.tab('處理規則')
        with ui.tab_panels(tabs, value=asr_tab).classes('w-full'):
            with ui.tab_panel(asr_tab).classes('gap-3 p-0'):
                model_dir = ui.input('ASR 模型資料夾', value=draft.asr_model_dir).classes('w-full')
                aligner_dir = ui.input('ForcedAligner 模型資料夾', value=draft.aligner_model_dir).classes('w-full')
                vocabulary = ui.textarea(
                    f'專有詞提示（ASR 與校稿共用，最多 {MAX_VOCABULARY_CHARS:,} 字元）',
                    value=draft.vocabulary).classes('w-full')
                device_field = ui.select({'auto': '自動（優先 CUDA）', 'cuda': 'NVIDIA GPU（CUDA）', 'cpu': 'CPU'},
                                         value=draft.asr_device, label='運算裝置').classes('w-full')
                ui.label('選擇包含 config.json、tokenizer 及 Safetensors 權重的 Qwen3-ASR 資料夾。模型不會上傳或自動下載。').classes('muted text-sm')
                load_result = ui.label('').classes('text-sm notice')

                async def test_load():
                    if state.loading:
                        return
                    state.loading = True
                    load_button.disable()
                    try:
                        draft.asr_model_dir = model_dir.value or ''
                        draft.aligner_model_dir = aligner_dir.value or ''
                        draft.asr_device = device_field.value
                        validate_model_dir(draft.asr_model_dir)
                        validate_model_dir(draft.aligner_model_dir, aligner=True)
                        load_result.set_text('正在載入本機模型…首次載入可能需要數十秒。')
                        info = await local_asr.load(draft)
                        load_result.set_text(f'ASR 與對齊模型載入成功 · {info["device"]}')
                    except Exception as exc:
                        load_result.set_text(f'載入失敗：{exc}')
                    finally:
                        state.loading = False
                        load_button.enable()

                load_button = ui.button('測試載入', on_click=test_load).props('outline')
            with ui.tab_panel(editor_tab).classes('gap-3 p-0'):
                editor_fields = {
                    'model': ui.input('模型 ID', value=draft.editor.model).classes('w-full'),
                    'base_url': ui.input('Base URL', value=draft.editor.base_url).classes('w-full'),
                    'api_key': ui.input('API Key', value=draft.editor.api_key, password=True, password_toggle_button=True).classes('w-full'),
                    'domain_terms': ui.textarea(
                        f'領域常見詞彙（僅供 LLM 校稿，最多 {MAX_VOCABULARY_CHARS:,} 字元）',
                        value=draft.editor.domain_terms).classes('w-full'),
                }
                temperature = ui.number('Temperature', value=draft.editor.temperature,
                                        min=0, max=2, step=0.05).classes('w-full')
                picker = ui.select([], label='從伺服器模型清單選擇').classes('w-full')
                picker.on_value_change(lambda e: editor_fields['model'].set_value(e.value) if e.value else None)
                result_label = ui.label('').classes('text-sm notice')

                def read_editor_fields():
                    for name, field in editor_fields.items():
                        setattr(draft.editor, name, field.value or '')
                    if temperature.value is None:
                        raise ValueError('Temperature 不可留空。')
                    draft.editor.temperature = float(temperature.value)
                    draft.editor.validate()

                async def check_editor(actual=False):
                    try:
                        read_editor_fields()
                        result_label.set_text('測試中…')
                        async with ModelAPI(draft) as api:
                            if actual:
                                result = await api.revise_recent('', [{
                                    'index': 1, 'start': 0, 'end': 1, 'reason': 'stop',
                                    'added': '今天的天企很好。', 'current': '今天的天企很好。',
                                    'raw': '今天的天企很好。'}])
                                result_label.set_text(result.warning or f'校稿請求成功：{result.texts[0]}')
                            else:
                                names = await api.models(draft.editor)
                                picker.set_options(names)
                                result_label.set_text(f'連線成功，取得 {len(names)} 個模型。')
                    except Exception as exc:
                        result_label.set_text(f'測試失敗：{exc}')

                async def list_models():
                    await check_editor()

                async def test_editor():
                    await check_editor(actual=True)

                with ui.row():
                    ui.button('取得模型清單', on_click=list_models).props('outline')
                    ui.button('測試校稿', on_click=test_editor).props('outline')
            with ui.tab_panel(audio_tab).classes('gap-3 p-0'):
                window = ui.number('音訊視窗長度 L（秒）', value=draft.window_seconds, min=5, max=30, step=1).classes('w-full')
                overlap = ui.number('左側參考 Overlap Z（秒）', value=draft.overlap_seconds, min=0, max=27, step=0.25).classes('w-full')
                ui.label('Z ≤ L−3，至少保留 3 秒新音訊。每隔 L−Z 秒產生一段辨識文字；重疊音訊只作參考，前一段已涵蓋的文字不重複插入。停止時處理不足 L 的尾段。').classes('muted text-sm')
                context = ui.number('提供給校稿的前文長度', value=draft.context_chars, min=100, max=20000, step=100).classes('w-full')
                timeout = ui.number('API 逾時（秒）', value=draft.timeout_seconds, min=5, max=600).classes('w-full')
                streaming = ui.switch('串流顯示校稿預覽', value=draft.streaming)
                ui.label('有效的三段校稿結果會直接套用；若回應格式錯誤、非空段回空白或請求失敗，重試後保留目前文字。').classes('muted text-sm')
        error_label = ui.label('').classes('text-red-700 text-sm')
        ui.label('設定保存在本機 data/settings.json；API Key 以明文儲存，匯出稿件不包含金鑰。').classes('muted text-xs')

        def save():
            try:
                read_editor_fields()
                draft.asr_model_dir = model_dir.value or ''
                draft.aligner_model_dir = aligner_dir.value or ''
                draft.vocabulary = vocabulary.value or ''
                draft.asr_device = device_field.value
                validate_model_dir(draft.asr_model_dir)
                validate_model_dir(draft.aligner_model_dir, aligner=True)
                draft.window_seconds = float(window.value)
                draft.overlap_seconds = float(overlap.value)
                if float(context.value) % 1:
                    raise ValueError('前文字數需為整數。')
                draft.context_chars = int(context.value)
                draft.timeout_seconds = float(timeout.value)
                draft.streaming = bool(streaming.value)
                draft.save()
                state.settings = draft
                stream_switch.set_value(state.settings.streaming)
                dialog.close()
                ui.notify('設定已儲存。', type='positive')
            except Exception as exc:
                error_label.set_text(str(exc))

        with ui.row().classes('w-full justify-end'):
            ui.button('返回', on_click=dialog.close).props('flat')
            ui.button('儲存設定', on_click=save)
    dialog.open()

