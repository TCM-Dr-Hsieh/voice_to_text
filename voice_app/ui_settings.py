"""Model and processing settings dialog for one transcription page."""
from copy import deepcopy

from nicegui import ui

from .api import ModelAPI
from .asr import local_asr, validate_model_dir
import json

from .config import MAX_VOCABULARY_CHARS, MIN_TRANSLATION_LAG, resolve_path, shared_mode
from .speaker.embedder import availability
from .ui_tuning import open_tuning_dialog, tuning_summary
from .translation import SYSTEM as TRANSLATE_SYSTEM, parse_translation


def show_settings_dialog(state, stream_switch, tab='asr'):
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
            meeting_tab = ui.tab('會議模式')
            translator_tab = ui.tab('翻譯')
        first = {'asr': asr_tab, 'editor': editor_tab, 'audio': audio_tab, 'meeting': meeting_tab,
                 'translator': translator_tab}.get(tab, asr_tab)
        with ui.tab_panels(tabs, value=first).classes('w-full'):
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
                overlap = ui.number('左側參考 Overlap Z（秒）', value=draft.overlap_seconds, min=0, max=29, step=0.25).classes('w-full')
                ui.label('Z ≤ L−1，至少保留 1 秒新音訊；建議每次新增的音訊（L−Z）大於 3 秒：每隔 L−Z 秒就要完成一次辨識、對齊和校稿，'
                         '太小時處理不及，即時錄音的待處理片段會累積，滿了就停止錄音。每隔 L−Z 秒產生一段辨識文字；重疊音訊只作參考，前一段已涵蓋的文字不重複插入。停止時處理不足 L 的尾段。').classes('muted text-sm')
                context = ui.number('提供給校稿的前文長度', value=draft.context_chars, min=100, max=20000, step=100).classes('w-full')
                timeout = ui.number('API 逾時（秒）', value=draft.timeout_seconds, min=5, max=600).classes('w-full')
                streaming = ui.switch('串流顯示校稿預覽', value=draft.streaming)
                ui.label('有效的三段校稿結果會直接套用；若回應格式錯誤、非空段回空白或請求失敗，重試後保留目前文字。').classes('muted text-sm')
            with ui.tab_panel(meeting_tab).classes('gap-3 p-0'):
                ui.label('會議模式：左欄即時語音（標示說話者），右欄即時翻譯。說話者以聲音自動分群，預設叫 Speaker-001、Speaker-002…，'
                         '可在會議畫面的「說話者」輸入名字。').classes('muted text-sm')
                speaker_model = ui.input('說話者聲紋模型檔（ONNX）', value=draft.meeting.speaker_model_path).classes('w-full')
                speaker_check = ui.label('').classes('text-sm notice')

                def check_speaker_model():
                    reason = availability(resolve_path(speaker_model.value or ''))
                    speaker_check.set_text('聲紋模型可以使用。' if not reason else reason)

                ui.button('檢查聲紋模型', on_click=check_speaker_model).props('outline')
                unknown_percentile = ui.number('「不明」百分位', value=draft.meeting.unknown_percentile, min=0, max=50,
                                               step=1).classes('w-full')
                ui.label('一句話與最像的說話者、第二像的說話者差距不夠大時標「不明」。數字越大越保守（較多不明、較少標錯），'
                         '0 表示幾乎都標。預設 10。').classes('muted text-sm')
                max_speakers = ui.number('說話者人數上限', value=draft.meeting.max_speakers, min=2, max=30, step=1).classes('w-full')
                live_translation = ui.switch('即時暫定翻譯：最新三段先翻成一整塊（黃色，不分說話者），定稿後換成分說話者的譯文（每次提交多一次 LLM 呼叫）',
                                             value=draft.meeting.live_translation)
                text_fill = ui.switch('用 LLM 依上下文補標「不明」的說話者（一階）', value=draft.meeting.text_fill)
                text_fill2 = ui.switch('二階補標：對一階仍不明的句子再問一次（需先開一階；較積極，準確度請自行觀察）',
                                       value=draft.meeting.text_fill2)
                ui.label('補標使用「文字校稿」的 LLM；只回答聲音無法判斷的句子，聲音已判定的說話者不會被改。補標的句子在左欄以 * 標示。'
                         ).classes('muted text-sm')
                background_filter = ui.switch('背景人聲過濾：小聲、又不像任何已知說話者的句子（例如麥克風收到的私下交談）標灰、不翻譯',
                                              value=draft.meeting.background_filter)
                background_rel_db = ui.number('背景：比常態音量小多少（dB）', value=draft.meeting.background_rel_db, min=1, max=30,
                                              step=0.5).classes('w-full')
                background_max_cos = ui.number('背景：與說話者的相似度要低於', value=draft.meeting.background_max_cos, min=0.1, max=0.9,
                                               step=0.025).classes('w-full')
                ui.label('兩個條件要同時成立才會標成背景：音量比最近的常態小這麼多，而且聲音不像任何已分出的說話者。'
                         '標成背景的句子仍然顯示（灰色）也會存檔，只是不送去翻譯；點灰色的「背景」字樣可以取消，點「不明」可以手動標成背景。'
                         '預設 6 dB、0.45；數字越小越保守（少標、少誤判）。').classes('muted text-sm')
                tuning_state = {'advanced': dict(draft.meeting.advanced)}          # the numbers being edited in the pop-up
                top_fields = {'unknown_percentile': unknown_percentile, 'max_speakers': max_speakers,
                              'background_rel_db': background_rel_db, 'background_max_cos': background_max_cos}
                with ui.row().classes('items-center gap-3'):
                    ui.button('進階參數…', icon='tune', on_click=lambda: open_tuning_dialog(tuning_state, top_fields, tuning_note)) \
                        .props('outline').mark('speaker-tuning-button')
                    tuning_note = ui.label(tuning_summary(tuning_state['advanced'], top_fields)).classes('muted text-sm')
                for box in top_fields.values():
                    box.on_value_change(lambda e: tuning_note.set_text(tuning_summary(tuning_state['advanced'], top_fields)))
                ui.label('「進階參數」可精調說話者標記的全部數字（含上面幾個），每個都說明調高、調低會怎樣；只對下一場會議有效。'
                         ).classes('muted text-sm')
                autosave = ui.number('會議自動存檔間隔（秒）', value=draft.meeting.autosave_seconds, min=5, max=600,
                                     step=5).classes('w-full')
            with ui.tab_panel(translator_tab).classes('gap-3 p-0'):
                ui.label(f'翻譯用的 LLM（OpenAI 相容服務），可與文字校稿不同。翻譯語言與延遲語音數（最小 {MIN_TRANSLATION_LAG}）'
                         '在會議畫面右上角設定。').classes('muted text-sm')
                translator_fields = {
                    'model': ui.input('模型 ID', value=draft.translator.model).classes('w-full'),
                    'base_url': ui.input('Base URL', value=draft.translator.base_url).classes('w-full'),
                    'api_key': ui.input('API Key', value=draft.translator.api_key, password=True,
                                        password_toggle_button=True).classes('w-full'),
                    'domain_terms': ui.textarea(
                        f'術語與專有名詞對照（翻譯時參考，最多 {MAX_VOCABULARY_CHARS:,} 字元；例如「手部衛生 = hand hygiene」）',
                        value=draft.translator.domain_terms).classes('w-full'),
                }
                translator_temperature = ui.number('Temperature', value=draft.translator.temperature, min=0, max=2,
                                                   step=0.05).classes('w-full')
                translator_picker = ui.select([], label='從伺服器模型清單選擇').classes('w-full')
                translator_picker.on_value_change(
                    lambda e: translator_fields['model'].set_value(e.value) if e.value else None)
                translator_result = ui.label('').classes('text-sm notice')

                def read_translator_fields():
                    for name, field in translator_fields.items():
                        setattr(draft.translator, name, field.value or '')
                    if translator_temperature.value is None:
                        raise ValueError('翻譯 Temperature 不可留空。')
                    draft.translator.temperature = float(translator_temperature.value)
                    draft.translator.validate()

                async def translator_models():
                    try:
                        read_translator_fields()
                        translator_result.set_text('測試中…')
                        async with ModelAPI(draft) as api:
                            names = await api.models(draft.translator)
                        translator_picker.set_options(names)
                        translator_result.set_text(f'連線成功，取得 {len(names)} 個模型。')
                    except Exception as exc:
                        translator_result.set_text(f'測試失敗：{exc}')

                async def translator_sample():
                    try:
                        read_translator_fields()
                        language = draft.meeting.target_language
                        translator_result.set_text('測試中…')
                        user = json.dumps({'translate': [{'i': 1, 'runs': [{'speaker': '王先生', 'text': '各位好，今天我們要討論手部衛生的重要性。'}]}],
                                           'already_translated': [], 'read_ahead': [],
                                           'glossary': draft.translator.domain_terms}, ensure_ascii=False)
                        async with ModelAPI(draft) as api:
                            texts = await api.chat_json(draft.translator, TRANSLATE_SYSTEM.format(language=language), user,
                                                        max_tokens=400, validator=lambda text: parse_translation(text, {1: 1}))
                        translator_result.set_text(f'翻譯成功（{language}）：{texts[1][0]}')
                    except Exception as exc:
                        translator_result.set_text(f'測試失敗：{exc}')

                with ui.row():
                    ui.button('取得模型清單', on_click=translator_models).props('outline')
                    ui.button('測試翻譯', on_click=translator_sample).props('outline')
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
                read_translator_fields()
                draft.meeting.speaker_model_path = speaker_model.value or ''
                draft.meeting.unknown_percentile = float(unknown_percentile.value)
                if float(max_speakers.value) % 1:
                    raise ValueError('說話者人數上限須為整數。')
                draft.meeting.max_speakers = int(max_speakers.value)
                draft.meeting.live_translation = bool(live_translation.value)
                draft.meeting.text_fill = bool(text_fill.value)
                draft.meeting.text_fill2 = bool(text_fill2.value) and bool(text_fill.value)
                draft.meeting.background_filter = bool(background_filter.value)
                draft.meeting.background_rel_db = float(background_rel_db.value)
                draft.meeting.background_max_cos = float(background_max_cos.value)
                draft.meeting.advanced = dict(tuning_state['advanced'])
                draft.meeting.autosave_seconds = float(autosave.value)
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

