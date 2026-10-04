import json

import httpx
import numpy as np
import pytest

from voice_app.api import ModelAPI, ModelError
from voice_app.config import Settings


@pytest.mark.asyncio
async def test_valid_edit_is_applied_without_manual_gate():
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        output = {'segments': [{'index': 1, 'text': '這是完全不一樣的句子'}]}
        return httpx.Response(200, json={'choices': [{'message': {
            'content': json.dumps(output, ensure_ascii=False)}, 'finish_reason': 'stop'}]})
    settings = Settings(streaming=False)
    settings.editor.temperature = 0.65
    settings.editor.domain_terms = '小兒夜尿、腎陽'
    async with ModelAPI(settings, httpx.MockTransport(handler)) as api:
        result = await api.revise_recent('禁止回改的前文', [{
            'index': 1, 'start': 0, 'end': 1, 'reason': 'stop',
            'added': '天企好。', 'current': '天企好。', 'raw': '天企好。'}])
    assert result.texts == ['這是完全不一樣的句子'] and not result.warning
    assert len(requests) == 1
    assert 'chat_template_kwargs' not in requests[0]
    assert requests[0]['temperature'] == 0.65
    assert '"領域常見詞彙": "小兒夜尿、腎陽"' in requests[0]['messages'][0]['content']
    assert '不得只因詞表有某個詞就把未說出的內容加入稿件' in requests[0]['messages'][0]['content']
    assert json.loads(requests[0]['messages'][1]['content'])['read_only_context_before'] == '禁止回改的前文'
    assert '領域常見詞彙' not in requests[0]['messages'][1]['content']


@pytest.mark.asyncio
async def test_number_and_negation_changes_do_not_create_hidden_pending_review():
    output = {'segments': [{'index': 1, 'text': '劑量是 20 毫克，我們今天去。'}]}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={
        'choices': [{'message': {'content': json.dumps(output, ensure_ascii=False)}}]}))
    async with ModelAPI(Settings(streaming=False), transport) as api:
        result = await api.revise_recent('', [{
            'index': 1, 'start': 0, 'end': 1, 'reason': 'stop',
            'added': '劑量是 10 毫克，我們今天不去。',
            'current': '劑量是 10 毫克，我們今天不去。',
            'raw': '劑量是 10 毫克，我們今天不去。'}])
    assert result.texts == [output['segments'][0]['text']]
    assert not result.warning


class BytesStream(httpx.AsyncByteStream):
    def __init__(self, payload):
        self.payload = payload

    async def __aiter__(self):
        for start in range(0, len(self.payload), 7):
            yield self.payload[start:start + 7]


def sse(events):
    return ''.join('data: ' + json.dumps(event, ensure_ascii=False) + '\r\n\r\n' for event in events).encode() + b'data: [DONE]\r\n\r\n'


@pytest.mark.asyncio
async def test_stream_unicode_reasoning_and_punctuation():
    output = json.dumps({'segments': [{'index': 1, 'text': '天氣，好！'}]}, ensure_ascii=False)
    data = sse([
        {'choices': [{'delta': {'reasoning_content': '不要顯示'}}]},
        {'choices': [{'delta': {'content': output[:12]}}]},
        {'choices': [{'delta': {'content': output[12:]}, 'finish_reason': 'stop'}]},
    ])
    previews = []
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, stream=BytesStream(data))
    settings = Settings(streaming=True)
    settings.editor.temperature = 0.25
    settings.editor.domain_terms = '中醫、針灸'
    async with ModelAPI(settings, httpx.MockTransport(handler)) as api:
        result = await api.revise_recent('', [{
            'index': 1, 'start': 0, 'end': 1, 'reason': 'stop',
            'added': '天企好', 'current': '天企好', 'raw': '天企好'}], previews.append)
    assert result.texts == ['天氣，好！'] and not result.warning
    assert 'chat_template_kwargs' not in requests[0]
    assert requests[0]['temperature'] == 0.25
    assert '"領域常見詞彙": "中醫、針灸"' in requests[0]['messages'][0]['content']
    assert '天氣，好！' in previews
    assert all('不要顯示' not in item for item in previews)


@pytest.mark.asyncio
async def test_truncated_stream_does_not_commit():
    data = sse([{'choices': [{'delta': {'content': '{"segments":[{"index":1,"text":"天氣好"}]}'}}]}])
    async with ModelAPI(Settings(streaming=True), httpx.MockTransport(lambda _: httpx.Response(200, stream=BytesStream(data)))) as api:
        result = await api.revise_recent('', [{
            'index': 1, 'start': 0, 'end': 1, 'reason': 'stop',
            'added': '天企好', 'current': '天企好', 'raw': '天企好'}])
    assert result.texts == ['天企好'] and result.warning


@pytest.mark.asyncio
async def test_three_segment_revision_uses_later_full_asr_to_repair_seam():
    segments = [
        {'index': 54, 'start': 164.8, 'end': 173.8, 'reason': 'window',
         'added': '就是絕對不怪誰。', 'current': '就是絕對不怪誰。', 'raw': '就是就是絕對不怪誰。'},
        {'index': 55, 'start': 167.8, 'end': 176.8, 'reason': 'window',
         'added': '水，那', 'current': '水，那', 'raw': '就是就是絕對不灌水，那會提高水準。'},
        {'index': 56, 'start': 170.8, 'end': 179.8, 'reason': 'window',
         'added': '後續', 'current': '後續', 'raw': '不灌水，那會提高水準。後續'},
    ]
    proposed = [{'index': 54, 'text': '就是絕對不灌'},
                {'index': 55, 'text': '水，那'}, {'index': 56, 'text': '後續'}]
    sent = []
    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {
            'content': json.dumps({'segments': proposed}, ensure_ascii=False)}, 'finish_reason': 'stop'}]})

    settings = Settings(streaming=False)
    settings.editor.temperature = 0.4
    settings.editor.domain_terms = '不灌水'
    async with ModelAPI(settings, httpx.MockTransport(handler)) as api:
        result = await api.revise_recent('較早已鎖定的前文', segments,
                                         context_after='游標後面的文章')
    assert result.texts == [item['text'] for item in proposed] and not result.warning
    payload = json.loads(sent[0]['messages'][1]['content'])
    assert 'read_only_context' not in payload
    assert 'chat_template_kwargs' not in sent[0]
    assert sent[0]['temperature'] == 0.4
    assert '"領域常見詞彙": "不灌水"' in sent[0]['messages'][0]['content']
    assert payload['read_only_context_before'] == '較早已鎖定的前文'
    assert payload['read_only_context_after'] == '游標後面的文章'
    assert [item['raw'] for item in payload['segments']] == [item['raw'] for item in segments]
    assert sent[0]['stream'] is False


@pytest.mark.asyncio
async def test_partial_llm_seam_fix_cannot_leave_duplicate_boundary_word():
    segments = [
        {'index': 54, 'start': 0, 'end': 9, 'reason': 'window',
         'added': '就是絕對不怪誰。', 'current': '就是絕對不怪誰。', 'raw': '就是絕對不怪誰。'},
        {'index': 55, 'start': 3, 'end': 12, 'reason': 'window',
         'added': '水，那會提高', 'current': '水，那會提高', 'raw': '就是絕對不灌水，那會提高'},
        {'index': 56, 'start': 6, 'end': 15, 'reason': 'stop',
         'added': '水準', 'current': '水準', 'raw': '不灌水，那會提高水準'},
    ]
    partial = {'segments': [{'index': 54, 'text': '就是絕對不灌水。'},
                            {'index': 55, 'text': '水，那會提高'},
                            {'index': 56, 'text': '水準'}]}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={
        'choices': [{'message': {'content': json.dumps(partial, ensure_ascii=False)}}]}))
    async with ModelAPI(Settings(), transport) as api:
        result = await api.revise_recent('', segments)
    assert not result.warning
    assert ''.join(result.texts) == '就是絕對不灌水，那會提高水準'
    assert '水。水' not in ''.join(result.texts)
    assert result.seam_repaired


@pytest.mark.asyncio
async def test_seam_repair_keeps_another_correction_from_the_same_llm_reply():
    segments = [
        {'index': 54, 'start': 0, 'end': 9, 'reason': 'window',
         'added': '就是絕對不怪誰。', 'current': '就是絕對不怪誰。', 'raw': '就是絕對不怪誰。'},
        {'index': 55, 'start': 3, 'end': 12, 'reason': 'window',
         'added': '水，那會提高', 'current': '水，那會提高', 'raw': '就是絕對不灌水，那會提高'},
        {'index': 56, 'start': 6, 'end': 15, 'reason': 'stop',
         'added': '天企', 'current': '天企', 'raw': '不灌水，那天氣'},
    ]
    partial = {'segments': [{'index': 54, 'text': '就是絕對不灌水。'},
                            {'index': 55, 'text': '水，那會提高'},
                            {'index': 56, 'text': '天氣'}]}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={
        'choices': [{'message': {'content': json.dumps(partial, ensure_ascii=False)}}]}))
    async with ModelAPI(Settings(streaming=False), transport) as api:
        result = await api.revise_recent('', segments)
    assert not result.warning and result.seam_repaired
    assert ''.join(result.texts) == '就是絕對不灌水，那會提高天氣'


@pytest.mark.asyncio
async def test_whole_latest_segment_rewrite_is_applied():
    segments = [
        {'index': index, 'start': index * 3, 'end': index * 3 + 9,
         'reason': 'window', 'added': char * 17, 'current': char * 17,
         'raw': char * 17}
        for index, char in enumerate('甲乙丙', 1)
    ]
    output = {'segments': [{'index': 1, 'text': '甲' * 17},
                           {'index': 2, 'text': '乙' * 17},
                           {'index': 3, 'text': '丁' * 17}]}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={
        'choices': [{'message': {'content': json.dumps(output, ensure_ascii=False)}}]}))
    async with ModelAPI(Settings(streaming=False), transport) as api:
        result = await api.revise_recent('', segments)
    assert result.texts == ['甲' * 17, '乙' * 17, '丁' * 17]
    assert not result.warning


@pytest.mark.asyncio
async def test_three_segment_revision_without_later_asr_evidence_is_applied():
    segments = [{'index': 1, 'start': 0, 'end': 4, 'reason': 'window',
                 'added': '今天下雨', 'current': '今天下雨', 'raw': '今天下雨'},
                {'index': 2, 'start': 3, 'end': 7, 'reason': 'stop',
                 'added': '出門', 'current': '出門', 'raw': '出門'}]
    output = {'segments': [{'index': 1, 'text': '今天下雪'}, {'index': 2, 'text': '出門'}]}
    async with ModelAPI(Settings(), httpx.MockTransport(lambda _: httpx.Response(200, json={
            'choices': [{'message': {'content': json.dumps(output, ensure_ascii=False)}}]}))) as api:
        result = await api.revise_recent('', segments)
    assert result.texts == ['今天下雪', '出門'] and not result.warning


@pytest.mark.asyncio
async def test_three_segment_revision_requires_valid_json_and_indexes():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={'choices': [{'message': {'content': '{"segments":[]}'}}]})
    segments = [{'index': 2, 'start': 0, 'end': 4, 'reason': 'stop',
                 'added': '原稿', 'current': '原稿', 'raw': '原稿'}]
    async with ModelAPI(Settings(), httpx.MockTransport(handler)) as api:
        result = await api.revise_recent('', segments)
    assert result.texts == ['原稿'] and '兩次未成功' in result.warning
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_blank_segment_retries_and_then_applies_complete_reply():
    calls = []
    def handler(request):
        calls.append(request)
        texts = ['今天下雨', ''] if len(calls) == 1 else ['今天下雨', '出門']
        output = {'segments': [{'index': index, 'text': text}
                               for index, text in enumerate(texts, 1)]}
        return httpx.Response(200, json={'choices': [{'message': {
            'content': json.dumps(output, ensure_ascii=False)}}]})
    segments = [{'index': index, 'start': index - 1, 'end': index,
                 'reason': 'window', 'added': text, 'current': text, 'raw': text}
                for index, text in enumerate(('今天下雨', '出門'), 1)]
    async with ModelAPI(Settings(streaming=False), httpx.MockTransport(handler)) as api:
        result = await api.revise_recent('', segments)
    assert result.texts == ['今天下雨', '出門'] and not result.warning
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_persistently_blank_segment_preserves_current_text_with_warning():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={'choices': [{'message': {
            'content': '{"segments":[{"index":1,"text":"。"}]}'}}]})
    segment = {'index': 1, 'start': 0, 'end': 1, 'reason': 'stop',
               'added': '原稿', 'current': '原稿', 'raw': '原稿'}
    async with ModelAPI(Settings(streaming=False), httpx.MockTransport(handler)) as api:
        result = await api.revise_recent('', [segment])
    assert result.texts == ['原稿'] and '回傳空白' in result.warning
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_empty_input_segment_may_remain_empty():
    output = {'segments': [{'index': 1, 'text': ''}]}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={
        'choices': [{'message': {'content': json.dumps(output)}}]}))
    segment = {'index': 1, 'start': 0, 'end': 1, 'reason': 'stop',
               'added': '', 'current': '', 'raw': ''}
    async with ModelAPI(Settings(streaming=False), transport) as api:
        result = await api.revise_recent('', [segment])
    assert result.texts == [''] and not result.warning


@pytest.mark.asyncio
async def test_asr_uses_local_engine_without_http(monkeypatch):
    from voice_app.api import local_asr
    async def transcribe(settings, samples):
        assert settings.asr_model_dir == Settings().asr_model_dir
        assert len(samples) == 16000
        return '你好'
    monkeypatch.setattr(local_asr, 'transcribe', transcribe)
    def reject_http(_):
        raise AssertionError('ASR must not send HTTP')
    async with ModelAPI(Settings(), httpx.MockTransport(reject_http)) as api:
        assert await api.transcribe(np.zeros(16000)) == '你好'
