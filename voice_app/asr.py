"""Async local Qwen ASR, isolated from the UI and its dependencies."""
import asyncio
import base64
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .config import ROOT, Settings, resolve_path


class ASRError(RuntimeError):
    pass


def worker_python() -> Path:
    """The ASR worker's interpreter: the project's own .venv-asr, created by setup.ps1."""
    return ROOT / '.venv-asr' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def validate_model_dir(value: str, *, aligner=False) -> Path:
    folder = resolve_path(value).resolve()
    role = 'ForcedAligner' if aligner else 'ASR'
    if not folder.is_dir():
        raise ASRError(f'{role} 模型資料夾不存在：{folder}')
    for name in ('config.json', 'preprocessor_config.json', 'tokenizer_config.json'):
        if not (folder / name).is_file():
            raise ASRError(f'{role} 模型資料夾缺少 {name}。請選擇完整的模型資料夾。')
    try:
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        if config.get('model_type') != 'qwen3_asr':
            raise ASRError(f'{role} 模型資料夾不是 Qwen3-ASR Transformers 模型（需要 model_type=qwen3_asr）。')
        if aligner != ('timestamp_token_id' in config):
            raise ASRError('請選擇 ForcedAligner 模型。' if aligner else '這是 ForcedAligner，請選擇語音辨識模型。')
        index = folder / 'model.safetensors.index.json'
        weights = (set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
                   if index.is_file() else {'model.safetensors'})
        if not weights:
            raise ASRError(f'{role} 模型權重索引為空。')
        for name in weights:
            path = (folder / name).resolve()
            if not path.is_relative_to(folder) or not path.is_file() or path.stat().st_size == 0:
                raise ASRError(f'{role} 模型資料夾缺少或有不合法的權重：{name}')
        if not (folder / 'tokenizer.json').is_file() and not all((folder / n).is_file() for n in ('vocab.json', 'merges.txt')):
            raise ASRError(f'{role} 模型資料夾缺少 tokenizer.json 或 vocab.json／merges.txt。')
    except (ValueError, KeyError, TypeError) as exc:
        raise ASRError(f'{role} 模型設定或權重索引無法解析：{exc}') from exc
    return folder


class LocalASR:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.process = None
        self.identity = self.info = None
        self.log_path = ROOT / 'data' / 'asr-worker.log'

    async def _stop(self):
        process, self.process = self.process, None
        self.identity = self.info = None
        if process and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()

    def _diagnostic(self):
        try:
            with self.log_path.open('rb') as stream:
                stream.seek(max(0, self.log_path.stat().st_size - 2500))
                return stream.read().decode('utf-8', errors='replace').strip()
        except OSError:
            return ''

    async def _read(self):
        line = await self.process.stdout.readline()
        if not line:
            raise ASRError('本機 ASR 程序已結束。' + self._diagnostic())
        try:
            result = json.loads(line)
        except ValueError as exc:
            raise ASRError('本機 ASR 回應格式錯誤。' + self._diagnostic()) from exc
        if 'error' in result:
            raise ASRError(result['error'])
        return result

    async def _ensure(self, settings):
        folder = validate_model_dir(settings.asr_model_dir)
        aligner = validate_model_dir(settings.aligner_model_dir, aligner=True)
        identity = (str(folder), str(aligner), settings.asr_device)
        if self.process and self.process.returncode is None and identity == self.identity:
            return self.info
        await self._stop()
        python = worker_python()
        if not python.is_file():
            raise ASRError('尚未安裝本機 ASR 執行環境，請雙擊 install.cmd（或執行 setup.ps1）。')
        self.log_path.parent.mkdir(exist_ok=True)
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_HUB_DISABLE_TELEMETRY='1', PYTHONIOENCODING='utf-8', TOKENIZERS_PARALLELISM='false')
        with self.log_path.open('wb') as log:
            self.process = await asyncio.create_subprocess_exec(
                str(python), '-u', str(ROOT / 'voice_app' / 'asr_worker.py'),
                '--model-dir', str(folder), '--device', settings.asr_device,
                '--aligner-dir', str(aligner),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=log, cwd=str(ROOT), env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                limit=1024 * 1024)
        try:
            self.info = await asyncio.wait_for(self._read(), timeout=300)
            if self.info.get('type') != 'ready':
                raise ASRError('ASR 未回報載入完成。')
            self.identity = identity
            return self.info
        except BaseException:
            await self._stop()
            raise

    async def load(self, settings: Settings):
        async with self.lock:
            try:
                return await self._ensure(settings)
            except asyncio.TimeoutError as exc:
                raise ASRError('ASR 載入超過 300 秒，請查看 data/asr-worker.log。') from exc

    async def transcribe(self, settings: Settings, samples):
        return (await self.recognize(settings, samples, align=False))['text']

    async def recognize(self, settings: Settings, samples, *, align=True):
        audio = np.asarray(samples, dtype='<f4').reshape(-1)
        if not len(audio):
            return {'text': '', 'items': [], 'language': ''}
        if not np.isfinite(audio).all():
            raise ASRError('音訊包含不合法數值。')
        async with self.lock:
            try:
                await self._ensure(settings)
                request = json.dumps({'audio': base64.b64encode(audio.tobytes()).decode('ascii'),
                                      'align': align, 'context': settings.vocabulary}) + '\n'
                self.process.stdin.write(request.encode('utf-8'))
                await self.process.stdin.drain()
                result = await asyncio.wait_for(self._read(), timeout=settings.timeout_seconds)
                if not isinstance(result.get('text'), str):
                    raise ASRError('ASR 沒有回傳辨識文字。')
                result['text'] = result['text'].strip()
                if align and not isinstance(result.get('items'), list):
                    raise ASRError('ForcedAligner 沒有回傳時間資訊。')
                return result
            except BaseException as exc:
                await self._stop()
                if isinstance(exc, asyncio.TimeoutError):
                    raise ASRError('本機 ASR 辨識逾時，程序已停止；可重試該片段。') from exc
                raise

    async def close(self):
        async with self.lock:
            await self._stop()


local_asr = LocalASR()
