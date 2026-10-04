"""Receive browser audio through the existing chunk/recording pipeline."""
import asyncio
import json
import queue
import struct
import time
from urllib.parse import urlsplit
import wave

import av
import numpy as np
from nicegui import app, ui
from starlette.websockets import WebSocket
from starlette.websockets import WebSocketDisconnect

from .audio import AudioSource, RATE
from .api import ModelAPI
from .config import ROOT


REMOTE_PAGES = {}
STOP_ACK_SECONDS = 8.0
RECONNECT_SECONDS = 30.0
STALE_SOCKET_SECONDS = 20.0
STARTUP_SECONDS = 360.0
PREPARING_INTERVAL_SECONDS = 5.0
app.add_static_file(local_file=ROOT / 'voice_app' / 'remote_capture.js', url_path='/remote_capture.js',
                    max_cache_age=0)
ui.add_head_html('<script src="/remote_capture.js?v=8"></script>', shared=True)


class RemoteAudioSource(AudioSource):
    def __init__(self, kind, *, settings, recording_path=None):
        super().__init__(kind, '', None, settings=settings, recording_path=recording_path)
        self.resampler = None
        self.recording = None
        self.connected = False
        self.session_id = ''
        self.next_sequence = 0
        self.socket_active = False
        self.socket_owner = None
        self.reconnect_task: asyncio.Task | None = None

    def attach(self, session_id: str, owner) -> int:
        if (not session_id or session_id != self.session_id or
                self.done.is_set() or self.cancelled.is_set()):
            raise ValueError('無法恢復此段遠端音訊。')
        if self.reconnect_task is not None:
            self.reconnect_task.cancel()
            self.reconnect_task = None
        self.socket_active = True
        self.socket_owner = owner
        return self.next_sequence

    def suspend(self, owner):
        if self.socket_owner is not owner:
            return
        self.socket_active = False
        self.socket_owner = None
        if self.done.is_set() or self.cancelled.is_set() or self.reconnect_task is not None:
            return

        async def expire():
            task = asyncio.current_task()
            try:
                await asyncio.sleep(RECONNECT_SECONDS)
                if not self.socket_active and not self.done.is_set():
                    self.finish('遠端音訊超過 30 秒未重新連線；已收到的音訊標示未完成。')
            except asyncio.CancelledError:
                pass
            finally:
                if self.reconnect_task is task:
                    self.reconnect_task = None

        self.reconnect_task = asyncio.create_task(expire())

    def start(self):
        if self.recording_path:
            self.recording = wave.open(str(self.recording_path), 'wb')
            self.recording.setnchannels(1)
            self.recording.setsampwidth(2)
            self.recording.setframerate(RATE)

    def _put(self, chunk):
        if self.cancelled.is_set():
            return
        if self.emergency:
            self.emergency.append(chunk)
            return
        try:
            self.queue.put_nowait(chunk)
        except queue.Full:
            if chunk.final:
                self.emergency.append(chunk)
                self.error = '模型處理速度落後，待處理片段已滿；已停止錄音，正在完成已接收片段。'
                self.stop.set()

    def connect(self, sample_rate: int):
        if self.connected or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 192000:
            raise ValueError('遠端音訊取樣率不合法或來源已連線。')
        self.resampler = av.AudioResampler(format='fltp', layout='mono', rate=RATE)
        self.sample_rate = sample_rate
        self.connected = True

    def _accept(self, samples):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not len(samples):
            return
        if self.recording is not None:
            self.recording.writeframesraw((np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes())
            self.recorded_samples += len(samples)
        self.level = min(1.0, float(np.sqrt(np.mean(samples ** 2))) * 5)
        for chunk in self.chunker.feed(samples):
            self._put(chunk)

    def receive(self, payload: bytes):
        if self.done.is_set() or self.cancelled.is_set():
            return
        if not self.connected or not payload or len(payload) > 65536 or len(payload) % 4:
            raise ValueError('遠端音訊封包格式或長度不合法。')
        samples = np.frombuffer(payload, dtype='<f4')
        if not np.isfinite(samples).all():
            raise ValueError('遠端音訊包含不合法數值。')
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format='flt', layout='mono')
        frame.sample_rate = self.sample_rate
        for converted in self.resampler.resample(frame):
            self._accept(converted.to_ndarray())

    def receive_packet(self, payload: bytes) -> int:
        if len(payload) < 8:
            raise ValueError('遠端音訊封包缺少序號或聲音資料。')
        sequence = struct.unpack_from('<I', payload)[0]
        if sequence < self.next_sequence:
            return self.next_sequence - 1  # An ACK was lost; do not write duplicated audio.
        if sequence != self.next_sequence:
            raise ValueError('遠端音訊封包順序不連續。')
        self.receive(payload[4:])
        self.next_sequence += 1
        return sequence

    def finish(self, error=''):
        if self.done.is_set():
            return
        if self.reconnect_task is not None:
            self.reconnect_task.cancel()
            self.reconnect_task = None
        self.socket_active = False
        self.socket_owner = None
        if error:
            self.error = error
        try:
            if not self.cancelled.is_set():
                if self.resampler is not None:
                    for converted in self.resampler.resample(None):
                        self._accept(converted.to_ndarray())
                tail = self.chunker.flush()
                if tail is not None:
                    self._put(tail)
        except Exception as exc:
            self.error = f'遠端音訊收尾失敗：{exc}'
        finally:
            if self.recording is not None:
                self.recording.close()
                self.recording = None
            self.level = 0.0
            self.done.set()

    async def close(self):
        if self.reconnect_task is not None:
            self.reconnect_task.cancel()
            self.reconnect_task = None
        self.socket_active = False
        self.socket_owner = None
        self.cancelled.set()
        self.stop.set()
        if self.recording is not None:
            self.recording.close()
            self.recording = None
        self.level = 0.0
        self.done.set()


@app.websocket('/remote-audio/{token}')
async def remote_audio_socket(websocket: WebSocket, token: str):
    state = REMOTE_PAGES.get(token)
    origin = websocket.headers.get('origin', '')
    if not state or not origin or urlsplit(origin).netloc != websocket.headers.get('host'):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    source = None
    owns_source = False
    loading_claimed = False
    try:
        hello = json.loads(await asyncio.wait_for(websocket.receive_text(), 15))
        kind = hello.get('kind')
        session_id = hello.get('session_id')
        if (kind not in ('remote_microphone', 'remote_system') or
                not isinstance(session_id, str) or not 8 <= len(session_id) <= 80):
            raise ValueError('遠端音訊來源不合法。')
        candidate = state.session.source
        rejoining_start = (hello.get('type') == 'start' and
                           ((state.loading and getattr(state, 'remote_start_id', '') == session_id) or
                            (isinstance(candidate, RemoteAudioSource) and
                             candidate.session_id == session_id)))
        if hello.get('type') == 'resume' or rejoining_start:
            while (state.loading and getattr(state, 'remote_start_id', '') == session_id and
                   (not isinstance(candidate, RemoteAudioSource) or
                    candidate.session_id != session_id)):
                try:
                    await asyncio.wait_for(websocket.send_json({'type': 'preparing'}), 1)
                except (OSError, RuntimeError, WebSocketDisconnect, asyncio.TimeoutError):
                    return
                await asyncio.sleep(PREPARING_INTERVAL_SECONDS)
                if REMOTE_PAGES.get(token) is not state:
                    return
                candidate = state.session.source
            if (not isinstance(candidate, RemoteAudioSource) or candidate.kind != kind or
                    candidate.session_id != session_id):
                raise ValueError('找不到可恢復的遠端音訊。')
            source = candidate
            if source.done.is_set():
                await websocket.send_json({'type': 'finished'})
                return
            next_sequence = source.attach(session_id, websocket)
            owns_source = True
            try:
                await websocket.send_json({'type': 'resumed', 'next_sequence': next_sequence})
            except (OSError, RuntimeError, WebSocketDisconnect):
                source.suspend(websocket)
                return
            state.session.status = '遠端音訊已重連，正在補傳暫存聲音'
        elif hello.get('type') == 'start':
            if (state.loading or state.session.active or
                    getattr(state, 'writing_pending', False)):
                raise ValueError('已有另一段轉錄進行中。')
            if state.controls is None:
                raise ValueError('頁面控制項尚未準備好。')
            if state.controls.source.value != kind:
                raise ValueError('音訊來源已變更，請重新按開始轉錄。')
            state.loading = loading_claimed = True
            state.remote_start_id = session_id
            writing_context = None
            if getattr(state, 'writing_mode', False):
                anchor = hello.get('writing_context')
                if (not isinstance(anchor, dict) or
                        not all(isinstance(anchor.get(key), str) and len(anchor[key]) <= 20000
                                for key in ('before', 'after'))):
                    raise ValueError('語音書寫的文章插入位置不合法。')
                writing_context = (anchor['before'], anchor['after'])
            state.settings.validate()
            state.session.status = '載入本機 ASR 與 ForcedAligner，準備接收遠端音訊'
            async with ModelAPI(state.settings) as api:
                prepare_task = asyncio.create_task(asyncio.wait_for(api.prepare_asr(), STARTUP_SECONDS))
                try:
                    while not prepare_task.done():
                        try:
                            await asyncio.wait_for(websocket.send_json({'type': 'preparing'}), 1)
                        except (OSError, RuntimeError, WebSocketDisconnect, asyncio.TimeoutError):
                            pass  # Model loading continues for a replacement socket.
                        await asyncio.wait({prepare_task}, timeout=PREPARING_INTERVAL_SECONDS)
                    try:
                        await prepare_task
                    except asyncio.TimeoutError as exc:
                        raise ValueError('ASR 準備逾時，請檢查模型與執行環境。') from exc
                finally:
                    if not prepare_task.done():
                        prepare_task.cancel()
                        await asyncio.gather(prepare_task, return_exceptions=True)
            if (REMOTE_PAGES.get(token) is not state or
                    state.controls.source.value != kind):
                raise ValueError('頁面已關閉或音訊來源已變更，未啟動遠端錄音。')
            state.session.start(state.settings, kind,
                                record_audio=bool(state.controls.record_switch.value),
                                writing_context=writing_context)
            state.writing_pending = bool(getattr(state, 'writing_mode', False))
            source = state.session.source
            assert isinstance(source, RemoteAudioSource)
            source.session_id = session_id
            source.socket_active = True
            source.socket_owner = websocket
            owns_source = True
            try:
                await websocket.send_json({'type': 'ready', 'next_sequence': 0})
            except (OSError, RuntimeError, WebSocketDisconnect):
                source.suspend(websocket)
                return
        else:
            raise ValueError('遠端音訊控制訊息不合法。')
        stop_sent_at = None
        ended_normally = False
        source_error = ''
        disconnected = False
        last_received = time.monotonic()
        while not source.cancelled.is_set() and not source.done.is_set():
            if source.socket_owner is not websocket:
                break
            if time.monotonic() - last_received > STALE_SOCKET_SECONDS:
                disconnected = True
                break
            if source.stop.is_set() and stop_sent_at is None:
                try:
                    await websocket.send_json({'type': 'stop'})
                except (OSError, RuntimeError, WebSocketDisconnect):
                    disconnected = True
                    break
                stop_sent_at = time.monotonic()
            if stop_sent_at is not None and time.monotonic() - stop_sent_at > STOP_ACK_SECONDS:
                source_error = '遠端停止時未收到音訊末尾；錄音標示未完成。'
                break
            try:
                message = await asyncio.wait_for(websocket.receive(), .25)
            except asyncio.TimeoutError:
                continue
            except (OSError, RuntimeError, WebSocketDisconnect):
                disconnected = True
                break
            if message['type'] == 'websocket.disconnect':
                disconnected = True
                break
            if source.socket_owner is not websocket:
                break
            last_received = time.monotonic()
            if message.get('bytes') is not None:
                sequence = source.receive_packet(message['bytes'])
                if stop_sent_at is not None:
                    stop_sent_at = time.monotonic()
                try:
                    await websocket.send_json({'type': 'ack', 'sequence': sequence})
                except (OSError, RuntimeError, WebSocketDisconnect):
                    disconnected = True
                    break
                continue
            if message.get('text') is None:
                continue
            control = json.loads(message['text'])
            if control.get('type') == 'format':
                sample_rate = control.get('sample_rate')
                if not source.connected:
                    source.connect(sample_rate)
                elif sample_rate != source.sample_rate:
                    raise ValueError('遠端音訊取樣率在重連後改變。')
            elif control.get('type') == 'stop':
                ended_normally = True
                break
            elif control.get('type') == 'ping':
                try:
                    await websocket.send_json({'type': 'pong'})
                except (OSError, RuntimeError, WebSocketDisconnect):
                    disconnected = True
                    break
            elif control.get('type') == 'error':
                source_error = '遠端音訊錯誤：' + str(control.get('message', '未知錯誤'))[:200]
                break
            else:
                raise ValueError('遠端音訊控制訊息不合法。')
        if source.socket_owner is not websocket:
            pass  # A newer socket now owns this session.
        elif source.cancelled.is_set():
            try:
                await websocket.send_json({'type': 'cancel'})
            except (OSError, RuntimeError, WebSocketDisconnect):
                pass
        elif disconnected:
            source.suspend(websocket)
            state.session.status = '遠端音訊暫時斷線，等待 30 秒內重連'
        else:
            source.finish(source_error if not ended_normally else '')
            try:
                await websocket.send_json({'type': 'finished'})
            except (OSError, RuntimeError, WebSocketDisconnect):
                pass
    except Exception as exc:
        message = f'遠端音訊失敗：{exc}'
        if owns_source and source is not None and source.socket_owner is websocket:
            source.finish(message)
        elif loading_claimed:
            state.session.error = message
            state.session.status = '無法開始遠端音訊'
        try:
            await websocket.send_json({'type': 'error', 'message': message})
        except Exception:
            pass
    finally:
        if owns_source and source is not None and not source.done.is_set() and not source.cancelled.is_set():
            source.suspend(websocket)
        if loading_claimed:
            state.loading = False
            state.remote_start_id = ''
        try:
            await websocket.close()
        except Exception:
            pass
