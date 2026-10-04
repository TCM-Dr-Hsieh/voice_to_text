from dataclasses import dataclass
import asyncio
from contextlib import nullcontext
from datetime import datetime
import io
from pathlib import Path
import queue
import threading
import wave

import av
import numpy as np

from .config import ROOT

RATE = 16000
HARDWARE_LOCK = threading.Lock()
RECORDINGS_DIR = ROOT / 'data' / 'recordings'


def reserve_recording_path(kind: str, directory: Path = RECORDINGS_DIR, now: datetime | None = None) -> Path:
    prefix = {'microphone': '麥克風', 'loopback': '電腦音訊',
              'remote_microphone': '遠端麥克風', 'remote_system': '遠端電腦音訊'}[kind]
    directory.mkdir(parents=True, exist_ok=True)
    day = (now or datetime.now()).strftime('%Y%m%d')
    number = 1
    while True:
        path = directory / f'{prefix}-{day}-{number:03d}.wav'
        if path.with_suffix('.json').exists():
            number += 1
            continue
        try:
            with path.open('xb'):
                return path
        except FileExistsError:
            number += 1


@dataclass
class Chunk:
    index: int
    start: float
    samples: np.ndarray
    final: bool = True
    reason: str = 'stop'
    commit_until: float | None = None

    @property
    def duration(self):
        return len(self.samples) / RATE


def wav_bytes(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes((np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes())
    return buffer.getvalue()


def decode_file(path: Path):
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError('檔案沒有可讀取的音軌。')
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=RATE)
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                yield converted.to_ndarray().reshape(-1)
        for converted in resampler.resample(None):
            yield converted.to_ndarray().reshape(-1)


def devices(kind: str) -> dict[str, str]:
    import soundcard as sc
    if kind == 'loopback':
        return {d.id: d.name for d in sc.all_speakers()}
    return {d.id: d.name for d in sc.all_microphones(include_loopback=False)}


def record_device(kind: str, device_id: str, stop: threading.Event):
    import soundcard as sc
    if not HARDWARE_LOCK.acquire(blocking=False):
        raise RuntimeError('已有其他視窗正在錄音，請先停止該工作。')
    try:
        mic = sc.get_microphone(id=device_id, include_loopback=kind == 'loopback')
        # SoundCard's WASAPI backend has a known single-channel capture issue.
        # Request all physical channels, then downmix explicitly.
        with mic.recorder(samplerate=RATE, channels=None, blocksize=1600) as recorder:
            while not stop.is_set():
                samples = recorder.record(numframes=1600)
                yield samples.mean(axis=1).astype(np.float32)
    finally:
        HARDWARE_LOCK.release()


class AudioSource:
    """Bounded producer. File decoding waits; live overflow stops capture explicitly."""
    def __init__(self, kind, device_id, path, *, settings,
                 recording_path: Path | None = None):
        self.kind, self.device_id, self.path = kind, device_id, path
        self.recording_path = recording_path
        self.recorded_samples = 0
        from .utterance import WindowChunker
        self.chunker = WindowChunker(settings)
        self.queue: queue.Queue[Chunk] = queue.Queue(maxsize=20)
        self.emergency: list[Chunk] = []
        self.stop = threading.Event()
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.error = ''
        self.level = 0.0
        self.thread = threading.Thread(target=self._produce, daemon=True)

    def start(self):
        self.thread.start()

    def _put(self, chunk):
        if self.emergency:
            self.emergency.append(chunk)
            return
        while not self.cancelled.is_set():
            try:
                self.queue.put(chunk, timeout=0.1)
                return
            except queue.Full:
                if not chunk.final:
                    return  # An obsolete preview never displaces final audio.
                if self.kind != 'file' or self.stop.is_set():
                    self.emergency.append(chunk)
                    if self.kind != 'file':
                        self.error = '模型處理速度落後，待處理片段已滿；已停止錄音，正在完成已接收片段。'
                    self.stop.set()
                    return

    def _produce(self):
        try:
            context = wave.open(str(self.recording_path), 'wb') if self.recording_path else nullcontext()
            with context as recording:
                if recording is not None:
                    recording.setnchannels(1)
                    recording.setsampwidth(2)
                    recording.setframerate(RATE)
                iterator = (decode_file(self.path) if self.kind == 'file' else
                            record_device(self.kind, self.device_id, self.stop))
                try:
                    for samples in iterator:
                        if self.cancelled.is_set() or (self.kind == 'file' and self.stop.is_set()):
                            break
                        if recording is not None:
                            recording.writeframesraw((np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes())
                            self.recorded_samples += len(samples)
                        self.level = min(1.0, float(np.sqrt(np.mean(samples ** 2))) * 5)
                        for chunk in self.chunker.feed(samples):
                            self._put(chunk)
                        if self.stop.is_set():
                            break
                finally:
                    iterator.close()
        except Exception as exc:
            self.error = f'音訊來源錯誤：{exc}'
            if self.kind != 'file' and '0x80070005' in str(exc).lower():
                self.error += (
                    '\nWindows 拒絕存取錄音裝置。若服務由受限帳號或沙箱啟動，'
                    '請先關閉服務，再以你自己的 Windows 帳號雙擊 start.cmd。'
                    '\n若仍失敗，請到 Windows「設定 → 隱私權與安全性 → 麥克風」，'
                    '確認已允許麥克風存取及桌面應用程式存取麥克風，再按「開始轉錄」。'
                )
        finally:
            if not self.cancelled.is_set():
                tail = self.chunker.flush()
                if tail is not None:
                    self._put(tail)
            self.level = 0
            self.done.set()

    async def next_chunk(self):
        while True:
            try:
                return self.queue.get_nowait()
            except queue.Empty:
                if self.done.is_set():
                    # Recheck after the producer's completion barrier to avoid
                    # losing a final chunk enqueued after the first queue read.
                    try:
                        return self.queue.get_nowait()
                    except queue.Empty:
                        return self.emergency.pop(0) if self.emergency else None
                await asyncio.sleep(0.05)

    async def close(self):
        self.cancelled.set()
        self.stop.set()
        if self.thread.is_alive():
            await asyncio.to_thread(self.thread.join, 3)
