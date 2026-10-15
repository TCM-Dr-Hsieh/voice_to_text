"""Replay of what was said in a meeting: a short WAV clip cut out of the recording while it is still being written.

In the meeting view the reader clicks the number of a segment (`語音#12`, or `Voice#12` on the translation side) and the page plays the
clip that starts at that audio time and is at most 30 s long, to compare the text with the voice. The browser asks for the clip by its
start (seconds from the start of the recording, the same clock as the times of the exported text) and the meeting's random token.

The clip is cut here, not streamed from the file, for two reasons: the recording is still being written, and the sizes in the header of
a WAV that is not closed yet are stale (a browser given the file itself would show a wrong duration), and a clip of its own length gives
the player's time bar the right range. So the number of samples comes from the size of the file, never from the header, and a clip that
reaches past what has been recorded so far is simply shorter. Cutting is a seek and a read of 30 s (about 1 MB), whatever the length of
the recording: a meeting of two hours (230 MB) costs the same as one of two minutes.

Only the audio of a meeting that is open on some page can be asked for, and only by time: there is no path in the request. The request
must carry the random `replay_token` of the meeting (its rows on the page hold it), so another page, another visitor of the same server
or a page left over from an earlier meeting is refused. A meeting whose audio would be recorded again by the replay is refused too (the
sources that capture what the computer plays: the replay would come out of the same speakers and be written down as new speech).
"""
from __future__ import annotations

import io
import math
import secrets
import struct
import wave
from pathlib import Path

from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from .audio import RATE

REPLAY_PATH = '/meeting-replay.wav'
CLIP_SECONDS = 30.0
SAMPLE_BYTES = 2                    # 16-bit mono, as AudioSource and RemoteAudioSource write it
NO_RECORDING = '這場會議沒有錄音可以回聽（「儲存錄音」是關的，或音訊來源是音訊檔）。'
ECHO = '這個音訊來源擷取的是電腦播放的聲音：回聽的聲音會從同一個輸出播出，被一起錄進逐字稿，所以無法回聽。'
GONE = '這個畫面的會議已經結束或過期，無法回聽；請重新整理頁面。'
MEETINGS: dict[str, object] = {}    # token -> the open meeting (what has `audio_path` and `replay_blocked`)


class ReplayError(Exception):
    """A request that cannot be answered; `status` is the HTTP status to send, the message is shown to the reader."""

    def __init__(self, message: str, status: int = 404):
        super().__init__(message)
        self.status = status


def blocked_reason(kind: str, audio_path: Path | None) -> str:
    """Why the audio of a meeting of this kind cannot be replayed ('' when it can)."""
    if audio_path is None:
        return NO_RECORDING
    if kind in ('loopback', 'remote_system'):
        return ECHO
    return ''


def register(meeting) -> None:
    MEETINGS[meeting.replay_token] = meeting


def unregister(meeting) -> None:
    MEETINGS.pop(getattr(meeting, 'replay_token', ''), None)


def _data_offset(handle) -> tuple[int, int]:
    """(offset of the first sample, sample rate) found by walking the RIFF chunks. The sizes in a header that is still being written
    are not trusted: only where the chunks start."""
    head = handle.read(12)
    if len(head) < 12 or head[:4] != b'RIFF' or head[8:12] != b'WAVE':
        raise ReplayError('錄音檔還沒有可用的內容，請稍後再試。')
    rate = RATE
    position = 12
    while True:
        handle.seek(position)
        chunk = handle.read(8)
        if len(chunk) < 8:
            raise ReplayError('錄音檔還沒有可用的內容，請稍後再試。')
        kind, size = chunk[:4], struct.unpack('<I', chunk[4:])[0]
        if kind == b'fmt ':
            body = handle.read(min(size, 16))
            if len(body) < 16:
                raise ReplayError('錄音檔格式不完整。', 500)
            tag, channels, rate, _, _, bits = struct.unpack('<HHIIHH', body)
            if tag != 1 or channels != 1 or bits != 16:
                raise ReplayError('錄音檔格式不是 16 位元單聲道，無法回放。', 500)
        elif kind == b'data':
            return position + 8, rate
        position += 8 + size + (size & 1)


def clip_wav(path: Path, start: float, seconds: float = CLIP_SECONDS) -> bytes:
    """A complete WAV of up to `seconds` of the recording from `start` (seconds from the start of the recording)."""
    if not math.isfinite(start) or start < 0:
        raise ReplayError('起點時間不合法。', 400)
    try:
        with open(path, 'rb') as handle:
            offset, rate = _data_offset(handle)
            handle.seek(0, 2)
            available = max(0, handle.tell() - offset) // SAMPLE_BYTES
            first = int(start * rate)
            if first >= available:
                raise ReplayError('這個時間點的聲音還沒有錄到。')
            count = min(int(seconds * rate), available - first)
            handle.seek(offset + first * SAMPLE_BYTES)
            pcm = handle.read(count * SAMPLE_BYTES)
    except FileNotFoundError:
        raise ReplayError('錄音檔還不存在。') from None
    out = io.BytesIO()
    with wave.open(out, 'wb') as clip:
        clip.setnchannels(1)
        clip.setsampwidth(SAMPLE_BYTES)
        clip.setframerate(rate)
        clip.writeframes(pcm[:len(pcm) // SAMPLE_BYTES * SAMPLE_BYTES])
    return out.getvalue()


def _refuse(message: str, status: int) -> Response:
    return PlainTextResponse(message, status_code=status, headers={'Cache-Control': 'no-store'})


def replay_response(request: Request, start: float, token: str = '') -> Response:
    """The clip of the meeting that `token` belongs to, as an HTTP response. The `Sec-Fetch-Site` check only rejects what the browser
    itself labels cross-site; the token is what keeps other pages (and stale ones) out."""
    if request.headers.get('sec-fetch-site') == 'cross-site':
        return _refuse('不允許跨網站讀取。', 403)
    meeting = None
    for known, candidate in list(MEETINGS.items()):
        if secrets.compare_digest(known.encode(), str(token).encode()):
            meeting = candidate
    if meeting is None:
        return _refuse(GONE, 409)
    if meeting.replay_blocked:
        return _refuse(meeting.replay_blocked, 409)
    try:
        body = clip_wav(meeting.audio_path, start)
    except ReplayError as exc:
        return _refuse(str(exc), exc.status)
    return Response(body, media_type='audio/wav', headers={'Cache-Control': 'no-store'})


def register_replay_routes(app) -> None:
    """Mount the clip route on a FastAPI-style app (NiceGUI's `app`, or a plain FastAPI app in tests)."""
    @app.get(REPLAY_PATH)
    def replay(request: Request, start: float = 0.0, token: str = ''):
        return replay_response(request, start, token)
