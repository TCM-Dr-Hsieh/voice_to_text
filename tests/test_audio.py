import io
import wave

import numpy as np
import pytest

from voice_app.config import Settings

from voice_app.audio import RATE, decode_file, wav_bytes


def test_wav_and_decode_roundtrip(tmp_path):
    samples = np.sin(np.arange(RATE * 2) * 0.04).astype(np.float32) * 0.4
    encoded = wav_bytes(samples)
    with wave.open(io.BytesIO(encoded)) as wav:
        assert (wav.getnchannels(), wav.getframerate(), wav.getnframes()) == (1, RATE, RATE * 2)
    path = tmp_path / 'test.wav'
    path.write_bytes(encoded)
    decoded = np.concatenate(list(decode_file(path)))
    np.testing.assert_allclose(decoded, samples, atol=0.0001)


def test_stereo_48k_resample(tmp_path):
    path = tmp_path / 'stereo.wav'
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(np.zeros((48000, 2), dtype='<i2').tobytes())
    assert sum(len(frame) for frame in decode_file(path)) == RATE


@pytest.mark.asyncio
async def test_completion_queue_race_preserves_last_chunk():
    import queue
    from voice_app.audio import AudioSource, Chunk
    source = AudioSource('file', '', None, settings=Settings())
    final = Chunk(0, 0, np.zeros(RATE))

    class RacingQueue:
        calls = 0
        def get_nowait(self):
            self.calls += 1
            if self.calls == 1:
                source.done.set()
                raise queue.Empty
            return final

    source.queue = RacingQueue()
    assert await source.next_chunk() is final


def test_overflow_tail_stays_in_order():
    from voice_app.audio import AudioSource, Chunk
    source = AudioSource('microphone', '', None, settings=Settings())
    for i in range(20):
        source.queue.put(Chunk(i, i * 3, np.zeros(1)))
    source._put(Chunk(20, 60, np.zeros(1)))
    source.queue.get_nowait()
    source._put(Chunk(21, 63, np.zeros(1)))
    assert [chunk.index for chunk in source.emergency] == [20, 21]
    assert source.stop.is_set() and source.error


@pytest.mark.asyncio
async def test_recording_access_denied_finishes_without_chunks(monkeypatch):
    from voice_app import audio

    def denied(*args):
        raise RuntimeError('Error 0x80070005')

    monkeypatch.setattr(audio, 'record_device', denied)
    source = audio.AudioSource('microphone', 'device', None, settings=Settings())
    source.start()
    assert await source.next_chunk() is None
    assert source.done.is_set()
    assert '0x80070005' in source.error and 'start.cmd' in source.error
    await source.close()


@pytest.mark.asyncio
async def test_file_access_error_does_not_suggest_microphone_permissions(monkeypatch):
    from voice_app import audio

    def denied(*args):
        raise OSError('Error 0x80070005')

    monkeypatch.setattr(audio, 'decode_file', denied)
    source = audio.AudioSource('file', '', 'denied.wav', settings=Settings())
    source.start()
    assert await source.next_chunk() is None
    assert source.error == '音訊來源錯誤：Error 0x80070005'
    await source.close()
