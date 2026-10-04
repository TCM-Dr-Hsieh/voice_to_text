import numpy as np
import pytest

from voice_app.audio import RATE
from voice_app.config import Settings
from voice_app.utterance import WindowChunker


def windows(seconds, block, *, length=12, overlap=1):
    audio = np.arange(round(seconds * RATE), dtype=np.float32)
    chunker = WindowChunker(Settings(window_seconds=length, overlap_seconds=overlap))
    result = []
    for offset in range(0, len(audio), block):
        result.extend(chunker.feed(audio[offset:offset + block]))
    tail = chunker.flush()
    if tail is not None:
        result.append(tail)
    assert chunker.flush() is None
    for chunk in result:
        offset = round(chunk.start * RATE)
        np.testing.assert_array_equal(chunk.samples, audio[offset:offset + len(chunk.samples)])
    return result


@pytest.mark.parametrize('block', [319, 1600, 300000])
def test_fixed_windows_are_independent_of_input_blocks(block):
    result = windows(25, block)
    assert [chunk.start for chunk in result] == [0, 11, 22]
    assert [chunk.duration for chunk in result] == [12, 12, 3]
    assert [chunk.reason for chunk in result] == ['window', 'window', 'stop']
    assert all(chunk.final and chunk.commit_until is None for chunk in result)


def test_exact_window_end_does_not_emit_overlap_only_tail():
    result = windows(12, 1600)
    assert len(result) == 1
    assert result[0].start == 0 and result[0].duration == 12


def test_overlap_above_four_seconds_uses_three_second_stride():
    result = windows(16, 1600, length=9, overlap=6)
    assert [chunk.start for chunk in result] == [0, 3, 6, 9]
    assert [chunk.duration for chunk in result] == [9, 9, 9, 7]


def test_short_audio_is_processed_on_stop():
    result = windows(3.25, 1600)
    assert len(result) == 1
    assert result[0].start == 0 and result[0].duration == 3.25


def test_no_two_second_right_holdback():
    result = windows(11, 1600, length=5, overlap=0)
    assert [chunk.start for chunk in result] == [0, 5, 10]
    assert [chunk.duration for chunk in result] == [5, 5, 1]


def test_silence_does_not_change_fixed_window_boundaries():
    chunker = WindowChunker(Settings(window_seconds=5, overlap_seconds=1))
    result = list(chunker.feed(np.zeros(RATE * 9, dtype=np.float32)))
    assert [(chunk.start, chunk.duration) for chunk in result] == [(0, 5), (4, 5)]
    assert chunker.flush() is None
