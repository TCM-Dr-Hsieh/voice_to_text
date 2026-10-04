"""Fixed, overlapping audio windows for both live and file sources."""

import numpy as np

from .audio import Chunk, RATE


class WindowChunker:
    """Emit [0, L], [L-Z, 2L-Z], ... and one unfinished tail on flush."""

    def __init__(self, settings):
        self.length = round(settings.window_seconds * RATE)
        self.overlap = round(settings.overlap_seconds * RATE)
        self.stride = self.length - self.overlap
        self.buffer = np.empty(0, dtype=np.float32)
        self.start_sample = 0
        self.received_samples = 0
        self.last_emitted_end = 0
        self.index = 0

    def feed(self, samples):
        incoming = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not len(incoming):
            return
        self.received_samples += len(incoming)
        self.buffer = np.concatenate((self.buffer, incoming))
        while len(self.buffer) >= self.length:
            end = self.start_sample + self.length
            yield Chunk(self.index, self.start_sample / RATE,
                        self.buffer[:self.length].copy(), True, 'window')
            self.index += 1
            self.last_emitted_end = end
            self.buffer = self.buffer[self.stride:].copy()
            self.start_sample += self.stride

    def flush(self):
        if self.received_samples <= self.last_emitted_end or not len(self.buffer):
            return None
        chunk = Chunk(self.index, self.start_sample / RATE,
                      self.buffer.copy(), True, 'stop')
        self.last_emitted_end = self.received_samples
        self.buffer = np.empty(0, dtype=np.float32)
        return chunk
