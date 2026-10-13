"""A synthetic meeting for the speaker tests: audio whose samples encode who is speaking, a fake voiceprint model that reads that
back, and a feeder that hands the tracker 6-second windows with a 3-second stride, like the real pipeline."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from voice_app.speaker.units import Token

RATE = 16000
DIM = 32


class FakeEmbedder:
    """Voiceprint = the speaker's base vector plus noise. The audio carries the speaker as a constant sample value (0.1 * (n + 1));
    a value of 0.9 is "noise": a voiceprint in a random direction each time (nobody)."""

    def __init__(self, bases: dict[int, np.ndarray], noise: float = 0.12, seed: int = 0):
        self.bases = bases
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def __call__(self, samples: np.ndarray):
        self.calls += 1
        level = float(np.mean(samples))
        if abs(level) < 0.05:
            return None
        if abs(level - 0.9) < 0.02:
            return self.rng.normal(0, 1, DIM)
        speaker = int(round(level / 0.1)) - 1
        return self.bases[speaker] + self.rng.normal(0, self.noise, DIM)


def bases(count: int, seed: int = 7) -> dict[int, np.ndarray]:
    """Unit base vectors, far apart (random directions in 32 dimensions are nearly orthogonal)."""
    rng = np.random.default_rng(seed)
    out = {}
    for n in range(count):
        v = rng.normal(0, 1, DIM)
        out[n] = v / np.linalg.norm(v)
    return out


@dataclass
class Turn:
    speaker: int
    start: float
    end: float


def make_turns(pattern: list[int], *, seed: int = 1, first: float = 0.5, lo: float = 1.6, hi: float = 5.5, pause: float = 0.7):
    rng = random.Random(seed)
    turns, t = [], first
    for speaker in pattern:
        length = round(rng.uniform(lo, hi) * 4) / 4
        turns.append(Turn(speaker, t, t + length))
        t += length + pause
    return turns


def window_of(t: float) -> int:
    """Which window first covers time t: window 0 is [0, 6), window k >= 1 adds [3k + 3, 3k + 6) (6 s windows, 3 s overlap)."""
    return 0 if t < 6.0 else 1 + int((t - 6.0) // 3.0)


def build(turns: list[Turn], *, step: float = 0.25):
    """(audio, tokens, truth): a token every `step` seconds inside each turn, the last one carrying the sentence mark.
    A token's `seg` is its window number + 1, like the pipeline's segment numbers. Speaker 8 is the noise."""
    total = turns[-1].end + 2.0
    audio = np.zeros(int(total * RATE), dtype=np.float32)
    tokens, truth, counters = [], [], {}
    for turn in turns:
        audio[int(turn.start * RATE):int(turn.end * RATE)] = 0.1 * (turn.speaker + 1)
        t = turn.start
        while t + step <= turn.end + 1e-9:
            seg = window_of(t) + 1
            c0 = counters.get(seg, 0)
            counters[seg] = c0 + 1
            last = t + 2 * step > turn.end + 1e-9
            tokens.append(Token(seg, c0, c0 + 1, t, t + step, last))
            truth.append(turn.speaker)
            t += step
    return audio, tokens, truth


def windows(audio) -> int:
    return int(np.ceil(len(audio) / RATE / 3.0))


def feed(tracker, audio, tokens, first: int = 0, last: int | None = None) -> set[int]:
    """Hand the tracker windows `first`..`last` (inclusive): the audio, then the tokens that window adds. Returns the segments touched."""
    by_window: dict[int, list] = {}
    for token in tokens:
        by_window.setdefault(window_of(token.start), []).append(token)
    changed: set[int] = set()
    for k in range(first, (windows(audio) if last is None else last + 1)):
        tracker.add_audio(3.0 * k, audio[int(3.0 * k * RATE):int((3.0 * k + 6.0) * RATE)])
        changed |= tracker.add_tokens(by_window.get(k, []))
    return changed


def accuracy(tracker, truth: list[int]) -> tuple[float, float, dict]:
    """(coverage, precision, group -> speaker): the share of tokens labelled, and how many of those carry the group their speaker mostly got."""
    votes: dict[int, dict] = {}
    for i, speaker in enumerate(truth):
        g = tracker.voice_group_of(i)
        if g is not None:
            votes.setdefault(speaker, {}).setdefault(g, 0)
            votes[speaker][g] += 1
    mapping = {s: max(c, key=c.get) for s, c in votes.items()}
    labelled = right = 0
    for i, speaker in enumerate(truth):
        g = tracker.voice_group_of(i)
        if g is not None:
            labelled += 1
            right += mapping.get(speaker) == g
    return labelled / max(len(truth), 1), right / max(labelled, 1), mapping
