"""Who is speaking: voice groups found while the meeting runs, any number of people. Pure computation, no I/O, no LLM.

Flow: aligned characters -> word-sized units -> one voiceprint per unit -> per sentence, the voice group it is closest to.

Open set. Nobody is known at the start. A unit that sounds like no known voice is an "orphan"; when enough orphans sound alike
(a dense little cloud of voiceprints) they become a new voice group. So the number of people is not fixed: a host, a lecturer and
students who ask one question each all get a group when they have said enough.

Constant cost. Everything done per sentence or per minute looks at a bounded amount of data: a running centre per group, a
capped sample of voiceprints per group (to keep the centre clean), the orphans of the last minutes and the sentences still
unknown. Nothing re-clusters the whole history, so a two-hour meeting costs the same per minute as the first minute.

A sentence is labelled with its group when it is clearly closer to that group than to the runner-up, otherwise it stays
"unknown" (a later step may fill it in from the text). Labels are stored per voice group number; merging two groups or naming a
group changes only how they are shown, so everything already decided moves with it.

Thread model: mutators take a lock and may run in a worker thread; readers (`labels_for`, `speaker_info`, ...) never wait, they
read immutable values that are replaced as a whole. Voiceprints live in memory only and are never written anywhere.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import BACKGROUND, UNKNOWN
from .mathutil import unit_rows
from .units import DEFAULT_PARAMS as DEFAULT_UNIT_PARAMS, Token, UnitParams, UnitStream

RATE = 16000


@dataclass(frozen=True)
class Params:
    unknown_percentile: float = 10.0     # with 2+ groups: a sentence whose lead over the runner-up is below this percentile of the past is unknown
    refit_seconds: float = 60.0          # the minute's housekeeping runs once per minute of audio
    first_units: int = 6                 # a new group needs this many orphan voiceprints that sound alike ...
    first_seconds: float = 4.0           # ... that add up to this many seconds of speech
    first_sentences: int = 2             # ... and come from at least this many different sentences (one noisy burst is not a person)
    link_cos: float = 0.50               # orphans this similar to the densest orphan belong to the same new voice
    tight_cos: float = 0.62              # ... and on average this similar to their own centre, else it is just noise
    new_cos: float = 0.45                # a sentence closer than this to every group belongs to nobody known yet
    assign_cos: float = 0.55             # an orphan joins a group (new or old) when at least this similar to its centre
    update_cos: float = 0.55             # units this similar to the group they were given to move its centre
    merge_cos: float = 0.88              # two groups whose centres are this similar are one person
    max_speakers: int = 12
    max_sentence_seconds: float = 15.0   # a sentence that runs on without a pause or full stop is cut here, so labels do not wait
    reservoir: int = 200                 # voiceprints kept per group to re-estimate its centre
    trim_cos: float = 0.40               # when re-estimating, voiceprints less similar than this to the centre are left out
    orphan_keep_seconds: float = 600.0
    max_orphans: int = 500
    rescore_seconds: float = 900.0       # unknown sentences this recent are looked at again after every change of the groups
    history_size: int = 1000
    min_history: int = 8                 # past sentences needed before the percentile is taken from the past
    abs_conf: float = 0.03               # lead over the runner-up below which a sentence is never labelled
    fill_after: int = 10                 # a text fill waits until this many later sentences are decided (they are its context)
    fill_batch: int = 6                  # ... and until this many sentences are waiting, or the oldest has waited `fill_max_age` seconds
    fill_max_age: float = 120.0
    fill2_batch: int = 12                # the second text fill waits until this many sentences are waiting for it
    fill2_refits: int = 1                # ... and each of them went to the first text fill at least this many minutes ago
    background_rel_db: float = 6.0       # background talk: a sentence this many dB quieter than the usual speech level ...
    background_max_cos: float = 0.45     # ... AND fitting no voice group better than this (mean best cosine of its units); <= 0 switches it off
    background_window_units: int = 300   # the usual level is the median of the latest this many measured units (about four minutes of speech) ...
    background_min_units: int = 40       # ... and nothing is judged before this many measured units were heard (about the first minute)
    quiet_first_seconds: float = 12.0    # a new voice made mostly of background-like (quiet, unknown) sentences needs this much speech ...
    quiet_first_sentences: int = 4       # ... from at least this many sentences: somebody who really takes part keeps talking, a whisper does not
    units: UnitParams = DEFAULT_UNIT_PARAMS


class AudioBuffer:
    """Recent audio by absolute time. ASR windows overlap, so a new window overwrites the overlap with the same samples."""

    def __init__(self, keep_seconds: float = 40.0):
        self.keep = int(keep_seconds * RATE)
        self.start = 0.0
        self.samples = np.empty(0, dtype=np.float32)

    def add(self, start: float, samples: np.ndarray):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        end = self.start + len(self.samples) / RATE
        if not len(self.samples) or start > end + 0.05 or start < self.start:
            self.start, self.samples = start, samples.copy()
        else:
            cut = int(round((start - self.start) * RATE))
            self.samples = np.concatenate((self.samples[:cut], samples))
        if len(self.samples) > self.keep:
            drop = len(self.samples) - self.keep
            self.start += drop / RATE
            self.samples = self.samples[drop:]

    def slice(self, t0: float, t1: float) -> np.ndarray | None:
        a, b = int(round((t0 - self.start) * RATE)), int(round((t1 - self.start) * RATE))
        if 0 <= a < b <= len(self.samples) + 8:          # (a word that ends with the window may round a sample or two past it)
            return self.samples[a:min(b, len(self.samples))]
        return None


@dataclass
class Unit:
    first: int                   # token indexes first..last
    last: int
    t0: float
    t1: float
    sentence: int
    vec: np.ndarray | None       # unit-length voiceprint, None when it could not be measured
    level: float = float('nan')  # mean-square level of its audio in dBFS, nan when the audio was not at hand

    @property
    def seconds(self) -> float:
        return self.t1 - self.t0


@dataclass
class Sentence:
    first: int
    last: int
    t0: float
    t1: float
    closed: bool = False
    units: list[int] = field(default_factory=list)
    decided: bool = False
    gid: int | None = None       # the voice group it is labelled with; None = unknown
    lean: int | None = None      # the group it sounds most like (even when too weak to label); None = no usable voiceprint
    best: float = float('nan')   # mean similarity to that group
    source: str = ''             # 'voice' | 'text' (first text fill) | 'text2' (second text fill)
    asked: bool = False          # already shown to the first text fill
    said: int | None = None      # the group the first text fill answered when that answer was not accepted
    said_refit: int = 0
    asked2: bool = False         # already shown to the second text fill
    auto_bg: bool = False        # the program judges it background talk (quiet, and like nobody known)
    manual_bg: bool | None = None   # the user's word on it: True = background, False = not, None = leave it to the program


class Voice:
    """One voice group: a running centre and a capped, uniformly sampled set of the voiceprints it was given."""

    def __init__(self, gid: int, cap: int):
        self.gid = gid
        self.cap = cap
        self.sum: np.ndarray = np.zeros(0)
        self.centroid: np.ndarray = np.zeros(0)
        self.seconds = 0.0
        self.units = 0
        self.seen = 0
        self.pool: list[np.ndarray] = []
        self.first_t = self.last_t = 0.0
        self._rng = np.random.default_rng(gid + 17)

    def add(self, vec: np.ndarray, seconds: float, t: float):
        if not self.units:
            self.first_t = t
        self.last_t = max(self.last_t, t)
        self.sum = self.sum + vec * seconds if self.units else vec * seconds
        self.centroid = unit_rows(self.sum)
        self.seconds += seconds
        self.units += 1
        self.seen += 1
        if len(self.pool) < self.cap:
            self.pool.append(vec)
        else:
            j = int(self._rng.integers(self.seen))
            if j < self.cap:
                self.pool[j] = vec

    def refit(self, trim_cos: float):
        """The centre from the sampled voiceprints, leaving out the ones far from it (a stray unit of somebody else)."""
        if len(self.pool) < 4:
            return
        X = np.stack(self.pool)
        c = unit_rows(X.mean(axis=0))
        for _ in range(2):
            keep = X[X @ c >= trim_cos]
            if len(keep) < max(3, len(X) // 4):
                break
            c = unit_rows(keep.mean(axis=0))
        self.centroid = c
        self.sum = c * self.seconds

    def absorb(self, other: 'Voice'):
        self.sum = self.sum + other.sum
        self.centroid = unit_rows(self.sum)
        self.seconds += other.seconds
        self.units += other.units
        self.first_t = min(self.first_t, other.first_t)
        self.last_t = max(self.last_t, other.last_t)
        merged = self.pool + other.pool
        if len(merged) > self.cap:
            keep = self._rng.permutation(len(merged))[:self.cap]
            merged = [merged[i] for i in sorted(keep)]
        self.pool = merged
        self.seen += other.seen


class SpeakerTracker:
    def __init__(self, embed: Callable[[np.ndarray], np.ndarray | None], params: Params | None = None):
        self._embed = embed
        self.params = params or Params()
        self._lock = threading.RLock()
        self.audio = AudioBuffer()
        self.stream = UnitStream(self.params.units)
        self.units: list[Unit] = []
        self.sentences: list[Sentence] = []
        self._sentence_of: list[int] = []            # token index -> sentence index
        self._unit_of: dict[int, int] = {}           # token index -> unit index
        self._seg_tokens: dict[int, list[int]] = {}
        self._voices: dict[int, Voice] = {}          # active voice groups (a merged one is gone, see `_alias`)
        self._alias: dict[int, int] = {}             # merged group -> the group it became part of
        self._next_gid = 0
        self._ids: list[int] = []
        self._matrix = np.zeros((0, 0))
        self._orphans: list[int] = []                # unit numbers that sound like nobody known
        self._orphan_set: set[int] = set()
        self._history: list[float] = []              # leads over the runner-up of the sentences decided so far
        self._ref_level: float | None = None         # the usual speech level (dBFS) of the last minutes: the yardstick for "quieter"
        self._token_audio: dict[int, np.ndarray] = {}   # audio of the tokens not settled yet: a unit is only closed by the NEXT word
        self._next_horizon = self.params.refit_seconds
        self._cursor = 0                             # sentences before it are decided
        self._view: dict[int, tuple] = {}
        self.events: list[dict] = []
        self.minutes: list[dict] = []
        self.version = 0                             # moves whenever the groups change (built, merged)
        self.rescored = 0
        self.finished = False

    # ------------------------------------------------------------------ read side (never waits)
    @property
    def tokens(self) -> list[Token]:
        return self.stream.tokens

    @property
    def group_count(self) -> int:
        return len(self._ids)

    @property
    def group_ids(self) -> list[int]:
        return list(self._ids)

    def resolve(self, gid: int | None) -> int | None:
        while gid in self._alias:
            gid = self._alias[gid]
        return gid

    def labels_for(self, seg: int) -> tuple | None:
        """((c0, c1, group, source), ...) for the segment's tokens in order, or None when the segment has no timed text."""
        return self._view.get(seg)

    def all_labels(self) -> dict[int, tuple]:
        return dict(self._view)

    def segment_token_indexes(self, seg: int) -> list[int]:
        return list(self._seg_tokens.get(seg, []))

    def sentence_range(self, index: int) -> tuple[int, int]:
        sentence = self.sentences[index]
        return sentence.first, sentence.last

    def voice_group_of(self, token: int) -> int | None:
        """The group a token's sentence is labelled with now, None while unknown."""
        s = self.sentences[self._sentence_of[token]]
        return self.resolve(s.gid) if s.decided else None

    def speaker_info(self) -> list[dict]:
        """Per active group: id, seconds and share of the speech heard, number of sentences and two recent example sentences."""
        decided = self.sentences[:self._cursor]
        total = sum(v.seconds for v in self._voices.values()) or 1.0
        out = []
        for gid in self._ids:
            voice = self._voices[gid]
            mine = [i for i, s in enumerate(decided) if self.resolve(s.gid) == gid]
            examples = [i for i in reversed(mine) if self.sentences[i].source == 'voice' and self.sentences[i].last - self.sentences[i].first >= 2][:2]
            merged = sorted(g for g in self._alias if self.resolve(g) == gid)
            out.append({'gid': gid, 'seconds': voice.seconds, 'share': voice.seconds / total, 'units': voice.units,
                        'sentences': len(mine), 'examples': tuple(examples), 'merged': merged,
                        'first_t': voice.first_t, 'last_t': voice.last_t})
        return out

    def stats(self) -> dict:
        decided = self.sentences[:self._cursor]
        background = [s for s in decided if self._is_background(s)]
        labelled = [s for s in decided if s.gid is not None and not self._is_background(s)]
        return {'sentences': len(self.sentences), 'decided': len(decided), 'labelled': len(labelled),
                'unknown': len(decided) - len(labelled) - len(background), 'background': len(background), 'groups': self.group_count,
                'text_filled': sum(1 for s in labelled if s.source == 'text'),
                'text_filled2': sum(1 for s in labelled if s.source == 'text2'),
                'rescored': self.rescored, 'orphans': len(self._orphans),
                'units': len(self.units), 'measured_units': sum(1 for u in self.units if u.vec is not None)}

    # ------------------------------------------------------------------ feeding
    def add_audio(self, start: float, samples: np.ndarray):
        with self._lock:
            self.audio.add(start, samples)

    def add_tokens(self, tokens: list[Token]) -> set[int]:
        """New aligned text, in time order. Returns the segment numbers whose labels may have changed."""
        with self._lock:
            base = len(self.stream.tokens)
            self.stream.extend(tokens)
            touched = set()
            for k, token in enumerate(tokens):
                self._add_token(base + k)
                touched.add(token.seg)
                piece = self.audio.slice(token.start, token.end) if token.end > token.start else np.empty(0, dtype=np.float32)
                if piece is not None:                                      # the audio is at hand now; it may be gone when the unit closes
                    self._token_audio[base + k] = piece.copy()
            self._take_units(self.stream.emit())
            return self._advance(touched)

    def finish(self) -> set[int]:
        """No more text: close the open sentence and decide whatever is left."""
        with self._lock:
            if self.finished:
                return set()
            self.finished = True
            if self.sentences:
                self.sentences[-1].closed = True
            self._take_units(self.stream.finish())
            self._token_audio.clear()
            return self._advance(set())

    def _add_token(self, i: int):
        token = self.stream.tokens[i]
        p = self.params
        if self.sentences and not self.sentences[-1].closed:
            last = self.sentences[-1]
            if token.start - self.stream.tokens[i - 1].end >= p.units.sent_pause or token.end - last.t0 > p.max_sentence_seconds:
                last.closed = True
        if not self.sentences or self.sentences[-1].closed:
            self.sentences.append(Sentence(i, i, token.start, token.end))
        sentence = self.sentences[-1]
        sentence.last, sentence.t1 = i, token.end
        self._sentence_of.append(len(self.sentences) - 1)
        self._seg_tokens.setdefault(token.seg, []).append(i)
        if token.sentence_end:
            sentence.closed = True

    def _take_units(self, finished: list[list[int]]):
        tokens = self.stream.tokens
        for group in finished:
            first, last = group[0], group[-1]
            t0, t1 = tokens[first].start, tokens[last].end
            vec = None
            level = float('nan')
            audio = self.audio.slice(t0, t1)
            if audio is None:                                              # a word before a long silence: its audio left the buffer
                parts = [self._token_audio.get(i) for i in range(first, last + 1)]
                audio = np.concatenate(parts) if all(part is not None for part in parts) else None
            if audio is not None:
                if len(audio):
                    level = 10.0 * np.log10(max(float(np.mean(np.square(audio, dtype=np.float64))), 1e-12))
                raw = self._embed(audio)
                if raw is not None and np.all(np.isfinite(raw)) and np.linalg.norm(raw) > 0:
                    vec = unit_rows(raw)
            sentence = self._sentence_of[first]
            number = len(self.units)
            self.units.append(Unit(first, last, t0, t1, sentence, vec, level))
            self.sentences[sentence].units.append(number)
            for i in range(first, last + 1):
                self._unit_of[i] = number
        for i in [i for i in self._token_audio if i < self.stream.settled]:       # what is settled is never asked for again
            del self._token_audio[i]

    def _advance(self, touched: set[int]) -> set[int]:
        newest = self.units[-1].t1 if self.units else 0.0
        changed = set(touched)
        while newest >= self._next_horizon:
            changed |= self._housekeeping(self._next_horizon)
            self._next_horizon += self.params.refit_seconds
        changed |= self._decide()
        changed |= self._discover()
        self._publish(changed)
        return changed

    def _sentence_segments(self, s: Sentence) -> set[int]:
        return {self.tokens[i].seg for i in range(s.first, s.last + 1)}

    # ------------------------------------------------------------------ the groups
    def _rebuild(self):
        self._ids = sorted(self._voices)
        self._matrix = np.stack([self._voices[g].centroid for g in self._ids]) if self._ids else np.zeros((0, 0))
        self.version += 1

    def _new_voice(self, units: list[int]) -> int:
        gid = self._next_gid
        self._next_gid += 1
        voice = Voice(gid, self.params.reservoir)
        for n in units:
            u = self.units[n]
            voice.add(u.vec, u.seconds, u.t1)
        voice.refit(self.params.trim_cos)
        self._voices[gid] = voice
        self._rebuild()
        return gid

    def _housekeeping(self, horizon: float) -> set[int]:
        """Once per minute of audio: clean the centres, join groups that are one person, forget old orphans, look at unknowns again."""
        p = self.params
        for voice in self._voices.values():
            voice.refit(p.trim_cos)
        if self._voices:
            self._rebuild()
        changed = self._auto_merge()
        cutoff = horizon - p.orphan_keep_seconds
        self._set_orphans([n for n in self._orphans if self.units[n].t1 >= cutoff][-p.max_orphans:])
        changed |= self._rescore()
        self._update_reference()
        changed |= self._remark_background()
        window = [s for s in self.sentences[:self._cursor] if horizon - p.refit_seconds <= s.t0 < horizon]
        self.minutes.append({'minute': round(horizon / 60), 'units': len(self.units), 'groups': self.group_count,
                             'sentences_decided': self._cursor, 'orphans': len(self._orphans),
                             'unknown_share': round(sum(s.gid is None for s in window) / len(window), 3) if window else None})
        return changed

    def _set_orphans(self, units: list[int]):
        self._orphans = units
        self._orphan_set = set(units)

    def _auto_merge(self) -> set[int]:
        changed: set[int] = set()
        while len(self._ids) >= 2:
            sims = self._matrix @ self._matrix.T
            np.fill_diagonal(sims, -1.0)
            a, b = np.unravel_index(int(np.argmax(sims)), sims.shape)
            if sims[a, b] < self.params.merge_cos:
                break
            ga, gb = self._ids[a], self._ids[b]
            keep, drop = (ga, gb) if self._voices[ga].seconds >= self._voices[gb].seconds else (gb, ga)
            self._merge(drop, keep)
            self.events.append({'event': 'speaker_merged_auto', 'merged': drop, 'into': keep, 'similarity': round(float(sims[a, b]), 3)})
            changed |= set(self._seg_tokens)
        return changed

    def _merge(self, drop: int, keep: int):
        self._voices[keep].absorb(self._voices.pop(drop))
        self._alias[drop] = keep
        self._rebuild()

    def merge(self, drop: int, keep: int) -> set[int]:
        """The user says two groups are one person."""
        with self._lock:
            drop, keep = self.resolve(drop), self.resolve(keep)
            if drop == keep or drop not in self._voices or keep not in self._voices:
                raise ValueError('找不到要合併的說話者，或兩者已經是同一位。')
            self._merge(drop, keep)
            self.events.append({'event': 'speaker_merged', 'merged': drop, 'into': keep})
            changed = set(self._seg_tokens)
            self._publish(changed)
            changed |= self._remark_background()
            self._publish(changed)
            return changed

    # ------------------------------------------------------------------ sentences
    def _score(self, s: Sentence):
        """(best similarity, group, lead over the runner-up) of a sentence against the groups, None without a voiceprint or groups."""
        if not self._ids:
            return None
        vecs, weights = [], []
        for n in s.units:
            u = self.units[n]
            if u.vec is not None:
                vecs.append(u.vec)
                weights.append(u.seconds)
        if not vecs:
            return None
        sims = np.stack(vecs) @ self._matrix.T                      # units x groups
        mean = np.average(sims, axis=0, weights=weights)
        order = np.argsort(mean)[::-1]
        best = float(mean[order[0]])
        runner = float(mean[order[1]]) if len(order) > 1 else None
        return best, self._ids[int(order[0])], (best - runner if runner is not None else None)

    def _threshold(self) -> float:
        p = self.params
        if len(self._history) < p.min_history:
            return p.abs_conf
        return max(p.abs_conf, float(np.percentile(self._history, p.unknown_percentile)))

    def _decide(self) -> set[int]:
        changed: set[int] = set()
        settled = self.stream.settled
        for i in range(self._cursor, len(self.sentences)):
            s = self.sentences[i]
            if not ((s.closed or self.finished) and s.last < settled):
                break
            self._decide_one(s)
            self._cursor = i + 1
            changed |= self._sentence_segments(s)
        return changed

    def _decide_one(self, s: Sentence):
        self._decide_voice(s)
        s.auto_bg = self._auto_background(s)

    def _decide_voice(self, s: Sentence):
        p = self.params
        s.decided, s.source = True, 'voice'
        s.auto_bg = self._auto_background(s)              # (known before the sentence's orphans are pooled: see `_densest`)
        got = self._score(s)
        if got is None:
            if not self._ids:
                self._add_orphans(s, everything=True)
            return
        best, top, lead = got
        s.best, s.lean = best, top
        if best < p.new_cos:
            self._add_orphans(s)
            return
        if lead is None:                                # only one group so far: it is that person
            s.gid = top
        else:
            self._history.append(lead)
            del self._history[:-p.history_size]
            if lead >= self._threshold():
                s.gid = top
        if s.gid is not None and best >= p.update_cos:
            self._learn(s, s.gid)

    def _learn(self, s: Sentence, gid: int):
        voice = self._voices[gid]
        changed = False
        for n in s.units:
            u = self.units[n]
            if u.vec is not None and float(u.vec @ voice.centroid) >= self.params.update_cos:
                voice.add(u.vec, u.seconds, u.t1)
                changed = True
        if changed:
            self._ids_dirty(gid)

    def _ids_dirty(self, gid: int):
        self._matrix[self._ids.index(gid)] = self._voices[gid].centroid

    def _add_orphans(self, s: Sentence, everything: bool = False):
        if s.manual_bg:                                  # the user said: not a person
            return
        for n in s.units:
            u = self.units[n]
            if u.vec is None or n in self._orphan_set:
                continue
            if everything or not self._ids or float((self._matrix @ u.vec).max()) < self.params.assign_cos:
                self._orphans.append(n)
                self._orphan_set.add(n)
        if len(self._orphans) > self.params.max_orphans:
            self._set_orphans(self._orphans[-self.params.max_orphans:])

    # ------------------------------------------------------------------ new voices
    def _discover(self) -> set[int]:
        """Turn a dense cloud of orphans into a new group (repeat while there is one), then look at the unknown sentences again."""
        p = self.params
        created = False
        while len(self._ids) < p.max_speakers and len(self._orphans) >= p.first_units:
            members = self._densest(self._orphans)
            if members is None:
                break
            gid = self._new_voice(members)
            created = True
            centre = self._voices[gid].centroid
            member_set = set(members)
            joined = [n for n in self._orphans if n in member_set or float(self.units[n].vec @ centre) >= p.assign_cos]
            for n in joined:
                if n not in member_set:
                    u = self.units[n]
                    self._voices[gid].add(u.vec, u.seconds, u.t1)
            self._voices[gid].refit(p.trim_cos)
            self._rebuild()
            taken = set(joined)
            self._set_orphans([n for n in self._orphans if n not in taken])
            self.events.append({'event': 'speaker_found', 'gid': gid, 'groups': self.group_count, 'units': len(joined),
                                'at': round(self.units[-1].t1, 1)})
        return (self._rescore() | self._remark_background()) if created else set()

    def _densest(self, orphans: list[int]) -> list[int] | None:
        """The orphans that sound like the most typical orphan, when they are enough, long enough and close enough to one another."""
        p = self.params
        X = np.stack([self.units[n].vec for n in orphans])
        S = X @ X.T
        near = S >= p.link_cos
        i = int(np.argmax(near.sum(axis=1) + S.mean(axis=1) * 1e-3))
        idx = np.flatnonzero(near[i])
        if len(idx) < p.first_units:
            return None
        seconds = sum(self.units[orphans[j]].seconds for j in idx)
        centre = unit_rows(X[idx].mean(axis=0))
        if seconds < p.first_seconds or float((X[idx] @ centre).mean()) < p.tight_cos:
            return None
        sentences = {self.units[orphans[j]].sentence for j in idx}
        if len(sentences) < p.first_sentences:
            return None
        quiet = sum(self.units[orphans[j]].seconds for j in idx if self._is_background(self.sentences[self.units[orphans[j]].sentence]))
        if quiet > seconds / 2 and (seconds < p.quiet_first_seconds or len(sentences) < p.quiet_first_sentences):
            return None                                 # background talk until it proves to be a person
        return [orphans[j] for j in idx]

    def _rescore(self) -> set[int]:
        """The groups changed: sentences still unknown are scored again. A text fill's answer or a user's choice is never overwritten."""
        if not self._ids or not self.units:
            return set()
        horizon = self.units[-1].t1 - self.params.rescore_seconds
        threshold = self._threshold()
        changed: set[int] = set()
        for s in self.sentences[:self._cursor]:
            if s.gid is not None or s.t1 < horizon:
                continue
            got = self._score(s)
            if got is None:
                continue
            best, top, lead = got
            s.best, s.lean = best, top
            if best >= self.params.new_cos and (lead is None or lead >= threshold):
                s.gid, s.source = top, 'voice'
                self.rescored += 1
                changed |= self._sentence_segments(s)
        return changed

    # ------------------------------------------------------------------ text fill
    def fill_candidates(self, final: bool = False) -> list[int]:
        """Unknown sentences not yet shown to the text fill, with enough decided sentences after them to read as context."""
        with self._lock:
            return [i for i in range(self._cursor)
                    if self.sentences[i].gid is None and not self.sentences[i].asked and not self._is_background(self.sentences[i])
                    and (final or self._cursor - 1 - i >= self.params.fill_after)]

    def fill_due(self, wanted: list[int]) -> bool:
        if not wanted:
            return False
        newest = self.sentences[self._cursor - 1].t1
        return len(wanted) >= self.params.fill_batch or newest - self.sentences[wanted[0]].t1 >= self.params.fill_max_age

    def mark_asked(self, indexes: list[int], asked: bool = True):
        with self._lock:
            for i in indexes:
                self.sentences[i].asked = asked

    def apply_fill(self, index: int, gid: int) -> bool:
        """Accept the text's answer only when it names the group the sentence's voice (weakly) leans to. A refused answer is kept
        (`said`) for the second text fill."""
        with self._lock:
            s = self.sentences[index]
            gid = self.resolve(gid)
            if index >= self._cursor or s.gid is not None or self._is_background(s) or gid not in self._voices:
                return False
            s.said, s.said_refit = gid, len(self.minutes)
            if s.lean is None or self.resolve(s.lean) != gid:
                return False
            s.gid, s.source = gid, 'text'
            self._publish(self._sentence_segments(s))
            return True

    def fill2_candidates(self) -> list[int]:
        """Unknown sentences the first text fill answered without being accepted, not yet shown to the second fill."""
        with self._lock:
            refits = len(self.minutes)
            return [i for i in range(self._cursor)
                    if self.sentences[i].gid is None and self.sentences[i].said is not None and not self.sentences[i].asked2
                    and not self._is_background(self.sentences[i])
                    and refits >= self.sentences[i].said_refit + self.params.fill2_refits]

    def fill2_due(self, wanted: list[int]) -> bool:
        return len(wanted) >= self.params.fill2_batch

    def mark_asked2(self, indexes: list[int], asked: bool = True):
        with self._lock:
            for i in indexes:
                self.sentences[i].asked2 = asked

    def apply_fill2(self, index: int, gid: int) -> bool:
        """The second answer decides whenever it names a speaker (it sees the neighbours' text-filled labels, the first did not)."""
        with self._lock:
            s = self.sentences[index]
            gid = self.resolve(gid)
            if index >= self._cursor or s.gid is not None or s.said is None or self._is_background(s) or gid not in self._voices:
                return False
            s.gid, s.source = gid, 'text2'
            self._publish(self._sentence_segments(s))
            return True

    # ------------------------------------------------------------------ background talk
    def _update_reference(self):
        """The usual speech level: the median level of the latest measured units (counted in units, not in minutes: speech may be sparse).
        None until there are enough."""
        p = self.params
        self._ref_level = None
        levels = []
        for u in reversed(self.units):
            if u.vec is not None and np.isfinite(u.level):
                levels.append(u.level)
                if len(levels) >= p.background_window_units:
                    break
        if len(levels) >= p.background_min_units:
            self._ref_level = float(np.median(levels))

    def _auto_background(self, s: Sentence) -> bool:
        """Background talk (a side conversation picked up by a microphone): clearly quieter than the usual speech level AND like no known
        voice group. Both cues are needed: a soft-spoken person still sounds like a group, and a new speaker at a normal volume is not
        quiet. A sentence that got a speaker (by voice or by text) is not background."""
        p = self.params
        if s.gid is not None or self._ref_level is None or not self._ids or p.background_max_cos <= 0:
            return False
        units = [u for u in (self.units[n] for n in s.units) if u.vec is not None and np.isfinite(u.level)]
        if not units:
            return False
        seconds = np.maximum(np.array([u.seconds for u in units]), 1e-3)
        power = float(np.average(10.0 ** (np.array([u.level for u in units]) / 10.0), weights=seconds))
        quieter = self._ref_level - 10.0 * np.log10(max(power, 1e-12))
        best = float(np.mean((np.stack([u.vec for u in units]) @ self._matrix.T).max(axis=1)))
        return quieter >= p.background_rel_db and best < p.background_max_cos

    def _is_background(self, s: Sentence) -> bool:
        """What is shown: the user's word when there is one, else the program's judgement."""
        if s.manual_bg is not None:
            return s.manual_bg
        return s.auto_bg and s.gid is None

    def _remark_background(self) -> set[int]:
        """The groups or the usual level changed: the recent decided sentences are judged again (either way)."""
        changed: set[int] = set()
        if not self.units:
            return changed
        horizon = self.units[-1].t1 - self.params.rescore_seconds
        for i in range(self._cursor - 1, -1, -1):
            s = self.sentences[i]
            if s.t1 < horizon:
                break
            before = self._is_background(s)
            s.auto_bg = self._auto_background(s)
            if self._is_background(s) != before:
                changed |= self._sentence_segments(s)
        return changed

    def sentences_of(self, seg: int, first: int, last: int) -> list[int]:
        """The sentences that the segment's timed pieces number `first` .. `last - 1` belong to, in order."""
        with self._lock:
            tokens = self._seg_tokens.get(seg, [])[first:last]
            return list(dict.fromkeys(self._sentence_of[i] for i in tokens))

    def set_background(self, indexes: list[int], flag: bool | None) -> set[int]:
        """The user's word on whether these sentences are background talk (None goes back to the program's judgement)."""
        with self._lock:
            changed: set[int] = set()
            for i in indexes:
                if not 0 <= i < self._cursor:
                    continue
                s = self.sentences[i]
                if s.manual_bg == flag:
                    continue
                s.manual_bg = flag
                changed |= self._sentence_segments(s)
                if flag is True:                         # "this is not a person": its voices leave the pool a new speaker is made from
                    taken = set(s.units)
                    self._set_orphans([n for n in self._orphans if n not in taken])
                elif s.gid is None:                      # "this is a person" (or: back to the program's judgement): a voice that is always
                    self._add_orphans(s)                 # quiet may become a speaker again
            if flag is not True:
                changed |= self._discover()
            self._publish(changed)
            return changed

    # ------------------------------------------------------------------ publishing
    def _label(self, i: int) -> tuple[int | None, str]:
        s = self.sentences[self._sentence_of[i]]
        if not s.decided:
            return UNKNOWN, ''
        if self._is_background(s):
            return UNKNOWN, BACKGROUND
        if s.gid is None:
            return UNKNOWN, ''
        return self.resolve(s.gid), s.source

    def _publish(self, segments: set[int]):
        for seg in segments:
            indexes = self._seg_tokens.get(seg)
            if indexes:
                self._view[seg] = tuple((self.tokens[i].c0, self.tokens[i].c1, *self._label(i)) for i in indexes)
