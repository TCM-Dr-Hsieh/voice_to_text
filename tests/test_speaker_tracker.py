import time

import numpy as np
import pytest

from voice_app.speaker.tracker import Params, SpeakerTracker
from voice_app.speaker.units import Token
from tests.speaker_fakes import FakeEmbedder, RATE, accuracy, bases, build, feed, make_turns


def run(pattern, *, count=None, seed=1, params=None, noise=0.12, **turn_kw):
    turns = make_turns(pattern, seed=seed, **turn_kw)
    audio, tokens, truth = build(turns)
    embedder = FakeEmbedder(bases(count or (max(pattern) + 1)), noise=noise)
    tracker = SpeakerTracker(embedder, params or Params())
    feed(tracker, audio, tokens)
    tracker.finish()
    return tracker, truth, (audio, tokens, embedder)


def test_two_speakers_are_found_and_labelled():
    tracker, truth, _ = run([0, 1] * 30)
    coverage, precision, mapping = accuracy(tracker, truth)
    assert tracker.group_count == 2
    assert coverage > 0.85 and precision > 0.98
    assert mapping[0] != mapping[1]


def test_one_speaker_is_one_group_and_nothing_is_unknown_afterwards():
    tracker, truth, _ = run([0] * 40, pause=0.8)
    coverage, precision, _ = accuracy(tracker, truth)
    assert tracker.group_count == 1
    assert coverage > 0.9 and precision == 1.0


def test_five_speakers_are_found_without_being_told_how_many():
    # a host and a lecturer talk most; three students ask now and then (the open-set case the meeting mode is for)
    rng = np.random.default_rng(3)
    pattern = []
    for _ in range(90):
        pattern.append(int(rng.choice([0, 1, 2, 3, 4], p=[.3, .3, .15, .15, .1])))
    pattern = [s for k, s in enumerate(pattern) if k == 0 or s != pattern[k - 1]]
    tracker, truth, _ = run(pattern, count=5)
    coverage, precision, mapping = accuracy(tracker, truth)
    assert tracker.group_count == 5
    assert len(set(mapping.values())) == 5
    assert coverage > 0.8 and precision > 0.97


def test_a_late_speaker_gets_a_group_and_the_unknowns_before_it_are_labelled():
    pattern = [0, 1] * 20 + [2, 0, 1, 2] * 5
    tracker, truth, _ = run(pattern, count=3)
    assert tracker.group_count == 3
    coverage, precision, mapping = accuracy(tracker, truth)
    assert len(set(mapping.values())) == 3 and precision > 0.97
    # the newcomer's first sentences were heard before the group existed: they are labelled afterwards, not left unknown
    first = next(i for i, s in enumerate(truth) if s == 2)
    early = [tracker.voice_group_of(i) for i in range(first, first + 12)]
    assert sum(g == mapping[2] for g in early) >= 8
    assert tracker.rescored > 0


def test_random_noise_is_not_a_speaker():
    # every noise voiceprint points in another direction: they never form a dense cloud
    pattern = [0, 1, 8, 0, 1, 8, 1, 0, 8] * 8
    tracker, truth, _ = run(pattern, count=2)
    assert tracker.group_count == 2
    noise = [i for i, s in enumerate(truth) if s == 8]
    assert all(tracker.voice_group_of(i) is None for i in noise)


def test_two_groups_can_be_joined_by_the_user_and_labels_follow():
    tracker, truth, _ = run([0, 1, 2] * 20, count=3)
    _, _, mapping = accuracy(tracker, truth)
    drop, keep = mapping[2], mapping[1]
    changed = tracker.merge(drop, keep)
    assert changed and tracker.group_count == 2
    assert tracker.resolve(drop) == keep
    labels = {tracker.voice_group_of(i) for i, s in enumerate(truth) if s == 2}
    assert labels <= {keep, None}
    assert drop not in tracker.group_ids
    info = {g['gid']: g for g in tracker.speaker_info()}
    assert drop in info[keep]['merged']
    with pytest.raises(ValueError):
        tracker.merge(drop, keep)                 # already one


def test_work_per_minute_does_not_grow_with_the_length_of_the_meeting():
    # an hour of four speakers: the per-minute housekeeping must cost the same at the end as near the start
    rng = np.random.default_rng(5)
    pattern = [int(x) for x in rng.choice(4, 700)]
    pattern = [s for k, s in enumerate(pattern) if k == 0 or s != pattern[k - 1]]
    turns = make_turns(pattern, seed=2, lo=3, hi=9)
    audio, tokens, truth = build(turns)
    tracker = SpeakerTracker(FakeEmbedder(bases(4)), Params())
    durations = []
    original = tracker._housekeeping

    def timed(horizon):
        started = time.perf_counter()
        out = original(horizon)
        durations.append(time.perf_counter() - started)
        return out

    tracker._housekeeping = timed
    feed(tracker, audio, tokens)
    tracker.finish()
    assert len(durations) >= 40
    early, late = np.mean(durations[5:15]), np.mean(durations[-10:])
    assert late < max(0.25, 8 * early)
    assert max(len(v.pool) for v in tracker._voices.values()) <= tracker.params.reservoir
    assert len(tracker._orphans) <= tracker.params.max_orphans
    assert tracker.group_count == 4


def test_a_long_sentence_without_a_pause_is_cut_so_labels_do_not_wait():
    p = Params(max_sentence_seconds=6.0)
    tokens = [Token(1 + int(i * 0.25 // 3), i % 12, i % 12 + 1, i * 0.25, i * 0.25 + 0.25, False) for i in range(160)]
    tracker = SpeakerTracker(FakeEmbedder(bases(1)), p)
    tracker.add_audio(0.0, np.full(RATE * 45, 0.1, dtype=np.float32))
    tracker.add_tokens(tokens)
    assert len(tracker.sentences) >= 6
    assert all(s.t1 - s.t0 <= 6.0 + 0.25 for s in tracker.sentences)


def test_finish_decides_the_last_sentences():
    tracker, truth, _ = run([0, 1] * 8)
    assert tracker.finished
    assert tracker.stats()['decided'] == tracker.stats()['sentences']
    assert tracker.finish() == set()              # a second call is harmless


def test_a_failing_voiceprint_model_does_not_break_the_feed_but_labels_nothing():
    tracker = SpeakerTracker(lambda samples: None)
    turns = make_turns([0, 1] * 6)
    audio, tokens, truth = build(turns)
    feed(tracker, audio, tokens)
    tracker.finish()
    assert tracker.group_count == 0
    assert all(tracker.voice_group_of(i) is None for i in range(len(truth)))


# --- the text fill's side of the tracker ------------------------------------------------------------------------------
def unknown_state():
    """Two known speakers and a handful of sentences the voice left unknown (silent audio: no voiceprint)."""
    pattern = [0, 1] * 12
    turns = make_turns(pattern, seed=3)
    audio, tokens, truth = build(turns)
    tracker = SpeakerTracker(FakeEmbedder(bases(2)), Params())
    feed(tracker, audio, tokens)
    tracker.finish()
    return tracker, truth


def test_fill_candidates_wait_for_context_unless_it_is_the_final_pass():
    tracker, _ = unknown_state()
    victim = next(i for i in range(5, tracker._cursor - 12))
    s = tracker.sentences[victim]
    s.gid, s.lean, s.asked = None, None, False
    assert victim in tracker.fill_candidates()
    tracker.sentences[tracker._cursor - 1].gid = None
    assert (tracker._cursor - 1) not in tracker.fill_candidates()
    assert (tracker._cursor - 1) in tracker.fill_candidates(final=True)
    tracker.mark_asked([victim])
    assert victim not in tracker.fill_candidates()


def test_first_fill_is_taken_only_when_it_agrees_with_the_voice_lean():
    tracker, _ = unknown_state()
    ids = tracker.group_ids
    a, b = ids[0], ids[1]
    s = tracker.sentences[2]
    s.gid, s.lean = None, a
    assert tracker.apply_fill(2, b) is False and s.gid is None and s.said == b
    assert tracker.apply_fill(2, a) is True and s.gid == a and s.source == 'text'
    t = tracker.sentences[3]
    t.gid, t.lean = None, None                    # no voiceprint: nothing to agree with
    assert tracker.apply_fill(3, a) is False and t.gid is None


def test_second_fill_takes_any_named_speaker_for_a_sentence_the_first_one_answered():
    tracker, _ = unknown_state()
    a, b = tracker.group_ids[:2]
    s = tracker.sentences[4]
    s.gid, s.lean, s.said, s.said_refit = None, a, b, 0
    assert 4 in tracker.fill2_candidates()
    assert tracker.apply_fill2(4, a) is True and s.gid == a and s.source == 'text2'
    u = tracker.sentences[5]
    u.gid, u.said = None, None                    # never given to the first fill
    assert tracker.apply_fill2(5, a) is False
    assert tracker.apply_fill2(6, 99) is False    # not a speaker


def test_labels_are_published_per_token_with_their_source():
    tracker, truth = unknown_state()[0], None
    seg = next(iter(tracker.all_labels()))
    labels = tracker.labels_for(seg)
    assert all(len(item) == 4 for item in labels)
    assert [item[0] for item in labels] == sorted(item[0] for item in labels)
    assert all(item[3] in ('', 'voice', 'text', 'text2') for item in labels)
