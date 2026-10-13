"""Background talk: a quiet voice that sounds like nobody known (a side conversation a microphone picked up) is judged background
(grey, not translated, never a speaker of its own unless it keeps talking). The rule needs both cues; the warm-up is honest; the
user's word wins; and the audio of a word before a long silence is not lost."""
import numpy as np
import pytest

from voice_app.speaker.tracker import Params, SpeakerTracker
from tests.speaker_fakes import DIM, FakeEmbedder, RATE, bases, build, feed, make_turns

WHISPER, QUIET_KNOWN = -0.6, -0.5      # speaker numbers whose audio value is 0.04 (-28 dBFS: a whisper nobody knows) / 0.05 (-26: speaker 0, quietly)
STRANGER = 2                           # a speaker nobody has heard, at a normal volume (audio value 0.3)


class WhisperEmbedder(FakeEmbedder):
    """The fake voiceprint model, plus: the audio value 0.04 is a quiet voice of its own, 0.05 is speaker 0 talking quietly."""

    def __init__(self, base_vectors, whisper, **kwargs):
        super().__init__(base_vectors, **kwargs)
        self.whisper = whisper

    def __call__(self, samples):
        level = float(np.mean(samples))
        if abs(level - 0.04) < 0.004:
            self.calls += 1
            return self.whisper + self.rng.normal(0, self.noise, DIM)
        if abs(level - 0.05) < 0.004:
            self.calls += 1
            return self.bases[0] + self.rng.normal(0, self.noise, DIM)
        return super().__call__(samples)


def scenario(pattern, *, params=None, seed=1, pause=0.7):
    turns = make_turns(pattern, seed=seed, lo=1.6, hi=2.6, pause=pause)
    audio, tokens, truth = build(turns)
    vectors = bases(3)
    tracker = SpeakerTracker(WhisperEmbedder({0: vectors[0], 1: vectors[1], 2: vectors[2]}, vectors[2] * 0 + bases(4, seed=99)[3]), params or Params())
    feed(tracker, audio, tokens)
    tracker.finish()
    return tracker, truth, tokens


def background_share(tracker, truth, who):
    mine = [i for i, s in enumerate(truth) if s == who]
    return sum(tracker._is_background(tracker.sentences[tracker._sentence_of[i]]) for i in mine) / len(mine)


BASE = [0, 1, 0] * 20                   # about two minutes of two people talking (speaker 0 most): the yardstick and the groups exist


def test_a_quiet_voice_like_nobody_is_background_and_the_speakers_never_are():
    tracker, truth, _ = scenario(BASE + [WHISPER, 0, 1] * 4)
    assert background_share(tracker, truth, WHISPER) >= 0.75
    assert background_share(tracker, truth, 0) == 0 and background_share(tracker, truth, 1) == 0
    assert tracker.group_count == 2                                   # the whispers did not become a person
    stats = tracker.stats()
    assert stats['background'] >= 3 and stats['unknown'] + stats['background'] + stats['labelled'] == stats['decided']


def test_a_quiet_voice_that_sounds_like_a_known_speaker_is_that_speaker_not_background():
    tracker, truth, _ = scenario(BASE + [QUIET_KNOWN, 1] * 5)
    assert background_share(tracker, truth, QUIET_KNOWN) == 0
    quiet = [i for i, s in enumerate(truth) if s == QUIET_KNOWN]
    assert sum(tracker.voice_group_of(i) is not None for i in quiet) / len(quiet) > 0.7      # still labelled with a speaker


def test_a_new_speaker_at_a_normal_volume_is_not_background_whatever_they_sound_like():
    tracker, truth, _ = scenario(BASE + [STRANGER, 0, 1] * 3)
    assert background_share(tracker, truth, STRANGER) == 0


def test_nothing_is_judged_before_there_is_a_usual_level_to_compare_with():
    tracker, truth, tokens = scenario([0, 1, 0] * 4 + [WHISPER, 0, 1])
    assert tracker._ref_level is None and background_share(tracker, truth, WHISPER) == 0     # the first minute: no yardstick yet
    early = Params(background_min_units=10 ** 6)                              # a yardstick can never be made: nothing is background
    tracker, truth, _ = scenario(BASE + [WHISPER, 0, 1] * 4, params=early)
    assert background_share(tracker, truth, WHISPER) == 0


def test_the_usual_level_is_the_median_of_the_latest_speech_not_of_the_whole_meeting():
    tracker, _, _ = scenario(BASE)
    level = tracker._ref_level
    assert level is not None and -20.5 < level < -19.5                         # speaker 0 (most of the speech) talks at -20 dBFS


def test_the_switch_turns_the_judgement_off():
    tracker, truth, _ = scenario(BASE + [WHISPER, 0, 1] * 4, params=Params(background_max_cos=0.0))
    assert background_share(tracker, truth, WHISPER) == 0 and tracker.stats()['background'] == 0


def test_a_voice_that_keeps_talking_is_a_person_after_all_and_gets_a_group():
    tracker, truth, _ = scenario(BASE + [WHISPER, 0] * 14)                    # about half a minute of the same quiet voice
    assert tracker.group_count == 3
    late = [i for i, s in enumerate(truth) if s == WHISPER][-6:]
    assert all(not tracker._is_background(tracker.sentences[tracker._sentence_of[i]]) for i in late)    # now somebody: not background


def test_the_users_word_wins_either_way_and_can_be_taken_back():
    tracker, truth, tokens = scenario(BASE + [WHISPER, 0, 1] * 4)
    flagged = [i for i, s in enumerate(tracker.sentences) if tracker._is_background(s)]
    spoken = [i for i, s in enumerate(tracker.sentences) if s.gid is not None and not tracker._is_background(s)]
    assert flagged and spoken
    first = tracker.sentences[flagged[0]].first
    seg = tracker.tokens[first].seg
    published = lambda: next(item for item in tracker.labels_for(seg) if item[0] == tracker.tokens[first].c0)
    assert published()[3] == 'background'
    changed = tracker.set_background([flagged[0]], False)                     # "this is not background"
    assert not tracker._is_background(tracker.sentences[flagged[0]]) and seg in changed and published()[3] != 'background'
    before = tracker.stats()['background']
    changed = tracker.set_background([spoken[0]], True)                       # "this is background", although it has a speaker
    assert tracker._is_background(tracker.sentences[spoken[0]]) and tracker.stats()['background'] == before + 1 and changed
    tracker.set_background([spoken[0]], None)                                 # back to the program's judgement
    assert not tracker._is_background(tracker.sentences[spoken[0]])
    tracker.set_background([flagged[0]], None)
    assert tracker._is_background(tracker.sentences[flagged[0]]) and published()[3] == 'background'


def test_a_sentence_marked_background_by_the_user_is_published_as_background():
    tracker, truth, tokens = scenario(BASE)
    first = next(i for i, s in enumerate(tracker.sentences) if s.decided and s.gid is not None)
    seg = tracker.tokens[tracker.sentences[first].first].seg
    assert all(item[3] != 'background' for item in tracker.labels_for(seg))
    tracker.set_background([first], True)
    sources = [item[3] for item in tracker.labels_for(seg)]
    index = [tracker.tokens[i].c0 for i in range(tracker.sentences[first].first, tracker.sentences[first].last + 1)]
    published = {item[0]: item for item in tracker.labels_for(seg)}
    assert all(published[c0][3] == 'background' and published[c0][2] is None for c0 in index) and 'background' in sources


def test_background_sentences_are_not_put_to_the_text_fill():
    tracker, truth, _ = scenario(BASE + [WHISPER, 0, 1] * 4)
    background = {i for i, s in enumerate(tracker.sentences) if tracker._is_background(s)}
    assert background
    assert not background & set(tracker.fill_candidates(final=True))
    assert not any(tracker.apply_fill(i, tracker.group_ids[0]) for i in background)
    assert not background & set(tracker.fill2_candidates())


def test_every_unit_has_a_voiceprint_even_the_word_before_a_long_silence():
    # a unit is only closed by the NEXT word: after a silence longer than the audio buffer its audio would be gone
    turns = make_turns([0, 1, 0], seed=3, lo=1.6, hi=2.6, pause=70.0)
    audio, tokens, truth = build(turns)
    tracker = SpeakerTracker(FakeEmbedder(bases(2)), Params())
    feed(tracker, audio, tokens)
    tracker.finish()
    assert tracker.stats()['units'] == tracker.stats()['measured_units'] > 3
    assert all(np.isfinite(u.level) for u in tracker.units)


def test_a_units_level_is_the_level_of_its_audio_in_dbfs():
    tracker, truth, _ = scenario(BASE[:10])
    by_speaker = {}
    for u in tracker.units:
        by_speaker.setdefault(truth[u.first], []).append(u.level)
    assert abs(np.mean(by_speaker[0]) - 20 * np.log10(0.1)) < 0.5 and abs(np.mean(by_speaker[1]) - 20 * np.log10(0.2)) < 0.5


def stranger_scenario(action):
    """Two minutes of two people, then a third voice (normal volume, nobody knows it) talks for three turns of two seconds. After the
    second turn is decided `action(tracker, sentences)` is applied to those sentences; returns the number of groups at the end."""
    turns = make_turns(BASE + [STRANGER] * 3, seed=1, lo=2.0, hi=2.0)
    audio, tokens, truth = build(turns)
    vectors = bases(3)
    tracker = SpeakerTracker(WhisperEmbedder({0: vectors[0], 1: vectors[1], 2: vectors[2]}, bases(4, seed=99)[3]), Params())
    from tests.speaker_fakes import window_of
    cut = window_of(turns[len(BASE) + 2].start)
    feed(tracker, audio, tokens, 0, cut)
    heard = [i for i in range(tracker._cursor) if truth[tracker.sentences[i].first] == STRANGER]
    assert heard and tracker.group_count == 2
    action(tracker, heard)
    feed(tracker, audio, tokens, cut + 1)
    tracker.finish()
    return tracker.group_count


def test_a_voice_the_user_called_background_is_not_used_to_found_a_speaker():
    assert stranger_scenario(lambda tracker, heard: None) == 3                     # left alone, three turns make a person
    assert stranger_scenario(lambda tracker, heard: tracker.set_background(heard, True)) == 2   # called background: they do not count


def test_taking_the_background_mark_back_lets_the_voice_count_again():
    def mark_then_unmark(tracker, heard):
        tracker.set_background(heard, True)
        tracker.set_background(heard, False)
    assert stranger_scenario(mark_then_unmark) == 3


def test_going_back_to_the_programs_judgement_lets_the_voice_count_again_too():
    def mark_then_leave_it_to_the_program(tracker, heard):
        tracker.set_background(heard, True)
        tracker.set_background(heard, None)
    assert stranger_scenario(mark_then_leave_it_to_the_program) == 3


@pytest.mark.parametrize('second', [False, None])
def test_marking_and_unmarking_before_any_speaker_exists_is_harmless(second):
    tracker, truth, _ = scenario([0, 1])                                          # two short turns: nobody has a group yet
    assert tracker.group_count == 0
    sentences = list(range(tracker._cursor))
    assert sentences
    tracker.set_background(sentences, True)
    tracker.set_background(sentences, second)                                      # (this used to fail on the empty matrix of groups)
    assert not any(tracker._is_background(tracker.sentences[i]) for i in sentences)
