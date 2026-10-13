"""The colours of the speakers: readable on the white page, told apart from one another, in step with what the page is sent."""
import itertools
import math

from voice_app.speaker.palette import SPEAKER_COLORS, color_of, palette_css


def _linear(value):
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _rgb(color):
    return [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]


def _luminance(color):
    r, g, b = (_linear(v) for v in _rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b):
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _lab(color):
    r, g, b = (_linear(v) for v in _rgb(color))
    x, y, z = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047, 0.2126 * r + 0.7152 * g + 0.0722 * b, (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))


def test_there_are_ten_different_colours_and_they_start_again_from_the_first():
    assert len(SPEAKER_COLORS) == len(set(SPEAKER_COLORS)) == 10
    assert [color_of(g) for g in range(10)] == list(SPEAKER_COLORS)
    assert color_of(10) == color_of(0) and color_of(11) == color_of(1) and color_of(23) == color_of(3)


def test_every_colour_is_readable_as_text_on_the_white_page():
    for color in SPEAKER_COLORS:
        assert _contrast(color, '#ffffff') >= 4.5, color


def test_no_two_colours_are_alike_and_none_is_like_the_grey_of_unknown_speakers():
    for a, b in itertools.combinations(SPEAKER_COLORS, 2):
        assert math.dist(_lab(a), _lab(b)) >= 20, (a, b)
    for color in SPEAKER_COLORS:
        assert math.dist(_lab(color), _lab('#8a9bb0')) >= 20, color


def test_the_stylesheet_has_a_rule_for_every_colour_and_the_page_is_told_how_many():
    css = palette_css()
    for n, color in enumerate(SPEAKER_COLORS):
        assert f'.m-c{n} {{border-left-color:{color};}}' in css and f'.m-c{n} .m-who {{color:{color};}}' in css
    assert f'.m-c{len(SPEAKER_COLORS)} ' not in css


def test_the_page_is_sent_the_number_of_colours(tmp_path):
    from tests.speaker_fakes import FakeEmbedder, bases
    from voice_app.config import Settings
    from voice_app.meeting import Meeting
    meeting = Meeting(Settings(), FakeEmbedder(bases(2)), path=tmp_path / 'm.json')
    assert meeting._labels()['palette'] == len(SPEAKER_COLORS)
