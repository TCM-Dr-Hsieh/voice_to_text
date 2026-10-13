"""The colours of the speakers: the first speaker has the first colour, the second the second, and so on; when the colours are used up
they start again from the first. A speaker keeps the colour of its number for the whole meeting (numbers are never reused or shifted by
a merge), so Speaker-003 is always the same colour on the screen.

Ten colours, chosen to be readable on the white page (contrast at least 4.5 to 1, so the name is legible in it) and as far apart from
one another as possible (the first five are the ones told apart most easily). The name is always printed too: colour is an extra, not
the only sign. Unknown and background talk stay grey, outside this palette.
"""
SPEAKER_COLORS = ('#1d4ed8', '#4d7c0f', '#e11d48', '#0e7490', '#a16207', '#a21caf', '#c2410c', '#334155', '#78350f', '#be185d')


def color_of(gid: int) -> str:
    return SPEAKER_COLORS[gid % len(SPEAKER_COLORS)]


def palette_css() -> str:
    """One rule per colour for the lines of the transcript and the translation: a thin bar at the left, and the name in the colour."""
    return ''.join(f'.m-c{n} {{border-left-color:{color};}}\n.m-c{n} .m-who {{color:{color};}}\n' for n, color in enumerate(SPEAKER_COLORS))
