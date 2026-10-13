"""Speaker marking for meetings: who said each part of the transcript.

The voice groups are anonymous (Speaker-001, Speaker-002, ...); the user may give them names. Nothing here decides who is
"the host": a voice group is just a person the program heard.
"""
UNKNOWN = None                       # the "group" of a sentence nobody could be assigned to yet
TEXT_SOURCES = ('text', 'text2')     # labels read from the text (first / second text fill), not heard: shown with a star
BACKGROUND = 'background'            # the source of a sentence judged to be background talk (quiet, and nobody known): grey, not translated


def default_name(gid: int) -> str:
    return f'Speaker-{gid + 1:03d}'
