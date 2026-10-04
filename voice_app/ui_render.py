"""Split the writing preview into locked and revisable voice text."""

from .text import join_texts


def revisable_split(corrected, segments, rolling_start=0):
    editable = join_texts(segment.corrected_tail for segment in segments[
        max(rolling_start, len(segments) - 2):])
    if not editable or not corrected.endswith(editable):
        return corrected, ''
    return corrected[:-len(editable)], editable
