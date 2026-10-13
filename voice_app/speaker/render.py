"""Turn per-token speaker labels into readable runs of text, on the CURRENT (proofread) text of a segment.

Labels live on the timed ASR text of a segment (as characters of its Traditional-script form, `Segment.added_conv`). The proofreader
may have rewritten the segment since, and its prompt and JSON contract do not carry speakers, so the label boundaries are
projected onto the proofread text with a diff of the two.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from ..text import join_text
from . import BACKGROUND

TEXT_MARK = '*'            # a label that came from reading the text, not from the voice (less reliable)


@dataclass(frozen=True)
class Run:
    gid: int | None        # the voice group (None = unknown)
    source: str            # '' | 'voice' | 'text' | 'text2' | 'background'
    text: str
    span: tuple[int, int] = (0, 0)     # the timed pieces of the segment it was made of: first, last + 1 (to find its sentences)

    @property
    def background(self) -> bool:
        return self.source == BACKGROUND


def project(source: str, target: str, positions: list[int]) -> list[int]:
    """Where each (non-decreasing) position of `source` falls in `target`."""
    if source == target:
        return list(positions)
    opcodes = SequenceMatcher(None, source, target, autojunk=False).get_opcodes()
    result, k = [], 0
    for p in positions:
        while k < len(opcodes) - 1 and opcodes[k][2] < p:
            k += 1
        tag, i1, i2, j1, j2 = opcodes[k] if opcodes else ('equal', 0, 0, 0, 0)
        if p <= i1:
            result.append(j1)
        elif p >= i2:
            result.append(j2)
        elif tag == 'equal':
            result.append(j1 + (p - i1))
        else:
            result.append(j1 + round((p - i1) / (i2 - i1) * (j2 - j1)))
    return result


def segment_runs(added: str, corrected: str, labels: tuple | None) -> list[Run]:
    """The segment's text cut into runs of one speaker. `labels` is ((c0, c1, gid, source), ...) over `added`, as the tracker publishes."""
    if not labels or not corrected.strip():
        return [Run(None, '', corrected.strip())] if corrected.strip() else []
    starts = [label[0] for label in labels] + [len(added)]
    cuts = project(added, corrected, starts)
    cuts[0], cuts[-1] = 0, len(corrected)
    runs: list[list] = []                              # gid, source, text, first piece, last piece + 1
    for k, (_, _, gid, source) in enumerate(labels):
        text = corrected[cuts[k]:cuts[k + 1]]
        if runs and runs[-1][0] == gid and runs[-1][1] == source:
            runs[-1][2] += text
            runs[-1][4] = k + 1
        else:
            runs.append([gid, source, text, k, k + 1])
    return [Run(gid, source, text.strip(), (first, end)) for gid, source, text, first, end in runs if text.strip()]


def speaker_runs(runs: list[Run]) -> list[Run]:
    """The same runs with neighbours of one speaker joined, whatever way each was decided: what the translator is given."""
    merged: list[Run] = []
    for run in runs:
        if merged and merged[-1].gid == run.gid:
            merged[-1] = Run(run.gid, merged[-1].source if merged[-1].source == run.source else '', join_text(merged[-1].text, run.text))
        else:
            merged.append(run)
    return merged
