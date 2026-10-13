"""Simplified -> Traditional Chinese on the server (OpenCC s2tw: characters and Taiwan variants, no change of vocabulary).

The ASR writes Simplified text; the proofreader turns it into Traditional. In meeting mode this conversion gives the text a sensible
starting point (so a proofreading failure does not leave Simplified text) and puts the ASR text and the proofread text in one script,
so the speaker boundaries can be carried from one to the other. Without OpenCC the text is returned unchanged.
"""
_converter = None
_unavailable = False


def to_traditional(text: str) -> str:
    global _converter, _unavailable
    if not text or _unavailable:
        return text
    if _converter is None:
        try:
            from opencc import OpenCC
            _converter = OpenCC('s2tw')
        except Exception:
            _unavailable = True
            return text
    return _converter.convert(text)
