import re
import unicodedata


def is_content(char: str) -> bool:
    return not char.isspace() and not unicodedata.category(char).startswith('P')


def content_chars(text: str) -> str:
    return ''.join(c for c in text if is_content(c))


def clean_editor(text: str) -> str:
    # Some servers expose reasoning inline rather than in a separate SSE field.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    if '<think>' in text:
        text = text.split('<think>', 1)[0]
    return text.strip()


def join_text(left: str, right: str) -> str:
    if left and right and left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum():
        return left + ' ' + right
    return left + right


def join_texts(parts) -> str:
    result = ''
    for part in parts:
        result = join_text(result, part)
    return result

