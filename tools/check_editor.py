"""Exercise the configured local editor using synthetic text only."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_app.api import ModelAPI
from voice_app.config import Settings


async def main():
    for streaming in (False, True):
        settings = Settings(streaming=streaming)
        async with ModelAPI(settings) as api:
            result = await api.revise_recent('我們正在測試語音辨識。', [{
                'index': 1, 'start': 0, 'end': 1, 'reason': 'stop',
                'added': '今天的天企很好。', 'current': '今天的天企很好。',
                'raw': '今天的天企很好。'}])
            print(f'stream={streaming} text={result.texts[0]} warning={result.warning}', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
