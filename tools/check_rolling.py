"""Probe the configured LLM with a known three-segment seam example."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_app.api import ModelAPI
from voice_app.config import Settings


SEGMENTS = [
    {'index': 54, 'start': 164.8, 'end': 173.8, 'reason': 'window',
     'added': '师哦，就是就是绝对不怪谁。', 'current': '師哦，就是就是絕對不怪誰。',
     'raw': '中医师呢都非常的哦，都是有非常优秀的医师哦，就是就是绝对不怪谁。那哦，会记得很很。'},
    {'index': 55, 'start': 167.8, 'end': 176.8, 'reason': 'window',
     'added': '水，那哦会提提很提高我们的', 'current': '水，那哦會提提很提高我們的',
     'raw': '非常优秀的医师，哦，就是就是绝对不灌水，那哦会提提很提高我们的很多学术的水平。'},
    {'index': 56, 'start': 170.8, 'end': 179.8, 'reason': 'window',
     'added': '很多学术的水平，那再来', 'current': '很多學術的水平，那再來',
     'raw': '不灌水，那呃会提提提高我们的很多学术的水平，那再来呢呃我们的。'},
]


async def main():
    settings = Settings.load()
    settings.streaming = False
    async with ModelAPI(settings) as api:
        result = await api.revise_recent('最值得參加的，因為它一定不灌水。', SEGMENTS)
    print(json.dumps({'texts': result.texts, 'warning': result.warning}, ensure_ascii=True))


if __name__ == '__main__':
    asyncio.run(main())
