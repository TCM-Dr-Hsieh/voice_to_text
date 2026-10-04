"""Verify local model loading and optionally transcribe a supplied audio file."""
import argparse
import asyncio
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding='utf-8')
from voice_app.asr import local_asr
from voice_app.audio import decode_file
from voice_app.config import Settings


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=Path)
    args = parser.parse_args()
    settings = Settings.load()
    try:
        started = time.monotonic()
        print(await local_asr.load(settings), flush=True)
        print(f'Load: {time.monotonic() - started:.2f}s', flush=True)
        audio = np.concatenate(list(decode_file(args.audio))) if args.audio else np.zeros(16000, dtype=np.float32)
        started = time.monotonic()
        result = await local_asr.recognize(settings, audio)
        print('Transcript:', result['text'], flush=True)
        print('Aligned units:', len(result['items']), flush=True)
        print(f'Inference: {time.monotonic() - started:.2f}s', flush=True)
    finally:
        await local_asr.close()


if __name__ == '__main__':
    asyncio.run(main())
