"""Run a supplied audio file through the actual local ASR and LM Studio editor."""
import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding='utf-8')
from voice_app.asr import local_asr
from voice_app.config import Settings
from voice_app.session import Session


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio', type=Path)
    parser.add_argument('--overlap-seconds', type=float, help='Override overlap for this test only; do not save settings')
    args = parser.parse_args()
    settings = Settings.load()
    if args.overlap_seconds is not None:
        settings.overlap_seconds = args.overlap_seconds
        settings.validate()
    session = Session()
    try:
        print(await local_asr.load(settings), flush=True)
        session.start(settings, 'file', path=args.audio)
        await session.task
        print(session.export_json(), flush=True)
        if session.error:
            raise RuntimeError(session.error)
        if not session.raw:
            raise RuntimeError('No speech recognized')
    finally:
        await session.cancel()
        await local_asr.close()


if __name__ == '__main__':
    asyncio.run(main())
