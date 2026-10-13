"""Run an audio file through the meeting mode and print who spoke: the real ASR, voiceprint model, proofreading LLM and (with
--translate) the translation LLM, exactly as the web page uses them.

    .venv\\Scripts\\python.exe tools\\check_meeting.py path\\to\\talk.m4a [--minutes 10] [--translate English] [--lag 6]
                                                       [--name 1=主持人 --name 2=講師] [--show 30] [--out folder]

The meeting is saved like a real one (data\\meetings\\ unless --out is given), so the bilingual text can be read afterwards.
"""
import argparse
import asyncio
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from voice_app.asr import local_asr                                    # noqa: E402
from voice_app.config import Settings, resolve_path                    # noqa: E402
from voice_app.meeting import Meeting, reserve_meeting_path            # noqa: E402
from voice_app.session import Session                                  # noqa: E402
from voice_app.speaker.embedder import load_embedder                   # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('audio', type=Path)
    parser.add_argument('--minutes', type=float, default=0, help='stop after this much audio (default: the whole file)')
    parser.add_argument('--translate', default='', help='also translate into this language, e.g. English')
    parser.add_argument('--lag', type=int, default=6, help='segments the translation stays behind (at least 3)')
    parser.add_argument('--name', action='append', default=[], help='N=name: call the N-th speaker (1 = Speaker-001) this')
    parser.add_argument('--show', type=int, default=30, help='lines of the bilingual text to print')
    parser.add_argument('--out', type=Path, help='folder for the meeting file (default data/meetings)')
    args = parser.parse_args()
    settings = Settings.load()
    embed = load_embedder(resolve_path(settings.meeting.speaker_model_path))
    path = reserve_meeting_path(args.out) if args.out else None
    meeting = Meeting(settings, embed, path=path)
    session = Session()
    started = time.monotonic()
    try:
        print(await local_asr.load(settings), flush=True)
        await session.start(settings, 'file', path=args.audio, meeting=meeting)
        if args.translate:
            meeting.translator.start(args.translate, args.lag)
        last = 0.0
        while session.task is not None and not session.task.done():
            await asyncio.sleep(1)
            if args.minutes and session.segments and session.segments[-1].end >= args.minutes * 60:
                session.request_stop()
            if time.monotonic() - last >= 10:
                last = time.monotonic()
                done = len(meeting.translator.results)
                print(f'  {len(session.segments)} segments, {meeting.speaker.tracker.group_count} speakers'
                      + (f', {done} translated' if args.translate else '') + f', {time.monotonic() - started:.0f}s', flush=True)
        if session.error:
            print('ERROR:', session.error)
            return 1
        for spec in args.name:
            number, _, name = spec.partition('=')
            people = meeting.speaker.speakers()
            if number.isdigit() and 1 <= int(number) <= len(people):
                meeting.speaker.set_name(people[int(number) - 1]['gid'], name)
        if args.translate:
            deadline = time.monotonic() + 600
            while meeting.translator.pending_count() > 0 and time.monotonic() < deadline:
                await asyncio.sleep(1)
        await meeting.save()
        print(f'\n{len(session.segments)} segments in {time.monotonic() - started:.0f} s; {meeting.speaker.status}')
        print('\nspeaker | speech | share | sentences | examples')
        for person in meeting.speaker.speakers():
            print(f'  {person["label"]:14s} {person["seconds"] / 60:5.1f} min {person["share"]:5.0%} {person["sentences"]:5d}   '
                  + ' | '.join(e[:28] for e in person['examples']))
        if args.translate:
            print('\ntranslation:', meeting.translator.status, meeting.translator.stats)
        print('\n' + '\n'.join(meeting.export_text().splitlines()[:args.show]))
        print(f'\nsaved: {meeting.path} (+ .txt)' + (f'  [{meeting.save_error}]' if meeting.save_error else ''))
        return 0
    finally:
        await session.cancel()
        await meeting.close()
        await local_asr.close()


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
