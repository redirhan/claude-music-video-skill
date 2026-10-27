#!/usr/bin/env python3
"""Entry point for the music-video skill helpers.

    python3 run.py song-folder SONG.mp3 OUT_DIR [options]    # measure the song (no browser)
    python3 run.py dance2d SONG.mp3 --out DIR [options]      # plan a 2D dance video (+ --render)
    python3 run.py analyze SONG.mp3                          # beats/bars/sections JSON to stdout

Run with -h after a command for its options.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "song-folder":
        from musicvideo.song_folder import main as run
        return run(rest)
    if cmd == "dance2d":
        from musicvideo.dance2d import main as run
        run(rest)
        return 0
    if cmd == "analyze":
        import argparse
        from musicvideo.analysis import analyze_music
        ap = argparse.ArgumentParser(prog="run.py analyze")
        ap.add_argument("audio", type=Path)
        ap.add_argument("--beats", default="auto", choices=("auto", "beat_this", "aubio", "numpy"))
        a = ap.parse_args(rest)
        info = analyze_music(a.audio, backend=a.beats)
        info.pop("path", None)
        print(json.dumps(info))
        return 0
    print(f"unknown command {cmd!r}\n{__doc__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
