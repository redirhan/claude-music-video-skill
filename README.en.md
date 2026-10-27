# music-video — a Claude Code skill that lets Claude "hear" a song

Claude cannot listen to audio. This skill gives it the next best thing: one command turns a song
into a **song folder** of measured data (beats, bars, energy sections, drops, per-frame loudness of
each stem, kick / snare / hat hits, optional notes, timed lyrics), plus a written method for turning
that data into a beat-synced motion-graphics music video. It ships a ready 2D dance-video planner
and a flat SVG character rig, so the first video works without any 3D models.

Free and open source (MIT). Made by Redirhan Studio.

## What's inside

```
skill/music-video/
  SKILL.md              the method Claude follows (song folder → plan → render → checks)
  run.py                entry point: song-folder | analyze | dance2d
  musicvideo/           Python helpers (song folder, beat tracking, 2D dance planner, whisper wrapper)
  runtime/              browser runtime for the 2D video: char2d.js (rig), dance2d.js, dance2d.css, Anton font (OFL)
tests/                  pytest suite (synthetic audio, offline)
```

## Install

1. Copy the skill folder into your Claude Code skills:
   ```bash
   mkdir -p ~/.claude/skills
   cp -R skill/music-video ~/.claude/skills/
   ```
2. Dependencies:
   - **Required:** Python 3.10+, `numpy`, and `ffmpeg` on PATH.
   - **Optional, recommended:** `pip install beat-this torch demucs` (real downbeats + stems),
     `whisper-cli` from whisper.cpp with a ggml model at `~/.cache/whisper-cpp/ggml-large-v3-turbo-q5_0.bin`
     (or set `MV_WHISPER_MODEL`) for timed lyrics, `basic-pitch` for notes
     (set `MV_BASIC_PITCH_PYTHON` if it lives in its own venv).
   - **Optional fallback:** the `aubio` command-line tool (`brew install aubio`). aubio is GPL-3.0; this
     skill only calls its CLI as a separate program and never bundles or imports it.
   - **Only for rendering video (`--render`):** Node.js 22+ (HyperFrames is fetched with `npx`). HyperFrames
     renders with a **headless Chrome/Chromium** browser. That is your choice: everything else runs
     without a browser.
3. In Claude Code, say: *"Make a music video for ~/Music/song.mp3"* or *"Analyse this song for a
   beat-synced edit"*. The skill loads by itself.

## Run it yourself

```bash
S=~/.claude/skills/music-video
python3 $S/run.py song-folder song.mp3 song_folder/ --lyrics-lang en --lyrics-text lyrics.txt
python3 $S/run.py dance2d song.mp3 --out dance/ --analysis song_folder/song-map.json --start 30 --dur 15
python3 $S/run.py dance2d song.mp3 --out dance/ --analysis song_folder/song-map.json --start 30 --dur 15 --render
```

Without the optional tools the run still finishes and says what it skipped: no Beat This! → aubio CLI
→ a built-in numpy beat tracker; no demucs → mix-only envelopes and hits; no whisper → no lyrics.

GSAP is not bundled (it has its own licence). Put `gsap.min.js` (3.14.x) into
`~/.claude/skills/music-video/runtime/` or set `MV_GSAP`; otherwise the generated project loads it from
the jsDelivr CDN when it renders.

## Tests

```bash
pip install numpy pytest
python3 -m pytest -q tests
```

All tests use synthetic audio and need no network, no models and no browser.

## Limits (honest list)

- Sections are energy phrases (calm / build / high), not verse / chorus labels.
- hits.json is approximate: good for flashes and shakes, not a drum transcription.
- Without Beat This! there are no real downbeats; bars assume 4/4.
- The built-in look is one style (flat 2D dancers). Other worlds are up to your plan and code.
- No 3D characters are shipped. Bring your own CC0 models or your own Mixamo downloads (see SKILL.md §4).

## Marketing stills / GIFs (placeholders, to be made from CC0 or own music)

- [ ] `media/hero.gif`: 6 s loop of the 2D dancers on a sunburst, cut on the downbeat
- [ ] `media/song-folder.png`: screenshot of a generated `song-map.md` table next to the folder tree
- [ ] `media/hits-overlay.gif`: waveform with kick / snare / hat markers lighting up in sync
- [ ] `media/calm-build-high.png`: three stills side by side (1 dancer / 3 dancers / 5-dancer row)
- [ ] `media/lyrics-accent.png`: kinetic lyric line with one accent word
- [ ] `media/terminal.png`: the `song-folder` run log with BPM, bars, sections

## Licence

MIT, see `LICENSE`. Bundled font and the tools used are listed in `THIRD_PARTY_NOTICES.md`.
