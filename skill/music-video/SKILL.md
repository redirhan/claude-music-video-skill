---
name: music-video
description: Make a motion-graphics music video (2D flat characters and kinetic lyrics, no filmed footage) from a song file. First builds a "song folder" with everything a model cannot hear (beats, bars, energy sections, drops, per-frame loudness per stem, kick/snare/hat hits, optional notes, timed lyrics), then plans a frame-deterministic 2D dance video on the beat and optionally renders it with HyperFrames. Load before planning, building or reviewing a music video, lyric video or beat-synced animation, or when the user says "make a music video for this song", "sync this animation to the beat", "analyse this song".
---

# Music video (motion graphics)

You cannot hear the song. Everything visual is keyed to measured data, so the first job is
always the song folder. If the song data is wrong, every frame is wrong.

Helpers live next to this file. Run them with the skill folder's `run.py`
(default install path shown; adjust if you installed elsewhere):

```bash
python3 ~/.claude/skills/music-video/run.py song-folder SONG.mp3 OUT_DIR [--fps 30] [--lyrics-lang en|tr] [--lyrics-text known.txt]
python3 ~/.claude/skills/music-video/run.py analyze SONG.mp3            # beats/bars/sections as JSON
python3 ~/.claude/skills/music-video/run.py dance2d SONG.mp3 --out DIR --analysis OUT_DIR/song-map.json --start 23.8 --dur 14.3 [--render]
```

Never claim to have listened to the song. Say "from the analysis".

## 1. Song folder (always first)

| File | Content | Tool |
|---|---|---|
| passage.<ext> | the audio, byte-identical copy (sha256 checked); use it as-is in the final mux | — |
| song-map.json / .md | bpm, beats, downbeats/bars, energy sections (calm/build/high), per-beat energy, drops | Beat This! → aubio CLI → built-in numpy tracker |
| stems/ | vocals / drums / bass / other (cached, reruns are quick) | demucs htdemucs (optional) |
| envelopes.json | per-frame loudness of the mix and each stem (dB + 0..1) | numpy |
| hits.json | kick / snare / hat onsets with strength 0..1 | band-split spectral flux on the drums stem |
| notes.json | melody and bass notes (optional) | basic-pitch |
| lyrics-passage.json / .md | lines with word times | whisper.cpp `whisper-cli` on the vocals stem |
| song-story.md | TEMPLATE you fill in: theme, voice, mood arc, key image per section | you, from lyrics + map |
| README.md | what each file is, units (seconds), fps | generated |

Rules:
- Sections are **energy phrases** (two-bar phrases ranked by the song's own energy), not verse/chorus.
  Name them yourself in song-story.md.
- `--lyrics-text` with the real lyrics (one line per line) aligns them to whisper's words; words
  that could not be matched get interpolated times (`matched: false`). Without it the lines are raw
  whisper output: check them. Lines over a silent vocals stem are flagged `suspect` (whisper
  hallucinates on instrumentals). Instrumental songs skip whisper.
- Every heavy step degrades instead of failing: no Beat This! → aubio (external CLI) → numpy tracker
  (no real downbeats, 4/4 guessed); no demucs → mix-only envelopes and hits; no basic-pitch → no
  notes.json; no whisper → no lyrics. The run's `warnings` say what was skipped. Tell the user.
- hits.json is good enough for flashes, shakes and scale pops, not a drum transcription: hats also
  fire on snares, bass notes can leak into the snare band.
- song-story.md is never overwritten on reruns. Write only what the lyrics and the analysis support;
  leave a heading as "unknown" rather than inventing meaning.

## 2. Creative plan (show it before rendering)

- One concept that comes from song-story.md, one world, one palette, one open-licence font (OFL)
  for lyric type.
- Section → scene table: a section change = a scene or camera change; the chorus / highest-energy
  section gets the biggest visual idea, repeated with variation each time; drops = cut, flash or
  scale on the downbeat.
- Lyric typography only on key lines (kinetic words on their word times), not karaoke for every line.
- On calm music pace by phrase and energy; do not hard-cut to every tracked beat. The number of
  distinct treatments matters more than the number of cuts.
- Show the plan (table + 3–4 described or rendered stills) and get an OK before the full render.
- Building blocks worth reading first: HyperFrames' own `music-to-video` skill
  (github.com/heygen-com/hyperframes, `skills/music-to-video/`, Apache-2.0) has a catalogue of beat-cut
  motion primitives (flash cut, crash zoom, whip, chromatic split, screen shake, kinetic letters…) and
  lyric layouts.

## 3. The built-in 2D dance video

Look: flat characters (round dark head, rounded-stroke limbs, coloured shirt and shoes), arched
mirror doors, rotating sunburst backgrounds in purple / teal / gold / pink, bold condensed (Anton)
lyric words with one accent word, a row of mirrored dancer copies in different colours, film strip,
crown and chess props. Inspired by the "low effort" 2D video in Devarf's YouTube experiment of one
music video made at every effort level.

**Planner** (`musicvideo/dance2d.py`, pure and seeded):
- the window snaps to downbeats; one pose key per beat, the pose ARRIVES on the beat;
- downbeats get a big pose (build: armsUp / point / clap, high: jump / armsUp) and a full knee dip;
- the move set (a 4-beat phrase) changes every bar by energy level and never repeats twice in a row;
- scenes cut at section starts and every 4 bars (palette swap, sunburst speed by level, direction
  alternates; doors in calm/build, chess + 5-dancer row in high, crown on the lead);
- dancer copies: calm 1, build 3, high 5;
- lyrics JSON `[{start, end, text, accent}]` in window seconds → each word lands on a beat.

**Rig** (`runtime/char2d.js`, `window.Char2D`): SVG groups with pivots (hips > legs > shins > shoes,
torso > arms > forearms, head). Limb angles are outward-positive degrees (0 down, 90 side, 180 up)
so one pose works for both sides; a ground lock keeps the lower foot on the floor (squats drop by
themselves), `air` lifts for jumps. Poses: idle, stepL/R, armsUp, swayL/R, pointL/R, clap, jump,
squat, handsOnHips (`mirror(p)` swaps sides). `poseAt(keys, t)` holds, then eases in the last
≤0.26 s with overshoot on big hits. `runtime/dance2d.js` draws the plan from `window.MV_DANCE`; one
GSAP tween drives a proxy setter → `draw(t)`, so every frame is a pure function of t.

Output of `dance2d`: `DIR/plan.json` and `DIR/project/` (a HyperFrames project). With `--render`
also `DIR/dance2d.mp4`: 1920×1080, 30 fps, H.264 yuv420p, the song's own audio for the window (AAC,
no fades). Music under CC BY needs the on-screen `--credit`.

**Rendering is the user's choice.** `--render` runs `npx hyperframes render`, which downloads and
drives a headless Chrome/Chromium. Ask before using it. Without it, everything up to the project
folder works with Python alone. GSAP is not bundled: put `gsap.min.js` in `runtime/` (or set
`MV_GSAP`), otherwise the project loads GSAP 3.14.2 from the jsDelivr CDN at render time.

## 4. Characters beyond the built-in rig (bring your own)

This skill ships no 3D models. If the user wants 3D (Three.js) characters:
- use models whose licence allows redistribution in a finished video, e.g. CC0 models, or
  characters and animations the user downloads from their own Mixamo account (fine inside a
  finished video; the raw files must not be redistributed or committed to a public repo);
- never copy example models from other projects without checking their licence;
- or keep to procedural primitives (capsules, spheres, boxes on a simple bone chain), which have no
  licence question at all.
Frame-determinism still applies: `mixer.setTime(t * rate)` each frame, never `mixer.update(dt)`;
`rate = song_bpm / clip_bpm`; switch clips on bar boundaries.

## 5. Rendering rules (any renderer)

- Everything is a pure function of `t = frame / fps`. No `Date`, no `requestAnimationFrame` deltas, no
  `Math.random()` without a seed.
- Small hits (camera shake, light flash, scale pop) on kick/snare from hits.json; brightness or glow
  from envelopes.json.
- Output: 1920×1080 (or 1080×1920 for Reels/Shorts), 30 fps, H.264 yuv420p, AAC, same length as the
  song (or the chosen window).
- Photosensitivity: never more than 3 full-frame flashes per second.
- Budget: more thinking time gives better motion, but review in one session. Multi-agent review loops
  on a single music video burn tokens fast and tend to restart after limits.

## 6. Checks before delivery

- Plays to the end; duration == song (or window) duration ±1 frame.
- The audio stream is the original song (compare decoded PCM hash, or a cross-correlation peak at lag 0).
- Sync spot check: render 5 frames on chorus downbeats and confirm the hit pose / flash lands there.
- A contact sheet per section (`ffmpeg -i out.mp4 -vf "fps=1,scale=480:-1,tile=4x4" sheet.jpg`) and
  actually look at it.
- `report.md`: what you made and why, how, what you checked, what you are unsure about.
