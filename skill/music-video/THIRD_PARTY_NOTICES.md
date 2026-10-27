# Third-party notices

## Bundled in this package

| File | Project | Licence |
|---|---|---|
| `skill/music-video/runtime/fonts/Anton-Regular.ttf` | Anton, © 2020 The Anton Project Authors (github.com/googlefonts/AntonFont) | SIL Open Font License 1.1, full text in `runtime/fonts/OFL-Anton.txt` |

Everything else in this package (Python helpers, `char2d.js`, `dance2d.js`, `dance2d.css`, SKILL.md,
tests) is original work under the MIT licence in `LICENSE`.

No 3D models are included. No code derived from OpenShorts, OpenMontage or any other project is
included. The built-in numpy beat tracker is our own implementation of the published dynamic
programming method (D. P. W. Ellis, "Beat Tracking by Dynamic Programming", J. New Music Research, 2007).

## Used but NOT bundled (you install them; their licences apply to them)

| Tool | Role | Licence | How it is used |
|---|---|---|---|
| FFmpeg | decode / mux | LGPL-2.1+ / GPL (depends on your build) | external command |
| NumPy | maths | BSD-3-Clause | Python import |
| Beat This! (CPJKU) | beat + downbeat tracking (optional) | MIT | Python import if installed |
| PyTorch | runtime for Beat This! / demucs | BSD-3-Clause | Python import if installed |
| demucs (htdemucs) | stem separation (optional) | MIT | `python -m demucs` subprocess |
| basic-pitch (Spotify) | note transcription (optional) | Apache-2.0 | subprocess |
| whisper.cpp (`whisper-cli`) + ggml model | timed lyrics (optional) | MIT | external command |
| aubio | beat tracking fallback (optional) | **GPL-3.0** | **external command only** (`aubio beat`); never imported, linked or bundled |
| HyperFrames (HeyGen) | rendering (optional, `--render`) | Apache-2.0 | `npx hyperframes render`; it drives a headless Chrome/Chromium |
| GSAP 3.14.2 (GreenSock / Webflow) | animation timeline in the browser runtime | GSAP "Standard" licence (gsap.com/standard-license), not an OSI licence | not bundled: you place `gsap.min.js` yourself or the project loads it from the jsDelivr CDN |

## Inspiration (no code taken)

- Devarf, YouTube video making the same music video at every model effort level (the "song folder"
  idea and the flat 2D look).
- HyperFrames `skills/music-to-video` (Apache-2.0): recommended reading for beat-cut primitives.
