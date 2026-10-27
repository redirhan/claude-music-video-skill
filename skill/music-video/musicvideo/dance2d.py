"""2D dance music video: a song (+ its beat analysis, + optional timed lyric lines) -> HyperFrames
project drawn by runtime/{char2d,dance2d}.js -> (optional, --render) 1920x1080 30 fps MP4 with the
original audio.

Planning is pure and deterministic (seeded random only); the browser side only draws the plan:
- pose keys land ON beats (the pose arrives at the beat; knee dip lowest on the beat), bigger moves
  and a full dip on downbeats, the move set (phrase) changes every bar and follows the energy level;
- scenes change on section boundaries and every `bars_per_scene` bars (palette swap, sunburst
  speed/direction, doors / chess / crown props); more energy -> more dancer copies in a row;
- lyric lines: each word lands on a beat inside its line, one accent word in the scene accent colour.

    python3 run.py dance2d SONG.mp3 --out DIR --start 23.78 --dur 14.3 [--analysis song-map.json]
        [--lyrics lines.json] [--credit "..."] [--seed 7] [--render] [--output final.mp4]

Planning needs only Python (+ ffmpeg/numpy when the analysis is computed here). Rendering is a
separate, opt-in step (--render): it runs the HyperFrames CLI (`npx hyperframes render`), which
drives a headless Chrome/Chromium browser. Without --render you get plan.json + a ready project
folder that you can render yourself, preview in HyperFrames Studio, or hand to another renderer.

GSAP is NOT bundled (its licence is GreenSock's own "Standard" licence). Put gsap.min.js at
runtime/gsap.min.js or point MV_GSAP at it; otherwise the project loads GSAP from the jsDelivr CDN
at render time (pinned version below).
"""

import argparse
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ANIM_DIR = Path(__file__).resolve().parents[1] / "runtime"
GSAP_VERSION = "3.14.2"
GSAP_CDN = f"https://cdn.jsdelivr.net/npm/gsap@{GSAP_VERSION}/dist/gsap.min.js"


def gsap_path() -> Optional[Path]:
    """Local gsap.min.js (MV_GSAP or runtime/gsap.min.js), or None -> CDN."""
    for cand in (os.environ.get("MV_GSAP", ""), str(ANIM_DIR / "gsap.min.js")):
        if cand and Path(cand).is_file():
            return Path(cand)
    return None
W, H, FPS = 1920, 1080, 30
GROUND = 930

POSES = ("idle", "stepL", "stepR", "armsUp", "swayL", "swayR", "pointL", "pointR", "clap", "jump",
         "squat", "handsOnHips")
LEVELS = ("calm", "build", "high")
# one phrase = the poses for the beats of one bar (cycled when a bar has more beats)
PHRASES: Dict[str, List[Sequence[str]]] = {
    "calm": [("stepL", "stepR", "stepL", "stepR"), ("swayL", "swayR", "swayL", "swayR"),
             ("handsOnHips", "stepL", "handsOnHips", "stepR"), ("stepL", "swayR", "stepR", "swayL")],
    "build": [("stepL", "clap", "stepR", "clap"), ("pointR", "swayL", "pointL", "swayR"),
              ("swayL", "clap", "swayR", "clap"), ("handsOnHips", "pointR", "handsOnHips", "pointL")],
    "high": [("armsUp", "squat", "armsUp", "squat"), ("clap", "pointR", "clap", "pointL"),
             ("squat", "armsUp", "stepL", "stepR"), ("armsUp", "clap", "squat", "clap")],
}
# the downbeat hit: a big pose per level (calm keeps the phrase's own first pose)
BIG = {"calm": (), "build": ("armsUp", "pointR", "clap"), "high": ("jump", "armsUp", "jump")}
COPIES = {"calm": 1, "build": 3, "high": 5}
RAY_SPEED = {"calm": 5.0, "build": 12.0, "high": 26.0}   # deg/s
RAYS = {"calm": 14, "build": 18, "high": 24}
# Devarf low-effort palette: purple / teal / gold / pink sunbursts (bg, ray, floor, accent, glass, ink)
PALETTES = [
    {"bg": "#3B1F6E", "ray": "#5A2E9C", "floor": "#24124A", "accent": "#FFC93C", "glass": "#CDB8FF", "ink": "#FFFFFF"},
    {"bg": "#0E8C8C", "ray": "#18B2A6", "floor": "#0A5F63", "accent": "#FF4F8B", "glass": "#BFF3EE", "ink": "#1E1B2E"},
    {"bg": "#F2A81D", "ray": "#FFC93C", "floor": "#C9760E", "accent": "#7A2FD0", "glass": "#FFF0C2", "ink": "#1E1B2E"},
    {"bg": "#E6397A", "ray": "#FF6FA0", "floor": "#A81F55", "accent": "#FFE14D", "glass": "#FFD1E2", "ink": "#FFFFFF"},
]
SHIRTS = ["#FFC93C", "#FF4F8B", "#18B2A6", "#9B5CFF", "#FF8A3D", "#4FC3FF"]
SHOES = ["#FF4F8B", "#18B2A6", "#FFC93C", "#FF8A3D", "#9B5CFF", "#FFFFFF"]


class DanceError(ValueError):
    pass


# ---------------------------------------------------------------- pure planning
def window(analysis: Dict[str, Any], start: float, dur: float, snap: bool = True) -> Dict[str, Any]:
    """Cut the analysis to [start, start+dur] (song seconds), snapping both ends to downbeats when
    one is within 0.6 s. Times in the result are relative to the window start."""
    dbs = analysis.get("downbeats") or []
    s, e = float(start), float(start) + float(dur)
    if snap and dbs:
        near = min(dbs, key=lambda d: abs(d - s))
        if abs(near - s) <= 0.6:
            s = near
        near = min(dbs, key=lambda d: abs(d - e))
        if abs(near - e) <= 0.6 and near > s + 1.0:
            e = near
    e = min(e, float(analysis.get("duration_seconds", e)))
    if e - s < 1.0:
        raise DanceError("window shorter than 1 s")
    beats_all = analysis.get("beats") or []
    energy_all = analysis.get("beat_energy_db") or [0.0] * len(beats_all)
    beats, energy = [], []
    for b, en in zip(beats_all, energy_all):
        if s - 1e-3 <= b < e - 1e-3:
            beats.append(round(b - s, 3))
            energy.append(float(en))
    downbeats = [round(d - s, 3) for d in dbs if s - 1e-3 <= d < e - 1e-3]
    sections = []
    for sec in analysis.get("sections") or []:
        a, b = max(sec["start"], s), min(sec["end"], e)
        if b - a > 0.05:
            sections.append({"start": round(a - s, 3), "end": round(b - s, 3), "level": sec["level"]})
    if not sections:
        sections = [{"start": 0.0, "end": round(e - s, 3), "level": "build"}]
    return {"offset": round(s, 3), "dur": round(e - s, 3), "beats": beats, "downbeats": downbeats,
            "energy_db": energy, "sections": sections, "bpm": analysis.get("bpm")}


def level_at(t: float, sections: List[Dict[str, Any]]) -> str:
    for sec in sections:
        if sec["start"] - 1e-6 <= t < sec["end"]:
            return sec["level"]
    return sections[-1]["level"] if sections else "build"


def normalize_energy(energy_db: List[float]) -> List[float]:
    """Per-beat energy -> 0..1 within the window (10th..90th percentile)."""
    if not energy_db:
        return []
    srt = sorted(energy_db)
    lo = srt[int(0.1 * (len(srt) - 1))]
    hi = srt[int(0.9 * (len(srt) - 1))]
    span = max(hi - lo, 1e-6)
    return [round(min(1.0, max(0.0, (v - lo) / span)), 3) for v in energy_db]


def bars_of(beats: List[float], downbeats: List[float]) -> List[List[int]]:
    """Beat indices grouped by bar (a bar starts at a downbeat; beats before the first downbeat form
    a pickup bar)."""
    dset = {round(d, 3) for d in downbeats}
    bars: List[List[int]] = []
    for i, b in enumerate(beats):
        if round(b, 3) in dset or not bars:
            bars.append([])
        bars[-1].append(i)
    return bars


def plan_keys(beats: List[float], downbeats: List[float], sections: List[Dict[str, Any]],
              energy: Optional[List[float]] = None, seed: int = 7) -> List[Dict[str, Any]]:
    """Pose keys: one per beat, the pose arrives on the beat. Each bar picks a phrase from its
    level's pool (never the same phrase twice in a row, random mirror); downbeats get the big hit
    pose and a full knee dip, other beats a dip scaled by energy."""
    rng = random.Random(seed)
    energy = energy or [0.5] * len(beats)
    dset = {round(d, 3) for d in downbeats}
    keys: List[Dict[str, Any]] = []
    last: Dict[str, int] = {}
    for bar in bars_of(beats, downbeats):
        lvl = level_at(beats[bar[0]], sections)
        pool = PHRASES[lvl]
        choices = [i for i in range(len(pool)) if i != last.get(lvl)]
        pi = rng.choice(choices)
        last[lvl] = pi
        phrase = pool[pi]
        mirror = rng.random() < 0.5
        for k, i in enumerate(bar):
            on_down = round(beats[i], 3) in dset
            pose = phrase[k % len(phrase)]
            if on_down and BIG[lvl]:
                pose = rng.choice(BIG[lvl])
            e = energy[i] if i < len(energy) else 0.5
            dip = 1.0 if on_down else round(0.3 + 0.4 * e, 3)
            if pose == "jump":
                dip = 1.0
            keys.append({"t": round(beats[i], 3), "pose": pose, "m": mirror, "dip": dip, "down": on_down,
                         "level": lvl})
    return keys


def plan_scenes(dur: float, downbeats: List[float], sections: List[Dict[str, Any]],
                bars_per_scene: int = 4, seed: int = 7) -> List[Dict[str, Any]]:
    """Scenes start at section starts and every `bars_per_scene` downbeats; each gets the next palette
    (never the previous one), a sunburst speed by level with alternating direction, and props."""
    rng = random.Random(seed * 31 + 1)
    cuts = {0.0}
    for sec in sections:
        cuts.add(round(sec["start"], 3))
    for k, d in enumerate(downbeats):
        if k and k % bars_per_scene == 0:
            cuts.add(round(d, 3))
    # merge cuts closer than one bar (keep the section start)
    ordered = sorted(c for c in cuts if c < dur - 0.5)
    sec_starts = {round(s["start"], 3) for s in sections}
    kept: List[float] = []
    for c in ordered:
        if kept and c - kept[-1] < 1.2:
            if c in sec_starts:
                kept[-1] = c
            continue
        kept.append(c)
    if not kept or kept[0] > 0:
        kept.insert(0, 0.0)
    kept[0] = 0.0
    scenes = []
    pal = -1
    for k, t0 in enumerate(kept):
        t1 = kept[k + 1] if k + 1 < len(kept) else dur
        lvl = level_at(t0 + 1e-3, sections)
        pal = rng.choice([i for i in range(len(PALETTES)) if i != pal])
        p = PALETTES[pal]
        direction = 1 if k % 2 == 0 else -1
        scenes.append({"t0": round(t0, 3), "t1": round(t1, 3), "level": lvl, "palette": pal, **p,
                       "rays": RAYS[lvl], "rot": direction * RAY_SPEED[lvl], "rot0": round(rng.uniform(0, 30), 2),
                       "doors": lvl != "high", "chess": lvl == "high", "crown": lvl != "calm",
                       "film": True})
    return scenes


def plan_dancers(scenes: List[Dict[str, Any]], energy_at=None, seed: int = 7) -> List[Dict[str, Any]]:
    """Dancer copies per scene: calm 1 (centre), build 3, high 5 (a row of mirrored copies in
    different colours); a scene whose mean energy is high gets the next level's count."""
    rng = random.Random(seed * 17 + 3)
    out = []
    for k, sc in enumerate(scenes):
        lvl = sc["level"]
        n = COPIES[lvl]
        if energy_at is not None and energy_at(sc["t0"], sc["t1"]) > 0.8 and n < 5:
            n += 2
        if n == 1:
            xs, scale = [W / 2], 1.35
        elif n == 3:
            xs, scale = [W / 2 - 560, W / 2, W / 2 + 560], 1.0
        else:
            xs, scale = [W / 2 - 720, W / 2 - 360, W / 2, W / 2 + 360, W / 2 + 720], 0.86
        shirt0 = rng.randrange(len(SHIRTS))
        for j, x in enumerate(xs):
            centre = abs(x - W / 2) < 1
            colors = {"shirt": SHIRTS[0] if centre else SHIRTS[(shirt0 + j) % len(SHIRTS)],
                      "sleeve": SHIRTS[0] if centre else SHIRTS[(shirt0 + j) % len(SHIRTS)],
                      "shoe": SHOES[(shirt0 + j) % len(SHOES)]}
            if colors["shirt"] == colors["shoe"]:
                colors["shoe"] = SHOES[(shirt0 + j + 1) % len(SHOES)]
            out.append({"id": f"s{k}d{j}", "t0": sc["t0"], "t1": sc["t1"], "x": round(x, 1),
                        "scale": scale * (1.0 if centre else 0.9), "m": (j % 2 == 1), "lead": centre,
                        "colors": colors})
    return out


def plan_lyrics(lines: List[Dict[str, Any]], beats: List[float], dur: float) -> List[Dict[str, Any]]:
    """Timed lines [{start, end, text, accent?}] (window seconds) -> words landing on the beats inside
    each line; accent = the given word or the longest one."""
    out = []
    for ln in lines:
        t0, t1 = float(ln["start"]), min(float(ln["end"]), dur)
        words = [w for w in str(ln["text"]).split() if w]
        if not words or t1 - t0 < 0.3:
            continue
        acc = (ln.get("accent") or max(words, key=len)).upper()
        inside = [b for b in beats if t0 - 0.05 <= b < t1 - 0.25]
        if len(inside) >= len(words):
            times = inside[:len(words)]
        else:
            step = (t1 - t0) * 0.6 / max(1, len(words))
            times = [t0 + i * step for i in range(len(words))]
        out.append({"t0": round(min(t0, times[0]), 3), "t1": round(t1, 3), "y": ln.get("y", 150),
                    "words": [{"text": w, "t": round(tt, 3), "accent": w.upper().strip(".,!?") == acc}
                              for w, tt in zip(words, times)]})
    return out


def plan(analysis: Dict[str, Any], start: float, dur: float, lyrics: Optional[List[Dict[str, Any]]] = None,
         seed: int = 7, bars_per_scene: int = 4, credit: str = "") -> Dict[str, Any]:
    """Whole spec for dance2d.js (pure)."""
    win = window(analysis, start, dur)
    en = normalize_energy(win["energy_db"])
    keys = plan_keys(win["beats"], win["downbeats"], win["sections"], en, seed)
    scenes = plan_scenes(win["dur"], win["downbeats"], win["sections"], bars_per_scene, seed)

    def energy_at(a: float, b: float) -> float:
        vals = [e for t, e in zip(win["beats"], en) if a <= t < b]
        return sum(vals) / len(vals) if vals else 0.0

    dancers = plan_dancers(scenes, energy_at, seed)
    return {"w": W, "h": H, "fps": FPS, "dur": win["dur"], "ground": GROUND, "offset": win["offset"],
            "bpm": win["bpm"], "beats": win["beats"], "downbeats": win["downbeats"],
            "sections": win["sections"], "keys": keys, "scenes": scenes, "dancers": dancers,
            "lyrics": plan_lyrics(lyrics or [], win["beats"], win["dur"]), "credit": credit}


# ---------------------------------------------------------------- project + render
INDEX = """<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width={w}, height={h}" />
<title>{title}</title>
<link rel="stylesheet" href="dance2d.css" />
<script src="{gsap_src}"></script>
</head>
<body>
<div id="root" data-composition-id="main" data-start="0" data-width="{w}" data-height="{h}" data-duration="{dur}"></div>
<script>window.MV_DANCE = {spec};</script>
<script src="char2d.js"></script>
<script src="dance2d.js"></script>
</body>
</html>
"""


def write_project(spec: Dict[str, Any], proj: Path, title: str = "dance2d") -> Path:
    """A self-contained HyperFrames project folder (runtime files are copied, not linked)."""
    proj = Path(proj)
    proj.mkdir(parents=True, exist_ok=True)
    for name in ("char2d.js", "dance2d.js", "dance2d.css"):
        shutil.copyfile(ANIM_DIR / name, proj / name)
    if (proj / "fonts").exists():
        shutil.rmtree(proj / "fonts")
    shutil.copytree(ANIM_DIR / "fonts", proj / "fonts")
    gsap = gsap_path()
    if gsap:
        shutil.copyfile(gsap, proj / "gsap.min.js")
    html = INDEX.format(w=spec["w"], h=spec["h"], dur=spec["dur"], title=title,
                        gsap_src="gsap.min.js" if gsap else GSAP_CDN,
                        spec=json.dumps(spec, ensure_ascii=False))
    (proj / "index.html").write_text(html, encoding="utf-8")
    return proj


def hyperframes_cmd() -> List[str]:
    """MV_HYPERFRAMES (e.g. a pinned local install's bin/hyperframes), else `npx --yes hyperframes`."""
    custom = os.environ.get("MV_HYPERFRAMES", "")
    if custom:
        return custom.split()
    npx = shutil.which("npx")
    if not npx:
        raise DanceError("npx not found (Node.js 22+ is needed for HyperFrames rendering)")
    return [npx, "--yes", "hyperframes"]


def render(proj: Path, out: Path, fps: int = FPS) -> Path:
    """HyperFrames render (headless browser). Only called with --render."""
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HYPERFRAMES_NO_TELEMETRY="1", DO_NOT_TRACK="1")
    r = subprocess.run([*hyperframes_cmd(), "render", ".", "--fps", str(fps), "--quality", "looks", "--output",
                        str(out), "--quiet", "--workers", "2"], cwd=str(proj), capture_output=True, text=True,
                       timeout=3600, env=env)
    if r.returncode != 0 or not out.is_file():
        raise DanceError(f"render failed for {proj}:\n{(r.stdout + r.stderr)[-2500:]}")
    return out


def mux(video: Path, audio: Path, offset: float, dur: float, out: Path) -> Path:
    """Final deliverable: H.264 yuv420p 30 fps + the song's own audio for [offset, offset+dur]
    (no fades, no level change; only AAC encoding)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-ss", f"{offset:.3f}", "-t", f"{dur:.3f}",
                    "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-t", f"{dur:.3f}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS),
                    "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(out)],
                   check=True, capture_output=True)
    return out


def build(audio: Path, out_dir: Path, start: float = 0.0, dur: float = 15.0, analysis_path: Optional[Path] = None,
          lyrics_path: Optional[Path] = None, seed: int = 7, credit: str = "", output: Optional[Path] = None,
          do_render: bool = False) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apath = Path(analysis_path) if analysis_path else out_dir / "analysis.json"
    if apath.is_file():
        analysis = json.loads(apath.read_text())
        if "duration_seconds" not in analysis and "duration" in analysis:   # a song-folder song-map.json
            analysis["duration_seconds"] = analysis["duration"]
            analysis["sections"] = [{"start": s["start"], "end": s["end"], "level": s["level"]}
                                    for s in analysis.get("sections", [])]
    else:
        from .analysis import analyze_music
        analysis = analyze_music(Path(audio))
        analysis.pop("path", None)
        apath.write_text(json.dumps(analysis))
    lyrics = json.loads(Path(lyrics_path).read_text()) if lyrics_path else []
    if isinstance(lyrics, dict):
        lyrics = lyrics.get("lines", [])
    spec = plan(analysis, start, dur, lyrics, seed=seed, credit=credit)
    (out_dir / "plan.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False))
    proj = write_project(spec, out_dir / "project", title=Path(audio).stem)
    res = {"plan": str(out_dir / "plan.json"), "project": str(proj), "offset": spec["offset"], "dur": spec["dur"],
           "gsap": "local" if gsap_path() else f"CDN {GSAP_CDN}"}
    if do_render:
        silent = render(proj, out_dir / "video_noaudio.mp4")
        final = mux(silent, Path(audio), spec["offset"], spec["dur"], Path(output) if output else out_dir / "dance2d.mp4")
        res["output"] = str(final)
    return res


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="run.py dance2d", description="2D dance music video from a song (HyperFrames)")
    ap.add_argument("audio", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="work folder (plan, project, renders)")
    ap.add_argument("--start", type=float, default=0.0, help="song seconds (snapped to a downbeat)")
    ap.add_argument("--dur", type=float, default=15.0)
    ap.add_argument("--analysis", type=Path, help="song-folder song-map.json or cached analysis JSON "
                                                  "(else computed into --out/analysis.json)")
    ap.add_argument("--lyrics", type=Path, help='JSON [{"start","end","text","accent"}] in window seconds')
    ap.add_argument("--credit", default="", help="on-screen music credit (CC BY attribution)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", type=Path, help="final mp4 path (default OUT/dance2d.mp4)")
    ap.add_argument("--render", action="store_true",
                    help="also render with HyperFrames (needs Node 22+; launches a headless Chrome/Chromium)")
    a = ap.parse_args(argv)
    res = build(a.audio, a.out, a.start, a.dur, a.analysis, a.lyrics, a.seed, a.credit, a.output, a.render)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
