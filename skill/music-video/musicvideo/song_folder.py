"""Song folder (music-video skill, step 1): the measured data a model needs to make a music video,
because it cannot hear the song.

    python3 run.py song-folder <audio> <out_dir> [--fps 30] [--lyrics-lang en|tr]
                               [--lyrics-text known_lyrics.txt] [--beats auto|beat_this|aubio|numpy]
                               [--no-stems] [--no-lyrics]

Writes into out_dir (all times in seconds, frames at --fps):
  passage.<ext>         the audio, byte-identical copy (sha256 checked)
  song-map.json/.md     bpm, beats, downbeats/bars, sections, per-beat energy, drops, duration
  stems/                demucs htdemucs stems vocals/drums/bass/other.wav (cached; optional)
  envelopes.json        per-frame loudness (dB + 0..1) of the mix and each stem
  hits.json             kick / snare / hat onsets from the drums stem (band-split spectral flux)
  notes.json            melody (other stem) / bass notes via basic-pitch (optional)
  lyrics-passage.json/.md  whisper-cli on the vocals stem, word + line times; known lyrics aligned
  song-story.md         TEMPLATE for the director (meaning is written by hand, never generated)
  README.md             what each file is

Read-only on the source audio. Heavy work (demucs, basic-pitch, whisper) runs in subprocesses
or behind small functions so tests can replace them. Without demucs (or with --no-stems) the folder
is still built from the mix alone: envelopes for the mix, hits from the mix (less clean), lyrics
from the mix.
"""

import argparse
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .analysis import AudioToolError, analyze_music, decode_mono
from .whisper_cli import DEFAULT_WHISPER_MODEL, transcribe_files

STEM_NAMES = ("vocals", "drums", "bass", "other")
DEMUCS_MODEL = "htdemucs"
ENV_SAMPLE_RATE = 22050
HIT_SAMPLE_RATE = 44100
ENV_FLOOR_DB = -60.0
BASIC_PITCH_TIMEOUT = int(os.environ.get("MV_BASIC_PITCH_TIMEOUT", "900"))
# basic-pitch pins older TensorFlow/Python versions, so it often lives in its own venv: point
# MV_BASIC_PITCH_PYTHON at that venv's python. Without it the current interpreter is tried.
BASIC_PITCH_PYTHON = os.environ.get("MV_BASIC_PITCH_PYTHON", "")
DEMUCS_PYTHON = os.environ.get("MV_DEMUCS_PYTHON", sys.executable)
# hit classes: (band Hz, min gap s, threshold k in median + k * MAD)
HIT_BANDS = {"kick": ((30.0, 150.0), 0.09), "snare": ((180.0, 2500.0), 0.09), "hat": ((6000.0, 16000.0), 0.05)}
VOCAL_ACTIVE_DB = -35.0       # a frame of the vocals stem louder than this counts as sung
HIT_COMPRESSION = 100.0
HIT_MIN_SHARE = 0.5           # a band's normalised rise vs the strongest band's at that moment
HIT_MIN_STRENGTH = 0.1
# which bands can leak into a class: a snare's thump into the kick band, a kick's body into the snare
# band. Hats are not gated (a snare's noise reaches the hat band, but hats usually play there anyway).
HIT_GATES = {"kick": ("snare",), "snare": ("kick",)}
HIT_LAG = 0.015               # calibrated on synthetic clicks (tests/test_song_folder.py)
MIN_VOCAL_ACTIVITY = 0.03     # below this share of sung frames the song is treated as instrumental


# --- small helpers -----------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(data: Any, path: Path) -> Path:
    """Readable JSON with lists of numbers kept on one line; written atomically."""
    text = json.dumps(data, ensure_ascii=False, indent=1)
    text = re.sub(r"\[\s*([-\d.eE+,\s]*?)\s*\]",
                  lambda m: "[" + re.sub(r"\s+", "", m.group(1)).replace(",", ", ") + "]", text)
    # small flat objects (a hit, a note, a word) on one line
    text = re.sub(r"\{\n\s*([^{}\[\]]*?)\n\s*\}", lambda m: "{" + re.sub(r",\n\s*", ", ", m.group(1)) + "}", text)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def fmt_time(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m)}:{s:05.2f}"


def n_frames(duration: float, fps: int) -> int:
    return int(math.ceil(duration * fps - 1e-6))


# --- passage -----------------------------------------------------------------------------

def copy_passage(audio: Path, out_dir: Path) -> Tuple[Path, str]:
    """Copy the audio unchanged to out_dir/passage.<ext>; raises if the bytes differ."""
    dst = out_dir / f"passage{audio.suffix.lower()}"
    src_hash = sha256(audio)
    if not (dst.is_file() and sha256(dst) == src_hash):
        shutil.copyfile(audio, dst)
    if sha256(dst) != src_hash:
        raise RuntimeError(f"passage copy differs from source: {dst}")
    return dst, src_hash


# --- song map ----------------------------------------------------------------------------

def norm_db(db: Sequence[float], floor: float = ENV_FLOOR_DB) -> List[float]:
    """dB -> 0..1 over the list's own range: 1 = its loudest value, 0 = its 5th-percentile level of
    the values above `floor` (or 6 dB below the top if the range is smaller); silence is 0."""
    arr = np.asarray(db, dtype=float)
    if not len(arr):
        return []
    top = float(arr.max())
    if top <= floor:
        return [0.0] * len(arr)
    lo = min(float(np.percentile(arr[arr > floor], 5)), top - 6.0)
    return [round(float(x), 3) for x in np.clip((arr - lo) / (top - lo), 0.0, 1.0)]


def build_song_map(info: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """song-map.json from analyze_music() output: adds bars, beat/bar indices per section and
    normalised per-beat energy."""
    beats = info["beats"]
    downbeats = info["downbeats"]
    energy = info["beat_energy_db"]
    energy_n = norm_db(energy)
    bars = []
    for k, start in enumerate(downbeats):
        end = downbeats[k + 1] if k + 1 < len(downbeats) else duration
        idx = [i for i, b in enumerate(beats) if start - 1e-6 <= b < end - 1e-6]
        bars.append({"bar": k, "start": start, "end": round(end, 3), "beats": len(idx),
                     "energy_db": round(float(np.mean([energy[i] for i in idx])), 1) if idx else None})
    sections = []
    for n, s in enumerate(info["sections"]):
        b_idx = [i for i, b in enumerate(beats) if s["start"] - 1e-6 <= b < s["end"] - 1e-6]
        bar_idx = [bar["bar"] for bar in bars if s["start"] - 1e-6 <= bar["start"] < s["end"] - 1e-6]
        e = [energy[i] for i in b_idx]
        sections.append({"section": n, "level": s["level"], "start": s["start"], "end": s["end"],
                         "bars": [bar_idx[0], bar_idx[-1]] if bar_idx else None, "bar_count": len(bar_idx),
                         "energy_db": round(float(np.mean(e)), 1) if e else None,
                         "energy": round(float(np.mean([energy_n[i] for i in b_idx])), 3) if b_idx else None})
    return {
        "duration": round(duration, 3), "bpm": info.get("bpm"), "beat_backend": info.get("beat_backend"),
        "beat_count": len(beats), "bar_count": len(bars),
        "beats": beats, "downbeats": downbeats, "bars": bars,
        "beat_energy_db": energy, "beat_energy": energy_n,
        "mean_energy_db": info.get("mean_energy_db"),
        "sections": sections, "drops": info.get("drops", []),
        "notes": ("sections are energy phrases (calm/build/high) from two-bar phrases, not verse/chorus labels; "
                  "drops are phrase starts with a >=6 dB energy jump; beat_energy is 0..1 over the song's own "
                  "range (5th-percentile beat = 0, loudest beat = 1)"),
    }


# --- stems -------------------------------------------------------------------------------

def stem_paths(out_dir: Path) -> Dict[str, Path]:
    return {name: out_dir / "stems" / f"{name}.wav" for name in STEM_NAMES}


def run_demucs(passage: Path, out_dir: Path, python: str = DEMUCS_PYTHON) -> Dict[str, Path]:
    """htdemucs 4-stem split into out_dir/stems/<name>.wav (cached: skipped when all exist)."""
    paths = stem_paths(out_dir)
    if all(p.is_file() and p.stat().st_size > 1000 for p in paths.values()):
        return paths
    stems_dir = out_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run([python, "-c", "import demucs"], capture_output=True)
    if probe.returncode != 0:
        raise AudioToolError("demucs is not installed (pip install demucs, or set MV_DEMUCS_PYTHON)")
    with tempfile.TemporaryDirectory(prefix="mv_demucs_", dir=stems_dir) as tmp:
        wav = Path(tmp) / "mix.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(passage), "-vn", "-ac", "2", "-ar", "44100",
                        str(wav)], check=True, capture_output=True)
        proc = subprocess.run([python, "-m", "demucs", "-n", DEMUCS_MODEL, "-o", tmp, str(wav)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise AudioToolError(f"demucs failed: {proc.stderr[-800:]}")
        for name, dst in paths.items():
            src = Path(tmp) / DEMUCS_MODEL / "mix" / f"{name}.wav"
            if not src.is_file():
                raise AudioToolError(f"demucs did not write {src.name}")
            shutil.move(str(src), dst)
    return paths


# --- envelopes ---------------------------------------------------------------------------

def frame_db(samples: np.ndarray, sample_rate: int, fps: int, frames: int) -> np.ndarray:
    """RMS dBFS of each video frame's window [i/fps, (i+1)/fps), floored at -90 dB."""
    edges = np.round(np.arange(frames + 1) * sample_rate / fps).astype(int)
    out = np.full(frames, -90.0)
    sq = np.concatenate([[0.0], np.cumsum(samples.astype(np.float64) ** 2)])
    n = len(samples)
    for i in range(frames):
        a, b = min(edges[i], n), min(edges[i + 1], n)
        if b > a:
            rms = math.sqrt(max(sq[b] - sq[a], 0.0) / (b - a))
            out[i] = max(20 * math.log10(max(rms, 1e-9)), -90.0)
    return out


def build_envelopes(tracks: Dict[str, np.ndarray], sample_rate: int, fps: int, duration: float) -> Dict[str, Any]:
    frames = n_frames(duration, fps)
    out: Dict[str, Any] = {"fps": fps, "frames": frames, "duration": round(duration, 3),
                           "floor_db": ENV_FLOOR_DB, "tracks": {}}
    for name, samples in tracks.items():
        db = frame_db(samples, sample_rate, fps, frames)
        out["tracks"][name] = {"peak_db": round(float(db.max()), 1) if frames else None,
                               "db": [round(float(x), 1) for x in db], "level": norm_db(db)}
    return out


# --- hits --------------------------------------------------------------------------------

def _stft_mag(samples: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    pad = np.concatenate([np.zeros(n_fft // 2), samples, np.zeros(n_fft)])
    count = max(1, 1 + (len(pad) - n_fft) // hop)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(count)[:, None]
    return np.abs(np.fft.rfft(pad[idx] * np.hanning(n_fft)[None, :], axis=1))


def _pick_peaks(flux: np.ndarray, hop_s: float, min_gap_s: float, k: float = 4.0,
                window_s: float = 1.0) -> List[int]:
    """Local maxima above a moving median + k * MAD threshold, at least min_gap_s apart."""
    w = max(3, int(window_s / hop_s) | 1)
    padded = np.pad(flux, w // 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, w)
    med = np.median(windows, axis=1)
    mad = np.median(np.abs(windows - med[:, None]), axis=1)
    global_floor = 0.05 * float(flux.max()) if len(flux) else 0.0
    thr = np.maximum(med + k * np.maximum(mad, 1e-6), global_floor)
    cand = [i for i in range(1, len(flux) - 1)
            if flux[i] > thr[i] and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]]
    gap = int(round(min_gap_s / hop_s))
    picked: List[int] = []
    for i in sorted(cand, key=lambda j: -flux[j]):          # strongest first
        if all(abs(i - p) >= gap for p in picked):
            picked.append(i)
    return sorted(picked)


def detect_hits(samples: np.ndarray, sample_rate: int, fps: int,
                bands: Dict[str, Tuple[Tuple[float, float], float]] = HIT_BANDS,
                min_share: float = HIT_MIN_SHARE) -> Dict[str, Any]:
    """Kick / snare / hat onsets from a drums signal.

    Per band: level-aware log energy -> positive flux -> adaptive peaks (candidates). Then a
    dominance gate removes cross-band leaks (a snare's low thump in the kick band, a kick's body in
    the snare band): at each candidate the linear energy rise of every band is divided by that
    band's typical rise (95th percentile of its rises at its own candidates); the candidate's band
    must reach `min_share` of its rival band's normalised rise there (HIT_GATES). Simultaneous
    kick + hat both pass. Hits weaker than HIT_MIN_STRENGTH of the class's strongest are dropped. Strength = flux / the class's strongest flux (0..1)."""
    n_fft, hop = 2048, 256
    hop_s = hop / sample_rate
    mag = _stft_mag(samples.astype(np.float64), n_fft, hop)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    names = list(bands)
    lin, flux, cand = {}, {}, {}
    for name in names:
        (lo, hi), min_gap = bands[name]
        band = mag[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
        lin[name] = np.maximum(np.diff(band, prepend=band[:1]), 0.0)
        comp = np.log1p(HIT_COMPRESSION * band / max(float(band.max()), 1e-9))   # level-aware log
        flux[name] = np.maximum(np.diff(comp, prepend=comp[:1]), 0.0)
        cand[name] = _pick_peaks(flux[name], hop_s, min_gap)

    def rise(name: str, i: int) -> float:
        return float(lin[name][max(0, i - 1):i + 2].max())

    typical = {n: (float(np.percentile([rise(n, i) for i in cand[n]], 95)) if cand[n] else 1.0) or 1.0
               for n in names}
    out: Dict[str, Any] = {}
    for name in names:
        kept = []
        for i in cand[name]:
            norm = {n: rise(n, i) / typical[n] for n in names}
            rivals = [norm[r] for r in HIT_GATES.get(name, ()) if r in norm]
            if not rivals or norm[name] >= min_share * max(rivals):
                kept.append(i)
        top = float(max((flux[name][i] for i in kept), default=1.0)) or 1.0
        kept = [i for i in kept if flux[name][i] >= HIT_MIN_STRENGTH * top]
        out[name] = [{"t": round(t, 3), "frame": int(math.floor(t * fps + 1e-6)),
                      "strength": round(float(flux[name][i]) / top, 3)}
                     for i in kept for t in [max(0.0, i * hop_s + HIT_LAG)]]   # frame centre -> onset
    return out


def build_hits(samples: np.ndarray, sample_rate: int, fps: int, duration: float,
               source: str = "stems/drums.wav") -> Dict[str, Any]:
    hits = detect_hits(samples, sample_rate, fps)
    counts = {k: len(v) for k, v in hits.items()}
    where = "the drums stem" if source.startswith("stems/") else "the full mix (no stems: other instruments leak in)"
    return {"source": source, "fps": fps, "duration": round(duration, 3),
            "method": f"band-split spectral flux on {where}: kick 30-150 Hz, snare 180-2500 Hz, "
                      "hat 6-16 kHz; strength 0..1 relative to the class's strongest hit",
            "counts": counts,
            "per_second": {k: round(c / duration, 2) if duration else 0.0 for k, c in counts.items()},
            **hits}


# --- notes (basic-pitch, optional) ------------------------------------------------------

_BP_SCRIPT = r"""
import json, sys
from basic_pitch.inference import predict
out = {}
for key, path in json.loads(sys.argv[1]).items():
    _, _, events = predict(path)
    out[key] = [[float(s), float(e), int(p), float(a)] for s, e, p, a, *_ in events]
print(json.dumps(out))
"""


def run_basic_pitch(stems: Dict[str, Path]) -> Tuple[Optional[Dict[str, List[List[float]]]], str]:
    """{key: [[start, end, midi, amplitude], ...]} via basic-pitch (current interpreter if it can import
    it, else MV_BASIC_PITCH_PYTHON). (None, reason) when unavailable."""
    arg = json.dumps({k: str(v) for k, v in stems.items()})
    for python in (sys.executable, BASIC_PITCH_PYTHON):
        if not python or not Path(python).is_file():
            continue
        try:
            probe = subprocess.run([python, "-c", "import basic_pitch"], capture_output=True,
                                   timeout=BASIC_PITCH_TIMEOUT)
            if probe.returncode != 0:
                continue
            proc = subprocess.run([python, "-c", _BP_SCRIPT, arg], capture_output=True, text=True,
                                  timeout=BASIC_PITCH_TIMEOUT)
        except subprocess.TimeoutExpired:
            return None, f"basic-pitch timed out after {BASIC_PITCH_TIMEOUT} s ({python}); notes.json skipped"
        if proc.returncode != 0:
            return None, f"basic-pitch failed: {proc.stderr.strip()[-400:]}"
        return json.loads(proc.stdout.strip().splitlines()[-1]), f"basic-pitch ({python})"
    return None, ("basic-pitch not installed (pip install basic-pitch, or point MV_BASIC_PITCH_PYTHON at "
                  "a venv that has it); notes.json skipped")


_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_name(p: int) -> str:
    return f"{_NOTE_NAMES[p % 12]}{p // 12 - 1}"


def build_notes(events: Dict[str, List[List[float]]], fps: int, min_amp: float = 0.3) -> Dict[str, Any]:
    out: Dict[str, Any] = {"fps": fps, "source": {"melody": "stems/other.wav", "bass": "stems/bass.wav"},
                           "min_amplitude": min_amp, "units": "seconds; pitch = MIDI number"}
    for key, ev in events.items():
        notes = sorted(([round(s, 3), round(e, 3), int(p), round(a, 3)] for s, e, p, a in ev if a >= min_amp),
                       key=lambda n: (n[0], n[2]))
        out[key] = [{"start": s, "end": e, "pitch": p, "name": midi_name(p), "amp": a,
                     "frame": int(math.floor(s * fps + 1e-6))} for s, e, p, a in notes]
        out[f"{key}_count"] = len(notes)
    return out


# --- lyrics ------------------------------------------------------------------------------

def norm_word(w: str) -> str:
    w = w.replace("İ", "i").replace("I", "ı").lower()
    return re.sub(r"[^\w]", "", w)


def _sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def align_words(known: List[str], heard: List[str], min_sim: float = 0.6) -> List[Optional[int]]:
    """Global alignment (Needleman-Wunsch) of known lyric words to whisper words.
    Returns, for each known word, the index of the matched whisper word or None."""
    a = [norm_word(w) for w in known]
    b = [norm_word(w) for w in heard]
    n, m = len(a), len(b)
    gap = -0.4
    score = np.zeros((n + 1, m + 1))
    score[:, 0] = gap * np.arange(n + 1)
    score[0, :] = gap * np.arange(m + 1)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)       # 0 diag, 1 up (skip known), 2 left (skip heard)
    back[1:, 0], back[0, 1:] = 1, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = _sim(a[i - 1], b[j - 1])
            diag = score[i - 1, j - 1] + (s if s >= min_sim else -0.6)
            up, left = score[i - 1, j] + gap, score[i, j - 1] + gap
            best = max(diag, up, left)
            score[i, j] = best
            back[i, j] = 0 if best == diag else 1 if best == up else 2
    match: List[Optional[int]] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        if back[i, j] == 0:
            if _sim(a[i - 1], b[j - 1]) >= min_sim:
                match[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif back[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return match


def align_known_lyrics(text: str, words: List[Dict[str, Any]], duration: float) -> List[Dict[str, Any]]:
    """Lines of the known lyrics with word times taken from matched whisper words; unmatched
    words are spread evenly between their matched neighbours (matched=false)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    flat: List[Tuple[int, str]] = [(li, w) for li, ln in enumerate(lines) for w in ln.split() if norm_word(w)]
    match = align_words([w for _, w in flat], [w["w"] for w in words])
    times: List[Optional[Tuple[float, float]]] = [
        (words[m]["start"], words[m]["end"]) if m is not None else None for m in match]
    # interpolate runs of unmatched words
    k = 0
    while k < len(times):
        if times[k] is not None:
            k += 1
            continue
        j = k
        while j < len(times) and times[j] is None:
            j += 1
        lo = times[k - 1][1] if k > 0 else (times[j][0] - 0.4 * (j - k) if j < len(times) else 0.0)
        hi = times[j][0] if j < len(times) else min(duration, lo + 0.4 * (j - k))
        lo = max(0.0, min(lo, hi))
        step = (hi - lo) / (j - k) if j > k else 0.0
        for q in range(k, j):
            times[q] = (round(lo + step * (q - k), 3), round(lo + step * (q - k + 1), 3))
        k = j
    out: List[Dict[str, Any]] = [{"line": li, "text": ln, "words": []} for li, ln in enumerate(lines)]
    for (li, w), t, m in zip(flat, times, match):
        out[li]["words"].append({"w": w, "start": t[0], "end": t[1], "matched": m is not None})
    for ln in out:
        ws = ln["words"]
        ln["start"] = ws[0]["start"] if ws else None
        ln["end"] = ws[-1]["end"] if ws else None
        ln["matched_share"] = round(sum(w["matched"] for w in ws) / len(ws), 2) if ws else 0.0
    return out


def lines_from_segments(segments: List[Dict[str, Any]], env_level: Optional[List[float]], fps: int) -> List[Dict[str, Any]]:
    """Whisper segments -> lyric lines; `vocal_level` = mean vocals-stem level (0..1) over the line,
    `suspect` when the vocals stem is nearly silent there (likely a hallucination)."""
    lines = []
    for n, s in enumerate(segments):
        level = None
        if env_level:
            a, b = int(s["start"] * fps), max(int(s["start"] * fps) + 1, int(s["end"] * fps))
            chunk = env_level[a:b]
            level = round(float(np.mean(chunk)), 3) if chunk else 0.0
        lines.append({"line": n, "start": s["start"], "end": s["end"], "text": s["text"], "p": s.get("p"),
                      "vocal_level": level, "suspect": bool(level is not None and level < 0.25),
                      "words": s.get("words", [])})
    return lines


def lyrics_md(data: Dict[str, Any]) -> str:
    out = ["# Lyrics (timed)", "",
           f"Source: {data['source']}. Language: {data['language']}. Times in seconds (m:ss.ss).", ""]
    if data.get("instrumental"):
        out += ["**No sung vocals detected** (vocals stem active in "
                f"{data['vocal_activity']:.1%} of frames) — treat the song as instrumental.", ""]
    for ln in data["lines"]:
        if ln.get("start") is None:
            out.append(f"- [—] {ln['text']}")
            continue
        flag = " ⚠ suspect (vocals stem silent)" if ln.get("suspect") else ""
        if "matched_share" in ln and ln["matched_share"] < 0.5:
            flag += f" ⚠ only {ln['matched_share']:.0%} of words matched (times interpolated)"
        out.append(f"- [{fmt_time(ln['start'])} – {fmt_time(ln['end'])}] {ln['text']}{flag}")
    return "\n".join(out) + "\n"


# --- markdown ----------------------------------------------------------------------------

def song_map_md(song_map: Dict[str, Any], lines: List[Dict[str, Any]], title: str) -> str:
    out = [f"# Song map — {title}", "",
           f"Duration {fmt_time(song_map['duration'])} ({song_map['duration']} s) · {song_map['bpm']} BPM "
           f"({song_map['beat_backend']}) · {song_map['beat_count']} beats · {song_map['bar_count']} bars · "
           f"mean energy {song_map['mean_energy_db']} dB", "",
           "Sections are energy phrases from the analysis (calm / build / high), not verse/chorus labels. "
           "Energy 0..1 = mean per-beat level over the song's own range (5th-percentile beat = 0, loudest = 1).", "",
           "| # | level | start – end | bars | energy | lyric lines |", "|---|---|---|---|---|---|"]
    for s in song_map["sections"]:
        lyr = [ln["text"] for ln in lines
               if ln.get("start") is not None and s["start"] - 1e-6 <= ln["start"] < s["end"] - 1e-6
               and not ln.get("suspect")]
        lyr_txt = " / ".join(lyr)
        if len(lyr_txt) > 160:
            lyr_txt = lyr_txt[:157] + "…"
        bars = f"{s['bars'][0]}–{s['bars'][1]} ({s['bar_count']})" if s["bars"] else "—"
        energy = f"{s['energy']:.2f} ({s['energy_db']} dB)" if s["energy"] is not None else "—"
        out.append(f"| {s['section']} | {s['level']} | {fmt_time(s['start'])} – {fmt_time(s['end'])} | "
                   f"{bars} | {energy} | {lyr_txt.replace('|', '/') or '—'} |")
    drops = ", ".join(fmt_time(d) for d in song_map["drops"]) or "none"
    out += ["", f"Drops (≥6 dB phrase jumps): {drops}", ""]
    return "\n".join(out)


SONG_STORY_TEMPLATE = """# Song story — {title}

> TEMPLATE. The director (Claude) fills this from lyrics-passage.md + song-map.md.
> Write only what the lyrics and the measured analysis support; say "from the analysis", never
> "I listened". Leave a heading as "unknown" rather than inventing meaning.
{instrumental_note}
## Theme
(one or two sentences: what the song is about, quoting the lines that show it)

## Characters / voice
(who sings to whom; point of view)

## Mood arc
(how the feeling moves across the sections — use the section numbers and energy from song-map.md)

## Key images per section
| # | level | time | key image / idea | lyric anchor |
|---|---|---|---|---|
{section_rows}
## Lines worth kinetic type
(3–6 lines with their times from lyrics-passage.md)

## Open questions
(anything unclear — slang, references, language)
"""


README_TEMPLATE = """# Song folder — {title}

Built by the music-video skill (`run.py song-folder`) from `{source}` on the analysis alone (nobody listened).
All times are **seconds** from the start of passage{ext}. Video frames are at **{fps} fps**
(frame = floor(t × {fps})); {frames} frames for {duration} s.

| File | What it is |
|---|---|
| passage{ext} | the audio, byte-identical to the source (sha256 `{sha}`) — use as-is in the final mux |
| song-map.json | `duration`, `bpm`, `beats` [t], `downbeats` [t], `bars` [{{bar, start, end, beats, energy_db}}], `sections` [{{section, level calm/build/high, start, end, bars [first, last], energy 0..1, energy_db}}], `beat_energy_db` / `beat_energy` (0..1) per beat, `drops` [t] |
| song-map.md | human table: section · start–end · bars · energy · lyric lines |
| stems/ | {stems_line} |
| envelopes.json | `tracks.<{tracks}>.db` (RMS dBFS per frame) and `.level` (0..1 over the track's own range: its 5th-percentile non-silent frame = 0, loudest = 1, silence = 0; compare tracks with `db`); arrays have `frames` entries |
| hits.json | `kick` / `snare` / `hat`: [{{t, frame, strength 0..1}}] from {hits_source}; `counts`, `per_second` |
| notes.json | {notes_line} |
| lyrics-passage.json | `lines` [{{line, start, end, text, words [{{w, start, end, …}}]}}]; whisper-cli on {lyrics_source}{lyrics_line} |
| lyrics-passage.md | the same as a readable list |
| song-story.md | TEMPLATE for the director to fill (theme, characters, mood arc, key images per section) |

Analysis notes: {notes}
"""


# --- main --------------------------------------------------------------------------------

def build_song_folder(audio: Path, out_dir: Path, fps: int = 30, lyrics_lang: str = "en",
                      lyrics_text: Optional[Path] = None, whisper_model: Path = DEFAULT_WHISPER_MODEL,
                      beats_backend: str = "auto", stems: bool = True, lyrics: bool = True,
                      log: Callable[[str], None] = print) -> Dict[str, Any]:
    audio, out_dir = Path(audio).resolve(), Path(out_dir).resolve()
    if not audio.is_file():
        raise FileNotFoundError(audio)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"out_dir": str(out_dir), "warnings": []}

    passage, digest = copy_passage(audio, out_dir)
    log(f"passage: {passage.name} (sha256 {digest[:12]}…, identical)")

    info = analyze_music(passage, backend=beats_backend)
    mix = decode_mono(passage, ENV_SAMPLE_RATE)
    duration = len(mix) / ENV_SAMPLE_RATE
    song_map = build_song_map(info, duration)
    song_map.update({"source": audio.name, "passage": passage.name, "sha256": digest})
    dump_json(song_map, out_dir / "song-map.json")
    log(f"song-map: {song_map['bpm']} BPM ({song_map['beat_backend']}), {song_map['beat_count']} beats, "
        f"{song_map['bar_count']} bars, {len(song_map['sections'])} sections, {len(song_map['drops'])} drops")
    if song_map["beat_backend"] != "beat_this":
        report["warnings"].append(f"beats from '{song_map['beat_backend']}' (no tracked downbeats: bar phase "
                                  "guessed from accents, assumes 4/4) - install beat-this for real downbeats")

    stem_files: Optional[Dict[str, Path]] = None
    if stems:
        try:
            stem_files = run_demucs(passage, out_dir)
            log("stems: " + ", ".join(p.name for p in stem_files.values()))
        except (AudioToolError, subprocess.CalledProcessError) as exc:
            report["warnings"].append(f"stems skipped: {exc}")
            log(f"stems: SKIPPED — {exc}")
    else:
        log("stems: skipped (--no-stems)")

    tracks = {"mix": mix}
    for name, path in (stem_files or {}).items():
        tracks[name] = decode_mono(path, ENV_SAMPLE_RATE)
    env = build_envelopes(tracks, ENV_SAMPLE_RATE, fps, duration)
    dump_json(env, out_dir / "envelopes.json")
    log(f"envelopes: {env['frames']} frames × {len(env['tracks'])} tracks")

    if stem_files:
        hits = build_hits(decode_mono(stem_files["drums"], HIT_SAMPLE_RATE), HIT_SAMPLE_RATE, fps, duration)
    else:
        hits = build_hits(decode_mono(passage, HIT_SAMPLE_RATE), HIT_SAMPLE_RATE, fps, duration,
                          source=passage.name)
    dump_json(hits, out_dir / "hits.json")
    log(f"hits: {hits['counts']} ({hits['per_second']} /s)")

    events, notes_msg = (run_basic_pitch({"melody": stem_files["other"], "bass": stem_files["bass"]})
                         if stem_files else (None, "needs stems (demucs); notes.json skipped"))
    if events is not None:
        notes = build_notes(events, fps)
        notes["tool"] = notes_msg
        dump_json(notes, out_dir / "notes.json")
        log(f"notes: melody {notes['melody_count']}, bass {notes['bass_count']}")
        notes_line = "basic-pitch notes: `melody` (other stem) / `bass`: [{start, end, pitch (MIDI), name, amp, frame}]"
    else:
        (out_dir / "notes.json").unlink(missing_ok=True)
        report["warnings"].append(notes_msg)
        log(f"notes: SKIPPED — {notes_msg}")
        notes_line = f"not built: {notes_msg}"

    # lyrics (on the vocals stem when there is one, else on the mix)
    vocal_track = env["tracks"].get("vocals")
    if vocal_track:
        vocal_db = np.asarray(vocal_track["db"])
        activity: Optional[float] = float(np.mean(vocal_db > VOCAL_ACTIVE_DB)) if len(vocal_db) else 0.0
        instrumental = activity < MIN_VOCAL_ACTIVITY
    else:
        activity, instrumental = None, False
    lyr_media = stem_files["vocals"] if stem_files else passage
    lyr_source = "the vocals stem" if stem_files else "the full mix (no stems: expect more errors)"
    known = Path(lyrics_text).read_text(encoding="utf-8") if lyrics_text else None
    segments: List[Dict[str, Any]] = []
    if lyrics and (not instrumental or known):
        try:
            segments = transcribe_files([("vocals", lyr_media)], model=whisper_model,
                                        language=lyrics_lang).get("vocals", [])
        except (AudioToolError, subprocess.CalledProcessError) as exc:
            report["warnings"].append(f"whisper failed: {exc}")
            log(f"lyrics: whisper failed — {exc}")
    whisper_lines = lines_from_segments(segments, vocal_track["level"] if vocal_track else None, fps)
    lyr: Dict[str, Any] = {"language": lyrics_lang,
                           "vocal_activity": round(activity, 3) if activity is not None else None,
                           "instrumental": instrumental, "units": "seconds"}
    if known:
        words = [w for ln in whisper_lines if not ln["suspect"] for w in ln["words"]]
        lyr.update({"source": f"known lyrics aligned to whisper-cli words ({lyr_source})",
                    "lines": align_known_lyrics(known, words, duration), "whisper_lines": whisper_lines})
    else:
        lyr.update({"source": f"whisper-cli on {lyr_source} (unchecked — fix against the real lyrics)"
                    if lyrics else "lyrics skipped (--no-lyrics)", "lines": whisper_lines})
    dump_json(lyr, out_dir / "lyrics-passage.json")
    (out_dir / "lyrics-passage.md").write_text(lyrics_md(lyr), encoding="utf-8")
    act_txt = f"vocal activity {activity:.1%}" if activity is not None else "vocal activity unknown (no stems)"
    log(f"lyrics: {len(lyr['lines'])} lines ({act_txt}{', instrumental' if instrumental else ''})")

    title = audio.stem.replace("_", " ")
    (out_dir / "song-map.md").write_text(song_map_md(song_map, lyr["lines"], title), encoding="utf-8")
    story = out_dir / "song-story.md"
    if not story.is_file():                       # never overwrite the director's filled-in story
        rows = "".join(f"| {s['section']} | {s['level']} | {fmt_time(s['start'])} – {fmt_time(s['end'])} | | |\n"
                       for s in song_map["sections"])
        note = ("\n> Analysis found no sung vocals: the story must come from the music (energy arc, "
                "instruments), the title and the brief.\n" if instrumental and not known else "")
        story.write_text(SONG_STORY_TEMPLATE.format(title=title, section_rows=rows, instrumental_note=note),
                         encoding="utf-8")
    lyrics_line = (", with the known lyrics from --lyrics-text aligned to its words (`matched` false = "
                   "interpolated time); `whisper_lines` keeps the raw transcript" if known else
                   " (raw; `suspect` = vocals stem silent there)")
    stems_line = (f"demucs {DEMUCS_MODEL} stems: vocals.wav, drums.wav, bass.wav, other.wav (44.1 kHz stereo)"
                  if stem_files else "not built (demucs missing or --no-stems): everything below comes from the mix")
    (out_dir / "README.md").write_text(README_TEMPLATE.format(
        title=title, source=audio.name, ext=passage.suffix, fps=fps, frames=env["frames"],
        duration=song_map["duration"], sha=digest, stems_line=stems_line, tracks="|".join(env["tracks"]),
        hits_source=hits["source"], lyrics_source=lyr_source, notes_line=notes_line,
        lyrics_line=lyrics_line, notes=song_map["notes"]), encoding="utf-8")
    report.update({"bpm": song_map["bpm"], "beat_backend": song_map["beat_backend"],
                   "duration": song_map["duration"], "beats": song_map["beat_count"],
                   "bars": song_map["bar_count"], "frames": env["frames"], "stems": bool(stem_files),
                   "hits": hits["counts"], "lyric_lines": len(lyr["lines"]), "instrumental": instrumental,
                   "notes": events is not None})
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="run.py song-folder", description=__doc__.split("\n\n")[0])
    ap.add_argument("audio", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--lyrics-lang", default="en", help="whisper language code (en, tr, …)")
    ap.add_argument("--lyrics-text", type=Path, default=None, help="known lyrics (one line per line)")
    ap.add_argument("--whisper-model", type=Path, default=DEFAULT_WHISPER_MODEL)
    ap.add_argument("--beats", default="auto", choices=("auto", "beat_this", "aubio", "numpy"),
                    help="beat tracker (auto = Beat This! > aubio CLI > built-in numpy)")
    ap.add_argument("--no-stems", action="store_true", help="skip demucs; analyse the mix only")
    ap.add_argument("--no-lyrics", action="store_true", help="skip whisper")
    args = ap.parse_args(argv)
    report = build_song_folder(args.audio, args.out_dir, fps=args.fps, lyrics_lang=args.lyrics_lang,
                               lyrics_text=args.lyrics_text, whisper_model=args.whisper_model,
                               beats_backend=args.beats, stems=not args.no_stems, lyrics=not args.no_lyrics)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
