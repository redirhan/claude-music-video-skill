"""Music analysis: beats, BPM, downbeats/bars, per-beat energy, energy sections and drops.

Beat tracking backends, best first ("auto" picks the first one that works):
  1. Beat This! (CPJKU, MIT) - pip install beat-this torch. Real downbeats, steady on half-time parts.
  2. aubio - called ONLY as an external command-line tool (`aubio beat`). aubio is GPL-3.0, so it is
     never imported or bundled here; install it yourself (brew install aubio / apt install aubio-tools).
  3. numpy - a small built-in tracker (onset envelope -> tempo by autocorrelation -> dynamic
     programming beat placement). Needs nothing but numpy + ffmpeg; less accurate, no real downbeats.

Audio is decoded by ffmpeg straight into memory; nothing is written next to the source file.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class AudioToolError(RuntimeError):
    """A local audio tool is missing or failed."""


def require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        hint = {"aubio": "brew install aubio (macOS) / apt install aubio-tools (Linux)",
                "whisper-cli": "brew install whisper-cpp, or build github.com/ggml-org/whisper.cpp"}.get(tool, tool)
        raise AudioToolError(f"'{tool}' not found on PATH (install: {hint})")
    return path


def decode_mono(path: Path, sample_rate: int) -> np.ndarray:
    """Decode an audio/video file's first audio stream to mono float32 in [-1, 1]."""
    ffmpeg = require("ffmpeg")
    raw = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate),
                          "-f", "s16le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def rms_db(samples: np.ndarray, window: int) -> np.ndarray:
    """dBFS of consecutive non-overlapping windows (silence floored at -90 dB)."""
    n = len(samples) // window
    if n == 0:
        return np.zeros(0)
    frames = samples[: n * window].reshape(n, window)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return np.maximum(20 * np.log10(np.maximum(rms, 1e-9)), -90.0)


def _median_bpm(beats: List[float]) -> float:
    intervals = sorted(b - a for a, b in zip(beats, beats[1:]))
    return round(60.0 / intervals[len(intervals) // 2], 1)


# --- beat backends ---------------------------------------------------------------------

def parse_beats(text: str) -> List[float]:
    """aubio prints one beat time (seconds) per line."""
    return sorted(round(float(line.split()[0]), 3) for line in text.splitlines() if line.strip())


def detect_beats_aubio(music_path: Path) -> Tuple[List[float], float]:
    """(beats, median BPM) from the external `aubio beat` command (GPL tool, run as a separate process)."""
    aubio, ffmpeg = require("aubio"), require("ffmpeg")
    with tempfile.TemporaryDirectory(prefix="mv_beats_") as tmp:
        wav = Path(tmp) / "music.wav"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(music_path), "-vn", "-ac", "1", "-ar", "44100",
                        str(wav)], check=True, capture_output=True)
        out = subprocess.run([aubio, "beat", str(wav)], check=True, capture_output=True, text=True).stdout
    beats = parse_beats(out)
    if len(beats) < 4:
        raise AudioToolError(f"too few beats detected in {music_path} ({len(beats)})")
    return beats, _median_bpm(beats)


BEAT_THIS_CHECKPOINT = "final0"


def detect_beats_neural(music_path: Path, device: str = "cpu") -> Tuple[List[float], List[float], float]:
    """(beats, downbeats, median BPM) with Beat This! (ISMIR 2024). CPU by default so reruns are identical.
    The model weights are downloaded once into the torch hub cache."""
    try:
        from beat_this.inference import Audio2Beats
    except ImportError as exc:
        raise AudioToolError("beat-this is not installed (pip install beat-this torch)") from exc
    sr = 22050
    signal = decode_mono(Path(music_path), sr)
    try:
        tracker = Audio2Beats(checkpoint_path=BEAT_THIS_CHECKPOINT, device=device, dbn=False)
        beats, downbeats = tracker(signal, sr)
    except Exception as exc:  # missing checkpoint, offline first run, ...
        raise AudioToolError(f"Beat This! failed on {music_path}: {exc}") from exc
    beats = sorted(round(float(b), 3) for b in beats)
    downbeats = sorted(round(float(d), 3) for d in downbeats)
    if len(beats) < 4:
        raise AudioToolError(f"too few beats detected in {music_path} ({len(beats)})")
    return beats, downbeats, _median_bpm(beats)


def onset_envelope(samples: np.ndarray, sample_rate: int, hop: int = 512, n_fft: int = 2048) -> np.ndarray:
    """Log-magnitude spectral flux (positive part, summed over bins), mean-removed, >= 0."""
    pad = np.concatenate([np.zeros(n_fft // 2), samples.astype(np.float64), np.zeros(n_fft)])
    count = max(1, 1 + (len(pad) - n_fft) // hop)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(count)[:, None]
    mag = np.log1p(100.0 * np.abs(np.fft.rfft(pad[idx] * np.hanning(n_fft)[None, :], axis=1)))
    flux = np.maximum(np.diff(mag, axis=0, prepend=mag[:1]), 0.0).sum(axis=1)
    k = max(1, int(round(0.5 * sample_rate / hop)))
    local = np.convolve(flux, np.ones(k) / k, mode="same")
    return np.maximum(flux - local, 0.0)


def detect_beats_numpy(samples: np.ndarray, sample_rate: int, hop: int = 512,
                       bpm_range: Tuple[float, float] = (60.0, 180.0), prior_bpm: float = 120.0,
                       tightness: float = 100.0) -> Tuple[List[float], float]:
    """Built-in fallback tracker (after Ellis 2007, "Beat tracking by dynamic programming").

    Tempo = autocorrelation peak of the onset envelope, weighted by a log-normal prior around
    `prior_bpm`; beats = the path that maximises onset strength while keeping the spacing close to
    that tempo."""
    env = onset_envelope(samples, sample_rate, hop)
    if len(env) < 16 or env.max() <= 0:
        raise AudioToolError("no onsets found (silent or too short audio)")
    env = env / env.std()
    fps = sample_rate / hop
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    lags = np.arange(len(ac))
    lo, hi = int(fps * 60.0 / bpm_range[1]), int(fps * 60.0 / bpm_range[0])
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        raise AudioToolError("audio too short for tempo estimation")
    bpm_at = 60.0 * fps / np.maximum(lags[lo:hi + 1], 1)
    weight = np.exp(-0.5 * (np.log2(bpm_at / prior_bpm) / 0.9) ** 2)
    period = lo + int(np.argmax(ac[lo:hi + 1] * weight))
    # dynamic programming
    score = env.copy()
    back = np.full(len(env), -1)
    window = np.arange(-2 * period, -(period // 2) + 1)
    penalty = -tightness * np.log(-window / period) ** 2
    for i in range(len(env)):
        prev = i + window
        ok = prev >= 0
        if not ok.any():
            continue
        cand = score[prev[ok]] + penalty[ok]
        j = int(np.argmax(cand))
        if cand[j] > 0:
            score[i] += cand[j]
            back[i] = prev[ok][j]
    # end at the best-scoring frame in the last period
    tail = score[-period:] if len(score) > period else score
    i = len(score) - len(tail) + int(np.argmax(tail))
    path = []
    while i >= 0:
        path.append(i)
        i = back[i]
    beats = sorted(round(p * hop / sample_rate, 3) for p in path)
    if len(beats) < 4:
        raise AudioToolError(f"too few beats detected ({len(beats)})")
    # BPM from the placed beats (the autocorrelation lag is quantised to whole hops)
    iv = np.diff(beats)
    iv = iv[np.abs(iv - np.median(iv)) < 0.25 * np.median(iv)]
    bpm = 60.0 / float(iv.mean()) if len(iv) else 60.0 * fps / period
    return beats, round(bpm, 1)


def track_beats(music_path: Path, backend: str = "auto") -> Dict[str, Any]:
    """Beats (+ downbeats when the backend provides them). backend: auto | beat_this | aubio | numpy."""
    errors = []
    if backend in ("auto", "beat_this"):
        try:
            beats, downbeats, bpm = detect_beats_neural(music_path)
            return {"beats": beats, "downbeats": downbeats, "bpm": bpm, "backend": "beat_this"}
        except AudioToolError as exc:
            if backend == "beat_this":
                raise
            errors.append(str(exc))
    if backend in ("auto", "aubio"):
        try:
            beats, bpm = detect_beats_aubio(music_path)
            return {"beats": beats, "downbeats": None, "bpm": bpm, "backend": "aubio"}
        except (AudioToolError, subprocess.CalledProcessError) as exc:
            if backend == "aubio":
                raise
            errors.append(str(exc))
    if backend in ("auto", "numpy"):
        sr = 22050
        beats, bpm = detect_beats_numpy(decode_mono(Path(music_path), sr), sr)
        return {"beats": beats, "downbeats": None, "bpm": bpm, "backend": "numpy",
                "backend_notes": errors}
    raise AudioToolError(f"unknown beat backend {backend!r}")


# --- structure --------------------------------------------------------------------------

def _downbeat_phase(beats: List[float], onset_db: np.ndarray, hop_s: float) -> int:
    """Bar phase (0-3) whose beats carry the strongest accents (4/4 guess)."""
    strength = []
    for b in beats:
        k = int(round(b / hop_s))
        strength.append(float(onset_db[max(0, k - 1):k + 2].max()) if k < len(onset_db) else 0.0)
    scores = [np.mean(strength[p::4]) if strength[p::4] else 0.0 for p in range(4)]
    return int(np.argmax(scores))


def _downbeat_indices(beats: List[float], downbeats: List[float]) -> List[int]:
    arr = np.asarray(beats)
    return sorted({int(np.argmin(np.abs(arr - d))) for d in downbeats})


def analyze_music_signal(samples: np.ndarray, sample_rate: int, beats: List[float],
                         downbeats: Optional[List[float]] = None) -> Dict[str, Any]:
    """Structure from a decoded signal and its beat list (pure; testable).

    With tracked downbeats bars follow the music (meter changes included); without them the bar
    phase is guessed from beat accents (4/4). Sections are two-bar phrases classified by the song's
    own energy tertiles (calm / build / high); drops are phrase starts with a >= 6 dB energy jump."""
    duration = len(samples) / sample_rate
    hop = int(0.02 * sample_rate)
    env = rms_db(samples, hop)
    onset = np.maximum(np.diff(env, prepend=env[:1]), 0.0)
    beat_energy = []
    for a, b in zip(beats, beats[1:] + [min(duration, beats[-1] + (beats[-1] - beats[-2]))]):
        seg = samples[int(a * sample_rate):int(b * sample_rate)]
        beat_energy.append(round(float(rms_db(seg, max(1, len(seg)))[0]) if len(seg) else -90.0, 1))
    bar_idx = _downbeat_indices(beats, downbeats) if downbeats else []
    if len(bar_idx) < 2:
        phase = _downbeat_phase(beats, onset, 0.02)
        bar_idx = list(range(phase, len(beats), 4))
    downbeats = [round(beats[i], 3) for i in bar_idx]
    phrase_starts = bar_idx[::2]
    phrase_ends = phrase_starts[1:] + [len(beats)]
    phrase_energy = [float(np.mean(beat_energy[i:j])) for i, j in zip(phrase_starts, phrase_ends)]
    lo, hi = (np.percentile(phrase_energy, [33, 67]) if phrase_energy else (0, 0))
    sections = []
    for idx, i in enumerate(phrase_starts):
        level = "calm" if phrase_energy[idx] <= lo else "high" if phrase_energy[idx] >= hi else "build"
        start = beats[i]
        j = phrase_ends[idx]
        end = beats[j] if j < len(beats) else duration
        if sections and sections[-1]["level"] == level:
            sections[-1]["end"] = round(end, 3)
        else:
            sections.append({"start": round(start, 3), "end": round(end, 3), "level": level})
    drops = [round(beats[phrase_starts[k]], 3) for k in range(1, len(phrase_starts))
             if phrase_energy[k] - phrase_energy[k - 1] >= 6.0]
    return {"duration_seconds": round(duration, 3), "beats": [round(b, 3) for b in beats],
            "downbeats": downbeats, "beat_energy_db": beat_energy,
            "mean_energy_db": round(float(np.mean(beat_energy)), 1) if beat_energy else -90.0,
            "sections": sections, "drops": drops}


def analyze_music(path: Path, sample_rate: int = 22050, backend: str = "auto") -> Dict[str, Any]:
    """Full music analysis of one file (read-only)."""
    tracked = track_beats(Path(path), backend)
    info = analyze_music_signal(decode_mono(Path(path), sample_rate), sample_rate, tracked["beats"],
                                tracked["downbeats"])
    info.update({"path": str(Path(path).resolve()), "bpm": tracked["bpm"], "beat_backend": tracked["backend"]})
    return info
