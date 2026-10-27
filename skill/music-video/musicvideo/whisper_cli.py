"""Word-timed transcription with the whisper.cpp command-line tool (`whisper-cli`, MIT).

The model file is read from MV_WHISPER_MODEL, else ~/.cache/whisper-cpp/ggml-large-v3-turbo-q5_0.bin.
Download a model with whisper.cpp's own script (models/download-ggml-model.sh large-v3-turbo-q5_0).
"""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .analysis import AudioToolError, require

DEFAULT_WHISPER_MODEL = Path(os.environ.get(
    "MV_WHISPER_MODEL", str(Path.home() / ".cache" / "whisper-cpp" / "ggml-large-v3-turbo-q5_0.bin")))
# whisper.cpp DTW token times lag the real word onsets by a roughly constant amount (measured ~0.12 s
# on speech against a forced alignment); subtract it.
DTW_LAG = 0.12
# whisper hallucinates subtitle credits on silence / music; matched as substrings of normalised text
_HALLUCINATIONS = ("altyazı mk", "izlediğiniz için teşekkür", "abone olmayı unutmayın", "altyazılar",
                   "thanks for watching", "subtitles by")
_MIN_SEGMENT_PROBABILITY = 0.45


def _normalize(text: str) -> str:
    return re.sub(r"[^\wçğıöşü ]", "", text.replace("İ", "i").replace("I", "ı").lower()).strip()


def is_hallucination(text: str) -> bool:
    norm = _normalize(text)
    return any(h in norm for h in _HALLUCINATIONS)


def _words_from_tokens(tokens: Iterable[Dict[str, Any]],
                       span: Tuple[float, float] = (0.0, 0.0)) -> List[Dict[str, Any]]:
    """Words [{w, start, end, p}] from whisper.cpp -ojf tokens of one segment. A token starting with a
    space opens a new word; other tokens extend it. A token without offsets takes the segment span."""
    words: List[Dict[str, Any]] = []
    for tok in tokens:
        text = str(tok.get("text", ""))
        if not text or text.startswith("[_"):
            continue
        offsets = tok.get("offsets")
        start, end = (offsets["from"] / 1000, offsets["to"] / 1000) if offsets else span
        p = float(tok.get("p", 0.0))
        if text.startswith(" ") or not words:
            words.append({"w": text.strip(), "start": start, "end": end, "_p": [p]})
        else:
            words[-1]["w"] += text
            words[-1]["end"] = max(words[-1]["end"], end)
            words[-1]["_p"].append(p)
    out = []
    for w in words:
        if not w["w"] or not re.search(r"\w", w["w"]):
            continue
        ps = w.pop("_p")
        out.append({"w": w["w"], "start": round(w["start"], 3), "end": round(max(w["end"], w["start"]), 3),
                    "p": round(sum(ps) / len(ps), 3)})
    return out


def words_from_tokens(tokens: Iterable[Dict[str, Any]],
                      span: Tuple[float, float] = (0.0, 0.0)) -> List[Dict[str, Any]]:
    """Like _words_from_tokens, but word starts come from DTW times (t_dtw, minus DTW_LAG) when present."""
    tokens = list(tokens)
    words = _words_from_tokens(tokens, span)
    if not any(t.get("t_dtw", -1) >= 0 for t in tokens):
        return words
    raw = [(str(t.get("text", "")), t.get("t_dtw", -1)) for t in tokens
           if t.get("text") and not str(t["text"]).startswith("[_")]
    texts: List[List[Any]] = []
    for tx, dtw in raw:
        if tx.startswith(" ") or not texts:
            texts.append([tx.strip(), dtw])
        else:
            texts[-1][0] += tx
    starts = [st for tx, st in texts if tx and re.search(r"\w", tx)]
    if len(starts) != len(words):
        return words
    for w, st in zip(words, starts):
        if st >= 0:
            w["start"] = round(max(0.0, st / 100 - DTW_LAG), 3)
    for i, w in enumerate(words):
        nxt = words[i + 1]["start"] if i + 1 < len(words) else None
        end = w["end"]
        if nxt is not None and nxt - w["start"] < 1.0:
            end = nxt
        w["end"] = round(max(end, w["start"] + 0.05), 3)
    return words


def parse_whisper_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Segments [{start, end, text, p, words}] from whisper.cpp -ojf output, without known
    hallucinations and low-confidence segments."""
    segments = []
    for seg in data.get("transcription", []):
        text = (seg.get("text") or "").strip()
        tokens = [t for t in seg.get("tokens", []) if not str(t.get("text", "")).startswith("[_")]
        p = sum(t.get("p", 0.0) for t in tokens) / len(tokens) if tokens else 0.0
        if not text or is_hallucination(text) or p < _MIN_SEGMENT_PROBABILITY:
            continue
        span = (seg["offsets"]["from"] / 1000, seg["offsets"]["to"] / 1000)
        segments.append({"start": round(span[0], 2), "end": round(span[1], 2), "text": text, "p": round(p, 3),
                         "words": words_from_tokens(tokens, span)})
    return segments


def dtw_preset(model: Path) -> Optional[str]:
    name = Path(model).name.lower()
    for key, preset in (("large-v3-turbo", "large.v3.turbo"), ("large-v3", "large.v3"), ("large-v2", "large.v2"),
                        ("medium", "medium"), ("small", "small"), ("base", "base"), ("tiny", "tiny")):
        if key in name:
            return preset
    return None


def transcribe_files(media: List[Tuple[str, Path]], model: Path = DEFAULT_WHISPER_MODEL,
                     language: str = "en") -> Dict[str, List[Dict[str, Any]]]:
    """Transcribe [(key, media_path)] with one whisper-cli run (audio decoded to a temp dir)."""
    if not media:
        return {}
    ffmpeg, whisper = require("ffmpeg"), require("whisper-cli")
    if not Path(model).is_file():
        raise AudioToolError(f"whisper model not found: {model} (set MV_WHISPER_MODEL)")
    results: Dict[str, List[Dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="mv_whisper_") as tmp:
        wavs = []
        for i, (key, path) in enumerate(media):
            wav = Path(tmp) / f"{i:05d}.wav"
            probe = subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                                    "-c:a", "pcm_s16le", str(wav)], capture_output=True)
            if probe.returncode == 0 and wav.is_file() and wav.stat().st_size > 1000:
                wavs.append((key, wav))
            else:
                results[key] = []
        if wavs:
            cmd = [whisper, "-m", str(model), "-l", language, "-ojf", "-np"]
            preset = dtw_preset(model)
            if preset:
                cmd += ["-nfa", "-dtw", preset]
            for _, wav in wavs:
                cmd += ["-f", str(wav)]
            subprocess.run(cmd, check=True, capture_output=True)
            for key, wav in wavs:
                out = Path(str(wav) + ".json")
                data = json.loads(out.read_text(encoding="utf-8", errors="replace")) if out.is_file() else {}
                results[key] = parse_whisper_json(data)
    return results
