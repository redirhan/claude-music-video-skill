"""Song folder builder (musicvideo/song_folder.py) on synthetic audio; demucs, basic-pitch,
whisper and the beat tracker are replaced so the tests stay fast and offline."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from musicvideo import song_folder as sf

SR = 44100
RNG = np.random.default_rng(0)


def _kick(n=int(0.15 * SR)):
    t = np.arange(n) / SR
    f = 60 + 80 * np.exp(-t * 40)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 18)


def _snare(n=int(0.12 * SR)):
    t = np.arange(n) / SR
    return (0.5 * np.sin(2 * np.pi * 200 * t) + RNG.normal(0, 0.6, n)) * np.exp(-t * 30)


def _hat(n=int(0.04 * SR)):
    t = np.arange(n) / SR
    x = RNG.normal(0, 1, n)
    x = np.diff(np.diff(x, prepend=0), prepend=0) / 4
    return x * np.exp(-t * 120) * 0.5


def drum_loop(duration=8.0, beat=0.5, start=0.25):
    """Kick on beats 1/3, snare on 2/4, hats on eighths (120 BPM)."""
    y = np.zeros(int(duration * SR) + SR)
    kicks, snares, hats = [], [], []
    for i in range(int((duration - start) / beat)):
        b = start + i * beat
        (kicks if i % 2 == 0 else snares).append(b)
        hats += [b, b + beat / 2]

    def add(x, t):
        a = int(t * SR)
        y[a:a + len(x)] += x

    for t in kicks:
        add(_kick(), t)
    for t in snares:
        add(_snare(), t)
    for t in hats:
        add(_hat(), t)
    y = y[:int(duration * SR)]
    return (0.5 * y / np.abs(y).max()).astype(np.float32), kicks, snares, [h for h in hats if h < duration - 0.05]


def _match(got, ref, tol=0.02):
    got = np.asarray(got)
    return sum(1 for r in ref if len(got) and np.min(np.abs(got - r)) <= tol)


def test_detect_hits_separates_kick_snare_hat():
    y, kicks, snares, hats = drum_loop()
    hits = sf.detect_hits(y, SR, 30)
    k, s, h = ([x["t"] for x in hits[c]] for c in ("kick", "snare", "hat"))
    assert len(k) == len(kicks) and _match(k, kicks) == len(kicks)
    assert len(s) == len(snares) and _match(s, snares) == len(snares)
    assert _match(h, hats) >= len(hats) - 1 and len(h) <= len(hats) + 1
    assert all(0 < x["strength"] <= 1 for c in hits.values() for x in c)
    assert all(x["frame"] == int(x["t"] * 30 + 1e-6) for x in hits["kick"])


def test_frame_db_and_envelopes():
    dur, fps = 2.0, 30
    t = np.arange(int(dur * SR)) / SR
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    tone[: SR // 2] = 0.0                                   # first half second silent
    env = sf.build_envelopes({"mix": tone, "vocals": tone * 0.1}, SR, fps, dur)
    assert env["frames"] == 60
    mix = env["tracks"]["mix"]
    assert len(mix["db"]) == len(mix["level"]) == 60
    assert mix["db"][0] == -90.0 and mix["level"][0] == 0.0
    assert abs(mix["db"][40] - 20 * np.log10(0.5 / np.sqrt(2))) < 0.2       # sine RMS
    assert max(mix["level"]) == 1.0 and mix["level"][40] == 1.0
    assert abs(env["tracks"]["vocals"]["db"][40] - (mix["db"][40] - 20)) < 0.2


def test_build_song_map_bars_and_sections():
    beats = [round(0.5 * i, 3) for i in range(16)]
    info = {"beats": beats, "downbeats": beats[::4], "beat_energy_db": [-30.0] * 8 + [-10.0] * 8,
            "sections": [{"start": 0.0, "end": 4.0, "level": "calm"}, {"start": 4.0, "end": 8.0, "level": "high"}],
            "drops": [4.0], "bpm": 120.0, "beat_backend": "test", "mean_energy_db": -20.0}
    m = sf.build_song_map(info, 8.0)
    assert m["bar_count"] == 4 and m["beat_count"] == 16
    assert [b["beats"] for b in m["bars"]] == [4, 4, 4, 4]
    assert m["bars"][-1]["end"] == 8.0
    assert m["sections"][0]["bars"] == [0, 1] and m["sections"][1]["bars"] == [2, 3]
    assert m["sections"][1]["energy"] == 1.0 and m["sections"][0]["energy"] == 0.0
    assert m["beat_energy"][0] == 0.0 and m["beat_energy"][-1] == 1.0


def test_align_known_lyrics_keeps_text_and_takes_times():
    heard = [{"w": w, "start": s, "end": s + 0.3} for w, s in
             (("hello", 1.0), ("darknes", 1.5), ("my", 2.0), ("old", 2.4), ("frend", 2.8),
              ("um", 3.3), ("i've", 4.0), ("come", 4.4), ("to", 4.8), ("talk", 5.0))]
    known = "Hello darkness, my old friend\nI've come to talk with you again\n"
    lines = sf.align_known_lyrics(known, heard, duration=10.0)
    assert [ln["text"] for ln in lines] == ["Hello darkness, my old friend", "I've come to talk with you again"]
    first = lines[0]["words"]
    assert first[1]["w"] == "darkness," and first[1]["start"] == 1.5 and first[1]["matched"]
    assert first[4]["start"] == 2.8                         # "friend" ~ "frend"
    second = lines[1]["words"]
    assert [w["matched"] for w in second] == [True, True, True, True, False, False, False]
    starts = [w["start"] for w in second]
    assert starts == sorted(starts) and second[4]["start"] >= 5.0   # interpolated after "talk"
    assert lines[0]["start"] == 1.0 and lines[1]["start"] == 4.0


def test_dump_json_keeps_number_lists_on_one_line(tmp_path):
    data = {"a": [1.0, 2.5, -3], "b": [{"t": 1, "s": 0.5}, {"w": "[x]", "n": None}], "c": {"d": {"e": 1}}}
    text = sf.dump_json(data, tmp_path / "x.json").read_text()
    assert "[1.0, 2.5, -3]" in text and '{"t": 1, "s": 0.5}' in text
    assert json.loads(text) == data


def _write_wav(path: Path, y: np.ndarray, sr: int = SR):
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg needed to decode")
def test_build_song_folder_end_to_end(tmp_path, monkeypatch):
    drums, kicks, _, _ = drum_loop(duration=6.0)
    t = np.arange(len(drums)) / SR
    voice = (0.3 * np.sin(2 * np.pi * 330 * t) * (t > 2.0)).astype(np.float32)
    mix = 0.6 * drums + voice
    src = tmp_path / "song.wav"
    _write_wav(src, mix)

    beats = [round(0.25 + 0.5 * i, 3) for i in range(11)]

    def fake_analyze(path, **kw):
        return {"beats": beats, "downbeats": beats[::4], "beat_energy_db": [-20.0] * len(beats),
                "sections": [{"start": 0.25, "end": 6.0, "level": "high"}], "drops": [], "bpm": 120.0,
                "beat_backend": "fake", "mean_energy_db": -20.0}

    def fake_demucs(passage, out_dir):
        paths = sf.stem_paths(out_dir)
        paths["vocals"].parent.mkdir(parents=True, exist_ok=True)
        for name, y in (("vocals", voice), ("drums", drums), ("bass", drums * 0.1), ("other", voice * 0.1)):
            _write_wav(paths[name], y)
        return paths

    calls = {}

    def fake_whisper(media, model=None, language="tr"):
        calls["language"] = language
        calls["media"] = media
        return {"vocals": [
            {"start": 0.5, "end": 1.5, "text": "ghost words", "p": 0.9,
             "words": [{"w": "ghost", "start": 0.5, "end": 1.0}, {"w": "words", "start": 1.0, "end": 1.5}]},
            {"start": 2.5, "end": 4.0, "text": "sing along now", "p": 0.9,
             "words": [{"w": "sing", "start": 2.5, "end": 3.0}, {"w": "along", "start": 3.0, "end": 3.5},
                       {"w": "now", "start": 3.5, "end": 4.0}]}]}

    monkeypatch.setattr(sf, "analyze_music", fake_analyze)
    monkeypatch.setattr(sf, "run_demucs", fake_demucs)
    monkeypatch.setattr(sf, "transcribe_files", fake_whisper)
    monkeypatch.setattr(sf, "run_basic_pitch",
                        lambda stems: ({"melody": [[2.0, 2.5, 64, 0.8], [3.0, 3.2, 40, 0.1]], "bass": []}, "fake"))
    known = tmp_path / "lyrics.txt"
    known.write_text("Sing along now\n", encoding="utf-8")

    out = tmp_path / "folder"
    report = sf.build_song_folder(src, out, fps=25, lyrics_lang="tr", lyrics_text=known, log=lambda m: None)

    assert (out / "passage.wav").read_bytes() == src.read_bytes()
    for name in ("song-map.json", "song-map.md", "envelopes.json", "hits.json", "notes.json",
                 "lyrics-passage.json", "lyrics-passage.md", "song-story.md", "README.md"):
        assert (out / name).is_file(), name
    env = json.loads((out / "envelopes.json").read_text())
    assert env["fps"] == 25 and env["frames"] == 150 == report["frames"]
    assert set(env["tracks"]) == {"mix", "vocals", "drums", "bass", "other"}
    assert all(len(tr["db"]) == 150 for tr in env["tracks"].values())
    hits = json.loads((out / "hits.json").read_text())
    assert _match([h["t"] for h in hits["kick"]], kicks) == len(kicks)
    notes = json.loads((out / "notes.json").read_text())
    assert notes["melody_count"] == 1 and notes["melody"][0]["name"] == "E4"
    lyr = json.loads((out / "lyrics-passage.json").read_text())
    assert calls["language"] == "tr" and calls["media"][0][1].name == "vocals.wav"
    assert lyr["whisper_lines"][0]["suspect"] and not lyr["whisper_lines"][1]["suspect"]
    assert lyr["lines"][0]["text"] == "Sing along now" and lyr["lines"][0]["start"] == 2.5
    md = (out / "song-map.md").read_text()
    assert "| 0 | high |" in md and "Sing along now" in md
    story = (out / "song-story.md").read_text()
    assert "TEMPLATE" in story and "## Theme" in story
    assert "25 fps" in (out / "README.md").read_text()

    # rerun: the director's song-story is never overwritten; passage still identical
    story_path = out / "song-story.md"
    story_path.write_text("filled in", encoding="utf-8")
    sf.build_song_folder(src, out, fps=25, lyrics_lang="tr", lyrics_text=known, log=lambda m: None)
    assert story_path.read_text() == "filled in"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg needed to decode")
def test_instrumental_skips_whisper_and_missing_notes(tmp_path, monkeypatch):
    drums, _, _, _ = drum_loop(duration=3.0)
    src = tmp_path / "beat.wav"
    _write_wav(src, drums)
    beats = [round(0.25 + 0.5 * i, 3) for i in range(5)]
    monkeypatch.setattr(sf, "analyze_music", lambda p, **k: {
        "beats": beats, "downbeats": beats[::4], "beat_energy_db": [-20.0] * 5,
        "sections": [{"start": 0.25, "end": 3.0, "level": "high"}], "drops": [], "bpm": 120.0,
        "beat_backend": "fake", "mean_energy_db": -20.0})

    def fake_demucs(passage, out_dir):
        paths = sf.stem_paths(out_dir)
        paths["vocals"].parent.mkdir(parents=True, exist_ok=True)
        for name in sf.STEM_NAMES:
            _write_wav(paths[name], drums if name == "drums" else drums * 0.0)
        return paths

    monkeypatch.setattr(sf, "run_demucs", fake_demucs)
    monkeypatch.setattr(sf, "transcribe_files", lambda *a, **k: pytest.fail("whisper must not run"))
    monkeypatch.setattr(sf, "run_basic_pitch", lambda stems: (None, "basic-pitch not installed"))
    report = sf.build_song_folder(src, tmp_path / "out", log=lambda m: None)
    assert report["instrumental"] and not report["notes"]
    assert not (tmp_path / "out" / "notes.json").exists()
    assert "basic-pitch not installed" in (tmp_path / "out" / "README.md").read_text()
    assert "No sung vocals" in (tmp_path / "out" / "lyrics-passage.md").read_text()


def test_copy_passage_detects_mismatch(tmp_path, monkeypatch):
    src = tmp_path / "a.MP3"
    src.write_bytes(b"abc" * 1000)
    dst, digest = sf.copy_passage(src, tmp_path)
    assert dst.name == "passage.mp3" and dst.read_bytes() == src.read_bytes()
    monkeypatch.setattr(sf.shutil, "copyfile", lambda a, b: Path(b).write_bytes(b"corrupt"))
    dst.write_bytes(b"old")
    with pytest.raises(RuntimeError):
        sf.copy_passage(src, tmp_path)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg needed to decode")
def test_no_stems_builds_from_the_mix(tmp_path, monkeypatch):
    """Without demucs the folder is still built: mix-only envelopes, hits from the mix, no notes."""
    drums, kicks, _, _ = drum_loop(duration=6.0)
    src = tmp_path / "beat.wav"
    _write_wav(src, drums)
    monkeypatch.setattr(sf, "transcribe_files", lambda *a, **k: {"vocals": []})
    report = sf.build_song_folder(src, tmp_path / "out", beats_backend="numpy", stems=False, log=lambda m: None)
    out = tmp_path / "out"
    assert not report["stems"] and not report["notes"]
    env = json.loads((out / "envelopes.json").read_text())
    assert set(env["tracks"]) == {"mix"}
    hits = json.loads((out / "hits.json").read_text())
    assert hits["source"] == "passage.wav"
    assert _match([h["t"] for h in hits["kick"]], kicks) >= len(kicks) - 1
    song_map = json.loads((out / "song-map.json").read_text())
    assert song_map["beat_backend"] == "numpy" and abs(song_map["bpm"] - 120) < 3
    assert "/" not in song_map["source"]                      # no local paths leak into the folder
    assert "not built" in (out / "README.md").read_text()
