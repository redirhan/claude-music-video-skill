"""Built-in numpy beat tracker and the structure analysis on synthetic audio (no downloads)."""

import numpy as np

from musicvideo import analysis as an

SR = 22050


def click_track(bpm=128.0, dur=20.0, accent_every=4):
    y = np.zeros(int(dur * SR), dtype=np.float32)
    step = 60.0 / bpm
    n = int(0.03 * SR)
    t = np.arange(n) / SR
    click = (np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 120)).astype(np.float32)
    times = []
    k = 0
    while (k * step + 0.5) < dur - 0.1:
        a = int((k * step + 0.5) * SR)
        y[a:a + n] += click * (1.0 if k % accent_every == 0 else 0.5)
        times.append(k * step + 0.5)
        k += 1
    return 0.5 * y / np.abs(y).max(), times


def test_numpy_tracker_finds_tempo_and_beats():
    y, times = click_track()
    beats, bpm = an.detect_beats_numpy(y, SR)
    assert abs(bpm - 128.0) < 2.5
    hits = sum(1 for t in times if min(abs(b - t) for b in beats) < 0.05)
    assert hits >= 0.9 * len(times)


def test_structure_bars_from_accents_and_sections():
    y, times = click_track(dur=24.0)
    info = an.analyze_music_signal(y, SR, [round(t, 3) for t in times])
    assert info["downbeats"][0] == round(times[0], 3)        # accented click = bar start
    assert len(info["downbeats"]) == len(times[::4])
    assert info["sections"] and all(s["level"] in ("calm", "build", "high") for s in info["sections"])


def test_parse_aubio_output():
    assert an.parse_beats("0.50\n1.00 extra\n\n0.25\n") == [0.25, 0.5, 1.0]
