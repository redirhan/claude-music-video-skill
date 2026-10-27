import re
from pathlib import Path

from musicvideo import dance2d as d

CHAR2D = Path(d.ANIM_DIR / "char2d.js").read_text()


def _analysis(bpm=100.0, bars=12, offset=1.0):
    step = 60.0 / bpm
    beats = [round(offset + i * step, 3) for i in range(bars * 4)]
    downbeats = beats[::4]
    energy = [-20.0 + (i % 4 == 0) * 3 + (i // 16) * 4 for i in range(len(beats))]
    third = beats[16], beats[32]
    sections = [{"start": 0.0, "end": third[0], "level": "calm"},
                {"start": third[0], "end": third[1], "level": "high"},
                {"start": third[1], "end": beats[-1] + step, "level": "build"}]
    return {"bpm": bpm, "beats": beats, "downbeats": downbeats, "beat_energy_db": energy,
            "sections": sections, "duration_seconds": beats[-1] + step}


def test_pose_names_exist_in_rig():
    js_poses = set(re.findall(r"^\s{4}(\w+):\s+P\(", CHAR2D, re.M))
    assert set(d.POSES) == js_poses
    used = {p for lvl in d.PHRASES.values() for ph in lvl for p in ph} | {p for v in d.BIG.values() for p in v}
    assert used <= js_poses


def test_window_snaps_to_downbeats_and_rebases():
    a = _analysis()
    w = d.window(a, start=a["downbeats"][2] + 0.3, dur=9.4)
    assert w["offset"] == a["downbeats"][2]
    assert w["beats"][0] == 0.0 and w["downbeats"][0] == 0.0
    assert all(0 <= b < w["dur"] for b in w["beats"])
    assert abs(w["dur"] - 4 * 2.4) < 0.01          # snapped to the downbeat 4 bars later


def test_keys_land_on_every_beat_with_big_downbeats():
    a = _analysis()
    w = d.window(a, 0.0, 30.0)
    keys = d.plan_keys(w["beats"], w["downbeats"], w["sections"], d.normalize_energy(w["energy_db"]), seed=3)
    assert [k["t"] for k in keys] == w["beats"]
    downs = [k for k in keys if k["down"]]
    assert [k["t"] for k in downs] == w["downbeats"]
    assert all(k["dip"] == 1.0 for k in downs)
    assert all(k["dip"] < 1.0 for k in keys if not k["down"] and k["pose"] != "jump")
    for k in downs:
        if k["level"] in ("build", "high"):
            assert k["pose"] in d.BIG[k["level"]]


def test_move_set_follows_section_level_and_changes_per_bar():
    a = _analysis()
    w = d.window(a, 0.0, 30.0)
    keys = d.plan_keys(w["beats"], w["downbeats"], w["sections"], None, seed=5)
    for k in keys:
        lvl = d.level_at(k["t"], w["sections"])
        allowed = {p for ph in d.PHRASES[lvl] for p in ph} | set(d.BIG[lvl])
        assert k["level"] == lvl and k["pose"] in allowed
    # consecutive bars of the same level never repeat the same phrase (+mirror) exactly
    bars = d.bars_of(w["beats"], w["downbeats"])
    seqs = [tuple((keys[i]["pose"], keys[i]["m"]) for i in bar[1:]) for bar in bars]
    lv = [keys[bar[0]]["level"] for bar in bars]
    for i in range(1, len(seqs)):
        if lv[i] == lv[i - 1]:
            assert seqs[i] != seqs[i - 1]


def test_scenes_cut_on_sections_and_change_palette():
    a = _analysis()
    w = d.window(a, 0.0, 30.0)
    scenes = d.plan_scenes(w["dur"], w["downbeats"], w["sections"], bars_per_scene=2, seed=1)
    starts = [s["t0"] for s in scenes]
    for sec in w["sections"]:
        assert round(sec["start"], 3) in starts
    assert scenes[0]["t0"] == 0.0 and scenes[-1]["t1"] == w["dur"]
    for s0, s1 in zip(scenes, scenes[1:]):
        assert s0["t1"] == s1["t0"] and s0["palette"] != s1["palette"]
        assert s1["t0"] in w["downbeats"]
    hi = [s for s in scenes if s["level"] == "high"]
    calm = [s for s in scenes if s["level"] == "calm"]
    assert hi and calm and abs(hi[0]["rot"]) > abs(calm[0]["rot"])


def test_more_energy_more_dancers():
    a = _analysis()
    spec = d.plan(a, 0.0, 30.0, seed=2)
    per_scene = {}
    for dn in spec["dancers"]:
        per_scene.setdefault(dn["t0"], []).append(dn)
    counts = {sc["level"]: len(per_scene[sc["t0"]]) for sc in spec["scenes"]}
    assert counts["calm"] <= counts["build"] <= counts["high"] and counts["high"] == 5
    for group in per_scene.values():
        assert sum(1 for dn in group if dn["lead"]) == 1
        assert all(0 < dn["x"] < d.W for dn in group)


def test_lyrics_words_on_beats_with_one_accent():
    beats = [i * 0.6 for i in range(20)]
    out = d.plan_lyrics([{"start": 1.2, "end": 4.0, "text": "feel the funk", "accent": "funk"}], beats, 12.0)
    words = out[0]["words"]
    assert [w["t"] for w in words] == [1.2, 1.8, 2.4]
    assert [w["accent"] for w in words] == [False, False, True]


def test_plan_is_deterministic_and_seeded():
    a = _analysis()
    p1 = d.plan(a, 0.0, 30.0, seed=11)
    p2 = d.plan(a, 0.0, 30.0, seed=11)
    p3 = d.plan(a, 0.0, 30.0, seed=12)
    assert p1 == p2
    assert [k["pose"] for k in p1["keys"]] != [k["pose"] for k in p3["keys"]]


def test_rig_runtime_has_no_clock_or_unseeded_random():
    js = CHAR2D + (d.ANIM_DIR / "dance2d.js").read_text()
    assert "Date" not in js and "performance.now" not in js and "requestAnimationFrame" not in js
    assert "Math.random" not in js


def test_write_project(tmp_path):
    spec = d.plan(_analysis(), 0.0, 10.0)
    proj = d.write_project(spec, tmp_path / "p")
    html = (proj / "index.html").read_text()
    assert "window.MV_DANCE" in html and 'data-duration="' in html
    for n in ("char2d.js", "dance2d.js", "dance2d.css", "fonts/Anton-Regular.ttf"):
        assert (proj / n).is_file() and not (proj / n).is_symlink()
    assert ("gsap.min.js" in html) and (d.gsap_path() is not None or d.GSAP_CDN in html)


def test_gsap_falls_back_to_pinned_cdn(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "gsap_path", lambda: None)
    proj = d.write_project(d.plan(_analysis(), 0.0, 10.0), tmp_path / "p")
    html = (proj / "index.html").read_text()
    assert d.GSAP_CDN in html and "@3.14.2/" in html and not (proj / "gsap.min.js").exists()


def test_build_without_render_never_starts_a_browser(tmp_path, monkeypatch):
    import json as _json
    monkeypatch.setattr(d, "render", lambda *a, **k: (_ for _ in ()).throw(AssertionError("render called")))
    a = tmp_path / "a.json"
    a.write_text(_json.dumps(_analysis()))
    res = d.build(tmp_path / "song.mp3", tmp_path / "out", 0.0, 10.0, analysis_path=a)
    assert "output" not in res and (tmp_path / "out" / "plan.json").is_file()
