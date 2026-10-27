// runtime/dance2d.js — draws one 2D dance music-video composition planned by musicvideo/dance2d.py.
//
// The planner decides every time and colour (pose keys on beats, scenes per section/phrase, dancer
// copies, lyric word times); this file only draws that plan as a pure function of t = frame / fps.
// One GSAP tween drives a proxy whose setter calls draw(t), so HyperFrames' seek renders each frame
// from t alone (no clock, no rAF deltas, no unseeded random).
//
// SPEC (window.MV_DANCE):
//   {w, h, dur, ground, keys: [{t, pose, m?, dip?}], beats: [s], downbeats: [s],
//    scenes: [{t0, t1, level, bg, ray, floor, accent, rays, rot, rot0, doors, crown, chess, film}],
//    dancers: [{id, t0, t1, x, scale, m, colors: {shirt, shoe, ...}, lead?}],
//    lyrics: [{t0, t1, y?, words: [{text, t, accent?}]}], credit?}
(function () {
  const S = window.MV_DANCE;
  const C = window.Char2D;
  const W = S.w, H = S.h, DUR = S.dur, GROUND = S.ground || Math.round(H * 0.86);
  const NS = "http://www.w3.org/2000/svg";
  const root = document.getElementById("root");
  const mk = (tag, attrs, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };
  const svg = mk("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "stage" }, root);

  // ---- background: flat colour + sunburst rays -----------------------------------------------------
  const bg = mk("rect", { x: 0, y: 0, width: W, height: H }, svg);
  const rayWrap = mk("g", { transform: `translate(${W / 2},${GROUND - 40})` }, svg);
  const rayG = mk("g", {}, rayWrap);
  const R = Math.hypot(W, H);
  const rayPaths = [];
  for (let i = 0; i < 32; i++) rayPaths.push(mk("path", { d: "" }, rayG));
  function setRays(n, color) {
    for (let i = 0; i < rayPaths.length; i++) {
      if (i >= n) { rayPaths[i].setAttribute("d", ""); continue; }
      const a0 = (i * 2 * Math.PI) / n, a1 = a0 + Math.PI / n;
      rayPaths[i].setAttribute("d", `M0 0 L${(R * Math.cos(a0)).toFixed(1)} ${(R * Math.sin(a0)).toFixed(1)} ` +
                                    `L${(R * Math.cos(a1)).toFixed(1)} ${(R * Math.sin(a1)).toFixed(1)} Z`);
      rayPaths[i].setAttribute("fill", color);
    }
  }

  // ---- arched mirror doors ----------------------------------------------------------------------------
  const doorG = mk("g", {}, svg);
  const doorXs = [W / 2 - 560, W / 2, W / 2 + 560];
  const doors = doorXs.map((cx, i) => {
    const g = mk("g", {}, doorG);
    const dw = i === 1 ? 330 : 270, dh = i === 1 ? 640 : 540, top = GROUND - dh;
    const arch = `M${cx - dw / 2} ${GROUND} L${cx - dw / 2} ${top + dw / 2} A${dw / 2} ${dw / 2} 0 0 1 ${cx + dw / 2} ${top + dw / 2} L${cx + dw / 2} ${GROUND} Z`;
    const frame = mk("path", { d: arch, fill: "none", "stroke-width": 22 }, g);
    const glass = mk("path", { d: arch, opacity: 0.92 }, g);
    g.insertBefore(glass, frame);
    const shine = mk("path", { d: `M${cx - dw * 0.3} ${GROUND - 40} L${cx - dw * 0.05} ${top + dw * 0.35} L${cx + dw * 0.08} ${top + dw * 0.35} L${cx - dw * 0.17} ${GROUND - 40} Z`, fill: "#fff", opacity: 0.22 }, g);
    return { g, frame, glass, shine };
  });

  // ---- floor + chess silhouettes ---------------------------------------------------------------------------
  const floor = mk("rect", { x: 0, y: GROUND - 6, width: W, height: H - GROUND + 6 }, svg);
  const floorLine = mk("rect", { x: 0, y: GROUND - 10, width: W, height: 8, fill: "#1E1B2E" }, svg);
  const chessG = mk("g", {}, svg);
  const KING = "M-34 0 L34 0 L28 -16 L18 -20 L14 -96 L26 -104 L22 -116 L-22 -116 L-26 -104 L-14 -96 L-18 -20 L-28 -16 Z M-6 -116 L-6 -128 L-16 -128 L-16 -138 L-6 -138 L-6 -150 L6 -150 L6 -138 L16 -138 L16 -128 L6 -128 L6 -116 Z";
    const chess = [[64, KING, 1.25], [W - 64, KING, 1.25]].map(([x, d, s]) =>
    mk("path", { d, transform: `translate(${x},${GROUND}) scale(${s})`, fill: "#1E1B2E" }, chessG));

  // ---- dancers -----------------------------------------------------------------------------------------
  const danceG = mk("g", {}, svg);
  const dancers = (S.dancers || []).map((d) => {
    const wrap = mk("g", {}, danceG);
    const rig = C.create(wrap, { colors: d.colors, scale: d.scale || 1 });
    return { d, wrap, rig };
  });

  // ---- crown (above the lead) -------------------------------------------------------------------------
  const crown = mk("path", { d: "M-46 0 L46 0 L52 -46 L26 -22 L0 -58 L-26 -22 L-52 -46 Z", fill: "#FFC93C", stroke: "#1E1B2E", "stroke-width": 6, "stroke-linejoin": "round" }, svg);

  // ---- film strip + tally marks ------------------------------------------------------------------------
  const filmG = mk("g", {}, svg);
  const holesTop = [], holesBot = [];
  mk("rect", { x: 0, y: 0, width: W, height: 54, fill: "#111" }, filmG);
  mk("rect", { x: 0, y: H - 54, width: W, height: 54, fill: "#111" }, filmG);
  for (let i = 0; i < 26; i++) {
    holesTop.push(mk("rect", { y: 15, width: 44, height: 24, rx: 5, fill: "#f4efe6" }, filmG));
    holesBot.push(mk("rect", { y: H - 39, width: 44, height: 24, rx: 5, fill: "#f4efe6" }, filmG));
  }
  const tallyG = mk("g", { transform: "translate(96,104)" }, svg);
  const tallies = [];
  for (let i = 0; i < 40; i++) {
    const grp = Math.floor(i / 5), k = i % 5;
    const x0 = grp * 84 + k * 15;
    const ln = k < 4 ? mk("line", { x1: x0, y1: 0, x2: x0, y2: 56 }, tallyG)
                     : mk("line", { x1: grp * 84 - 8, y1: 46, x2: grp * 84 + 54, y2: 8 }, tallyG);
    ln.setAttribute("stroke-width", 9); ln.setAttribute("stroke-linecap", "round");
    tallies.push(ln);
  }

  // ---- flash overlay -----------------------------------------------------------------------------------
  const flash = mk("rect", { x: 0, y: 0, width: W, height: H, fill: "#fff", opacity: 0 }, svg);

  // ---- lyrics (HTML on top, condensed caps) --------------------------------------------------------------
  const lyr = document.createElement("div");
  lyr.className = "lyrics";
  root.appendChild(lyr);
  const lines = (S.lyrics || []).map((L) => {
    const n = document.createElement("div");
    n.className = "line";
    n.style.top = (L.y || 150) + "px";
    const words = L.words.map((w) => {
      const s = document.createElement("span");
      s.className = "word" + (w.accent ? " accent" : "");
      s.textContent = w.text;
      n.appendChild(s);
      return { s, w };
    });
    lyr.appendChild(n);
    return { L, n, words };
  });
  if (S.credit) {
    const cr = document.createElement("div");
    cr.className = "credit";
    cr.textContent = S.credit;
    root.appendChild(cr);
  }

  // ---- helpers -----------------------------------------------------------------------------------------
  const sceneAt = (t) => {
    const sc = S.scenes;
    for (let i = sc.length - 1; i >= 0; i--) if (t >= sc[i].t0) return sc[i];
    return sc[0];
  };
  const lastBefore = (arr, t) => {
    let v = null;
    for (const x of arr || []) { if (x <= t) v = x; else break; }
    return v;
  };
  const countUpTo = (arr, t) => { let n = 0; for (const x of arr || []) { if (x <= t + 1e-6) n++; else break; } return n; };
  const decay = (d, tau) => (d === null || d < 0 ? 0 : Math.exp(-d / tau));

  function draw(t) {
    const sc = sceneAt(t);
    const db = lastBefore(S.downbeats, t), bt = lastBefore(S.beats, t);
    const hitD = decay(db === null ? null : t - db, 0.12);
    const hitB = decay(bt === null ? null : t - bt, 0.09);

    bg.setAttribute("fill", sc.bg);
    setRays(sc.rays, sc.ray);
    rayG.setAttribute("transform", `rotate(${(sc.rot0 + sc.rot * (t - sc.t0)).toFixed(3)}) scale(${(1 + 0.05 * hitD).toFixed(4)})`);

    for (const d of doors) {
      d.g.setAttribute("display", sc.doors ? "inline" : "none");
      d.frame.setAttribute("stroke", sc.accent);
      d.glass.setAttribute("fill", sc.glass || "#CFE9F2");
    }
    floor.setAttribute("fill", sc.floor);
    chessG.setAttribute("display", sc.chess ? "inline" : "none");

    // dancers
    let lead = null;
    for (const D of dancers) {
      const on = t >= D.d.t0 && t < D.d.t1;
      D.wrap.setAttribute("display", on ? "inline" : "none");
      if (!on) continue;
      let p = C.poseAt(S.keys, t - (D.d.lag || 0));
      if (D.d.m) p = Object.assign(C.mirror(p), { dip: p.dip });
      const k = C.clamp01((t - D.d.t0) / 0.16);
      const pop = D.d.t0 > 0.01 ? 0.6 + 0.4 * C.easeOutBack(k) : 1;
      D.wrap.setAttribute("transform", `translate(${D.d.x},${GROUND}) scale(${pop.toFixed(4)}) translate(${-D.d.x},${-GROUND})`);
      D.rig.apply(p, D.d.x, GROUND);
      if (D.d.lead) lead = { D, p };
    }
    // crown hovers above the lead's head and bobs on beats
    if (sc.crown && lead) {
      const s = lead.D.d.scale || 1;
      const headTop = GROUND - (C.hipHeight(lead.p) + lead.p.air) * s - (C.DIM.headR * 2 + 160) * s;
      const cx = lead.D.d.x + lead.p.x * s;
      crown.setAttribute("display", "inline");
      crown.setAttribute("transform", `translate(${cx.toFixed(1)},${(headTop - 18 * hitB).toFixed(1)}) rotate(${(6 * Math.sin(t * 2.1)).toFixed(2)}) scale(${(s * (1 + 0.12 * hitD)).toFixed(3)})`);
    } else {
      crown.setAttribute("display", "none");
    }

    // film strip slides with time; tally = bars counted so far
    filmG.setAttribute("display", sc.film ? "inline" : "none");
    const off = (t * 160) % 76;
    holesTop.forEach((h, i) => h.setAttribute("x", (i * 76 - off).toFixed(1)));
    holesBot.forEach((h, i) => h.setAttribute("x", (i * 76 + off - 76).toFixed(1)));
    const bars = countUpTo(S.downbeats, t);
    tallies.forEach((ln, i) => {
      ln.setAttribute("display", i < bars ? "inline" : "none");
      ln.setAttribute("stroke", sc.ink || "#1E1B2E");
    });
    tallyG.setAttribute("transform", `translate(96,${sc.film ? 104 : 70})`);

    // scene-change flash (only at scene starts, max one per scene)
    const ds = t - sc.t0;
    flash.setAttribute("opacity", sc.t0 > 0.01 && ds >= 0 && ds < 0.2 ? (0.3 * (1 - ds / 0.2)).toFixed(3) : "0");

    // lyrics: words pop in at their time, the line leaves at t1
    for (const Ln of lines) {
      const on = t >= Ln.L.t0 && t < Ln.L.t1;
      Ln.n.style.display = on ? "block" : "none";
      if (!on) continue;
      const out = C.clamp01((t - (Ln.L.t1 - 0.18)) / 0.18);
      Ln.n.style.transform = `translateX(-50%) translateY(${(-60 * out).toFixed(1)}px)`;
      Ln.n.style.opacity = (1 - out).toFixed(3);
      for (const Wd of Ln.words) {
        const u = C.clamp01((t - Wd.w.t) / 0.14);
        const vis = t >= Wd.w.t;
        Wd.s.style.visibility = vis ? "visible" : "hidden";
        const sc2 = vis ? 1.45 - 0.45 * C.easeOutBack(u) : 1;
        Wd.s.style.transform = `scale(${sc2.toFixed(3)}) rotate(${((1 - u) * (Wd.w.accent ? -8 : 4)).toFixed(2)}deg)`;
        if (Wd.w.accent) Wd.s.style.color = sc.accent;
      }
    }
  }

  // GSAP drives a proxy whose setter draws the frame: seek(t) -> draw(t), with or without callbacks.
  const proxy = {};
  let cur = 0;
  Object.defineProperty(proxy, "t", { get: () => cur, set: (v) => { cur = v; draw(v); } });
  const tl = gsap.timeline({ paused: true });
  tl.fromTo(proxy, { t: 0 }, { t: DUR, duration: DUR, ease: "none", immediateRender: true }, 0);
  tl.set({}, {}, DUR);
  draw(0);
  window.__timelines = window.__timelines || {};
  window.__timelines["main"] = tl;
})();
