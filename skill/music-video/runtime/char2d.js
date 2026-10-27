// runtime/char2d.js — flat 2D SVG character rig (Devarf "low effort" look: yellow shirt, round dark
// head, limbs as rounded strokes, coloured shoes).
//
// Frame-deterministic: nothing here reads a clock. Callers compute everything from t = frame / fps
// (Char2D.poseAt(keys, t) -> pose, rig.apply(pose)). Seeded random only (Char2D.rng(seed)).
//
// Rig hierarchy (SVG <g> groups, each rotating about its own pivot; local y points down):
//   root (x, ground, scale) > hips (sway x, air, hip tilt)
//     > legL/legR (hip joint)  > shinL/shinR (knee)  > shoe (kept level)
//     > torso (waist)          > armL/armR (shoulder) > foreL/foreR (elbow) > hand
//                              > head (neck)
// Angles are degrees, OUTWARD-positive for limbs (0 = hanging straight down, 90 = out to the side,
// 180 = straight up), so one pose number works for both sides; L = screen left.
// Ground lock: the lower ankle always rests on the floor (squats lower the hips by themselves);
// `air` lifts the whole body (jumps).
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const DIM = { hipW: 17, shoulderW: 27, shoulderY: -112, neckY: -128, headR: 40, headY: -166,
                upperArm: 66, foreArm: 60, thigh: 86, shin: 82, foot: 13 };
  const DEFAULT_COLORS = { shirt: "#FFC93C", ink: "#1E1B2E", pants: "#1E1B2E", shoe: "#FF4F8B",
                           skin: "#1E1B2E", sleeve: "#FFC93C" };

  // ---- pose library (outward-positive degrees; x/air in px) -------------------------------------
  const P = (o) => Object.assign({ x: 0, air: 0, hip: 0, torso: 0, head: 0,
                                   aL: [10, 8], aR: [10, 8], lL: [5, 0], lR: [5, 0] }, o);
  const POSES = {
    idle:        P({}),
    stepL:       P({ x: -22, hip: 5, torso: -3, head: -5, aL: [34, 40], aR: [18, -40], lL: [2, 0], lR: [24, -18] }),
    stepR:       P({ x: 22, hip: -5, torso: 3, head: 5, aR: [34, 40], aL: [18, -40], lR: [2, 0], lL: [24, -18] }),
    armsUp:      P({ aL: [162, 8], aR: [162, 8], lL: [11, 0], lR: [11, 0], head: 0 }),
    swayL:       P({ x: -34, hip: 9, torso: -9, head: 7, aL: [50, -108], aR: [58, 46], lL: [3, 0], lR: [16, -10] }),
    swayR:       P({ x: 34, hip: -9, torso: 9, head: -7, aR: [50, -108], aL: [58, 46], lR: [3, 0], lL: [16, -10] }),
    pointL:      P({ torso: 6, head: 12, aL: [138, -4], aR: [50, -108], lL: [14, -8], lR: [7, 0] }),
    pointR:      P({ torso: -6, head: -12, aR: [138, -4], aL: [50, -108], lR: [14, -8], lL: [7, 0] }),
    clap:        P({ aL: [24, -104], aR: [24, -104], lL: [8, 0], lR: [8, 0], head: 3 }),
    jump:        P({ air: 84, aL: [140, 16], aR: [140, 16], lL: [27, 6], lR: [27, 6] }),
    squat:       P({ aL: [84, -52], aR: [84, -52], lL: [44, -74], lR: [44, -74], head: 0 }),
    handsOnHips: P({ aL: [50, -110], aR: [50, -110], lL: [10, 0], lR: [10, 0], head: 8 }),
  };
  const NUM = ["x", "air", "hip", "torso", "head"];
  const LIMB = ["aL", "aR", "lL", "lR"];

  function mirror(p) {
    return { x: -p.x, air: p.air, hip: -p.hip, torso: -p.torso, head: -p.head,
             aL: p.aR.slice(), aR: p.aL.slice(), lL: p.lR.slice(), lR: p.lL.slice() };
  }
  function lerpPose(a, b, w) {
    const o = {};
    for (const k of NUM) o[k] = a[k] + (b[k] - a[k]) * w;
    for (const k of LIMB) o[k] = [a[k][0] + (b[k][0] - a[k][0]) * w, a[k][1] + (b[k][1] - a[k][1]) * w];
    return o;
  }
  const clamp01 = (v) => Math.max(0, Math.min(1, v));
  const easeInOut = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
  const easeOutBack = (u) => { const c = 1.6; return 1 + (c + 1) * Math.pow(u - 1, 3) + c * Math.pow(u - 1, 2); };

  function resolve(key) {
    const base = POSES[key.pose] || POSES.idle;
    return key.m ? mirror(base) : base;
  }

  // keys: [{t, pose, m?, dip?}] sorted by t. The pose ARRIVES at key.t (the beat): it holds, then
  // moves to the next key during the last `move` seconds before it, with a small overshoot on big
  // hits; `dip` (0..1) is a knee-bounce whose lowest point is exactly on the key time.
  function poseAt(keys, t, opt) {
    opt = opt || {};
    if (!keys || !keys.length) return Object.assign({}, POSES.idle);
    let i = 0;
    while (i + 1 < keys.length && keys[i + 1].t <= t) i++;
    let p;
    if (t < keys[0].t) {
      p = resolve(keys[0]);
    } else if (i + 1 >= keys.length) {
      p = resolve(keys[i]);
    } else {
      const k0 = keys[i], k1 = keys[i + 1];
      const span = k1.t - k0.t;
      const move = Math.min(span * 0.6, opt.move || 0.26);
      const u = clamp01((t - (k1.t - move)) / move);
      const w = (k1.dip || 0) >= 0.9 ? Math.min(easeOutBack(u), 1.08) : easeInOut(u);
      p = lerpPose(resolve(k0), resolve(k1), w);
    }
    p = { x: p.x, air: p.air, hip: p.hip, torso: p.torso, head: p.head,
          aL: p.aL.slice(), aR: p.aR.slice(), lL: p.lL.slice(), lR: p.lR.slice() };
    // knee bounce around the nearest key
    let dip = 0;
    for (let j = Math.max(0, i - 1); j <= Math.min(keys.length - 1, i + 1); j++) {
      const d = t - keys[j].t, a = keys[j].dip || 0;
      if (!a) continue;
      const g = d < 0 ? Math.exp(-Math.pow(d / 0.07, 2)) : Math.exp(-Math.pow(d / 0.16, 2));
      dip = Math.max(dip, a * g);
    }
    if (dip > 0 && p.air < 1) {
      p.lL[0] += 9 * dip; p.lL[1] -= 22 * dip;
      p.lR[0] += 9 * dip; p.lR[1] -= 22 * dip;
      p.head += (p.head >= 0 ? 5 : -5) * dip;
    }
    p.dip = dip;
    return p;
  }

  // ---- forward kinematics (for ground lock) -----------------------------------------------------
  const rad = (d) => (d * Math.PI) / 180;
  function ankleY(side, leg, hipDeg) {
    const s = side === "L" ? -1 : 1;
    const a = rad(leg[0]), b = rad(leg[0] + leg[1]);
    // limb direction for outward-positive angle: L (-sin, cos), R (sin, cos)
    const x = s * DIM.hipW + s * Math.sin(a) * DIM.thigh + s * Math.sin(b) * DIM.shin;
    const y = Math.cos(a) * DIM.thigh + Math.cos(b) * DIM.shin;
    const h = rad(hipDeg);
    return x * Math.sin(h) + y * Math.cos(h);
  }
  function hipHeight(p) {
    return Math.max(ankleY("L", p.lL, p.hip), ankleY("R", p.lR, p.hip)) + DIM.foot;
  }

  // ---- SVG construction ---------------------------------------------------------------------------
  function mk(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  const line = (len, w, color, parent) =>
    mk("line", { x1: 0, y1: 0, x2: 0, y2: len, stroke: color, "stroke-width": w, "stroke-linecap": "round" }, parent);

  // create(parent <g|svg>, {colors, scale}) -> rig with .apply(pose, x, groundY) and .node
  function create(parent, opts) {
    opts = opts || {};
    const c = Object.assign({}, DEFAULT_COLORS, opts.colors || {});
    const root = mk("g", { class: "c2d" }, parent);
    const shadow = mk("ellipse", { cx: 0, cy: 0, rx: 62, ry: 11, fill: "#000", opacity: 0.18 }, root);
    const body = mk("g", {}, root);
    const hips = mk("g", {}, body);
    const parts = {};
    for (const side of ["L", "R"]) {
      const s = side === "L" ? -1 : 1;
      const leg = mk("g", {}, hips);
      line(DIM.thigh, 26, c.pants, leg);
      const shin = mk("g", {}, leg);
      line(DIM.shin, 22, c.pants, shin);
      const shoe = mk("g", {}, shin);
      mk("rect", { x: side === "L" ? -34 : -10, y: -6, width: 44, height: 22, rx: 11, fill: c.shoe }, shoe);
      mk("rect", { x: side === "L" ? -34 : -10, y: 10, width: 44, height: 6, rx: 3, fill: "#fff", opacity: 0.85 }, shoe);
      parts["leg" + side] = leg; parts["shin" + side] = shin; parts["shoe" + side] = shoe; parts["s" + side] = s;
    }
    const torso = mk("g", {}, hips);
    // shirt: rounded trapezoid waist -> shoulders
    mk("path", { d: "M -24 6 L 24 6 Q 30 6 31 -2 L 36 -104 Q 37 -120 20 -120 L -20 -120 Q -37 -120 -36 -104 L -31 -2 Q -30 6 -24 6 Z",
                 fill: c.shirt }, torso);
    mk("rect", { x: -26, y: -6, width: 52, height: 14, rx: 6, fill: c.pants }, torso); // belt line
    const head = mk("g", {}, torso);
    line(14, 16, c.ink, mk("g", { transform: `translate(0,${DIM.neckY - 4})` }, head));
    mk("circle", { cx: 0, cy: DIM.headY, r: DIM.headR, fill: c.ink }, head);
    mk("circle", { cx: -13, cy: DIM.headY - 13, r: 8, fill: "#fff", opacity: 0.16 }, head);
    for (const side of ["L", "R"]) {
      const arm = mk("g", {}, torso);
      line(DIM.upperArm, 20, c.ink, arm);
      line(DIM.upperArm * 0.45, 26, c.sleeve, arm);
      const fore = mk("g", {}, arm);
      line(DIM.foreArm, 18, c.skin, fore);
      mk("circle", { cx: 0, cy: DIM.foreArm + 4, r: 13, fill: c.skin }, fore);
      parts["arm" + side] = arm; parts["fore" + side] = fore;
    }
    const scale = opts.scale || 1;

    function apply(p, x, groundY) {
      const lift = hipHeight(p);
      const air = Math.max(0, p.air);
      root.setAttribute("transform", `translate(${x.toFixed(2)},${groundY.toFixed(2)}) scale(${scale})`);
      // shadow shrinks while airborne
      const k = 1 - Math.min(0.55, air / 220);
      shadow.setAttribute("rx", (62 * k).toFixed(1));
      shadow.setAttribute("cx", (p.x).toFixed(1));
      hips.setAttribute("transform", `translate(${p.x.toFixed(2)},${(-lift - air).toFixed(2)}) rotate(${p.hip.toFixed(2)})`);
      for (const side of ["L", "R"]) {
        const s = parts["s" + side];
        const lg = p["l" + side];
        parts["leg" + side].setAttribute("transform", `translate(${s * DIM.hipW},0) rotate(${(-s * lg[0]).toFixed(2)})`);
        parts["shin" + side].setAttribute("transform", `translate(0,${DIM.thigh}) rotate(${(-s * lg[1]).toFixed(2)})`);
        // keep the shoe level with the floor
        parts["shoe" + side].setAttribute("transform", `translate(0,${DIM.shin}) rotate(${(s * (lg[0] + lg[1]) - p.hip).toFixed(2)})`);
        const am = p["a" + side];
        parts["arm" + side].setAttribute("transform", `translate(${s * DIM.shoulderW},${DIM.shoulderY}) rotate(${(-s * am[0]).toFixed(2)})`);
        parts["fore" + side].setAttribute("transform", `translate(0,${DIM.upperArm}) rotate(${(-s * am[1]).toFixed(2)})`);
      }
      torso.setAttribute("transform", `rotate(${(p.torso - p.hip * 0.5).toFixed(2)})`);
      head.setAttribute("transform", `translate(0,${DIM.neckY}) rotate(${p.head.toFixed(2)}) translate(0,${-DIM.neckY})`);
    }
    return { node: root, apply, colors: c };
  }

  // mulberry32: seeded, deterministic
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  window.Char2D = { POSES, DIM, DEFAULT_COLORS, mirror, lerpPose, poseAt, hipHeight, create, rng,
                       easeInOut, easeOutBack, clamp01 };
})();
