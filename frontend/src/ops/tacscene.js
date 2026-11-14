// The tactical map: the theatre as a three.js scene. Terrain with the land cover painted on
// it, the fire drawn from the simulation's own arrival times, forecast isochrones, the
// control line, and every unit with its track. One instance per theatre; React hands it the
// world each frame and never re-renders at frame rate.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { C } from "../lib/theme";
import { COVER, toLatLon } from "./engine/theatre.js";
import { UNBURNED } from "./engine/fire.js";
import { contours } from "./engine/contour.js";
import { SPECS } from "./engine/units.js";
import { KIND } from "./engine/line.js";
import { fireName } from "./engine/world.js";
import { drawSymbol } from "./icons.js";
import { makeAircraft, disposeAircraft } from "./aircraft3d.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";

const FOV = 32, TAN = Math.tan((FOV / 2) * (Math.PI / 180));
const SEGMENTS = 160;
const ISO_HOURS = [1, 3, 6, 12, 24];
const COVER_RGB = {
  [COVER.land]: [17, 27, 37], [COVER.water]: [4, 9, 14], [COVER.wetland]: [19, 33, 41], [COVER.wood]: [14, 28, 37], [COVER.scrub]: [21, 32, 39],
  [COVER.grass]: [27, 37, 40], [COVER.rock]: [31, 40, 50], [COVER.built]: [27, 37, 49], [COVER.bare]: [24, 32, 40],
};
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const reduceMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

const FIRE_VERT = `
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const FIRE_FRAG = `
  precision highp float;
  uniform sampler2D tFire; uniform sampler2D tState; uniform float uTime; uniform float uN;
  uniform vec3 uRetard; uniform vec3 uWet; uniform vec3 uGuard;
  varying vec2 vUv;
  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
  }
  vec3 ramp(float t) { // ember, flame, gold, white: the fire ramp of the rest of the console
    vec3 a = vec3(0.33, 0.13, 0.09), b = vec3(0.69, 0.24, 0.09), c = vec3(0.96, 0.48, 0.16), d = vec3(1.0, 0.89, 0.59), e = vec3(1.0, 0.98, 0.93);
    t = clamp(t, 0.0, 1.0);
    return t < 0.25 ? mix(a, b, t * 4.0) : t < 0.5 ? mix(b, c, (t - 0.25) * 4.0) : t < 0.75 ? mix(c, d, (t - 0.5) * 4.0) : mix(d, e, (t - 0.75) * 4.0);
  }
  void main() {
    vec4 f = texture2D(tFire, vUv);
    vec4 s = texture2D(tState, vUv);
    vec2 cell = vUv * uN;
    float burned = smoothstep(0.44, 0.52, f.r);
    vec3 col = vec3(0.0);
    float a = 0.0;
    // suppression laid on unburned ground: wetted fuel, retardant, guard
    if (s.b > 0.5) { col = uWet; a = 0.34; }
    if (s.r > 0.5) { float hatch = step(0.5, fract((cell.x + cell.y) * 1.5)); col = uRetard; a = 0.5 + 0.3 * hatch; }
    if (s.g > 0.25) { col = mix(uGuard, uRetard, step(0.5, s.r) * 0.45); a = 0.92; }
    // burned ground: char, with embers that fade over a few hours
    float ember = f.a * (0.3 + 0.7 * noise(cell * 5.0 + uTime * 0.12));
    vec3 scar = mix(vec3(0.034, 0.03, 0.032), vec3(0.42, 0.1, 0.03), ember * 0.5) * (0.78 + 0.44 * noise(cell * 2.3));
    scar = mix(scar, vec3(0.045, 0.05, 0.06), s.a * 0.6); // out: mopped up or burned off
    col = mix(col, scar, burned);
    a = mix(a, 0.9, burned);
    // flame: colour by intensity, flicker in time
    float flick = 0.7 + 0.6 * noise(cell * 6.0 + vec2(uTime * 1.9, -uTime * 2.6));
    float heat = clamp(f.g * flick * 1.5, 0.0, 1.0);
    col = mix(col, ramp(0.22 + 0.78 * f.b) * (0.9 + 0.3 * flick), heat * burned);
    // the glow on fuel the front is about to take
    float ahead = smoothstep(0.1, 0.5, f.r) * (1.0 - burned) * smoothstep(0.03, 0.35, f.g);
    col = mix(col, ramp(0.45), ahead * 0.8);
    a = max(a, ahead * 0.75);
    if (a < 0.01) discard;
    gl_FragColor = vec4(col, a);
    #include <colorspace_fragment>
  }`;

function symbolTexture(type) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 128;
  drawSymbol(canvas.getContext("2d"), type, 128);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

export class TacScene {
  constructor(mount, labels, theatre, { onPick, onCursor, onCamera } = {}) {
    this.mount = mount;
    this.labels = labels;
    this.theatre = theatre;
    this.onPick = onPick;
    this.onCursor = onCursor;
    this.onCamera = onCamera;
    this.disposables = [];
    this.keep = (o) => (this.disposables.push(o), o);
    const renderer = (this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" }));
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    mount.appendChild(renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x030609, 90, 260);
    this.camera = new THREE.PerspectiveCamera(FOV, 1, 0.05, 600);
    this.size = { w: 1, h: 1 };
    this.still = reduceMotion();

    const { n, half } = theatre, elev = theatre.grid.elev;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < elev.length; i++) { lo = Math.min(lo, elev[i]); hi = Math.max(hi, elev[i]); }
    this.base = lo;
    this.lift = hi - lo > 700 ? 1.5 / 1000 : 3 / 1000; // vertical exaggeration: more where the land is nearly flat
    this.elevAt = (x, y) => {
      const fx = Math.max(0, Math.min(n - 1.001, (x + half) / theatre.cell - 0.5)), fy = Math.max(0, Math.min(n - 1.001, (y + half) / theatre.cell - 0.5));
      const c = Math.floor(fx), r = Math.floor(fy), u = fx - c, v = fy - r, i = r * n + c;
      return ((elev[i] * (1 - u) + elev[i + 1] * u) * (1 - v) + (elev[i + n] * (1 - u) + elev[i + n + 1] * u) * v - lo) * this.lift;
    };
    this.at = (x, y, up = 0, out = new THREE.Vector3()) => out.set(x, this.elevAt(x, y) + up, -y);

    // Aircraft fly over a smoothed terrain, not over every ridge: the highest ground within a
    // couple of kilometres, blended in as they climb away from the surface.
    const E = 32, cellsPer = n / E, peak = new Float32Array(E * E);
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const k = Math.floor(r / cellsPer) * E + Math.floor(c / cellsPer);
      if (elev[r * n + c] > peak[k]) peak[k] = elev[r * n + c];
    }
    const over = (x, y) => {
      const fx = Math.max(0, Math.min(E - 1.001, ((x + half) / (2 * half)) * E - 0.5)), fy = Math.max(0, Math.min(E - 1.001, ((y + half) / (2 * half)) * E - 0.5));
      const c = Math.floor(fx), r = Math.floor(fy), u = fx - c, v = fy - r, i = r * E + c;
      return ((peak[i] * (1 - u) + peak[i + 1] * u) * (1 - v) + (peak[i + E] * (1 - u) + peak[i + E + 1] * u) * v - lo) * this.lift;
    };
    /** Where a point at a height above ground (m) is in the scene. */
    this.aloft = (x, y, alt, out = new THREE.Vector3()) => {
      const ground = this.elevAt(x, y), k = Math.min(1, Math.max(0, alt) / 300);
      return out.set(x, ground + (Math.max(ground, over(x, y)) - ground) * k + Math.max(0, alt) * this.lift + 0.03, -y);
    };
    // aircraft and their courses are drawn after the map, over it, with light on them
    this.airScene = new THREE.Scene();
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(-0.6, 1, -0.5);
    this.airScene.add(new THREE.AmbientLight(0xffffff, 1.1), sun);
    const wide = (color, width, opacity, dashed) => {
      const material = this.keep(new LineMaterial({ color, linewidth: width, transparent: true, opacity, depthTest: false, dashed: Boolean(dashed), dashSize: dashed?.[0] ?? 1, gapSize: dashed?.[1] ?? 0 }));
      const mesh = new LineSegments2(new LineSegmentsGeometry(), material);
      mesh.frustumCulled = false;
      mesh.visible = false;
      this.airScene.add(mesh);
      return { mesh, material, dashed: Boolean(dashed) };
    };
    // every aircraft's course ahead of it, the drop and scoop runs on it, the selected course, and trails
    this.course = { ahead: wide(C.accent, 2.2, 0.9), run: wide("#ffffff", 3.6, 1), picked: wide("#ffffff", 3.2, 1, [0.5, 0.28]), trail: wide(C.accent, 1.4, 0.35) };
    // the same for a plan that has not been sent yet: the routes it proposes
    this.proposed = { way: wide(C.accent, 1.8, 0.85, [0.45, 0.22]), run: wide("#ffffff", 3.4, 1) };
    this.lastCourse = -1e9;

    this.buildTerrain();
    this.buildVectors();
    this.buildFire();
    this.dynamic = {};
    this.unitNodes = new Map();
    this.tags = new Map();
    this.fireTags = new Map();
    this.textures = {};
    this.forecastSeen = null;
    this.lineSeen = null;
    this.lastFireUpdate = -1e9;

    const controls = (this.controls = new OrbitControls(this.camera, renderer.domElement));
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.screenSpacePanning = false;
    controls.minDistance = 2.5;
    controls.maxDistance = 150;
    controls.maxPolarAngle = 1.28;
    controls.zoomSpeed = 0.8;
    controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };
    controls.touches = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_ROTATE };
    controls.addEventListener("start", () => (this.flight = null));
    controls.addEventListener("change", () => {
      const t = controls.target;
      t.x = Math.max(-half, Math.min(half, t.x));
      t.z = Math.max(-half, Math.min(half, t.z));
    });
    // open high over the theatre, looking straight down; the first flight tilts in
    this.setView({ x: 0, y: 0, dist: 150, polar: 0.05, az: 0 });

    const dom = renderer.domElement;
    this.raycaster = new THREE.Raycaster();
    let down = null;
    const ground = (e) => {
      const box = dom.getBoundingClientRect();
      this.raycaster.setFromCamera(new THREE.Vector2(((e.clientX - box.left) / box.width) * 2 - 1, -((e.clientY - box.top) / box.height) * 2 + 1), this.camera);
      const hit = this.raycaster.intersectObject(this.terrain, false)[0];
      return hit ? { x: hit.point.x, y: -hit.point.z, sx: e.clientX - box.left, sy: e.clientY - box.top } : { sx: e.clientX - box.left, sy: e.clientY - box.top };
    };
    this.handlers = {
      pointerdown: (e) => (down = { x: e.clientX, y: e.clientY }),
      pointerup: (e) => {
        if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return;
        const p = ground(e);
        // a unit under the pointer wins over the ground behind it
        let best = null, bestD = 20;
        for (const [id, node] of this.unitNodes) {
          const d = Math.hypot(node.sx - p.sx, node.sy - p.sy);
          if (node.shown && d < bestD) { bestD = d; best = id; }
        }
        this.onPick?.(best ? { unit: best } : p.x === undefined ? {} : { x: p.x, y: p.y });
      },
      pointermove: (e) => {
        const p = ground(e);
        this.onCursor?.(p.x === undefined ? null : p);
      },
      pointerleave: () => this.onCursor?.(null),
    };
    for (const [k, f] of Object.entries(this.handlers)) dom.addEventListener(k, f);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(mount);
    this.resize();
  }

  // ---- static scene

  buildTerrain() {
    const { theatre, keep } = this, { n, half, cover } = theatre, size = half * 2, elev = theatre.grid.elev;
    const G = SEGMENTS, position = new Float32Array((G + 1) * (G + 1) * 3), uv = new Float32Array((G + 1) * (G + 1) * 2), index = [];
    for (let j = 0; j <= G; j++)
      for (let i = 0; i <= G; i++) {
        const x = -half + (size * i) / G, y = -half + (size * j) / G, k = j * (G + 1) + i;
        position.set([x, this.elevAt(x, y), -y], k * 3);
        uv.set([i / G, j / G], k * 2);
      }
    for (let j = 0; j < G; j++)
      for (let i = 0; i < G; i++) {
        const a = j * (G + 1) + i, b = a + 1, c = a + G + 1, d = c + 1;
        index.push(a, b, c, b, d, c);
      }
    const geometry = (this.terrainGeometry = keep(new THREE.BufferGeometry()));
    geometry.setAttribute("position", new THREE.BufferAttribute(position, 3));
    geometry.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    geometry.setIndex(index);

    // land cover with relief shading, one pixel per cell, then scaled up smoothly
    const small = document.createElement("canvas");
    small.width = small.height = n;
    const sctx = small.getContext("2d"), image = sctx.createImageData(n, n);
    const cellM = theatre.cell * 1000;
    for (let r = 0; r < n; r++)
      for (let c = 0; c < n; c++) {
        const i = r * n + c, rgb = COVER_RGB[cover[i]] || COVER_RGB[0];
        const dx = (elev[r * n + Math.min(n - 1, c + 1)] - elev[r * n + Math.max(0, c - 1)]) / (2 * cellM), dy = (elev[Math.min(n - 1, r + 1) * n + c] - elev[Math.max(0, r - 1) * n + c]) / (2 * cellM);
        // light from the north-west
        const shade = cover[i] === COVER.water ? 1 : Math.max(0.55, Math.min(1.75, 1 + (-dx + dy) * 1.6 * (this.lift * 700)));
        const o = ((n - 1 - r) * n + c) * 4;
        image.data[o] = rgb[0] * shade;
        image.data[o + 1] = rgb[1] * shade;
        image.data[o + 2] = rgb[2] * shade;
        image.data[o + 3] = 255;
      }
    sctx.putImageData(image, 0, 0);
    const S = Math.min(4096, this.renderer.capabilities.maxTextureSize);
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = S;
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(small, 0, 0, S, S);
    const px = (x) => ((x + half) / size) * S, py = (y) => ((half - y) / size) * S;
    const trace = (ring) => {
      ctx.moveTo(px(ring[0][0]), py(ring[0][1]));
      for (let i = 1; i < ring.length; i++) ctx.lineTo(px(ring[i][0]), py(ring[i][1]));
      ctx.closePath();
    };
    const data = theatre.data;
    // crisp shorelines from the vector water, where the data has it
    if (data.water?.length) {
      ctx.fillStyle = "rgb(4,9,14)";
      ctx.beginPath();
      for (const ring of data.water) trace(ring);
      ctx.fill();
      ctx.fillStyle = "rgb(15,27,37)";
      ctx.beginPath();
      for (const ring of data.islands || []) trace(ring);
      ctx.fill();
      ctx.strokeStyle = "rgba(74,108,138,0.75)";
      ctx.lineWidth = S / 2048;
      ctx.beginPath();
      for (const ring of data.water) trace(ring);
      for (const ring of data.islands || []) trace(ring);
      ctx.stroke();
    }
    ctx.fillStyle = "rgba(60,84,108,0.3)";
    ctx.beginPath();
    for (const ring of data.built || []) trace(ring);
    ctx.fill();
    ctx.lineCap = ctx.lineJoin = "round";
    const stroke = (lines, style, width, dash) => {
      ctx.strokeStyle = style;
      ctx.lineWidth = (width * S) / 2048;
      ctx.setLineDash(dash ? dash.map((d) => (d * S) / 2048) : []);
      ctx.beginPath();
      for (const p of lines) {
        ctx.moveTo(px(p[0][0]), py(p[0][1]));
        for (let i = 1; i < p.length; i++) ctx.lineTo(px(p[i][0]), py(p[i][1]));
      }
      ctx.stroke();
    };
    const roads = (c) => (data.roads || []).filter((r) => r.c === c).map((r) => r.p);
    stroke((data.streams || []).map((s) => s.p), "rgba(74,108,138,0.55)", 0.8);
    stroke(roads("track"), "rgba(120,146,170,0.3)", 0.6, [3, 3]);
    stroke(roads("local"), "rgba(140,166,190,0.42)", 0.6);
    stroke(roads("main"), "rgba(160,186,208,0.6)", 1);
    stroke((data.rail || []), "rgba(150,160,175,0.5)", 0.7, [5, 2]);
    stroke(roads("hwy"), "rgba(3,6,9,0.9)", 3);
    stroke(roads("hwy"), "rgba(196,214,230,0.85)", 1.5);
    for (const a of data.airports || []) stroke((a.runways || []).map((r) => [[r[0], r[1]], [r[2], r[3]]]), "rgba(233,238,244,0.9)", 2.2);
    ctx.setLineDash([]);
    // buildings as specks, so a town reads as a town from any height
    ctx.fillStyle = "rgba(172,190,208,0.55)";
    const b = Math.max(1, S / 2048);
    for (const [x, y] of data.buildings || []) ctx.fillRect(px(x) - b / 2, py(y) - b / 2, b, b);
    // a 4 km reference grid
    ctx.strokeStyle = "rgba(111,211,255,0.07)";
    ctx.lineWidth = S / 2048;
    ctx.beginPath();
    for (let k = -half; k <= half + 1e-6; k += 4) {
      ctx.moveTo(px(k), 0);
      ctx.lineTo(px(k), S);
      ctx.moveTo(0, py(k));
      ctx.lineTo(S, py(k));
    }
    ctx.stroke();
    const texture = keep(new THREE.CanvasTexture(canvas));
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());
    this.terrain = new THREE.Mesh(geometry, keep(new THREE.MeshBasicMaterial({ map: texture })));
    this.scene.add(this.terrain);

    // the table the theatre sits on, and its frame
    const table = new THREE.GridHelper(512, 64, 0x16222e, 0x0d1620);
    table.position.y = -0.05;
    table.material.transparent = true;
    table.material.opacity = 0.7;
    keep(table.geometry);
    keep(table.material);
    this.scene.add(table);
    const frame = [];
    for (let k = 0; k < 64; k++) {
      const t0 = -half + k, t1 = t0 + 1;
      for (const [ax, ay, bx, by] of [[t0, -half, t1, -half], [t0, half, t1, half], [-half, t0, -half, t1], [half, t0, half, t1]]) {
        const a = this.at(ax, ay, 0.03), bb = this.at(bx, by, 0.03);
        frame.push(a.x, a.y, a.z, bb.x, bb.y, bb.z);
      }
    }
    this.scene.add(this.lines(frame, C.accent, 0.55));
  }

  lines(positions, color, opacity = 1, dashed = null) {
    const geometry = this.keep(new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(positions, 3)));
    const material = this.keep(dashed ? new THREE.LineDashedMaterial({ color, transparent: true, opacity, dashSize: dashed[0], gapSize: dashed[1], depthWrite: false }) : new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity, depthWrite: false }));
    const mesh = new THREE.LineSegments(geometry, material);
    if (dashed) mesh.computeLineDistances();
    return mesh;
  }

  /** Line segments along a path of [x, y] points, draped on the terrain. */
  drape(points, up, out, step = 0.5) {
    const a = new THREE.Vector3(), b = new THREE.Vector3();
    for (let i = 1; i < points.length; i++) {
      const p = points[i - 1], q = points[i], parts = Math.max(1, Math.ceil(Math.hypot(q[0] - p[0], q[1] - p[1]) / step));
      for (let k = 0; k < parts; k++) {
        this.at(p[0] + ((q[0] - p[0]) * k) / parts, p[1] + ((q[1] - p[1]) * k) / parts, up, a);
        this.at(p[0] + ((q[0] - p[0]) * (k + 1)) / parts, p[1] + ((q[1] - p[1]) * (k + 1)) / parts, up, b);
        out.push(a.x, a.y, a.z, b.x, b.y, b.z);
      }
    }
  }

  buildVectors() {
    const { theatre } = this, data = theatre.data;
    // the town's edge: what is being protected
    const edge = [], { n, cell, half } = theatre, isTown = theatre.isTown;
    for (const c of theatre.town.cells) {
      const col = c % n, row = Math.floor(c / n), x0 = col * cell - half, y0 = row * cell - half, x1 = x0 + cell, y1 = y0 + cell;
      if (col === 0 || !isTown[c - 1]) this.drape([[x0, y0], [x0, y1]], 0.035, edge, 1);
      if (col === n - 1 || !isTown[c + 1]) this.drape([[x1, y0], [x1, y1]], 0.035, edge, 1);
      if (row === 0 || !isTown[c - n]) this.drape([[x0, y0], [x1, y0]], 0.035, edge, 1);
      if (row === n - 1 || !isTown[c + n]) this.drape([[x0, y1], [x1, y1]], 0.035, edge, 1);
    }
    this.townEdge = this.lines(edge, "#ffffff", 0.75);
    this.scene.add(this.townEdge);
    // the real final burn perimeter, off until asked for
    const final = [];
    for (const ring of data.perimeter || []) this.drape(ring, 0.05, final, 1.2);
    this.perimeter = this.lines(final, "#ffffff", 0.55, [0.45, 0.3]);
    this.perimeter.visible = false;
    this.scene.add(this.perimeter);
    // satellite detections, as points: before the scenario opens, and after it
    const make = (list, color, size) => {
      const pos = [];
      const v = new THREE.Vector3();
      for (const h of list) {
        if (!theatre.inside(h.x, h.y)) continue;
        this.at(h.x, h.y, 0.06, v);
        pos.push(v.x, v.y, v.z);
      }
      const points = new THREE.Points(this.keep(new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(pos, 3))), this.keep(new THREE.PointsMaterial({ color, size, sizeAttenuation: false, transparent: true, opacity: 0.9, depthWrite: false })));
      points.visible = false;
      this.scene.add(points);
      return points;
    };
    this.makeHotspots = make;
  }

  /** Split the satellite detections at the moment the scenario opens. */
  setHotspots(opens) {
    const all = this.theatre.data.hotspots || [];
    // before the opening time in pale gold; after it in the violet the analysis mode uses for VIIRS
    this.hotBefore = this.makeHotspots(all.filter((h) => Date.parse(h.t) <= opens), "#ffe9b0", 3 * this.renderer.getPixelRatio());
    this.hotAfter = this.makeHotspots(all.filter((h) => Date.parse(h.t) > opens), C.viirs, 3 * this.renderer.getPixelRatio());
  }

  buildFire() {
    const { n } = this.theatre, gl = this.renderer;
    const linear = gl.capabilities.isWebGL2 && gl.extensions.has("OES_texture_float_linear");
    this.fireData = new Float32Array(n * n * 4);
    this.stateData = new Uint8Array(n * n * 4);
    this.fireTexture = this.keep(new THREE.DataTexture(this.fireData, n, n, THREE.RGBAFormat, THREE.FloatType));
    this.fireTexture.magFilter = this.fireTexture.minFilter = linear ? THREE.LinearFilter : THREE.NearestFilter;
    this.stateTexture = this.keep(new THREE.DataTexture(this.stateData, n, n, THREE.RGBAFormat, THREE.UnsignedByteType));
    this.stateTexture.magFilter = this.stateTexture.minFilter = THREE.NearestFilter;
    this.fireMaterial = this.keep(new THREE.ShaderMaterial({
      uniforms: {
        tFire: { value: this.fireTexture }, tState: { value: this.stateTexture }, uTime: { value: 0 }, uN: { value: n },
        uRetard: { value: new THREE.Color(C.accent) }, uWet: { value: new THREE.Color("#2f6fb0") }, uGuard: { value: new THREE.Color("#f2f6fa") },
      },
      vertexShader: FIRE_VERT, fragmentShader: FIRE_FRAG, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
    }));
    const mesh = new THREE.Mesh(this.terrainGeometry, this.fireMaterial);
    mesh.renderOrder = 2;
    this.scene.add(mesh);
  }

  /** Copy the state of the fire into the textures the shader reads. */
  paintFire(world) {
    const fire = world.fire, { fuel } = this.theatre.grid, t = fire.t, out = this.fireData, state = this.stateData;
    const { arrival, tent, tset, hfi, held, retard, guard, wet, knock } = fire;
    const live = (this.liveMask ??= new Uint8Array(arrival.length));
    live.fill(0);
    for (const c of fire.front) if (!held[c] && knock[c] <= t) live[c] = 1;
    for (let i = 0, o = 0; i < arrival.length; i++, o += 4) {
      const at = arrival[i];
      if (at !== UNBURNED) {
        const age = t - at;
        out[o] = 0.5 + 0.5 * Math.min(1, age / 10);
        // flaming for half an hour or so, then the live edge keeps a glow
        out[o + 1] = held[i] ? 0 : Math.max(live[i] ? 0.4 : 0, 1 - age / (hfi[i] > 4000 ? 45 : 28));
        out[o + 2] = Math.max(0, Math.min(1, (Math.log10(Math.max(hfi[i], 10)) - 1) / 3.4));
        out[o + 3] = Math.exp(-Math.max(0, age) / 90);
        state[o] = 0;
        state[o + 1] = 0;
        state[o + 2] = 0;
        state[o + 3] = held[i] ? 255 : 0;
      } else {
        const pending = tent[i];
        out[o] = pending !== UNBURNED && pending > tset[i] ? 0.5 * Math.max(0, Math.min(1, (t - tset[i]) / (pending - tset[i]))) : 0;
        out[o + 1] = 0;
        out[o + 2] = 0;
        out[o + 3] = 0;
        state[o] = retard[i] ? 255 : 0;
        state[o + 1] = guard[i] ? (guard[i] === 2 ? 255 : 140) : 0;
        state[o + 2] = wet[i] > t && fuel[i] ? 255 : 0;
        state[o + 3] = 0;
      }
    }
    this.fireTexture.needsUpdate = true;
    this.stateTexture.needsUpdate = true;
  }

  // ---- overlays that follow the plan and the forecast

  replace(key, object) {
    const old = this.dynamic[key];
    if (old) {
      this.scene.remove(old);
      old.traverse((o) => (o.geometry?.dispose(), o.material?.dispose()));
    }
    this.dynamic[key] = object;
    if (object) this.scene.add(object);
  }

  dynamicLines(positions, color, opacity, dashed) {
    const geometry = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const material = dashed ? new THREE.LineDashedMaterial({ color, transparent: true, opacity, dashSize: dashed[0], gapSize: dashed[1], depthWrite: false }) : new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false });
    const mesh = new THREE.LineSegments(geometry, material);
    if (dashed) mesh.computeLineDistances();
    mesh.renderOrder = 4;
    return mesh;
  }

  /** Forecast isochrones: where the fire is expected in 1, 3, 6, 12 and 24 hours. */
  paintForecast(world, show) {
    const forecast = world.forecast;
    if (!forecast || !show) return this.replace("forecast", null), (this.isoLabels = []), (this.forecastSeen = null);
    if (this.forecastSeen === forecast) return;
    this.forecastSeen = forecast;
    const { theatre } = this, { n, cell, half } = theatre, fuel = theatre.grid.fuel;
    const hours = ISO_HOURS.filter((h) => forecast.from + h * 60 <= forecast.to);
    const sets = contours(forecast.arrival, n, (i) => fuel[i] !== 0, hours.map((h) => forecast.from + h * 60), forecast.to + 1);
    const group = new THREE.Group(), a = new THREE.Vector3(), b = new THREE.Vector3();
    this.isoLabels = [];
    sets.forEach((seg, k) => {
      if (!seg.length) return;
      const pos = [];
      let best = null, bestD = Infinity;
      for (let i = 0; i < seg.length; i += 4) {
        const x0 = (seg[i] + 0.5) * cell - half, y0 = (seg[i + 1] + 0.5) * cell - half, x1 = (seg[i + 2] + 0.5) * cell - half, y1 = (seg[i + 3] + 0.5) * cell - half;
        this.at(x0, y0, 0.07, a);
        this.at(x1, y1, 0.07, b);
        pos.push(a.x, a.y, a.z, b.x, b.y, b.z);
        const d = Math.hypot(x0 - theatre.town.cx, y0 - theatre.town.cy);
        if (d < bestD) { bestD = d; best = [x0, y0]; }
      }
      group.add(this.dynamicLines(pos, C.warn, 0.95 - k * 0.13));
      if (best) this.isoLabels.push({ key: `iso${hours[k]}`, x: best[0], y: best[1], text: `+${hours[k]}H` });
    });
    this.replace("forecast", group);
  }

  /** The control line of a plan (previewed or committed) and the projected run to the town. */
  paintPlan(plan) {
    const key = plan ? `${plan.key}:${plan.line?.cells.length}:${plan.line?.cells[0]}:${plan.route?.length}:${plan.route?.[0]}:${plan.segments?.length}:${plan.missions?.length}:${plan.missions?.[0]?.at}` : "";
    if (key === this.planSeen) return;
    this.planSeen = key;
    const group = new THREE.Group(), { theatre } = this;
    this.planLabels = [];
    // proposed routes: aircraft at the height they would fly, ground units along the road
    const way = [], run = [], v0 = new THREE.Vector3(), v1 = new THREE.Vector3();
    for (const m of plan?.missions || []) {
      for (const leg of m.legs) {
        const list = leg.kind === "drop" || leg.kind === "scoop" ? run : way;
        for (let i = 1; i < leg.pts.length; i++) {
          const p = leg.pts[i - 1], q = leg.pts[i];
          if (m.cls === "air") {
            this.aloft(p[0], p[1], leg.alt0 + ((leg.alt1 - leg.alt0) * leg.cum[i - 1]) / (leg.km || 1), v0);
            this.aloft(q[0], q[1], leg.alt0 + ((leg.alt1 - leg.alt0) * leg.cum[i]) / (leg.km || 1), v1);
          } else {
            this.at(p[0], p[1], 0.08, v0);
            this.at(q[0], q[1], 0.08, v1);
          }
          list.push(v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
        }
        if (m.label && leg.does && leg.does !== "work" && !m.labelled) {
          const end = leg.pts[leg.pts.length - 1];
          this.planLabels.push({ key: `mission:${m.id}`, x: end[0], y: end[1], text: m.label, tone: "accent" });
          m.labelled = true;
        }
      }
    }
    for (const [line, list] of [[this.proposed.way, way], [this.proposed.run, run]]) {
      line.mesh.visible = list.length > 0;
      if (!list.length) continue;
      line.mesh.geometry.dispose();
      line.mesh.geometry = new LineSegmentsGeometry().setPositions(list);
      if (line.dashed) line.mesh.computeLineDistances();
      line.material.resolution.set(this.size.w, this.size.h);
    }
    if (plan?.route?.length > 1) {
      const pos = [];
      this.drape(plan.route.filter((_, i) => i % 2 === 0 || i === plan.route.length - 1), 0.1, pos, 0.4);
      group.add(this.dynamicLines(pos, C.warn, 0.95, [0.5, 0.3]));
      const mid = plan.route[plan.route.length >> 1];
      this.planLabels.push({ key: "run", x: mid[0], y: mid[1], text: plan.routeLabel || "PROJECTED RUN", tone: "warn" });
    }
    const line = plan?.line;
    if (line?.cells?.length) {
      const by = { [KIND.natural]: [], [KIND.road]: [], [KIND.build]: [] };
      for (let i = 1; i < line.cells.length; i++) {
        const p = theatre.xy(line.cells[i - 1]), q = theatre.xy(line.cells[i]);
        if (Math.hypot(q[0] - p[0], q[1] - p[1]) > theatre.cell * 3) continue; // the line left the map and came back
        this.drape([p, q], 0.09, by[line.kinds[i]], 1);
      }
      group.add(this.dynamicLines(by[KIND.natural], C.accent, 0.5, [0.12, 0.2]));
      group.add(this.dynamicLines(by[KIND.road], C.accent, 0.8, [0.35, 0.15]));
      group.add(this.dynamicLines(by[KIND.build], C.accent, 1));
      const mid = theatre.xy(line.cells[line.cells.length >> 1]);
      this.planLabels.push({ key: "line", x: mid[0], y: mid[1], text: `CONTROL LINE · ${line.km.toFixed(1)} KM`, tone: "accent" });
    }
    for (const seg of plan?.segments || []) {
      const pos = [];
      this.drape(seg.cells.map((c) => theatre.xy(c)), 0.11, pos, 1);
      group.add(this.dynamicLines(pos, C.accent, 0.95));
    }
    this.replace("plan", group.children.length ? group : null);
  }

  // ---- units

  unitNode(u) {
    let node = this.unitNodes.get(u.id);
    if (node) return node;
    let sprite = null, material = null, model = null, shadow = null;
    if (SPECS[u.type].cls === "air") {
      // a solid model, and the shadow it throws on the ground under it
      model = makeAircraft(u.type);
      this.airScene.add(model.group);
      shadow = new THREE.Mesh((this.shadowGeometry ??= this.keep(new THREE.CircleGeometry(0.5, 20).rotateX(-Math.PI / 2))), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.4, depthWrite: false }));
      shadow.renderOrder = 6;
      this.scene.add(shadow);
    } else {
      const texture = (this.textures[u.type] ??= this.keep(symbolTexture(u.type)));
      material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, depthWrite: false, sizeAttenuation: false });
      sprite = new THREE.Sprite(material);
      sprite.renderOrder = 10;
      this.scene.add(sprite);
    }
    const tag = document.createElement("button");
    tag.className = "tag";
    tag.tabIndex = -1;
    tag.innerHTML = "<b></b><span></span><i></i>";
    tag.addEventListener("click", () => this.onPick?.({ unit: u.id }));
    this.labels.appendChild(tag);
    node = { sprite, material, model, shadow, tag, text: ["", "", ""], sx: 0, sy: 0, shown: false, cls: "" };
    this.unitNodes.set(u.id, node);
    return node;
  }

  label(map, key, cls) {
    let node = map.get(key);
    if (!node) {
      node = document.createElement("div");
      node.className = cls;
      this.labels.appendChild(node);
      map.set(key, node);
    }
    return node;
  }

  project(x, y, up, out) {
    const v = this.at(x, y, up, (this.tmp ??= new THREE.Vector3()));
    v.project(this.camera);
    out.x = (v.x * 0.5 + 0.5) * this.size.w;
    out.y = (-v.y * 0.5 + 0.5) * this.size.h;
    out.on = v.z > -1 && v.z < 1 && out.x > -60 && out.x < this.size.w + 60 && out.y > -40 && out.y < this.size.h + 40;
    return out;
  }

  // ---- camera

  setView({ x, y, dist, polar, az }) {
    const target = this.at(x, y, 0);
    this.controls?.target.copy(target);
    this.camera.position.set(target.x + dist * Math.sin(polar) * Math.sin(az), target.y + dist * Math.cos(polar), target.z + dist * Math.sin(polar) * Math.cos(az));
    this.camera.lookAt(target);
    this.view = { x, y, dist, polar, az };
  }
  currentView() {
    const t = this.controls.target, p = this.camera.position, d = p.distanceTo(t);
    return { x: t.x, y: -t.z, dist: d, polar: Math.acos(Math.max(-1, Math.min(1, (p.y - t.y) / d))), az: Math.atan2(p.x - t.x, p.z - t.z) };
  }
  /** Fly the camera to look at a point: from `dist` km away, tilted `polar` from overhead, from bearing `az`. */
  flyTo(to, seconds = 1.6) {
    const from = this.currentView(), half = this.theatre.half;
    const goal = { ...from, ...to };
    goal.x = Math.max(-half, Math.min(half, goal.x));
    goal.y = Math.max(-half, Math.min(half, goal.y));
    let turn = goal.az - from.az;
    while (turn > Math.PI) turn -= 2 * Math.PI;
    while (turn < -Math.PI) turn += 2 * Math.PI;
    if (this.still || seconds <= 0) return this.setView(goal), this.controls.update();
    this.flight = { from, goal, turn, start: performance.now(), ms: seconds * 1000 };
  }

  resize() {
    const w = Math.max(1, this.mount.clientWidth), h = Math.max(1, this.mount.clientHeight);
    this.size = { w, h };
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.applyInset();
  }

  /** Keep the centre of the view in the part of the stage the boards leave free. */
  setInset(left, right) {
    this.insetGoal = (left - right) / 2;
    if (this.inset === undefined) { this.inset = this.insetGoal; this.applyInset(); }
  }
  applyInset() {
    const { w, h } = this.size, shift = this.inset || 0;
    if (Math.abs(shift) > 0.5) this.camera.setViewOffset(w, h, -shift, 0, w, h);
    else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
  }

  // ---- one frame

  frame(world, view, now) {
    const { camera, controls } = this;
    if (this.flight) {
      const f = this.flight, k = ease(Math.min(1, (now - f.start) / f.ms));
      const mix = (a, b) => a + (b - a) * k;
      this.setView({ x: mix(f.from.x, f.goal.x), y: mix(f.from.y, f.goal.y), dist: Math.exp(mix(Math.log(f.from.dist), Math.log(f.goal.dist))), polar: mix(f.from.polar, f.goal.polar), az: f.from.az + f.turn * k });
      if (k >= 1) this.flight = null;
    } else controls.update();
    if (this.insetGoal !== undefined && Math.abs(this.insetGoal - this.inset) > 0.5) {
      this.inset += (this.insetGoal - this.inset) * (this.still ? 1 : 0.16);
      this.applyInset();
    }
    const cam = this.currentView();
    this.fireMaterial.uniforms.uTime.value = now / 1000;
    if (now - this.lastFireUpdate > 120) {
      this.paintFire(world);
      this.lastFireUpdate = now;
    }
    this.paintForecast(world, view.layers.forecast);
    this.paintPlan(view.plan);
    this.perimeter.visible = Boolean(view.layers.actual);
    if (this.hotBefore) this.hotBefore.visible = Boolean(view.layers.hotspots);
    if (this.hotAfter) this.hotAfter.visible = Boolean(view.layers.actual);

    const pxScale = (px) => (px * 2 * TAN) / this.size.h;
    const p = { x: 0, y: 0, on: false };
    const stalks = [], marks = [], groundTrack = [];
    const a = new THREE.Vector3(), b = new THREE.Vector3();
    const far = cam.dist > 55, k = view.rem / 13;
    const renewCourse = now - this.lastCourse > 110;
    const ahead = [], runs = [], picked = [], trail = [];
    const segment = (list, v0, v1) => list.push(v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
    // aircraft standing at a base are drawn in a rank whose spacing grows with their size on
    // screen, so that from far out they stand side by side instead of in a heap
    const ramps = new Map();
    for (const u of world.units) {
      if (u.cls !== "air" || u.alt > 0 || u.speed > 0) continue;
      const list = ramps.get(u.home.name) || ramps.set(u.home.name, []).get(u.home.name);
      list.push(u);
    }
    const stand = new Map();
    for (const list of ramps.values()) {
      const cx = list.reduce((t, u) => t + u.x, 0) / list.length, cy = list.reduce((t, u) => t + u.y, 0) / list.length;
      list.forEach((u, i) => stand.set(u.id, { i, n: list.length, cx, cy }));
    }
    for (const u of world.units) {
      const node = this.unitNode(u), spec = SPECS[u.type], air = spec.cls === "air";
      const selected = view.unit === u.id, committed = Boolean(u.op), moving = u.legs.length > 0;
      const colour = selected ? "#ffffff" : committed ? C.accent : moving ? "#c6d3e0" : "#8fa0b3";
      let size, anchor;
      if (air) {
        const m = node.model, group = m.group, parked = stand.get(u.id);
        this.aloft(u.x, u.y, u.alt, group.position);
        const eye = camera.position.distanceTo(group.position);
        size = m.px * (selected ? 1.2 : 1) * (parked ? 0.6 : 1) * k;
        // a constant size on screen from far out; closer in it stops shrinking with the map
        const length = Math.max(parked ? 0.05 : 0.12, (size * 2 * eye * TAN) / this.size.h);
        group.scale.setScalar(length);
        // where it is drawn: its place in the rank while parked, else where it is; eased between the two
        let tx = u.x, ty = u.y;
        if (parked) {
          const per = Math.min(parked.n, 7), gap = Math.max(0.09, length * 0.95);
          tx = parked.cx + ((parked.i % per) - (per - 1) / 2) * gap;
          ty = parked.cy - Math.floor(parked.i / per) * gap * 1.15;
        }
        if (node.dx === undefined || Math.hypot(tx - node.dx, ty - node.dy) > 6) { node.dx = tx; node.dy = ty; }
        node.dx += (tx - node.dx) * 0.25;
        node.dy += (ty - node.dy) * 0.25;
        this.aloft(node.dx, node.dy, u.alt, group.position);
        anchor = group.position;
        // ease into the bank and the climb: the simulation reports them leg by leg
        const flying = u.alt > 1;
        m.bank += ((flying ? u.bank : 0) - m.bank) * 0.14;
        m.pitch += ((flying ? Math.max(-0.3, Math.min(0.3, Math.atan(u.climb * this.lift * 1000 * 3))) : 0) - m.pitch) * 0.14;
        group.rotation.set(m.pitch, (-u.hdg * Math.PI) / 180, (-m.bank * Math.PI) / 180, "YXZ");
        m.material.color.set(colour);
        m.material.emissive.set(colour).multiplyScalar(selected ? 0.12 : 0.3);
        m.lines.opacity = selected ? 0.9 : 0.5;
        if (m.rotor && (flying || moving)) m.rotor.rotation.y = now * 0.02;
        this.at(node.dx, node.dy, 0.03, node.shadow.position);
        node.shadow.scale.set(length * 0.9, 1, length * 0.9);
        node.shadow.material.opacity = flying ? Math.max(0.12, 0.42 - u.alt / 4000) : 0;
      } else {
        this.at(u.x, u.y, 0.04, node.sprite.position);
        anchor = node.sprite.position;
        size = 26 * (selected ? 1.25 : 1) * k;
        node.sprite.scale.setScalar(pxScale(size));
        node.material.color.set(colour);
      }
      a.copy(anchor).project(camera);
      p.x = (a.x * 0.5 + 0.5) * this.size.w;
      p.y = (-a.y * 0.5 + 0.5) * this.size.h;
      p.on = a.z > -1 && a.z < 1 && p.x > -60 && p.x < this.size.w + 60 && p.y > -40 && p.y < this.size.h + 40;
      node.sx = p.x;
      node.sy = p.y;
      // parked units of the same base pile up from far out: show only those that are doing something
      const shown = p.on && (!far || air || committed || moving || selected);
      node.shown = shown;
      if (air) {
        node.model.group.visible = shown;
        node.shadow.visible = shown;
        if (u.alt > 30 && shown) {
          this.at(node.dx, node.dy, 0.02, b);
          stalks.push(anchor.x, anchor.y, anchor.z, b.x, b.y, b.z);
        }
        // the course: every leg still to fly, at the height it is flown at, from where the aircraft is now
        // (helicopters shuttle a few hundred metres between a dip site and the edge: their
        // courses are drawn only when one is selected, or the fire disappears under them)
        if (renewCourse && moving && shown && (selected || !spec.heli)) {
          const list = selected ? picked : ahead;
          let from = a.copy(anchor);
          for (let i = u.li; i < u.legs.length; i++) {
            const leg = u.legs[i];
            if (leg.kind === "ground" || leg.km <= 0) continue;
            const done = i === u.li ? ((world.t - leg.t0) / (leg.t1 - leg.t0)) * leg.km : -1;
            const work = leg.kind === "drop" || leg.kind === "scoop";
            for (let q = 0; q < leg.pts.length; q++) {
              if (leg.cum[q] <= done) continue;
              this.aloft(leg.pts[q][0], leg.pts[q][1], leg.alt0 + ((leg.alt1 - leg.alt0) * leg.cum[q]) / leg.km, b);
              if (i > u.li && q === 0) {
                from = from.copy(b);
                continue;
              }
              segment(work && !selected ? runs : list, from, b);
              if (work && selected) segment(runs, from, b);
              from = from.copy(b);
            }
            if (leg.mark) {
              const end = leg.pts[leg.pts.length - 1];
              // a post from the ground to where the course turns in, drops or touches down
              this.at(end[0], end[1], 0.02, a);
              stalks.push(b.x, b.y, b.z, a.x, a.y, a.z);
              if (selected) marks.push({ key: `m${u.id}:${leg.t1.toFixed(1)}`, x: end[0], y: end[1], text: leg.mark });
              from = a.copy(b);
            }
          }
          for (let i = 1; i < u.trail.length; i++) {
            this.aloft(u.trail[i - 1][0], u.trail[i - 1][1], u.trail[i - 1][2], a);
            this.aloft(u.trail[i][0], u.trail[i][1], u.trail[i][2], b);
            segment(trail, a, b);
          }
        }
      } else {
        node.sprite.visible = shown;
        // the route of a selected ground unit, along the road
        if (selected && moving) for (let i = u.li; i < u.legs.length; i++) if (u.legs[i].pts.length > 1) this.drape(u.legs[i].pts, 0.07, groundTrack, 0.6);
      }
      // the data tag
      const lines = [u.call, "", ""];
      if (selected || (air && (committed || u.alt > 0))) lines[1] = air ? `${spec.short} ${String(Math.round(u.alt / 10) * 10).padStart(4, "0")} ${Math.round(u.speed)}` : spec.short;
      if (selected || (committed && !far)) lines[2] = u.status;
      const cls = `tag${selected ? " on" : ""}${committed ? " task" : ""}${air ? " air" : ""}`;
      if (node.cls !== cls) node.tag.className = node.cls = cls;
      for (let i = 0; i < 3; i++) if (node.text[i] !== lines[i]) node.tag.children[i].textContent = node.text[i] = lines[i];
      node.want = shown && (air || selected || committed || cam.dist < 22);
      // which tags win where they collide: the selected unit, then aircraft in the air, then units at work
      node.rank = (selected ? 1000 : 0) + (air && u.alt > 0 ? 100 : 0) + (committed ? 30 : 0) + (moving ? 10 : 0) + (air ? 5 : 0);
      node.tx = p.x + size * 0.55;
      node.ty = p.y - size * 0.5;
      node.tw = (Math.max(...lines.map((l) => l.length)) * 0.62 + 0.9) * view.rem * 0.85;
      node.th = lines.filter(Boolean).length * view.rem * 1.08 + 3;
    }
    if (renewCourse) {
      this.lastCourse = now;
      for (const [key, list] of [["ahead", ahead], ["run", runs], ["picked", picked], ["trail", trail]]) {
        const line = this.course[key];
        line.mesh.visible = list.length > 0;
        if (!list.length) continue;
        line.mesh.geometry.dispose();
        line.mesh.geometry = new LineSegmentsGeometry().setPositions(list);
        if (line.dashed) line.mesh.computeLineDistances();
        line.material.resolution.set(this.size.w, this.size.h);
        line.material.linewidth = (key === "ahead" ? 2.2 : key === "run" ? 3.6 : key === "picked" ? 3.2 : 1.4) * k;
      }
    }
    const placed = [];
    for (const node of [...this.unitNodes.values()].sort((m, k) => k.rank - m.rank)) {
      let show = node.want;
      if (show && placed.some((b) => node.tx < b.tx + b.tw && node.tx + node.tw > b.tx && node.ty < b.ty + b.th && node.ty + node.th > b.ty)) show = false;
      if (show) placed.push(node);
      node.tag.style.display = show ? "" : "none";
      if (show) node.tag.style.transform = `translate(${Math.round(node.tx)}px, ${Math.round(node.ty)}px)`;
    }
    this.replace("selTrack", groundTrack.length ? this.dynamicLines(groundTrack, "#ffffff", 0.95, [0.3, 0.18]) : null);
    if (renewCourse || !this.dynamic.stalks) this.replace("stalks", stalks.length ? this.dynamicLines(stalks, C.accent, 0.4) : null);

    // fixed labels: places, incidents, isochrones, plan
    const seen = new Set();
    const place = (key, x, y, text, cls, up = 0.1) => {
      this.project(x, y, up, p);
      const node = this.label(this.fireTags, key, cls);
      seen.add(key);
      if (node.className !== cls) node.className = cls;
      if (node.textContent !== text) node.textContent = text;
      node.style.display = p.on ? "" : "none";
      if (p.on) node.style.transform = `translate(${Math.round(p.x)}px, ${Math.round(p.y)}px)`;
    };
    for (const pl of view.places) place(`p:${pl.n}`, pl.x, pl.y, pl.n, `maplabel ${pl.k}`);
    for (const f of view.incidents) place(`f:${f.id}`, f.head[0], f.head[1], `FIRE ${fireName(f.id)}${f.cls ? ` · CLASS ${f.cls}` : " · OUT"}`, `firetag${view.fire === f.id ? " on" : ""}${f.cls ? "" : " out"}`, 0.3);
    if (view.layers.forecast) for (const l of this.isoLabels || []) place(l.key, l.x, l.y, l.text, "isolabel");
    for (const l of this.planLabels || []) place(l.key, l.x, l.y, l.text, `planlabel ${l.tone}`, 0.15);
    for (const m of marks) place(m.key, m.x, m.y, m.text, "waypoint", 0.12);
    for (const [key, node] of this.fireTags) if (!seen.has(key)) { node.remove(); this.fireTags.delete(key); }

    this.onCamera?.(cam);
    // the map first; then the aircraft and their courses over it, sorted among themselves
    this.renderer.autoClear = false;
    this.renderer.clear();
    this.renderer.render(this.scene, camera);
    this.renderer.clearDepth();
    this.renderer.render(this.airScene, camera);
  }

  dispose() {
    this.observer.disconnect();
    for (const [k, f] of Object.entries(this.handlers)) this.renderer.domElement.removeEventListener(k, f);
    this.controls.dispose();
    for (const key of Object.keys(this.dynamic)) this.replace(key, null);
    for (const node of this.unitNodes.values()) {
      node.material?.dispose();
      node.model?.dispose();
      node.shadow?.material.dispose();
      node.tag.remove();
    }
    for (const line of [...Object.values(this.course), ...Object.values(this.proposed)]) line.mesh.geometry.dispose();
    disposeAircraft();
    for (const node of this.fireTags.values()) node.remove();
    for (const d of this.disposables) d.dispose?.();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}

export { toLatLon };
