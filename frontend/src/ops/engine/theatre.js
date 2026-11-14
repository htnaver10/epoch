// A theatre: the 64 km square around a town, turned from bundled map data into what the
// simulation needs. Fuels and breaks on the fire grid, lakes an aircraft can scoop from, a
// road network to route on, and the town itself as the thing to protect.
import { FUEL } from "./fbp.js";
import { BARRIER, lineCells } from "./fire.js";

export const COVER = { land: 0, water: 1, wetland: 2, wood: 3, scrub: 4, grass: 5, rock: 6, built: 7, bare: 8 };
const ROAD_SPEED = { hwy: 90, main: 70, local: 40, track: 25 }; // km/h
const ROAD_BREAK = { hwy: BARRIER.highway, main: BARRIER.main, local: BARRIER.local, track: BARRIER.track };
const SCOOP_AXES = [[1, 0], [0, 1], [1, 1], [1, -1]];

/** Smooth, repeatable noise in 0..1 for the fuel mosaic. */
export function noise2(x, y, seed = 0) {
  const hash = (i, j) => {
    let h = (i * 374761393 + j * 668265263 + seed * 1442695041) | 0;
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
  };
  const xi = Math.floor(x), yi = Math.floor(y), fx = x - xi, fy = y - yi;
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const top = hash(xi, yi) + (hash(xi + 1, yi) - hash(xi, yi)) * sx, bottom = hash(xi, yi + 1) + (hash(xi + 1, yi + 1) - hash(xi, yi + 1)) * sx;
  return top + (bottom - top) * sy;
}

function decodeRle(rle, size) {
  const out = new Uint8Array(size);
  let k = 0;
  for (let i = 0; i < rle.length; i += 2) {
    out.fill(rle[i], k, k + rle[i + 1]);
    k += rle[i + 1];
  }
  return out;
}

function decodeElevation(elev, n) {
  const out = new Float32Array(n * n);
  if (!elev?.b64) return out;
  const raw = typeof atob === "function" ? Uint8Array.from(atob(elev.b64), (c) => c.charCodeAt(0)) : new Uint8Array(Buffer.from(elev.b64, "base64"));
  const src = new Int16Array(raw.buffer, raw.byteOffset, raw.byteLength >> 1), m = elev.n;
  // bilinear resample from the coarser elevation grid
  for (let r = 0; r < n; r++) {
    const y = Math.min(m - 1.001, Math.max(0, ((r + 0.5) / n) * m - 0.5)), y0 = Math.floor(y), fy = y - y0;
    for (let c = 0; c < n; c++) {
      const x = Math.min(m - 1.001, Math.max(0, ((c + 0.5) / n) * m - 0.5)), x0 = Math.floor(x), fx = x - x0;
      const a = src[y0 * m + x0], b = src[y0 * m + x0 + 1], d = src[(y0 + 1) * m + x0], e = src[(y0 + 1) * m + x0 + 1];
      out[r * n + c] = (a + (b - a) * fx) * (1 - fy) + (d + (e - d) * fx) * fy;
    }
  }
  return out;
}

/** Fuel type for a cell of land cover. The mosaic of forest types is modelled, not mapped. */
function fuelFor(cover, x, y, elev, profile) {
  switch (cover) {
    case COVER.water:
    case COVER.rock:
    case COVER.bare:
      return FUEL.NF;
    case COVER.built:
      return FUEL.URB;
    case COVER.wetland:
      return FUEL.WET;
    case COVER.scrub:
      return FUEL.SCR;
    case COVER.grass:
      return FUEL.O1;
    default: {
      if (profile.treeline && elev > profile.treeline) return FUEL.NF;
      if (profile.treeline && elev > profile.treeline - 250) return FUEL.C1;
      const v = 0.65 * noise2(x / 2.6, y / 2.6, profile.seed) + 0.35 * noise2(x / 0.7, y / 0.7, profile.seed + 7);
      let k = 0;
      for (const [share, fuel] of profile.mix) if (v < (k += share)) return FUEL[fuel];
      return FUEL[profile.mix.at(-1)[1]];
    }
  }
}

export const PROFILES = {
  // taiga shield: black spruce and jack pine between bedrock and water
  shield: { seed: 11, mix: [[0.2, "C1"], [0.38, "C2"], [0.14, "C3"], [0.18, "M2"], [0.1, "D1"]] },
  // boreal plain: spruce, mixedwood and aspen
  plain: { seed: 23, mix: [[0.42, "C2"], [0.1, "C3"], [0.3, "M2"], [0.18, "D1"]] },
  // montane valley: lodgepole pine on the floor, sparse subalpine forest below the treeline
  montane: { seed: 37, treeline: 2150, mix: [[0.5, "C3"], [0.18, "C2"], [0.14, "M2"], [0.1, "D1"], [0.08, "O1"]] },
};

class MinQueue {
  constructor() {
    this.k = [];
    this.v = [];
  }
  push(key, value) {
    const { k, v } = this;
    let i = k.length;
    k.push(key);
    v.push(value);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (k[p] <= key) break;
      k[i] = k[p];
      v[i] = v[p];
      i = p;
    }
    k[i] = key;
    v[i] = value;
  }
  pop() {
    const { k, v } = this;
    const top = v[0], key = k.pop(), value = v.pop(), n = k.length;
    if (n) {
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && k[c + 1] < k[c]) c++;
        if (k[c] >= key) break;
        k[i] = k[c];
        v[i] = v[c];
        i = c;
      }
      k[i] = key;
      v[i] = value;
    }
    return top;
  }
  get size() {
    return this.k.length;
  }
}

/** The road network: junctions where polylines share a coordinate, routed by travel time. */
export function roadGraph(roads) {
  const index = new Map(), nodes = [];
  const node = (p) => {
    const key = `${p[0].toFixed(3)},${p[1].toFixed(3)}`;
    let i = index.get(key);
    if (i === undefined) {
      i = nodes.length;
      index.set(key, i);
      nodes.push({ x: p[0], y: p[1], edges: [] });
    }
    return i;
  };
  for (const road of roads) {
    const speed = ROAD_SPEED[road.c] || 40;
    for (let i = 1; i < road.p.length; i++) {
      const a = node(road.p[i - 1]), b = node(road.p[i]);
      if (a === b) continue;
      const km = Math.hypot(nodes[a].x - nodes[b].x, nodes[a].y - nodes[b].y), min = (km / speed) * 60;
      nodes[a].edges.push({ to: b, km, min });
      nodes[b].edges.push({ to: a, km, min });
    }
  }
  // which part of the network each junction belongs to: a route needs both ends in one part
  const part = new Int32Array(nodes.length).fill(-1);
  let parts = 0;
  for (let s = 0; s < nodes.length; s++) {
    if (part[s] >= 0) continue;
    const stack = [s];
    part[s] = parts;
    while (stack.length) for (const e of nodes[stack.pop()].edges) if (part[e.to] < 0) { part[e.to] = parts; stack.push(e.to); }
    parts++;
  }
  const size = new Int32Array(parts);
  for (const p of part) size[p]++;
  let main = 0;
  for (let i = 1; i < parts; i++) if (size[i] > size[main]) main = i;
  const BIN = 1; // km
  const bins = new Map();
  nodes.forEach((n, i) => {
    if (part[i] !== main) return;
    const key = `${Math.floor(n.x / BIN)},${Math.floor(n.y / BIN)}`;
    if (!bins.has(key)) bins.set(key, []);
    bins.get(key).push(i);
  });
  /** Nearest junction on the main network. */
  const nearest = (x, y) => {
    let best = -1, bestD = Infinity;
    const bx = Math.floor(x / BIN), by = Math.floor(y / BIN);
    for (let ring = 0; ring < 80 && (best < 0 || ring * BIN <= bestD + BIN); ring++) {
      for (let i = bx - ring; i <= bx + ring; i++)
        for (let j = by - ring; j <= by + ring; j++) {
          if (Math.max(Math.abs(i - bx), Math.abs(j - by)) !== ring) continue;
          for (const k of bins.get(`${i},${j}`) || []) {
            const d = Math.hypot(nodes[k].x - x, nodes[k].y - y);
            if (d < bestD) { bestD = d; best = k; }
          }
        }
    }
    return { node: best, km: bestD };
  };
  /** Fastest drive between two points: the path, its length and its time. Null if off the network. */
  const route = (from, to) => {
    const a = nearest(from[0], from[1]), b = nearest(to[0], to[1]);
    if (a.node < 0 || b.node < 0) return null;
    const cost = new Float32Array(nodes.length).fill(Infinity), prev = new Int32Array(nodes.length).fill(-1);
    const goal = nodes[b.node], queue = new MinQueue();
    cost[a.node] = 0;
    queue.push(0, a.node);
    while (queue.size) {
      const u = queue.pop();
      if (u === b.node) break;
      for (const e of nodes[u].edges) {
        const c = cost[u] + e.min;
        if (c < cost[e.to]) {
          cost[e.to] = c;
          prev[e.to] = u;
          queue.push(c + (Math.hypot(nodes[e.to].x - goal.x, nodes[e.to].y - goal.y) / 90) * 60, e.to);
        }
      }
    }
    if (cost[b.node] === Infinity) return null;
    const path = [];
    for (let u = b.node; u >= 0; u = prev[u]) path.push([nodes[u].x, nodes[u].y]);
    path.reverse();
    let km = 0;
    for (let i = 1; i < path.length; i++) km += Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
    return { path, km, min: cost[b.node], off: [a.km, b.km] };
  };
  return { nodes, nearest, route, main, part };
}

function nearestCellOf(cover, n, c0, r0, code, reach) {
  let best = -1, bestD = Infinity;
  for (let r = Math.max(0, r0 - reach); r <= Math.min(n - 1, r0 + reach); r++)
    for (let c = Math.max(0, c0 - reach); c <= Math.min(n - 1, c0 + reach); c++) {
      const d = Math.hypot(r - r0, c - c0);
      if (cover[r * n + c] === code && d < bestD) { bestD = d; best = r * n + c; }
    }
  return best;
}

/** For every cell, the nearest cell of a given cover (breadth-first from all of them at once). */
function nearestOf(cover, n, code) {
  const near = new Int32Array(n * n).fill(-1), queue = new Int32Array(n * n);
  let head = 0, tail = 0;
  for (let i = 0; i < n * n; i++) if (cover[i] === code) { near[i] = i; queue[tail++] = i; }
  while (head < tail) {
    const u = queue[head++], c = u % n, r = (u - c) / n;
    if (c > 0 && near[u - 1] < 0) { near[u - 1] = near[u]; queue[tail++] = u - 1; }
    if (c < n - 1 && near[u + 1] < 0) { near[u + 1] = near[u]; queue[tail++] = u + 1; }
    if (r > 0 && near[u - n] < 0) { near[u - n] = near[u]; queue[tail++] = u - n; }
    if (r < n - 1 && near[u + n] < 0) { near[u + n] = near[u]; queue[tail++] = u + n; }
  }
  return near;
}

/** Lakes as connected water, and for each the straight runs long enough to scoop on. */
function findLakes(cover, n, cell, minRunKm) {
  const label = new Int32Array(n * n).fill(-1), lakes = [];
  const stack = [];
  for (let s = 0; s < n * n; s++) {
    if (cover[s] !== COVER.water || label[s] >= 0) continue;
    const id = lakes.length;
    let count = 0, sx = 0, sy = 0;
    label[s] = id;
    stack.push(s);
    while (stack.length) {
      const u = stack.pop(), c = u % n, r = (u - c) / n;
      count++;
      sx += c;
      sy += r;
      if (c > 0 && cover[u - 1] === COVER.water && label[u - 1] < 0) { label[u - 1] = id; stack.push(u - 1); }
      if (c < n - 1 && cover[u + 1] === COVER.water && label[u + 1] < 0) { label[u + 1] = id; stack.push(u + 1); }
      if (r > 0 && cover[u - n] === COVER.water && label[u - n] < 0) { label[u - n] = id; stack.push(u - n); }
      if (r < n - 1 && cover[u + n] === COVER.water && label[u + n] < 0) { label[u + n] = id; stack.push(u + n); }
    }
    lakes.push({ id, cells: count, ha: count * cell * cell * 100, c: sx / count, r: sy / count, runs: [] });
  }
  // longest run of water through each cell along four axes; keep the best few per lake and axis
  SCOOP_AXES.forEach(([dx, dy], axis) => {
    const run = new Uint16Array(n * n);
    const step = Math.hypot(dx, dy) * cell;
    for (let r = 0; r < n; r++)
      for (let c = 0; c < n; c++) {
        const i = r * n + c;
        if (cover[i] !== COVER.water) continue;
        const pc = c - dx, pr = r - dy;
        run[i] = pc >= 0 && pr >= 0 && pr < n && cover[pr * n + pc] === COVER.water ? run[pr * n + pc] + 1 : 1;
      }
    // a run ends where the next cell along the axis is not water
    for (let r = 0; r < n; r++)
      for (let c = 0; c < n; c++) {
        const i = r * n + c;
        if (!run[i]) continue;
        const nc = c + dx, nr = r + dy;
        if (nc < n && nr >= 0 && nr < n && cover[nr * n + nc] === COVER.water) continue;
        const km = run[i] * step;
        if (km < minRunKm) continue;
        const lake = lakes[label[i]];
        lake.runs.push({ axis, km, c0: c - dx * (run[i] - 1), r0: r - dy * (run[i] - 1), c1: c, r1: r });
      }
  });
  for (const lake of lakes) {
    lake.runs.sort((a, b) => b.km - a.km);
    // the longest runs on each axis, spread out: a big lake has thousands
    const kept = [];
    for (const run of lake.runs) {
      if (kept.length >= 24) break;
      const mc = (run.c0 + run.c1) / 2, mr = (run.r0 + run.r1) / 2;
      if (kept.some((k) => k.axis === run.axis && Math.hypot((k.c0 + k.c1) / 2 - mc, (k.r0 + k.r1) / 2 - mr) < 12)) continue;
      kept.push(run);
    }
    lake.runs = kept;
  }
  return { lakes, label };
}

/**
 * @param data the theatre JSON (see scripts/prepare_ops.py for the schema)
 */
export function buildTheatre(data, { profile = PROFILES.shield, minScoopKm = 1.4, place = null } = {}) {
  const n = data.cover.n, cell = data.cover.cell, half = (n * cell) / 2;
  const cover = decodeRle(data.cover.rle, n * n);
  const elev = decodeElevation(data.elev, n);
  const col = (x) => Math.max(0, Math.min(n - 1, Math.floor((x + half) / cell)));
  const toCell = (x, y) => col(y) * n + col(x);
  const xy = (i) => [((i % n) + 0.5) * cell - half, (Math.floor(i / n) + 0.5) * cell - half];
  const inside = (x, y) => Math.abs(x) < half && Math.abs(y) < half;

  // buildings per cell; a cell with several is built-up whatever the land-use polygons say
  const buildings = new Uint16Array(n * n);
  for (const [x, y] of data.buildings || []) if (inside(x, y)) buildings[toCell(x, y)]++;
  for (let i = 0; i < n * n; i++) if (buildings[i] >= 3 && cover[i] !== COVER.water) cover[i] = COVER.built;

  const fuel = new Uint8Array(n * n);
  for (let i = 0; i < n * n; i++) {
    const [x, y] = xy(i);
    fuel[i] = fuelFor(cover[i], x, y, elev[i], profile);
  }

  // linear breaks: roads and rivers narrower than a cell
  const barrier = new Uint8Array(n * n);
  const draw = (points, value) => {
    for (let i = 1; i < points.length; i++) {
      const a = points[i - 1], b = points[i];
      if (!inside(a[0], a[1]) && !inside(b[0], b[1])) continue;
      for (const k of lineCells(n, (a[0] + half) / cell - 0.5, (a[1] + half) / cell - 0.5, (b[0] + half) / cell - 0.5, (b[1] + half) / cell - 0.5)) if (barrier[k] < value) barrier[k] = value;
    }
  };
  for (const s of data.streams || []) draw(s.p, s.k === "river" ? BARRIER.river : BARRIER.track);
  for (const r of data.roads || []) draw(r.p, ROAD_BREAK[r.c] || BARRIER.local);
  for (let i = 0; i < n * n; i++) if (!fuel[i] || fuel[i] === FUEL.URB) barrier[i] = 0; // only fuel needs a break

  // the town: the built-up ground around its reference point, grown outward across gaps of
  // up to half a kilometre, so that an outlying lodge or camp is not counted as the town
  const town = { ...(place || data.town), cells: [], edge: [], buildings: 0 };
  {
    const seen = new Uint8Array(n * n), GAP = 3;
    const start = nearestCellOf(cover, n, col(town.x), col(town.y), COVER.built, 20);
    const queue = start >= 0 ? [start] : [];
    if (start >= 0) seen[start] = 1;
    while (queue.length) {
      const u = queue.pop(), c0 = u % n, r0 = (u - c0) / n;
      town.cells.push(u);
      town.buildings += buildings[u];
      for (let r = Math.max(0, r0 - GAP); r <= Math.min(n - 1, r0 + GAP); r++)
        for (let c = Math.max(0, c0 - GAP); c <= Math.min(n - 1, c0 + GAP); c++) {
          const v = r * n + c;
          if (!seen[v] && cover[v] === COVER.built) { seen[v] = 1; queue.push(v); }
        }
    }
  }
  const isTown = new Uint8Array(n * n);
  for (const i of town.cells) isTown[i] = 1;
  let reach = 0, sx = 0, sy = 0;
  for (const i of town.cells) {
    const [x, y] = xy(i);
    sx += x;
    sy += y;
    const c = i % n, r = (i - c) / n;
    if ((c > 0 && !isTown[i - 1]) || (c < n - 1 && !isTown[i + 1]) || (r > 0 && !isTown[i - n]) || (r < n - 1 && !isTown[i + n])) town.edge.push(i);
  }
  if (town.cells.length) {
    town.cx = sx / town.cells.length;
    town.cy = sy / town.cells.length;
    for (const i of town.cells) {
      const [x, y] = xy(i);
      reach = Math.max(reach, Math.hypot(x - town.cx, y - town.cy));
    }
  } else {
    town.cx = town.x;
    town.cy = town.y;
  }
  town.radius = reach;

  const { lakes, label } = findLakes(cover, n, cell, minScoopKm);
  let waterNear = null;
  const elevAt = (x, y) => elev[toCell(x, y)];
  /** The nearest cell to a point that passes `test`, searching outward ring by ring. -1 if none. */
  const nearestCell = (x, y, test, reach = 40) => {
    const c0 = col(x), r0 = col(y);
    for (let ring = 0; ring <= reach; ring++) {
      let best = -1, bestD = Infinity;
      for (let r = r0 - ring; r <= r0 + ring; r++)
        for (let c = c0 - ring; c <= c0 + ring; c++) {
          if (r < 0 || c < 0 || r >= n || c >= n || Math.max(Math.abs(r - r0), Math.abs(c - c0)) !== ring) continue;
          const i = r * n + c;
          if (!test(i)) continue;
          const d = Math.hypot(r - r0, c - c0);
          if (d < bestD) { bestD = d; best = i; }
        }
      if (best >= 0) return best;
    }
    return -1;
  };
  return {
    id: data.id, name: data.name, region: data.region, date: data.date, utcOffset: data.utcOffset ?? -6, centre: data.centre,
    n, cell, half, cover, buildings, isTown, town, lakes, lakeOf: label,
    grid: { n, cell, fuel, barrier, elev },
    roads: roadGraph(data.roads || []),
    data, toCell, xy, inside, elevAt, nearestCell,
    /** Nearest ground that can burn, and nearest open water. */
    nearestFuel: (x, y) => nearestCell(x, y, (i) => fuel[i] !== 0 && fuel[i] !== FUEL.URB),
    nearestWater: (x, y) => (waterNear ??= nearestOf(cover, n, COVER.water))[toCell(x, y)],
  };
}

/** Latitude and longitude of a point in the theatre's local kilometres, and back. */
export function toLatLon(theatre, x, y) {
  const [lat0, lon0] = theatre.centre;
  return [lat0 + y / 110.574, lon0 + x / (111.32 * Math.cos((lat0 * Math.PI) / 180))];
}
export function fromLatLon(theatre, lat, lon) {
  const [lat0, lon0] = theatre.centre;
  return [(lon - lon0) * 111.32 * Math.cos((lat0 * Math.PI) / 180), (lat - lat0) * 110.574];
}
