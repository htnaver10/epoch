// The control line: where to hold a fire short of the town. It is chosen as the least-effort
// line that (a) stands between the town and the projected fire, (b) the fire does not reach
// before it can be built, and (c) uses what is already there: lakes, rock, roads and rivers
// cost little or nothing, so the line ties them together. Found by dynamic programming over
// bearings from the town: one radius per bearing, a bounded change between neighbours.
import { FUEL } from "./fbp.js";
import { UNBURNED, lineCells } from "./fire.js";

export const KIND = { natural: 0, road: 1, build: 2 };
const STEPS = 240; // bearings round the town
const EFFORT = { [FUEL.C1]: 0.8, [FUEL.C2]: 1, [FUEL.C3]: 1, [FUEL.C4]: 1.1, [FUEL.D1]: 0.5, [FUEL.M2]: 0.8, [FUEL.O1]: 0.4, [FUEL.WET]: 0.45, [FUEL.SCR]: 0.6 };
const UNANCHORED = 14; // the price of an end that ties into nothing, in cells of line through forest
const BIG = 1e9;

/**
 * @param theatre   built theatre
 * @param fire      the running FireSim
 * @param forecast  project(fire, hours) with no further suppression
 * @param options   reachKm: how far out the line may go. bufferKm: clear of the built-up edge.
 *                  leadMin: minutes needed before the fire arrives anywhere on the line.
 *                  burnDepth: cells of fuel to burn off on the fire's side of the line.
 * @returns null when the town is not threatened inside the forecast, else the line
 */
export function planLine(theatre, fire, forecast, { reachKm = 12, bufferKm = 0.5, leadMin = 90, burnDepth = 3 } = {}) {
  const { n, cell, half, town, isTown } = theatre, { fuel, barrier } = theatre.grid;
  const now = fire.t, rings = Math.ceil(reachKm / cell), dTheta = (2 * Math.PI) / STEPS;
  const at = (i, j) => {
    const a = i * dTheta, r = (j + 0.5) * cell, x = town.cx + Math.sin(a) * r, y = town.cy + Math.cos(a) * r;
    if (Math.abs(x) >= half || Math.abs(y) >= half) return -1;
    return Math.floor((y + half) / cell) * n + Math.floor((x + half) / cell);
  };
  // along each bearing: where the town ends, where the black begins, and when the fire shows up
  const inner = new Int16Array(STEPS), outer = new Int16Array(STEPS).fill(rings - 1), first = new Float32Array(STEPS).fill(UNBURNED);
  for (let i = 0; i < STEPS; i++) {
    let last = -1, black = rings;
    for (let j = 0; j < rings; j++) {
      const k = at(i, j);
      if (k < 0) break;
      if (isTown[k]) last = j;
      if (fire.arrival[k] <= now && black === rings) black = j;
      if (forecast.arrival[k] < first[i]) first[i] = forecast.arrival[k];
    }
    inner[i] = last;
    outer[i] = Math.min(rings - 1, black - 2);
  }
  const buffer = Math.max(1, Math.round(bufferKm / cell)), lo = new Int16Array(STEPS);
  for (let i = 0; i < STEPS; i++) {
    let m = 1;
    for (let d = -4; d <= 4; d++) m = Math.max(m, inner[(i + d + STEPS) % STEPS]);
    lo[i] = m + buffer;
  }
  // the threatened arc: the smallest arc holding every bearing the projected fire touches
  const hit = Array.from({ length: STEPS }, (_, i) => first[i] !== UNBURNED && first[i] <= forecast.to);
  const hits = hit.filter(Boolean).length;
  if (!hits) return null;
  let gapStart = 0, gap = 0;
  for (let i = 0; i < STEPS; i++) {
    if (hit[i]) continue;
    let len = 0;
    while (len < STEPS && !hit[(i + len) % STEPS]) len++;
    if (len > gap) { gap = len; gapStart = i; }
    i += len - 1;
  }
  const PAD = 10, EXT = 26;
  const ring = gap < 2 * PAD + 4; // threatened from all sides: the line goes all the way round
  const a0 = ring ? 0 : gapStart + gap - PAD, count = ring ? STEPS : STEPS - gap + 2 * PAD; // bearings a0 .. a0+count-1 must be covered
  const startZone = ring ? 0 : Math.min(EXT, Math.floor((gap - 2 * PAD) / 2)); // may start this many bearings early, and end as many late
  const total = count + 2 * startZone, base = a0 - startZone;

  const cost = (i, j) => {
    const k = at(i, j);
    if (k < 0) return 0.2; // the edge of the mapped area
    if (j < lo[i] || j > outer[i]) return BIG;
    const f = fuel[k];
    if (!f) return 0;
    if (f === FUEL.URB) return BIG;
    const lead = forecast.arrival[k] - now;
    if (lead < leadMin) return BIG;
    const hurry = lead === Infinity ? 0 : (0.3 * leadMin) / lead;
    return (barrier[k] >= 3 ? 0.25 : barrier[k] ? 0.5 : EFFORT[f] || 1) + hurry;
  };
  const anchored = (i, j) => {
    const k = at(i, j);
    return k < 0 || !fuel[k] || fire.arrival[k] <= now;
  };

  const best = new Float32Array(total * rings).fill(BIG), prev = new Int16Array(total * rings).fill(-1);
  const REACH = 2;
  for (let s = 0; s < total; s++) {
    const i = (((base + s) % STEPS) + STEPS) % STEPS, ip = (i - 1 + STEPS) % STEPS;
    for (let j = 0; j < rings; j++) {
      const w = cost(i, j);
      if (w >= BIG) continue;
      let b = BIG, from = -1;
      if (s <= startZone) b = anchored(i, j) ? 0 : UNANCHORED; // the line may begin here
      if (s > 0) {
        for (let d = -REACH; d <= REACH; d++) {
          const q = j + d;
          if (q < 0 || q >= rings) continue;
          const c = best[(s - 1) * rings + q];
          if (c >= BIG) continue;
          // length of the step in cells: along the bearing change and across it
          const len = Math.max(0.35, Math.hypot(d, (j + 0.5) * dTheta));
          const v = c + len * 0.5 * (w + cost(ip, q));
          if (v < b) { b = v; from = q; }
        }
      }
      if (b < BIG) { best[s * rings + j] = b + w * 0.15; prev[s * rings + j] = from; }
    }
  }
  let end = -1, endCost = BIG;
  for (let s = total - 1 - startZone; s < total; s++) {
    if (s < 0) continue;
    const i = (((base + s) % STEPS) + STEPS) % STEPS;
    for (let j = 0; j < rings; j++) {
      const c = best[s * rings + j];
      if (c >= BIG) continue;
      const v = c + (ring || anchored(i, j) ? 0 : UNANCHORED);
      if (v < endCost) { endCost = v; end = s * rings + j; }
    }
  }
  if (end < 0) return { feasible: false, reason: "No line can be finished before the fire arrives", cells: [] };

  // walk back, then lay the line on the grid cell by cell
  const nodes = [];
  for (let s = Math.floor(end / rings), j = end % rings; s >= 0 && j >= 0; ) {
    nodes.push([(((base + s) % STEPS) + STEPS) % STEPS, j]);
    const from = prev[s * rings + j];
    if (from < 0) break;
    j = from;
    s--;
  }
  nodes.reverse();
  if (ring) nodes.push(nodes[0]);
  const seen = new Set(), cells = [];
  const radius = new Float32Array(STEPS).fill(-1);
  let last = -1;
  for (const [i, j] of nodes) {
    radius[i] = (j + 0.5) * cell;
    const k = at(i, j);
    if (k < 0) {
      last = -1;
      continue;
    }
    const run = last < 0 ? [k] : lineCells(n, last % n, Math.floor(last / n), k % n, Math.floor(k / n));
    for (const c of run) if (!seen.has(c)) { seen.add(c); cells.push(c); }
    last = k;
  }
  const kinds = new Uint8Array(cells.length), deadline = new Float32Array(cells.length);
  let build = 0, road = 0, natural = 0;
  cells.forEach((k, idx) => {
    const kind = !fuel[k] || fire.arrival[k] <= now ? KIND.natural : barrier[k] ? KIND.road : KIND.build;
    kinds[idx] = kind;
    deadline[idx] = forecast.arrival[k];
    if (kind === KIND.build) build++;
    else if (kind === KIND.road) road++;
    else natural++;
  });
  // the strip to burn off on the fire's side of each cell of line: the next cells outward from the town
  const band = cells.map((k) => {
    const c = k % n, r = Math.floor(k / n);
    const dx = (c + 0.5) * cell - half - town.cx, dy = (r + 0.5) * cell - half - town.cy, d = Math.hypot(dx, dy) || 1;
    const out = [];
    for (let s = 1; s <= burnDepth + 1 && out.length < burnDepth; s += 0.5) {
      const cc = Math.round(c + (dx / d) * s), rr = Math.round(r + (dy / d) * s);
      if (cc < 0 || rr < 0 || cc >= n || rr >= n) break;
      const q = rr * n + cc;
      if (q !== k && !seen.has(q) && !out.includes(q) && fuel[q] && fuel[q] !== FUEL.URB) out.push(q);
    }
    return out;
  });
  const [i0, j0] = nodes[0], [i1, j1] = nodes[nodes.length - 1];
  return {
    feasible: true, cells, kinds, deadline, radius, ring, band,
    km: cells.length * cell, buildKm: build * cell, roadKm: road * cell, naturalKm: natural * cell,
    anchored: [ring || anchored(i0, j0), ring || anchored(i1, j1)],
    firstArrival: Math.min(...Array.from(deadline)),
    /** Is a point on the town's side of the line? Outside the line's arc: unknown, taken as no. */
    townSide(x, y) {
      const dx = x - town.cx, dy = y - town.cy;
      const i = Math.round((((Math.atan2(dx, dy) + 2 * Math.PI) % (2 * Math.PI)) / dTheta)) % STEPS;
      return radius[i] >= 0 && Math.hypot(dx, dy) < radius[i] - cell * 0.75;
    },
  };
}
