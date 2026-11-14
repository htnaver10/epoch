// Fire growth on the theatre grid. Each cell records when the fire reached it; the fire moves
// from a burning cell to its sixteen neighbours at the rate the FBP ellipse gives for that
// direction, fuel and slope (a minimum-travel-time spread, the approach of Prometheus and
// Cell2Fire). Because weather and suppression change under a running fire, arrival times that
// are still in the future are re-derived, with credit for progress made: everywhere when the
// hour's weather changes, and round any ground that suppression has just changed.
import { FUEL, behaviour, crownFraction, ellipseNormalRate, ellipseRate, intensity, slopeFactor } from "./fbp.js";
import { LIMIT } from "./doctrine.js";

const FUEL_URBAN = FUEL.URB;

export const UNBURNED = Infinity;
// king moves, then knight moves: [dx, dy]
export const MOVES = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1], [2, 1], [1, 2], [-1, 2], [-2, 1], [-2, -1], [-1, -2], [1, -2], [2, -1]];
const DIST = MOVES.map(([x, y]) => Math.hypot(x, y));
const BEARING = MOVES.map(([x, y]) => Math.atan2(x, y)); // clockwise from north

// What a fire has to carry to cross each kind of break (kW/m). Linear breaks: tracks and
// streams, local roads, main roads, highways with their cleared right of way, rivers.
export const BARRIER_HOLD = [0, 350, 800, 1500, 3000, 6000];
export const BARRIER = { none: 0, track: 1, local: 2, main: 3, highway: 4, river: 5 };
export const GUARD_HOLD = [0, LIMIT.handLine, LIMIT.machineGuard]; // hand line, machine guard
export const WET_FACTOR = 0.3; // spread into fuel wetted by a drop or a sprinkler line
const MIN_ROS = 0.08; // m/min: below this the edge is treated as stalled
const CHECK_MIN = 5; // minutes between checks for changed ground and expired drops
const DEG = Math.PI / 180;

class Heap {
  constructor(size = 4096) {
    this.t = new Float32Array(size);
    this.c = new Int32Array(size);
    this.n = 0;
  }
  push(time, cell) {
    if (this.n === this.t.length) {
      const t = new Float32Array(this.n * 2), c = new Int32Array(this.n * 2);
      t.set(this.t);
      c.set(this.c);
      this.t = t;
      this.c = c;
    }
    let i = this.n++;
    const { t, c } = this;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (t[p] <= time) break;
      t[i] = t[p];
      c[i] = c[p];
      i = p;
    }
    t[i] = time;
    c[i] = cell;
  }
  pop() {
    const { t, c } = this;
    const top = c[0], n = --this.n;
    if (n > 0) {
      const time = t[n], cell = c[n];
      let i = 0;
      for (;;) {
        let k = 2 * i + 1;
        if (k >= n) break;
        if (k + 1 < n && t[k + 1] < t[k]) k++;
        if (t[k] >= time) break;
        t[i] = t[k];
        c[i] = c[k];
        i = k;
      }
      t[i] = time;
      c[i] = cell;
    }
    return top;
  }
  clone() {
    const h = new Heap(this.t.length);
    h.t.set(this.t);
    h.c.set(this.c);
    h.n = this.n;
    return h;
  }
}

/** Seeded generator (mulberry32), so every run of a scenario is repeatable. */
export function rng(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  next.state = () => a;
  next.restore = (s) => (a = s >>> 0);
  return next;
}

/**
 * @param grid    { n, cell (km), fuel: Uint8Array, barrier: Uint8Array, elev: Float32Array (m) }
 * @param weather (minutes) => { ws, wd, ffmc, bui }   wd is the direction the wind blows from
 */
export class FireSim {
  constructor(grid, weather, { seed = 1, spotting = true } = {}) {
    const size = grid.n * grid.n;
    this.grid = grid;
    this.n = grid.n;
    this.cellM = grid.cell * 1000;
    this.weather = weather;
    this.spotting = spotting;
    this.random = rng(seed);
    this.t = 0;
    this.arrival = new Float32Array(size).fill(UNBURNED);
    this.tent = new Float32Array(size).fill(UNBURNED); // pending arrival
    this.tset = new Float32Array(size); // when that pending arrival was derived
    this.prog = new Float32Array(size); // share of the way the fire has already come
    this.pros = new Float32Array(size); // rate of the pending arrival
    this.psrc = new Int32Array(size).fill(-1);
    this.ros = new Float32Array(size); // rate at which the front crossed the cell (m/min)
    this.hfi = new Float32Array(size); // intensity when it burned, or potential at the edge (kW/m)
    this.src = new Int32Array(size).fill(-1); // the cell the fire came from
    this.owner = new Uint16Array(size); // fire number
    this.retard = new Uint8Array(size);
    this.guard = new Uint8Array(size);
    this.wet = new Float32Array(size); // wetted until (minutes)
    this.knock = new Float32Array(size); // burning edge knocked down until (minutes)
    this.held = new Uint8Array(size); // edge mopped up for good
    this.heap = new Heap();
    this.front = []; // burned cells that still have fuel to burn beside them
    this.inFront = new Uint8Array(size);
    this.touched = []; // cells whose state suppression has changed since the last check
    this.expiry = []; // [time, cell]: drops that wear off
    this.stamp = new Int32Array(size);
    this.epoch = 0;
    this.burned = 0;
    this.fires = 0;
    this.stale = true; // every pending arrival needs re-deriving
    this.refreshHour = -1;
    this.table = null;
    this.tableKey = -1;
  }

  clone({ weather = this.weather, seed, spotting = this.spotting } = {}) {
    const c = Object.create(FireSim.prototype);
    Object.assign(c, this, { weather, spotting });
    for (const k of ["arrival", "tent", "tset", "prog", "pros", "psrc", "ros", "hfi", "src", "owner", "retard", "guard", "wet", "knock", "held", "inFront", "stamp"]) c[k] = this[k].slice();
    c.heap = this.heap.clone();
    c.front = this.front.slice();
    c.touched = this.touched.slice();
    c.expiry = this.expiry.slice();
    c.random = rng(seed ?? this.random.state());
    c.table = null;
    c.tableKey = -1;
    return c;
  }

  /** Per fuel and direction, the rate of spread for the weather of this hour. */
  tableAt(t) {
    const key = Math.floor(t / 60);
    if (key === this.tableKey && this.table) return this.table;
    const wx = this.weather(t);
    const b = behaviour(wx);
    const toward = (wx.wd + 180) * DEG;
    // rate: how fast the fire travels that way. front: what share of that is the front's own
    // advance along its normal, which is what its intensity depends on.
    const rate = new Float32Array(b.fuels.length * 16), front = new Float32Array(b.fuels.length * 16);
    b.fuels.forEach((f, i) => {
      for (let d = 0; d < 16; d++) {
        const r = ellipseRate(f.head, f.back, f.lb, BEARING[d] - toward);
        rate[i * 16 + d] = r;
        front[i * 16 + d] = r > 0 ? ellipseNormalRate(f.head, f.back, f.lb, BEARING[d] - toward) / r : 0;
      }
    });
    this.tableKey = key;
    return (this.table = { wx, fuels: b.fuels, rate, front, toward });
  }

  solid(i) {
    return this.grid.fuel[i] === 0 || this.grid.barrier[i] !== 0 || this.guard[i] !== 0 || this.retard[i] !== 0;
  }

  /** Start a fire in a cell. Returns the fire number, or 0 if the cell cannot burn. */
  ignite(cell, { fire = 0, at = this.t } = {}) {
    if (this.arrival[cell] !== UNBURNED || !this.grid.fuel[cell]) return 0;
    const id = fire || ++this.fires;
    this.owner[cell] = id;
    this.burn(cell, at, 0.5, -1);
    return id;
  }

  /** Mark ground that burned before the scenario opened. `live` cells keep spreading. */
  seed(cells, { fire = 0, live = true, at = 0 } = {}) {
    const id = fire || ++this.fires;
    for (const cell of cells) {
      if (this.arrival[cell] !== UNBURNED || !this.grid.fuel[cell]) continue;
      this.arrival[cell] = at;
      this.owner[cell] = id;
      this.burned++;
      if (live) { this.front.push(cell); this.inFront[cell] = 1; }
      else this.held[cell] = 1;
    }
    this.stale = true;
    return id;
  }

  burn(cell, t, ros, from) {
    const tb = this.tableAt(t);
    this.arrival[cell] = t;
    this.ros[cell] = ros;
    this.src[cell] = from;
    const fb = tb.fuels[this.grid.fuel[cell]];
    this.hfi[cell] = intensity(fb, ros);
    if (from >= 0) this.owner[cell] = this.owner[from];
    this.burned++;
    this.front.push(cell);
    this.inFront[cell] = 1;
    this.relax(cell, t, tb);
    if (this.spotting && tb.wx.ws >= 15) {
      const cf = crownFraction(fb, ros);
      if (cf > 0.5 && this.random() < 0.012 * cf * Math.min(2, tb.wx.ws / 25)) this.spot(cell, t, tb);
    }
  }

  /** An ember carried downwind starts a new fire ahead of the front. */
  spot(cell, t, tb) {
    const n = this.n, random = this.random;
    const reach = 2 + Math.min(12, -Math.log(1 - random()) * 3); // cells: mostly a few hundred metres, rarely 2 km
    const bearing = tb.toward + (random() - 0.5) * 0.6;
    const col = Math.round((cell % n) + Math.sin(bearing) * reach), row = Math.round(Math.floor(cell / n) + Math.cos(bearing) * reach);
    if (col < 0 || row < 0 || col >= n || row >= n) return;
    const v = row * n + col;
    if (this.arrival[v] !== UNBURNED || !this.grid.fuel[v] || this.retard[v] || this.guard[v] || this.wet[v] > t) return;
    const when = Math.fround(t + 6 + random() * 14);
    if (when < this.tent[v]) {
      this.tent[v] = when;
      this.tset[v] = t;
      this.pros[v] = 0.5;
      this.psrc[v] = cell;
      this.heap.push(when, v);
    }
  }

  /**
   * Offer the fire in `u` to its neighbours. Returns how many neighbours could still burn,
   * whether or not the fire can reach them now: zero means the cell has left the perimeter.
   */
  relax(u, t, tb = this.tableAt(t)) {
    const n = this.n, { fuel, barrier, elev } = this.grid;
    const col = u % n, row = (u - col) / n, fu = fuel[u];
    const stopped = this.held[u] || this.knock[u] > t;
    let live = 0, potential = 0;
    const moves = this.moves || 16;
    for (let d = 0; d < moves; d++) {
      const dx = MOVES[d][0], dy = MOVES[d][1], c = col + dx, r = row + dy;
      if (c < 0 || r < 0 || c >= n || r >= n) continue;
      const v = r * n + c, fv = fuel[v];
      if (!fv || this.arrival[v] !== UNBURNED) continue;
      if (d >= 4) {
        // a diagonal or knight move passes between two cells: it is shut if both are breaks
        const sx = Math.sign(dx), sy = Math.sign(dy);
        const a = d < 8 ? row * n + col + sx : Math.abs(dx) === 2 ? row * n + col + sx : (row + sy) * n + col;
        const b = d < 8 ? (row + sy) * n + col : (row + sy) * n + col + sx;
        if (this.solid(a) && this.solid(b)) continue;
      }
      live++;
      if (stopped) continue;
      const ra = tb.rate[fu * 16 + d], rb = tb.rate[fv * 16 + d];
      if (!(ra > 0) || !(rb > 0)) continue;
      const run = DIST[d] * this.cellM;
      const rise = elev[v] - elev[u];
      let ros = (2 * ra * rb) / (ra + rb);
      if (rise > 2 || rise < -2) ros *= slopeFactor(rise / run);
      if (this.wet[v] > t) ros *= WET_FACTOR;
      const normal = ros * tb.front[fv * 16 + d];
      const hfi = intensity(tb.fuels[fv], normal);
      if (hfi > potential) potential = hfi;
      if (barrier[v] && hfi < BARRIER_HOLD[barrier[v]]) continue;
      if (this.guard[v] || this.retard[v]) {
        // a guard, a retardant line, or a guard coated with retardant
        const hold = this.retard[v] ? (this.guard[v] === 2 ? LIMIT.coatedGuard : LIMIT.retardant) : GUARD_HOLD[this.guard[v]];
        if (hfi < hold) continue;
        ros *= 0.5;
      }
      if (ros < MIN_ROS) continue;
      const when = Math.fround(t + (run / ros) * (1 - this.prog[v]));
      if (when < this.tent[v]) {
        this.tent[v] = when;
        this.tset[v] = t;
        this.pros[v] = normal;
        this.psrc[v] = u;
        this.heap.push(when, v);
      }
    }
    if (live && !stopped) this.hfi[u] = potential; // on the perimeter: what the edge could do now
    return live;
  }

  /** Credit a pending cell with the progress the fire has made towards it, and withdraw its arrival. */
  withdraw(v, t) {
    const span = this.tent[v] - this.tset[v];
    const f = span > 0 ? Math.max(0, Math.min(0.95, (t - this.tset[v]) / span)) : 0;
    this.prog[v] += (1 - this.prog[v]) * f;
    this.tent[v] = UNBURNED;
  }

  /** Re-derive every pending arrival under the conditions that hold now. */
  refresh(t) {
    const { heap, tent, arrival } = this;
    for (let i = 0; i < heap.n; i++) {
      const v = heap.c[i];
      if (arrival[v] === UNBURNED && tent[v] === heap.t[i]) this.withdraw(v, t);
    }
    heap.n = 0;
    const tb = this.tableAt(t);
    const front = this.front;
    let keep = 0, active = 0;
    for (let i = 0; i < front.length; i++) {
      const u = front[i];
      const before = heap.n;
      if (this.relax(u, t, tb) > 0) {
        front[keep++] = u;
        if (heap.n > before) active++;
      } else this.inFront[u] = 0;
    }
    front.length = keep;
    this.active = active;
    this.refreshHour = Math.floor(t / 60);
    this.stale = false;
    this.touched.length = 0;
  }

  /** The same, but only round cells whose state has changed: a drop, a guard, a mopped-up edge. */
  refreshLocal(cells, t) {
    const { n, arrival, tent, inFront, stamp } = this, tb = this.tableAt(t), epoch = ++this.epoch;
    // pending arrivals within a move of the change are withdrawn...
    for (const d of cells) {
      const c0 = d % n, r0 = (d - c0) / n;
      for (let r = Math.max(0, r0 - 2); r <= Math.min(n - 1, r0 + 2); r++)
        for (let c = Math.max(0, c0 - 2); c <= Math.min(n - 1, c0 + 2); c++) {
          const v = r * n + c;
          if (arrival[v] === UNBURNED && tent[v] !== UNBURNED) this.withdraw(v, t);
        }
    }
    // ...and every perimeter cell that could reach them offers its fire again
    for (const d of cells) {
      const c0 = d % n, r0 = (d - c0) / n;
      for (let r = Math.max(0, r0 - 4); r <= Math.min(n - 1, r0 + 4); r++)
        for (let c = Math.max(0, c0 - 4); c <= Math.min(n - 1, c0 + 4); c++) {
          const u = r * n + c;
          if (!inFront[u] || stamp[u] === epoch) continue;
          stamp[u] = epoch;
          this.relax(u, t, tb);
        }
    }
  }

  /** Bring pending arrivals up to date with the weather and with what suppression has done. */
  service(t) {
    if (this.expiry.length) {
      let keep = 0;
      for (const e of this.expiry) {
        if (e[0] <= t) this.touched.push(e[1]);
        else this.expiry[keep++] = e;
      }
      this.expiry.length = keep;
    }
    if (this.stale || Math.floor(t / 60) !== this.refreshHour) this.refresh(t);
    else if (this.touched.length) {
      this.refreshLocal(this.touched, t);
      this.touched.length = 0;
    }
  }

  /** Run the fire forward to time `to` (minutes). */
  advance(to) {
    const { heap, tent, arrival } = this;
    for (;;) {
      this.service(this.t);
      if (this.t >= to) break;
      const stop = Math.min(to, (Math.floor(this.t / CHECK_MIN + 1e-9) + 1) * CHECK_MIN);
      while (heap.n && heap.t[0] <= stop) {
        const when = heap.t[0], v = heap.pop();
        if (arrival[v] !== UNBURNED || tent[v] !== when) continue; // superseded
        this.burn(v, when, this.pros[v], this.psrc[v]);
      }
      this.t = stop;
    }
  }

  // ---- suppression: each returns how many cells it changed

  /** Long-term retardant on unburned fuel. */
  layRetardant(cells) {
    let k = 0;
    for (const c of cells) if (this.arrival[c] === UNBURNED && this.grid.fuel[c] && !this.retard[c]) { this.retard[c] = 1; this.touched.push(c); k++; }
    return k;
  }

  /** A hand line (1) or machine guard (2) through unburned fuel. */
  buildGuard(cells, kind = 2) {
    let k = 0;
    for (const c of cells) if (this.arrival[c] === UNBURNED && this.guard[c] < kind) { this.guard[c] = kind; this.touched.push(c); k++; }
    return k;
  }

  /**
   * Water or foam. On a burning edge it knocks the fire down for `minutes` if the edge is
   * below `limit` kW/m; on unburned fuel it wets it. Without a crew to follow up, a
   * knocked-down edge comes back.
   */
  dropWater(cells, { minutes = 45, limit = LIMIT.bucket } = {}) {
    let k = 0;
    const until = this.t + minutes;
    for (const c of cells) {
      if (this.arrival[c] === UNBURNED) k += this.wetUntil(c, until);
      else if (!this.held[c] && this.hfi[c] <= limit && this.knock[c] < until) {
        this.knock[c] = until;
        this.touched.push(c);
        this.expiry.push([until, c]);
        k++;
      }
    }
    return k;
  }

  /** Keep a cell of unburned fuel wet until a time: a drop, a sprinkler line, an engine's hose. */
  wetUntil(cell, until) {
    if (this.arrival[cell] !== UNBURNED || !this.grid.fuel[cell] || this.wet[cell] >= until) return 0;
    const was = this.wet[cell] > this.t;
    this.wet[cell] = until;
    if (!was) this.touched.push(cell); // already wet: nothing changes now
    this.expiry.push([until, cell]);
    return 1;
  }

  /** Put a burning cell out for good (an engine on a spot fire in town). */
  hold(cell) {
    if (this.arrival[cell] === UNBURNED || this.held[cell]) return 0;
    this.held[cell] = 1;
    this.touched.push(cell);
    return 1;
  }

  /** A crew works a section of edge until it is out: only below `limit` kW/m, or once knocked down. */
  mopUp(cells, limit = LIMIT.crew) {
    let k = 0;
    for (const c of cells) {
      if (this.arrival[c] === UNBURNED || this.held[c]) continue;
      if (this.hfi[c] <= limit || this.knock[c] > this.t) { this.held[c] = 1; this.touched.push(c); k++; }
    }
    return k;
  }

  /**
   * Planned ignition from a finished line: the fuel between the line and the fire is burned
   * off under control, so the main fire arrives at ground that is already black.
   */
  burnOut(cells) {
    let k = 0;
    for (const c of cells) {
      if (this.arrival[c] !== UNBURNED || !this.grid.fuel[c] || this.grid.fuel[c] === FUEL_URBAN) continue;
      this.arrival[c] = this.t;
      this.ros[c] = 0.5;
      this.hfi[c] = 300;
      this.held[c] = 1;
      this.burnedOut = (this.burnedOut || 0) + 1;
      this.touched.push(c);
      k++;
    }
    return k;
  }

  /** Burning or pending cells that are on the perimeter and still able to spread. */
  perimeter() {
    return this.front.filter((u) => !this.held[u]);
  }
}

/** Cells on a straight line between two cells, four-connected so that a fire cannot slip through. */
export function lineCells(n, c0, r0, c1, r1) {
  let c = Math.round(c0), r = Math.round(r0);
  const dc = Math.abs(Math.round(c1) - c), dr = Math.abs(Math.round(r1) - r);
  const sc = c1 > c0 ? 1 : -1, sr = r1 > r0 ? 1 : -1;
  const out = [];
  const put = () => c >= 0 && r >= 0 && c < n && r < n && out.push(r * n + c);
  put();
  for (let ic = 0, ir = 0; ic < dc || ir < dr; ) {
    // one axis at a time, whichever keeps closer to the line
    if ((ic + 0.5) / dc < (ir + 0.5) / dr) { c += sc; ic++; }
    else { r += sr; ir++; }
    put();
  }
  return out;
}
