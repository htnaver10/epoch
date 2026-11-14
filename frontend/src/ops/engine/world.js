// The running world: the fire, the clock, every unit, and the operations under way. One
// step moves units along their legs, applies what they did to the fire, grows the fire, and
// lets idle units decide what to do next. The planner runs copies of this world forward to
// see how a course of action turns out, so everything here is deterministic for a seed.
import { FUEL } from "./fbp.js";
import { FireSim, UNBURNED, rng } from "./fire.js";
import { SPECS, advanceUnit, insertTurn, orbit, positionAt, racetrack, schedule } from "./units.js";
import { downstream, project, threat } from "./forecast.js";
import { KIND } from "./line.js";
import { DROP, LIMIT } from "./doctrine.js";

const STEP = 1; // minutes
const PACKAGE = 2; // cells of line per heavy tanker load (320 m)
const BURN_RATE = 0.8; // km of line burned off per crew hour. Assumption.
const CLAIM_MIN = 12;
const SCOOP_KM = 1.34, SCOOP_SPEED = 130;
const WET_SHARE = 0.3; // intensity of a fire in wetted fuel against the same fire in dry fuel (its spread factor)
const clampHdg = (a) => ((a % 360) + 360) % 360;

export class World {
  constructor({ theatre, weather, units = [], seed = 1, clock = 12 * 60, daylight = [5 * 60 + 30, 21 * 60 + 30] }) {
    this.theatre = theatre;
    this.weather = weather;
    this.units = units;
    this.seed = seed;
    this.clock = clock; // local minutes after midnight at t = 0
    this.daylight = daylight;
    this.fire = new FireSim(theatre.grid, weather, { seed });
    this.t = 0;
    this.ops = new Map();
    this.log = [];
    this.soak = new Float32Array(theatre.n * theatre.n);
    this.claims = new Map(); // cell -> [litres on the way, until]: water pledged by aircraft inbound
    this.crewClaims = new Map(); // cell -> until: edge a crew is on its way to
    this.random = rng(seed * 7919 + 13);
    this.quiet = false; // a scoring copy: straight flight tracks, forecast renewed as it runs
    this.silent = false; // any copy: no idle traffic, no log, no trails
    this.forecast = null;
    this.forecastAt = -Infinity;
    this.forecastHours = 36;
    this.threat = null;
    this.changes = 0;
    this.scoops = scoopRuns(theatre);
    this.runways = new Map(); // base -> when its runway is next free for a departure
  }

  /** A copy to run forward in planning. */
  /**
   * A copy to run forward in planning. A `preview` copy flies exactly as the live world
   * would (same tracks, same takeoffs), so that what it does can be shown as the plan.
   */
  clone({ seed = this.seed + 1, weather = this.weather, preview = false } = {}) {
    const w = Object.create(World.prototype);
    Object.assign(w, this, { weather, seed, quiet: !preview, silent: true, log: [], forecastHours: preview ? this.forecastHours : 6 });
    w.fire = this.fire.clone({ weather, seed: seed * 31 + 7 });
    w.units = this.units.map((u) => ({ ...u, legs: u.legs.slice(), work: u.work ? { ...u.work } : null, trail: [] }));
    w.ops = new Map([...this.ops].map(([k, op]) => [k, cloneOp(op)]));
    w.soak = this.soak.slice();
    w.claims = new Map([...this.claims].map(([k, v]) => [k, v.slice()]));
    w.crewClaims = new Map(this.crewClaims);
    w.runways = new Map(this.runways);
    w.random = rng(seed * 7919 + 13);
    return w;
  }

  localMinute(t = this.t) {
    return (((this.clock + t) % 1440) + 1440) % 1440;
  }
  /** The local calendar date, as YYYY-MM-DD. */
  dateText(t = this.t) {
    if (!this.opens) return this.theatre.date;
    return new Date(this.opens + t * 60000 + (this.theatre.utcOffset || 0) * 3600e3).toISOString().slice(0, 10);
  }
  isDay(t = this.t) {
    const m = this.localMinute(t);
    return m >= this.daylight[0] && m <= this.daylight[1];
  }
  say(call, text, tone = "") {
    if (!this.silent) this.log.push({ t: this.t, call, text, tone });
  }

  // ---- the forecast, shared by the display and the tactics

  /**
   * The current forecast. A planning copy renews it every hour and a half as it runs; the
   * live world keeps the one it has until told to renew (the page does that in slices).
   */
  getForecast(force = false) {
    if (force || !this.forecast || (this.quiet && this.t - this.forecastAt >= 100)) this.setForecast(project(this.fire, this.forecastHours));
    return this.forecast;
  }
  setForecast(forecast) {
    this.forecast = forecast;
    this.forecastAt = forecast.from;
    this.threat = threat(forecast, this.theatre.isTown);
    this.flow = downstream(forecast, this.theatre.isTown);
  }

  // ---- stepping

  step(minutes) {
    const end = this.t + minutes;
    while (this.t < end - 1e-9) {
      const to = Math.min(end, this.t + STEP), dt = to - this.t;
      const events = [];
      for (const u of this.units) {
        const flying = u.alt > 0;
        advanceUnit(u, to, events);
        if (flying && u.cls === "air") u.fuel -= dt;
      }
      events.sort((a, b) => a.at - b.at);
      this.t = to;
      for (const e of events) this.handle(e);
      if (!this.quiet) this.separate();
      this.work(dt);
      this.fire.advance(to);
      for (const u of this.units) if (!u.legs.length && !u.work) this.think(u);
      if (!this.silent) this.trails();
      if ((Math.floor(to) % 30 === 0 && Math.floor(to) !== Math.floor(to - dt)) || this.opsDirty) this.review();
    }
  }

  /**
   * Every aircraft keeps its own lookout. It looks half a minute down its track and down
   * everyone else's; if two will come within a kilometre at about the same height, the one
   * with less right of way turns a full circle where it is and carries on behind the other.
   * An aircraft on a drop, a scoop or final approach holds its course.
   */
  separate() {
    const air = this.units.filter((u) => u.cls === "air" && u.alt > 20 && u.legs.length);
    if (air.length < 2) return;
    const t = this.t, rank = (u) => {
      const leg = u.legs[u.li];
      if (!leg || leg.kind === "giveway") return 5; // already turning: leave it alone
      if (leg.kind === "drop" || leg.kind === "scoop" || leg.status === "FINAL") return 3;
      return leg.kind === "orbit" ? 0 : 1;
    };
    for (let i = 0; i < air.length; i++)
      for (let j = i + 1; j < air.length; j++) {
        const a = air[i], b = air[j];
        if (Math.abs(a.x - b.x) > 14 || Math.abs(a.y - b.y) > 14) continue;
        const near = SPECS[a.type].heli && SPECS[b.type].heli ? 0.45 : 0.95;
        let close = false;
        for (const ahead of [0, 0.17, 0.34, 0.5]) {
          const p = ahead ? positionAt(a, t + ahead) : [a.x, a.y, a.alt], q = ahead ? positionAt(b, t + ahead) : [b.x, b.y, b.alt];
          if (Math.hypot(p[0] - q[0], p[1] - q[1]) < near && Math.abs(p[2] - q[2]) < 130) { close = true; break; }
        }
        if (!close) continue;
        let ra = rank(a), rb = rank(b);
        if (ra === 5 || rb === 5) continue;
        // a helicopter always gives way to a fixed-wing aircraft: it can slow, turn and wait where a bomber cannot
        const ha = Boolean(SPECS[a.type].heli), hb = Boolean(SPECS[b.type].heli);
        if (ha !== hb) {
          ra = ha ? -1 : 9;
          rb = hb ? -1 : 9;
        }
        // the one with less right of way gives way; between equals, the one with further still to go
        const remain = (u) => u.legs[u.legs.length - 1].t1;
        const yields = ra !== rb ? (ra < rb ? a : b) : remain(a) >= remain(b) ? a : b, other = yields === a ? b : a;
        if (yields.gaveWay && t - yields.gaveWay < 1.5) continue;
        if (insertTurn(yields, t, `GIVING WAY · ${other.call}`)) {
          yields.gaveWay = t;
          yields.status = `GIVING WAY · ${other.call}`;
        }
      }
  }

  /** Where each aircraft has just been: a point every few hundred metres, the last forty kept. */
  trails() {
    for (const u of this.units) {
      if (u.cls !== "air" || u.alt <= 0) continue;
      const last = u.trail[u.trail.length - 1];
      if (last && Math.hypot(u.x - last[0], u.y - last[1]) <= 0.35) continue;
      u.trail.push([u.x, u.y, u.alt]);
      if (u.trail.length > 40) u.trail.shift();
    }
  }

  // ---- operations

  /** Light a new fire at a point: the nearest ground that can burn. Returns its number, or 0. */
  igniteAt(x, y) {
    const cell = this.theatre.nearestFuel(x, y);
    if (cell < 0) return 0;
    const id = this.fire.ignite(cell);
    if (id) this.say("DETECT", `New start reported: fire ${fireName(id)}`, "warn");
    return id;
  }

  /** The fire burning at or right beside a point, if any. */
  fireAt(x, y) {
    const { n } = this.theatre, k = this.theatre.toCell(x, y), c0 = k % n, r0 = Math.floor(k / n);
    for (let ring = 0; ring <= 3; ring++)
      for (let r = r0 - ring; r <= r0 + ring; r++)
        for (let c = c0 - ring; c <= c0 + ring; c++) {
          if (r < 0 || c < 0 || r >= n || c >= n) continue;
          const i = r * n + c;
          if (this.fire.arrival[i] !== UNBURNED && this.fire.owner[i]) return this.fire.owner[i];
        }
    return 0;
  }

  /** Commit units to a course of action against a fire. */
  commit(coa) {
    const op = {
      fire: coa.fire, coa: coa.key, name: coa.name, since: this.t, status: "ACTIVE", line: coa.line || null, segments: (coa.segments || []).map((s) => ({ ...s, by: 0, set: 0 })),
      ret: coa.line ? new Uint8Array(coa.line.cells.length) : null, guard: coa.line ? new Uint8Array(coa.line.cells.length) : null, burn: coa.line ? new Uint8Array(coa.line.cells.length) : null,
      stats: { retardant: 0, water: 0, drops: 0, guardKm: 0, burnKm: 0, mopKm: 0, sprinklerKm: 0, aheadKm: 0 }, quietSince: null,
    };
    if (op.line) op.line.kinds.forEach((k, i) => k === KIND.natural && ((op.ret[i] = 2), (op.guard[i] = 2), (op.burn[i] = 2)));
    this.ops.set(coa.fire, op);
    for (const id of coa.units) {
      const u = this.units.find((x) => x.id === id);
      if (!u || (u.op && u.op !== coa.fire)) continue;
      u.op = coa.fire;
      u.state = "committed";
      u.work = null;
      // whatever it was doing, it breaks off where it is
      if (u.legs.length) { u.legs = []; u.li = 0; }
      if (u.cls === "air" && u.alt <= 0) schedule(u, this.t, [{ hold: u.role === "helo" ? 4 : 6, status: "SCRAMBLE", kind: "ground" }]);
    }
    this.say("PLAN", `${coa.name} started on fire ${fireName(coa.fire)}: ${coa.units.length} crews and aircraft`, "accent");
    this.getForecast(true);
    return op;
  }

  standDown(fireId, reason = "stood down") {
    const op = this.ops.get(fireId);
    if (!op || op.status !== "ACTIVE") return;
    op.status = reason === "held" ? "HELD" : "STOOD DOWN";
    for (const u of this.units) if (u.op === fireId) this.release(u);
    this.say("PLAN", `Fire ${fireName(fireId)} ${reason}: crews and aircraft released`, reason === "held" ? "good" : "");
  }

  release(u) {
    u.op = 0;
    u.work = null;
    u.state = "busy";
    u.legs = [];
    u.li = 0;
    this.returnHome(u, "RTB");
  }

  /** Housekeeping twice an hour: is each fire still spreading, and is anything left to do. */
  review() {
    this.opsDirty = false;
    for (const [id, op] of this.ops) {
      if (op.status !== "ACTIVE") continue;
      let live = 0;
      for (const c of this.fire.front) if (this.fire.owner[c] === id && !this.fire.held[c]) live++;
      const active = live > 0 && this.fire.heap.n > 0 && this.hasPending(id);
      if (!active) {
        op.quietSince ??= this.t;
        if (this.t - op.quietSince >= 60) this.standDown(id, "held");
      } else op.quietSince = null;
    }
    for (const [cell, p] of this.claims) if (p[1] < this.t) this.claims.delete(cell);
    for (const [cell, until] of this.crewClaims) if (until < this.t) this.crewClaims.delete(cell);
  }
  hasPending(id) {
    const { heap, tent, arrival, psrc, owner } = this.fire;
    for (let i = 0; i < heap.n; i++) {
      const v = heap.c[i];
      if (arrival[v] === UNBURNED && tent[v] === heap.t[i] && psrc[v] >= 0 && owner[psrc[v]] === id) return true;
    }
    return false;
  }

  // ---- events raised when a leg completes

  handle(e) {
    const u = e.unit, op = this.ops.get(u.op), spec = SPECS[u.type];
    switch (e.type) {
      case "scoop":
      case "fill":
        u.load = spec.load;
        break;
      case "reload":
        u.load = spec.load;
        u.fuel = spec.enduranceMin;
        u.status = "LOADED";
        break;
      case "refuel":
        u.fuel = spec.enduranceMin;
        break;
      case "landed":
        u.alt = 0;
        u.state = u.op ? "committed" : "idle";
        u.status = u.op ? "AT BASE · HOLD" : "AT BASE";
        if (!u.op) u.nextIdle = this.t + 20 + this.random() * 60;
        break;
      case "home":
        u.state = "idle";
        u.status = u.cls === "air" ? "AT BASE" : "AT BASE";
        u.nextIdle = this.t + 15 + this.random() * 50;
        break;
      case "retardant": {
        if (!op || (!e.cells && !op.line)) break;
        const cells = e.cells || e.idx.map((i) => op.line.cells[i]);
        const laid = this.fire.layRetardant(cells);
        if (e.cells) {
          for (const c of cells) this.claims.delete(c);
          op.stats.aheadKm += laid * this.theatre.cell;
        } else for (const i of e.idx) op.ret[i] = 2;
        op.stats.retardant += u.load;
        op.stats.drops++;
        u.used = true;
        u.delivered += u.load;
        u.sorties++;
        this.say(u.call, `Retardant away: ${fmt(u.load)} L on ${Math.round(cells.length * this.theatre.cell * 1000)} m ${e.cells ? "across the fire's path" : "of line"}${laid < cells.length ? " (part already burned)" : ""}`, "accent");
        u.load = 0;
        break;
      }
      case "water": {
        const c = e.cell, fire = this.fire;
        this.soak[c] += u.load;
        const pledge = this.claims.get(c);
        if (pledge && (pledge[0] -= u.load) <= 0) this.claims.delete(c);
        const burning = fire.arrival[c] !== UNBURNED;
        if (burning) {
          const need = DROP.knockdownBase + DROP.knockdownPerKw * fire.hfi[c];
          if (this.soak[c] >= need && fire.dropWater([c], { limit: spec.limit, minutes: DROP.knockMinutes })) this.soak[c] = 0;
        } else if (this.soak[c] >= DROP.wetLitres) {
          if (fire.dropWater([c], { minutes: DROP.wetMinutes }) && op) op.stats.aheadKm += this.theatre.cell;
          this.soak[c] = 0;
        }
        if (op) { op.stats.water += u.load; op.stats.drops++; }
        u.used = true;
        u.delivered += u.load;
        u.sorties++;
        if (u.sorties % 6 === 1) this.say(u.call, `Drop ${u.sorties}: ${fmt(u.load)} L ${burning ? "on the edge" : "ahead of the fire"}`);
        u.load = 0;
        break;
      }
      case "work":
        u.used = true;
        u.work = { ...e.work, progress: 0, since: this.t };
        u.status = e.work.status;
        break;
      case "status":
        u.status = e.status;
        if (e.idle) { u.state = "idle"; u.nextIdle = this.t + e.idle; }
        break;
      default:
    }
  }

  // ---- continuous work by ground units, a step at a time

  work(dt) {
    const { fire, theatre } = this, cellKm = theatre.cell;
    for (const u of this.units) {
      const w = u.work;
      if (!w) continue;
      const op = this.ops.get(u.op), spec = SPECS[u.type];
      if (!op || op.status !== "ACTIVE") {
        u.work = null;
        continue;
      }
      // pull back if the fire is about to arrive where the unit stands
      if (w.kind !== "structure" && w.kind !== "mop" && this.fireNear(u.x, u.y, 2, LIMIT.crew)) {
        this.say(u.call, "Fire at the line: pulling back", "warn");
        if (w.kind === "guard" || w.kind === "handline") op.guard[w.idx] = 0;
        if (w.kind === "burn") op.burn[w.idx] = 0;
        u.work = null;
        u.retreat = this.t;
        continue;
      }
      switch (w.kind) {
        case "guard":
        case "handline": {
          const rate = w.kind === "guard" ? spec.rate : spec.lineRate;
          w.progress += (rate * dt) / 60 / cellKm;
          while (w.progress >= 1 && u.work) {
            w.progress -= 1;
            const level = w.kind === "guard" ? 2 : 1, cell = op.line.cells[w.idx];
            fire.buildGuard([cell], level);
            op.guard[w.idx] = Math.max(op.guard[w.idx], level === 2 ? 2 : 3); // 3: hand line, a dozer may still improve it
            op.stats.guardKm += cellKm;
            const next = this.nextLineCell(op, w.idx, (i) => op.line.kinds[i] === KIND.build && op.guard[i] === 0);
            if (next < 0) u.work = null;
            else {
              op.guard[next] = 1;
              w.idx = next;
              [u.x, u.y] = theatre.xy(op.line.cells[next]);
            }
          }
          break;
        }
        case "burn": {
          w.progress += (BURN_RATE * dt) / 60 / cellKm;
          while (w.progress >= 1 && u.work) {
            w.progress -= 1;
            fire.burnOut(op.line.band[w.idx]);
            op.burn[w.idx] = 2;
            op.stats.burnKm += cellKm;
            const next = this.nextLineCell(op, w.idx, (i) => this.canBurn(op, i));
            if (next < 0) u.work = null;
            else {
              op.burn[next] = 1;
              w.idx = next;
              [u.x, u.y] = theatre.xy(op.line.cells[next]);
            }
          }
          break;
        }
        case "mop": {
          w.progress += (spec.rate * dt) / 60 / cellKm;
          if (w.progress >= 1) {
            w.progress = 0;
            if (fire.mopUp([w.cell], LIMIT.crew)) op.stats.mopKm += cellKm;
            u.work = null; // look for the next piece of edge
          } else if (fire.hfi[w.cell] > LIMIT.crew && fire.knock[w.cell] <= this.t) {
            u.work = null; // it flared up: not workable
            u.retreat = this.t;
          }
          break;
        }
        case "sprinkler": {
          const seg = op.segments[w.seg];
          w.progress += (spec.rate * dt) / 60 / cellKm;
          while (w.progress >= 1 && seg.set < seg.cells.length) {
            w.progress -= 1;
            const until = this.t + 36 * 60;
            for (const c of [seg.cells[seg.set], ...(seg.ring[seg.set] || [])]) fire.wetUntil(c, until);
            seg.set++;
            op.stats.sprinklerKm += cellKm;
          }
          if (seg.set >= seg.cells.length) {
            u.work = { kind: "structure", cell: seg.cells[seg.cells.length >> 1], status: "SPRINKLERS RUNNING" };
            u.status = "SPRINKLERS RUNNING";
            this.say(u.call, `Sprinkler line running: ${Math.round(seg.cells.length * cellKm * 1000)} m at ${seg.label}`, "good");
          }
          break;
        }
        case "structure": {
          // keep the ground around wet, and put out what lands in reach
          if (Math.floor(this.t) % 10 !== 0) break;
          const n = theatre.n, c0 = w.cell % n, r0 = Math.floor(w.cell / n), until = this.t + 30;
          for (let r = r0 - 3; r <= r0 + 3; r++)
            for (let c = c0 - 3; c <= c0 + 3; c++) {
              if (r < 0 || c < 0 || r >= n || c >= n) continue;
              const i = r * n + c, d = Math.hypot(r - r0, c - c0);
              if (fire.arrival[i] === UNBURNED) {
                if (d <= 2) fire.wetUntil(i, until);
              } else if (d <= 3.2 && fire.hfi[i] <= LIMIT.crew && theatre.grid.fuel[i] === FUEL.URB) fire.hold(i);
            }
          break;
        }
        default:
      }
    }
  }

  canBurn(op, i) {
    // burn off only from a finished piece of line, with fuel on the fire's side, and not too late
    if (op.burn[i] !== 0 || !op.line.band[i].length) return false;
    const done = op.line.kinds[i] === KIND.road || op.ret[i] === 2 || op.guard[i] >= 2;
    return done && op.line.deadline[i] - this.t > 40;
  }
  nextLineCell(op, idx, ok) {
    const n = op.line.cells.length;
    const a = idx + 1 < n && ok(idx + 1) ? idx + 1 : -1, b = idx - 1 >= 0 && ok(idx - 1) ? idx - 1 : -1;
    if (a < 0 || b < 0) return Math.max(a, b);
    return op.line.deadline[a] <= op.line.deadline[b] ? a : b;
  }
  /** Is there fire hotter than `limit` (or any fire, with no limit) within `reach` cells of a point? */
  fireNear(x, y, reach, limit = 0) {
    const { n } = this.theatre, fire = this.fire, k = this.theatre.toCell(x, y), c0 = k % n, r0 = Math.floor(k / n);
    for (let r = r0 - reach; r <= r0 + reach; r++)
      for (let c = c0 - reach; c <= c0 + reach; c++) {
        if (r < 0 || c < 0 || r >= n || c >= n) continue;
        const i = r * n + c;
        if (fire.arrival[i] !== UNBURNED && !fire.held[i] && this.t - fire.arrival[i] < 90 && fire.hfi[i] > limit) return true;
      }
    return false;
  }

  // ---- what an idle or finished unit does next

  think(u) {
    const op = u.op ? this.ops.get(u.op) : null;
    if (u.op && (!op || op.status !== "ACTIVE")) return this.release(u);
    if (!op) {
      if (u.cls === "air" && u.alt > 0) return this.returnHome(u, "RTB");
      if (!this.silent && u.state === "idle" && this.t >= u.nextIdle) this.idle(u);
      return;
    }
    const spec = SPECS[u.type];
    if (spec.cls === "air") {
      if (!this.isDay()) return this.ground(u, "GROUNDED · NIGHT", this.minutesToDawn());
      if (u.fuel < 35 && u.alt > 0) return this.returnHome(u, "RTB · FUEL", true);
      if (u.role === "tanker") return this.tanker(u, op, spec);
      if (u.role === "skimmer" || u.role === "helo") return this.bomber(u, op, spec);
      if (u.role === "lead") return this.lead(u, op, spec);
    }
    if (u.role === "dozer") return this.dozer(u, op);
    if (u.role === "crew") return this.crew(u, op, spec);
    if (u.role === "engine" || u.role === "spu") return this.protect(u, op);
    if (u.role === "medic" || u.role === "police") return this.stage(u, op);
  }

  minutesToDawn() {
    const m = this.localMinute();
    return m > this.daylight[1] ? 1440 - m + this.daylight[0] : Math.max(0, this.daylight[0] - m);
  }
  ground(u, status, minutes) {
    if (u.alt > 0) return this.returnHome(u, status);
    u.status = status;
    schedule(u, this.t, [{ hold: Math.max(5, Math.min(minutes, 60)), status, kind: "ground" }]);
  }

  /** The runway an aircraft uses, flown into the wind: where the roll starts and the way it points. */
  runway(u) {
    const r = u.home.runway || [u.home.x - 0.85, u.home.y + 0.3, u.home.x + 0.85, u.home.y + 0.6];
    let a = [r[0], r[1]], b = [r[2], r[3]];
    const wind = (this.weather(this.t).wd * Math.PI) / 180;
    if ((b[0] - a[0]) * Math.sin(wind) + (b[1] - a[1]) * Math.cos(wind) < 0) [a, b] = [b, a];
    const d = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
    return { a, b, ux: (b[0] - a[0]) / d, uy: (b[1] - a[1]) / d, km: d };
  }

  /**
   * Give an aircraft legs to fly. From the ground a fixed-wing aircraft taxies to the
   * runway, rolls and climbs out straight ahead before turning; a helicopter lifts where it is.
   */
  fly(u, legs) {
    const spec = SPECS[u.type];
    if (u.alt > 0 || this.quiet) return schedule(u, this.t, legs, { round: !this.quiet });
    if (spec.heli) return schedule(u, this.t, [{ hold: 0.5, status: "LIFTING", kind: "climb", alt: [0, 60] }, ...legs]);
    const rw = this.runway(u), roll = Math.min(rw.km, 1.3);
    const off = [rw.a[0] + rw.ux * roll, rw.a[1] + rw.uy * roll];
    // one aircraft on the runway at a time: each departure a minute and a half behind the last
    const taxi = (Math.hypot(rw.a[0] - u.x, rw.a[1] - u.y) / 35) * 60, base = u.home.name;
    const wait = Math.max(0, (this.runways.get(base) || 0) - (this.t + taxi));
    this.runways.set(base, this.t + taxi + wait + 1.5);
    // those waiting stand in a queue back from the threshold, off to one side of the runway
    const place = Math.round(wait / 1.5), short = [rw.a[0] - rw.ux * 0.2 * place + rw.uy * 0.12, rw.a[1] - rw.uy * 0.2 * place - rw.ux * 0.12];
    schedule(u, this.t, [
      ...(wait > 0.05 ? [{ pts: [short], speed: 35, status: "TAXI", kind: "taxi" }, { hold: wait, status: "HOLDING SHORT", kind: "ground" }] : []),
      { pts: [rw.a], speed: 35, status: wait > 0.05 ? "LINING UP" : "TAXI", kind: "taxi" },
      { pts: [off], speed: 150, status: "TAKEOFF ROLL", kind: "taxi" },
      { pts: [[off[0] + rw.ux * 2.4, off[1] + rw.uy * 2.4]], speed: spec.work, status: "CLIMB OUT", kind: "transit", alt: [0, 260] },
      ...legs,
    ]);
  }

  returnHome(u, status = "RTB", refuel = false) {
    const spec = SPECS[u.type], home = [u.home.x, u.home.y];
    if (spec.cls === "air") {
      if (u.alt <= 0 && Math.hypot(u.x - home[0], u.y - home[1]) < 0.3) {
        if (u.op) return schedule(u, this.t, [{ hold: 20, status: "AT BASE · HOLD", kind: "ground" }]);
        u.state = "idle";
        u.status = "AT BASE";
        u.nextIdle = this.t + 30 + this.random() * 60;
        return;
      }
      const legs = [];
      if (this.quiet) legs.push({ pts: [home], speed: spec.cruise, status, kind: "rtb", alt: [u.alt, 0], end: { type: "landed" } });
      else if (spec.heli) legs.push({ pts: [home], speed: spec.cruise, status, kind: "rtb", alt: [u.alt, 60], smooth: true }, { hold: 0.5, status: "LANDING", kind: "climb", alt: [60, 0], end: { type: "landed" } });
      else {
        // onto the extended centreline 5 km out, down the approach, roll out, taxi in
        const rw = this.runway(u), stop = [rw.a[0] + rw.ux * Math.min(rw.km, 1.1), rw.a[1] + rw.uy * Math.min(rw.km, 1.1)];
        legs.push(
          { pts: [[rw.a[0] - rw.ux * 7, rw.a[1] - rw.uy * 7], [rw.a[0] - rw.ux * 4.5, rw.a[1] - rw.uy * 4.5]], speed: spec.cruise, status, kind: "rtb", alt: [u.alt, 280], smooth: true, heading: Math.atan2(rw.ux, rw.uy) },
          { pts: [rw.a], speed: spec.work * 0.85, status: "FINAL", kind: "rtb", alt: [280, 0], mark: "RWY" },
          { pts: [stop], speed: 90, status: "ROLLOUT", kind: "taxi" },
          { pts: [home], speed: 35, status: "TAXI", kind: "taxi", end: { type: "landed" } },
        );
      }
      if (refuel || u.fuel < spec.enduranceMin * 0.5) legs.push({ hold: 25, status: "REFUELLING", kind: "ground", end: { type: "refuel" } });
      return schedule(u, this.t, legs, { round: !this.quiet });
    }
    if (Math.hypot(u.x - home[0], u.y - home[1]) < 0.15) {
      u.state = "idle";
      u.status = "AT BASE";
      u.nextIdle = this.t + 20 + this.random() * 60;
      return;
    }
    const legs = this.drive(u, home, status);
    legs[legs.length - 1].end = { type: "home" };
    schedule(u, this.t, legs);
  }

  /** Legs for a ground unit to reach a point: by road as far as roads go, then across country. */
  drive(u, to, status, air = false) {
    const spec = SPECS[u.type];
    if (air && spec.heliborne) return [{ pts: [to], speed: spec.heliborne, status: `${status} · HELI`, kind: "heli", alt: [u.alt, 0], mark: "LZ" }];
    const route = this.theatre.roads.route([u.x, u.y], to);
    const cross = spec.cross || 4;
    if (!route || route.off[0] > 3) return [{ pts: [to], speed: cross, status, kind: "cross" }];
    const legs = [];
    if (route.off[0] > 0.03) legs.push({ pts: [route.path[0]], speed: route.off[0] < 0.4 ? 20 : cross, status, kind: "cross" });
    if (route.path.length > 1) legs.push({ pts: route.path, speed: (route.km / Math.max(route.min, 0.01)) * 60 * (spec.road / 90), status, kind: "road" });
    if (route.off[1] > 0.03) legs.push({ pts: [to], speed: route.off[1] < 0.4 ? 20 : cross, status, kind: "cross" });
    if (!legs.length) legs.push({ hold: 0.5, status, kind: "ground" });
    return legs;
  }

  // -- land-based airtanker: retardant on the line, most urgent first, then back to reload

  tanker(u, op, spec) {
    const home = [u.home.x, u.home.y];
    if (u.load <= 0) {
      if (u.alt <= 0 && Math.hypot(u.x - home[0], u.y - home[1]) < 0.4) return schedule(u, this.t, [{ hold: spec.reloadMin, status: "RELOADING", kind: "ground", end: { type: "reload" } }]);
      return this.returnHome(u, "RTB · RELOAD");
    }
    const { theatre } = this;
    const pkg = this.retardantPackage(op);
    let cells = null;
    if (!pkg && !op.line) {
      // no control line in this plan: lay retardant across the fire's forecast path instead,
      // two cells a load, where it is due in the next three hours and the retardant will hold
      const first = this.aheadTarget(u, op, spec, { lasts: 170, holds: LIMIT.retardant * WET_SHARE, skip: (v) => this.fire.retard[v] || this.claims.has(v) }).cell;
      if (first >= 0) {
        const forecast = this.forecast, n = theatre.n;
        let mate = -1, near = Infinity;
        for (const w of [first + 1, first - 1, first + n, first - n, first + n + 1, first + n - 1, first - n + 1, first - n - 1]) {
          if (w < 0 || w >= forecast.arrival.length || this.fire.arrival[w] !== UNBURNED || !theatre.grid.fuel[w] || this.fire.retard[w] || this.claims.has(w) || forecast.arrival[w] === UNBURNED) continue;
          // the neighbour the fire reaches at about the same time: along the front, not along its path
          const gap = Math.abs(forecast.arrival[w] - forecast.arrival[first]);
          if (gap < near) { near = gap; mate = w; }
        }
        cells = mate >= 0 ? [first, mate] : [first];
        for (const c of cells) this.claims.set(c, [1e9, this.t + 30]);
      }
    }
    if (!pkg && !cells) {
      if (u.alt > 0) return this.returnHome(u, op.line ? "RTB · LINE COMPLETE" : "RTB · NOTHING TO COAT");
      return schedule(u, this.t, [{ hold: 20, status: "LOADED · HOLD", kind: "ground" }]);
    }
    if (pkg) for (const i of pkg) op.ret[i] = 1;
    const run = cells || [op.line.cells[pkg[0]], op.line.cells[pkg[pkg.length - 1]]];
    let a = theatre.xy(run[0]), b = theatre.xy(run[run.length - 1]);
    if (Math.hypot(b[0] - u.x, b[1] - u.y) < Math.hypot(a[0] - u.x, a[1] - u.y)) [a, b] = [b, a];
    let dx = b[0] - a[0], dy = b[1] - a[1];
    if (!dx && !dy) { dx = a[0] - u.x; dy = a[1] - u.y; }
    const d = Math.hypot(dx, dy) || 1, ux = dx / d, uy = dy / d;
    // lined up on the run well before the initial point, so the drop itself is flown straight
    const ip = [a[0] - ux * 2.6, a[1] - uy * 2.6], lineUp = [a[0] - ux * 5.2, a[1] - uy * 5.2], exit = [b[0] + ux * 2.2, b[1] + uy * 2.2];
    this.fly(u, [
      { pts: [lineUp, ip], speed: spec.cruise * 0.7, status: "INBOUND · RETARDANT", kind: "transit", alt: [Math.max(u.alt, 260), 160], smooth: true, heading: Math.atan2(ux, uy), mark: "IP" },
      { pts: [a, b], speed: spec.work, status: "DROP RUN", kind: "drop", alt: [160, 60], end: cells ? { type: "retardant", cells } : { type: "retardant", idx: pkg }, mark: "DROP" },
      { pts: [exit], speed: spec.work, status: "OFF THE DROP", kind: "transit", alt: [60, 400] },
    ]);
  }

  /** The next piece of line to coat: the stretch the fire reaches soonest. */
  retardantPackage(op) {
    const line = op.line;
    if (!line) return null;
    let best = -1, bestKey = Infinity;
    for (let i = 0; i < line.cells.length; i++) {
      if (op.ret[i] !== 0 || line.kinds[i] === KIND.natural) continue;
      if (this.fire.arrival[line.cells[i]] !== UNBURNED) {
        op.ret[i] = 2; // too late for this cell
        continue;
      }
      // built line first; roads are a break already and are coated when there is time
      const key = (line.deadline[i] === UNBURNED ? 1e7 : line.deadline[i]) + (line.kinds[i] === KIND.road ? 120 : 0);
      if (key < bestKey) { bestKey = key; best = i; }
    }
    if (best < 0) return null;
    const pkg = [best];
    for (let k = 1; pkg.length < PACKAGE; k++) {
      const up = best + k, down = best - k;
      if (up < line.cells.length && op.ret[up] === 0 && line.kinds[up] !== KIND.natural && pkg.at(-1) === up - 1) pkg.push(up);
      else if (down >= 0 && op.ret[down] === 0 && line.kinds[down] !== KIND.natural && pkg[0] === down + 1) pkg.unshift(down);
      else break;
    }
    return pkg;
  }

  // -- skimmer or bucket helicopter: water on the edge, from the nearest water that works

  bomber(u, op, spec) {
    const { theatre, fire } = this;
    const target = this.waterTarget(u, op, spec);
    if (target < 0) {
      // nothing workable now. With the fire hours from the line, wait on the ground and save the fuel;
      // with it close, hold near the fire and look again
      // (a small fire under direct attack counts as close: the aircraft stay over it until the crews have it)
      const due = op.line ? op.line.firstArrival - this.t : Infinity, close = due < 150 || (op.coa === "direct" && this.edgeOf(op).cells.length <= 60);
      if (u.alt <= 0) return schedule(u, this.t, [{ hold: 10, status: close ? "AT BASE · READY" : "AT BASE · STANDBY", kind: "ground" }]);
      if (!close) return this.returnHome(u, "RTB · STANDBY");
      const focus = this.focusOf(op), wind = (this.weather(this.t).wd * Math.PI) / 180;
      // each aircraft waits in its own place round the fire, so that they do not stack up
      const slot = (u.id * 2.4) % (2 * Math.PI);
      // a helicopter with nothing to work goes back to its base and waits there
      if (spec.heli) return this.returnHome(u, "RTB · NOTHING TO WORK");
      const centre = [focus[0] + Math.sin(slot) * 4.5, focus[1] + Math.cos(slot) * 4.5];
      return this.fly(u, [{ pts: racetrack(centre[0], centre[1], wind, 3, 1.1), speed: spec.work, status: op.coa === "direct" ? "HOLDING · WATCHING THE EDGE" : "HOLDING · LINE WATCH", kind: "orbit", alt: [u.alt, spec.alt], smooth: true, track: true }]);
    }
    const [tx, ty] = theatre.xy(target);
    this.claims.set(target, [this.pledged(target) + spec.load, this.t + CLAIM_MIN]);
    if (u.load <= 0) {
      if (spec.heli) {
        const w = theatre.nearestWater(tx, ty);
        const dip = w >= 0 ? theatre.xy(w) : [u.home.x, u.home.y];
        return this.fly(u, [
          { pts: [dip], speed: spec.cruise, status: "TO DIP SITE", kind: "transit", alt: [Math.max(u.alt, 60), 20], smooth: true, mark: "DIP" },
          { hold: 0.8, status: "FILLING BUCKET", kind: "scoop", alt: [20, 20], end: { type: "fill" } },
        ]);
      }
      // the group works one stretch of water in trail, as skimmers do
      if (!op.scoop || this.t - op.scoop.at > 40) op.scoop = { run: this.scoopRun({ x: tx, y: ty }, tx, ty), at: this.t };
      const run = op.scoop.run;
      if (!run) return this.returnHome(u, "RTB · NO SCOOP WATER");
      const lead = [run.a[0] - run.ux * 2.4, run.a[1] - run.uy * 2.4];
      return this.fly(u, [
        { pts: [lead, run.a], speed: spec.cruise * 0.8, status: "TO SCOOP", kind: "transit", alt: [Math.max(u.alt, 120), 0], smooth: true, heading: Math.atan2(run.ux, run.uy), mark: "SCOOP" },
        { pts: [run.b], speed: SCOOP_SPEED, status: "SCOOPING", kind: "scoop", alt: [0, 0], end: { type: "scoop" } },
        { pts: [[run.b[0] + run.ux * 1.2, run.b[1] + run.uy * 1.2]], speed: spec.work, status: "LOADED", kind: "transit", alt: [0, 150] },
      ]);
    }
    const dx = tx - u.x, dy = ty - u.y, d = Math.hypot(dx, dy) || 1, ux = dx / d, uy = dy / d;
    const back = Math.min(spec.heli ? 0.8 : 2, d * 0.5);
    this.fly(u, [
      { pts: [[tx - ux * back, ty - uy * back]], speed: spec.cruise * (spec.heli ? 1 : 0.8), status: fire.arrival[target] === UNBURNED ? "INBOUND · WETTING AHEAD" : "INBOUND · KNOCKDOWN", kind: "transit", alt: [Math.max(u.alt, 90), 90], smooth: true, heading: Math.atan2(ux, uy) },
      { pts: [[tx, ty]], speed: spec.work, status: "DROP RUN", kind: "drop", alt: [90, 40], end: { type: "water", cell: target }, mark: "DROP" },
      { pts: [[tx + ux * (spec.heli ? 0.5 : 1.4), ty + uy * (spec.heli ? 0.5 : 1.4)]], speed: spec.work, status: "OFF THE DROP", kind: "transit", alt: [40, 150] },
    ]);
  }

  /** Where the centre of the action is: the head of the fire, or the stretch of line it reaches first. */
  focusOf(op) {
    const { theatre, fire } = this;
    if (op.line) {
      let best = 0;
      for (let i = 1; i < op.line.cells.length; i++) if (op.line.deadline[i] < op.line.deadline[best]) best = i;
      return theatre.xy(op.line.cells[best]);
    }
    let best = -1, top = -1;
    const weight = this.threat;
    for (const c of fire.front) if (fire.owner[c] === op.fire && (weight ? weight[c] : fire.hfi[c]) > top) { top = weight ? weight[c] : fire.hfi[c]; best = c; }
    return best >= 0 ? theatre.xy(best) : [theatre.town.cx, theatre.town.cy];
  }

  /** Litres already on their way to a cell. */
  pledged(cell) {
    const p = this.claims.get(cell);
    return p && p[1] > this.t ? p[0] : 0;
  }
  /** Does a cell still want water, beyond what has landed and what is inbound? */
  wants(cell) {
    const fire = this.fire;
    const need = fire.arrival[cell] !== UNBURNED ? DROP.knockdownBase + DROP.knockdownPerKw * fire.hfi[cell] : DROP.wetLitres;
    return this.soak[cell] + this.pledged(cell) < need;
  }

  /**
   * The edge worth working on an operation, with what makes each cell matter. Rebuilt every
   * few minutes: the perimeter of a large fire is thousands of cells and every aircraft asks.
   */
  edgeOf(op) {
    const cached = op.edge;
    if (cached && this.t - cached.at < 8) return cached;
    const { fire, theatre } = this, { n, cell, half, town } = theatre, line = op.line, direct = op.coa === "direct";
    if (direct) this.getForecast();
    const weight = this.threat, cells = [], xs = [], ys = [], base = [], inside = [];
    for (const c of fire.front) {
      if (fire.held[c]) continue;
      const x = ((c % n) + 0.5) * cell - half, y = (Math.floor(c / n) + 0.5) * cell - half;
      const behind = theatre.isTown[c] === 1 || (line ? line.townSide(x, y) : false);
      if (!behind && fire.owner[c] !== op.fire) continue;
      let score;
      if (behind) score = 1e7 - Math.hypot(x - town.cx, y - town.cy);
      else if (direct) score = 1 + (weight ? weight[c] : 0);
      else {
        // holding a line: only edge that is close to it or to the town
        const d = Math.hypot(x - town.cx, y - town.cy) - town.radius;
        if (d > 3.5 && !(line && this.nearLine(line, c, 6))) continue;
        score = 1000 / (1 + Math.max(0, d));
      }
      cells.push(c);
      xs.push(x);
      ys.push(y);
      base.push(score);
      inside.push(behind);
    }
    return (op.edge = { at: this.t, front: fire.front.length, cells, xs, ys, base, inside });
  }

  /**
   * Unburned fuel this fire is forecast to take in the next three hours, with how much
   * burns downstream of each cell. Kept per operation and renewed with the forecast.
   */
  aheadOf(op) {
    const forecast = this.getForecast(), cached = op.ahead;
    if (cached && cached.forecast === forecast && this.t - cached.at < 10) return cached;
    const { fire, theatre } = this, { n, cell, half } = theatre, flow = this.flow.weight, cells = [], xs = [], ys = [];
    for (const v of this.flow.order) {
      if (forecast.arrival[v] - this.t > 180) continue;
      if (forecast.sim.owner[v] !== op.fire || fire.arrival[v] !== UNBURNED || theatre.isTown[v] || flow[v] < 2) continue;
      cells.push(v);
      xs.push(((v % n) + 0.5) * cell - half);
      ys.push((Math.floor(v / n) + 0.5) * cell - half);
    }
    return (op.ahead = { forecast, at: this.t, cells, xs, ys });
  }

  /**
   * Where to treat fuel before the fire gets to it. Of the cells the forecast burns inside
   * the window this aircraft can meet (after it can arrive, before the treatment wears off),
   * the one with the most fire downstream of it, with a preference for extending a strip
   * already laid. `lasts` is how long the treatment works; `holds` the intensity it stops.
   */
  aheadTarget(u, op, spec, { lasts, holds, skip }) {
    const { fire, t } = this, { n } = this.theatre, ahead = this.aheadOf(op), forecast = ahead.forecast, flow = this.flow.weight;
    let best = -1, top = 0;
    for (let i = 0; i < ahead.cells.length; i++) {
      const v = ahead.cells[i];
      if (fire.arrival[v] !== UNBURNED || skip(v)) continue;
      const eta = (Math.hypot(ahead.xs[i] - u.x, ahead.ys[i] - u.y) / spec.cruise) * 60 + (u.load > 0 ? 2 : 7);
      const due = forecast.arrival[v] - t;
      if (due < eta + DROP.aheadMinMinutes || due > eta + lasts) continue;
      // worth more where the treated fuel will hold the fire, not just slow it
      const stops = forecast.sim.hfi[v] * WET_SHARE <= holds ? 1 : 0.45;
      // and where it joins what is already down, so that drops build a strip and not a scatter
      let joins = 1;
      for (const w of [v + 1, v - 1, v + n, v - n]) if (w >= 0 && w < flow.length && (fire.wet[w] > t || fire.retard[w] || this.claims.has(w))) joins = 1.6;
      const score = (flow[v] * stops * joins * Math.min(1, spec.load / DROP.wetLitres)) / (1 + eta / 20);
      if (score > top) { top = score; best = v; }
    }
    return { cell: best, score: top };
  }

  /**
   * The cell to put water on. Anything burning behind the line or in the town comes first.
   * Then the choice is between knocking down edge that is under the aircraft's limit and
   * wetting fuel the forecast says the fire takes next: whichever saves more downstream.
   */
  waterTarget(u, op, spec) {
    const { fire, t } = this, limit = spec.limit, line = op.line, edge = this.edgeOf(op);
    let best = -1, top = 0, urgent = false;
    for (let i = 0; i < edge.cells.length; i++) {
      const c = edge.cells[i];
      if (fire.hfi[c] > limit || fire.held[c] || fire.knock[c] > t || !this.wants(c)) continue;
      // what a load buys: a hot edge takes several, so one load there is worth a share of the result
      const share = Math.min(1, spec.load / (DROP.knockdownBase + DROP.knockdownPerKw * fire.hfi[c]));
      // a helicopter's small load goes where a crew is already on its way to follow it up
      const backed = spec.heli && (this.crewClaims.get(c) || 0) > t ? 2 : 1;
      const score = (edge.base[i] * share * backed) / (1 + Math.hypot(edge.xs[i] - u.x, edge.ys[i] - u.y) / 25);
      if (score > top) { top = score; best = c; urgent = edge.inside[i]; }
    }
    // treating fuel ahead of the fire is work for the big loads; a bucket is too little to wet a cell
    if (spec.heli) return best;
    if (!urgent) {
      const ahead = this.aheadTarget(u, op, spec, { lasts: DROP.wetMinutes - 5, holds: LIMIT.crew, skip: (v) => fire.wet[v] > t + 15 || !this.wants(v) });
      // on a held line the edge near it still comes first; in direct attack the two compete
      if (ahead.cell >= 0 && (best < 0 || (!line && ahead.score * DROP.aheadShare > top))) return ahead.cell;
    }
    if (best >= 0) return best;
    if (!line) {
      // direct attack with every workable cell already knocked down: wet the fuel beside them,
      // so that what comes back when the knockdown wears off comes back slowly
      const { n } = this.theatre, fuel = this.theatre.grid.fuel;
      for (let i = 0; i < edge.cells.length && best < 0; i++) {
        const c = edge.cells[i];
        if (fire.held[c] || fire.knock[c] <= t) continue;
        for (const v of [c + 1, c - 1, c + n, c - n, c + n + 1, c + n - 1, c - n + 1, c - n - 1]) {
          if (v < 0 || v >= fuel.length || !fuel[v] || fire.arrival[v] !== UNBURNED || fire.wet[v] > t + 15 || !this.wants(v)) continue;
          best = v;
          break;
        }
      }
      return best;
    }
    // pre-wet the strip in front of the line where the fire is due within the next hour and a half
    let soon = Infinity;
    for (let i = 0; i < line.cells.length; i++) {
      const due = line.deadline[i] - t;
      if (due < 0 || due > 90 || due >= soon) continue;
      const c = line.band[i].find((q) => fire.arrival[q] === UNBURNED && fire.wet[q] < t + 15 && this.wants(q));
      if (c !== undefined) { soon = due; best = c; }
    }
    return best;
  }
  nearLine(line, cell, reach) {
    line.set ??= new Set(line.cells);
    const n = this.theatre.n, c0 = cell % n, r0 = Math.floor(cell / n);
    for (let r = r0 - reach; r <= r0 + reach; r += 2) for (let c = c0 - reach; c <= c0 + reach; c += 2) if (line.set.has(r * n + c) || line.set.has(r * n + c + 1)) return true;
    return false;
  }

  /** The scooping run that makes the shortest circuit to a target, flown into the wind. */
  scoopRun(u, tx, ty) {
    const wind = (this.weather(this.t).wd * Math.PI) / 180, wx = Math.sin(wind), wy = Math.cos(wind); // unit vector towards where the wind comes from
    let best = null, bestCost = Infinity;
    for (const run of this.scoops) {
      const into = run.ux * wx + run.uy * wy >= 0 ? 1 : -1;
      const ux = run.ux * into, uy = run.uy * into;
      const a = [run.mx - (ux * SCOOP_KM) / 2, run.my - (uy * SCOOP_KM) / 2], b = [run.mx + (ux * SCOOP_KM) / 2, run.my + (uy * SCOOP_KM) / 2];
      const cost = Math.hypot(a[0] - u.x, a[1] - u.y) * 0.5 + Math.hypot(b[0] - tx, b[1] - ty) + Math.hypot(a[0] - tx, a[1] - ty) * 0.5;
      if (cost < bestCost) { bestCost = cost; best = { a, b, ux, uy, lake: run.lake }; }
    }
    return best;
  }

  // -- bird dog: over the work, for as long as there are tankers on it

  lead(u, op, spec) {
    const working = this.units.some((x) => x.op === u.op && x.cls === "air" && x !== u && x.alt > 0);
    if (working) u.used = true;
    if (!working && u.alt > 0 && this.t - op.since > 60) return this.returnHome(u, "RTB");
    const [fx, fy] = this.focusOf(op);
    const d = Math.hypot(fx - u.x, fy - u.y), radius = 3.4;
    const start = Math.atan2(u.x - fx, u.y - fy);
    const legs = [];
    if (d > radius * 1.6) legs.push({ pts: [[fx + Math.sin(start) * radius, fy + Math.cos(start) * radius]], speed: spec.cruise, status: "INBOUND · AIR ATTACK", kind: "transit", alt: [Math.max(u.alt, 300), spec.alt], smooth: true });
    legs.push({ pts: orbit(fx, fy, radius, start, 2), speed: spec.work, status: "AIR ATTACK · ORBIT", kind: "orbit", alt: [Math.max(u.alt, 300), spec.alt], smooth: true, track: true });
    this.fly(u, legs);
  }

  // -- dozers: machine guard along the line, where there is still time to finish it

  dozer(u, op) {
    const line = op.line;
    if (!line) return this.stage(u, op);
    if (u.retreat && this.t - u.retreat < 20) return schedule(u, this.t, [{ hold: 20, status: "HOLDING BACK", kind: "ground" }]);
    const idx = this.lineWork(u, op, (i) => line.kinds[i] === KIND.build && (op.guard[i] === 0 || op.guard[i] === 3), SPECS[u.type].rate);
    if (idx < 0) return this.park(u, op, "GUARD COMPLETE");
    op.guard[idx] = 1;
    const legs = this.drive(u, this.theatre.xy(line.cells[idx]), "TO THE LINE");
    legs[legs.length - 1].end = { type: "work", work: { kind: "guard", idx, status: "CUTTING GUARD" } };
    schedule(u, this.t, legs);
  }

  /** The cell of line to start on: the most urgent one this unit can still reach with time to work. */
  lineWork(u, op, ok, rateKmH) {
    const line = op.line, { theatre, t } = this;
    let best = -1, bestKey = Infinity, fallback = -1, latest = -Infinity;
    for (let i = 0; i < line.cells.length; i++) {
      if (!ok(i) || this.fire.arrival[line.cells[i]] !== UNBURNED) continue;
      const [x, y] = theatre.xy(line.cells[i]);
      const travel = (Math.hypot(x - u.x, y - u.y) / 35) * 60 + 10; // rough: the route is worked out once chosen
      const due = line.deadline[i] === UNBURNED ? t + 1e5 : line.deadline[i];
      const slack = due - t - travel - (theatre.cell / rateKmH) * 60;
      if (slack > 30) {
        if (due + travel < bestKey) { bestKey = due + travel; best = i; }
      } else if (due > latest) { latest = due; fallback = i; }
    }
    return best >= 0 ? best : fallback >= 0 && latest - t > 45 ? fallback : -1;
  }

  // -- crews: put out what is behind the line, burn off from finished line, cut hand line, mop up edge

  crew(u, op, spec) {
    const { fire, theatre, t } = this;
    if (u.retreat && t - u.retreat < 15) return schedule(u, t, [{ hold: 15, status: "IN SAFETY ZONE", kind: "ground" }]);
    const line = op.line, direct = op.coa === "direct", edge = this.edgeOf(op);
    // edge a crew can work: quiet enough, or knocked down by a drop
    let best = -1, top = 0;
    for (let i = 0; i < edge.cells.length; i++) {
      const c = edge.cells[i];
      if (fire.held[c] || (this.crewClaims.get(c) || 0) > t) continue;
      if (!(fire.hfi[c] <= LIMIT.crew || fire.knock[c] > t)) continue;
      const inside = edge.inside[i];
      if (!inside && !direct && !(line && this.nearLine(line, c, 4))) continue;
      const d = Math.hypot(edge.xs[i] - u.x, edge.ys[i] - u.y);
      // knocked-down edge first: it comes back if nobody gets there
      const score = ((inside ? 1e6 : 1) + (fire.knock[c] > t ? 400 : 0) + (direct ? Math.min(edge.base[i], 300) : 0) + 20) / (1 + d * (spec.heliborne ? 0.25 : 1.5));
      if (score > top) { top = score; best = c; }
    }
    if (best >= 0) {
      const [x, y] = theatre.xy(best), d = Math.hypot(x - u.x, y - u.y);
      this.crewClaims.set(best, t + 90);
      if (d < 0.4) {
        u.x = x;
        u.y = y;
        u.used = true;
        u.work = { kind: "mop", cell: best, progress: 0, status: "MOPPING UP EDGE" };
        u.status = "MOPPING UP EDGE";
        return;
      }
      const legs = d < 1.2 ? [{ pts: [[x, y]], speed: spec.cross, status: "ALONG THE EDGE", kind: "cross" }] : this.drive(u, [x, y], "TO THE EDGE", d > 4);
      legs[legs.length - 1].end = { type: "work", work: { kind: "mop", cell: best, status: "MOPPING UP EDGE" } };
      return schedule(u, t, legs);
    }
    if (line) {
      let idx = this.lineWork(u, op, (i) => this.canBurn(op, i), BURN_RATE);
      let job = "burn";
      if (idx < 0) {
        idx = this.lineWork(u, op, (i) => line.kinds[i] === KIND.build && op.guard[i] === 0, spec.lineRate);
        job = "handline";
      }
      if (idx >= 0) {
        if (job === "burn") op.burn[idx] = 1;
        else op.guard[idx] = 1;
        const to = theatre.xy(line.cells[idx]), d = Math.hypot(to[0] - u.x, to[1] - u.y);
        const legs = d < 1 ? [{ pts: [to], speed: spec.cross, status: "ALONG THE LINE", kind: "cross" }] : this.drive(u, to, "TO THE LINE", d > 6);
        legs[legs.length - 1].end = { type: "work", work: { kind: job, idx, status: job === "burn" ? "BURNING OFF" : "CUTTING HAND LINE" } };
        return schedule(u, t, legs);
      }
    }
    if (direct) {
      // nothing a crew can work yet: wait at the quiet end of the fire, upwind, for the drops to land
      let rear = -1, least = Infinity;
      for (let i = 0; i < edge.cells.length; i++) if (!edge.inside[i] && fire.hfi[edge.cells[i]] < least) { least = fire.hfi[edge.cells[i]]; rear = i; }
      if (rear >= 0 && edge.cells.length <= 400) {
        const wind = (this.weather(t).wd * Math.PI) / 180;
        const to = [edge.xs[rear] + Math.sin(wind) * 0.45, edge.ys[rear] + Math.cos(wind) * 0.45], d = Math.hypot(to[0] - u.x, to[1] - u.y);
        if (d < 0.25) return schedule(u, t, [{ hold: 6, status: "AT THE REAR · WAITING ON DROPS", kind: "ground" }]);
        const legs = d < 1.2 ? [{ pts: [to], speed: spec.cross, status: "TO THE REAR", kind: "cross" }] : this.drive(u, to, "TO THE REAR", d > 4);
        return schedule(u, t, legs);
      }
    }
    this.park(u, op, direct ? "STANDING BY · EDGE TOO HOT" : "HOLDING THE LINE");
  }

  /** Nothing to do for now: wait near the work, on a road. */
  park(u, op, status) {
    const [fx, fy] = this.focusOf(op), { theatre } = this;
    // a third of the way from the town to the action, never closer than 1.5 km to live fire
    const to = [theatre.town.cx + (fx - theatre.town.cx) * 0.35, theatre.town.cy + (fy - theatre.town.cy) * 0.35];
    const node = theatre.roads.nearest(to[0], to[1]);
    const spot = node.node >= 0 ? [theatre.roads.nodes[node.node].x, theatre.roads.nodes[node.node].y] : to;
    if (Math.hypot(spot[0] - u.x, spot[1] - u.y) < 0.6 || this.fireNear(spot[0], spot[1], 9)) return schedule(u, this.t, [{ hold: 20, status, kind: "ground" }]);
    const legs = this.drive(u, spot, "REPOSITIONING");
    legs.push({ hold: 20, status, kind: "ground" });
    schedule(u, this.t, legs);
  }

  // -- engines and structure protection units: the edge of town the fire reaches first

  protect(u, op) {
    const seg = op.segments.find((s) => s.by === u.id) || op.segments.find((s) => !s.by && (u.role === "spu" ? s.set < s.cells.length : true));
    if (!seg) return this.stage(u, op);
    seg.by = u.id;
    const mid = seg.cells[seg.cells.length >> 1], to = this.theatre.xy(mid);
    const legs = this.drive(u, to, `TO ${seg.label}`);
    legs[legs.length - 1].end = { type: "work", work: u.role === "spu" ? { kind: "sprinkler", seg: op.segments.indexOf(seg), status: "LAYING SPRINKLER LINE" } : { kind: "structure", cell: mid, status: "STRUCTURE DEFENCE" } };
    schedule(u, this.t, legs);
  }

  /** Medics and police, and anything with no job: stage on the way out of town. */
  stage(u, op) {
    const { theatre } = this, [fx, fy] = this.focusOf(op);
    const d = Math.hypot(fx - theatre.town.cx, fy - theatre.town.cy) || 1;
    const out = u.role === "police" ? theatre.town.radius + 0.6 : theatre.town.radius * 0.6;
    const to = [theatre.town.cx + ((fx - theatre.town.cx) / d) * out, theatre.town.cy + ((fy - theatre.town.cy) / d) * out];
    const node = theatre.roads.nearest(to[0], to[1]);
    const spot = node.node >= 0 ? [theatre.roads.nodes[node.node].x, theatre.roads.nodes[node.node].y] : to;
    const status = u.role === "police" ? "ROAD CONTROL" : u.role === "medic" ? "STAGED · MEDICAL" : "STAGED";
    if (u.role === "police" || u.role === "medic") u.used = true;
    if (Math.hypot(spot[0] - u.x, spot[1] - u.y) < 0.2) return schedule(u, this.t, [{ hold: 30, status, kind: "ground" }]);
    const legs = this.drive(u, spot, "TO STAGING");
    legs.push({ hold: 30, status, kind: "ground" });
    schedule(u, this.t, legs);
  }

  // ---- everyday traffic for units with nothing to do

  idle(u) {
    const { theatre, random } = this, spec = SPECS[u.type], town = theatre.town;
    const address = () => {
      const b = theatre.data.buildings;
      if (b?.length) return b[Math.floor(random() * b.length)];
      return [town.cx + (random() - 0.5) * town.radius, town.cy + (random() - 0.5) * town.radius];
    };
    u.state = "busy";
    if (u.role === "medic") {
      const call = address(), hospital = theatre.data.stations?.find((s) => s.k === "hospital");
      const legs = this.drive(u, call, "RESPONDING · MEDICAL");
      legs.push({ hold: 8 + random() * 10, status: "ON SCENE", kind: "ground" });
      schedule(u, this.t, legs);
      if (hospital) {
        const at = { ...u, x: call[0], y: call[1] };
        schedule(u, this.t, [...this.drive(at, [hospital.x, hospital.y], "TRANSPORT"), { hold: 10, status: "AT HOSPITAL", kind: "ground" }], { append: true });
        const back = this.drive({ ...u, x: hospital.x, y: hospital.y }, [u.home.x, u.home.y], "RETURNING");
        back[back.length - 1].end = { type: "home" };
        schedule(u, this.t, back, { append: true });
      } else schedule(u, this.t, [{ hold: 1, status: "CLEAR", kind: "ground", end: { type: "home" } }], { append: true });
      return;
    }
    if (u.role === "police") {
      let at = { ...u };
      schedule(u, this.t, [{ hold: 0.2, status: "PATROL", kind: "ground" }]);
      for (let i = 0; i < 3; i++) {
        const stop = address();
        schedule(u, this.t, this.drive(at, stop, "PATROL"), { append: true });
        at = { ...u, x: stop[0], y: stop[1] };
      }
      const back = this.drive(at, [u.home.x, u.home.y], "PATROL");
      back[back.length - 1].end = { type: "home" };
      schedule(u, this.t, back, { append: true });
      return;
    }
    if (u.role === "engine" && random() < 0.5) {
      const call = address(), legs = this.drive(u, call, "RESPONDING · ALARM");
      legs.push({ hold: 12 + random() * 15, status: "ON SCENE", kind: "ground" });
      schedule(u, this.t, legs);
      const back = this.drive({ ...u, x: call[0], y: call[1] }, [u.home.x, u.home.y], "RETURNING");
      back[back.length - 1].end = { type: "home" };
      schedule(u, this.t, back, { append: true });
      return;
    }
    if (spec.cls === "air" && this.isDay() && (u.role === "lead" || u.role === "helo")) {
      // a detection patrol: out round a few points and home
      const pts = [], reach = u.role === "lead" ? 24 : 13;
      const a0 = random() * Math.PI * 2;
      for (let i = 0; i < 4; i++) pts.push([clampKm(u.home.x + Math.sin(a0 + i * 1.4) * reach * (0.6 + random() * 0.4), theatre.half), clampKm(u.home.y + Math.cos(a0 + i * 1.4) * reach * (0.6 + random() * 0.4), theatre.half)]);
      this.fly(u, [
        { pts: [pts[0]], speed: spec.cruise * 0.8, status: "DETECTION PATROL", kind: "patrol", alt: [u.alt, spec.alt], smooth: true },
        { pts: pts.slice(1), speed: spec.cruise * 0.8, status: "DETECTION PATROL", kind: "patrol", alt: [spec.alt, spec.alt], smooth: true, end: { type: "status", status: "RTB" } },
      ]);
      return;
    }
    if (spec.cls === "air" && this.isDay() && u.role === "skimmer" && random() < 0.35) {
      // a proficiency circuit: scoop and drop on open water
      const run = this.scoopRun(u, u.home.x, u.home.y);
      if (run && Math.hypot(run.a[0] - u.home.x, run.a[1] - u.home.y) < 30) {
        this.fly(u, [
          { pts: [[run.a[0] - run.ux * 2.4, run.a[1] - run.uy * 2.4], run.a], speed: spec.cruise * 0.8, status: "PROFICIENCY CIRCUIT", kind: "patrol", alt: [Math.max(u.alt, 150), 0], smooth: true, heading: Math.atan2(run.ux, run.uy), mark: "SCOOP" },
          { pts: [run.b], speed: SCOOP_SPEED, status: "SCOOPING", kind: "scoop", alt: [0, 0] },
          { pts: [[run.b[0] + run.ux * 2, run.b[1] + run.uy * 2]], speed: spec.work, status: "PROFICIENCY CIRCUIT", kind: "patrol", alt: [0, 200], end: { type: "status", status: "RTB" } },
        ]);
        return;
      }
    }
    u.state = "idle";
    u.nextIdle = this.t + 30 + random() * 90;
  }
}

const clampKm = (v, half) => Math.max(-half + 1, Math.min(half - 1, v));
const fmt = (n) => Math.round(n).toLocaleString("en-CA");
/** Fires are lettered in the order they were found, as in the field: A, B, C. */
export const fireName = (id) => (id <= 26 ? String.fromCharCode(64 + id) : `F${id}`);

function cloneOp(op) {
  return {
    ...op, edge: null, ahead: null, ret: op.ret?.slice() ?? null, guard: op.guard?.slice() ?? null, burn: op.burn?.slice() ?? null,
    segments: op.segments.map((s) => ({ ...s })), stats: { ...op.stats },
  };
}

/** Every scooping run in the theatre, as a midpoint and a direction, in kilometres. */
function scoopRuns(theatre) {
  const { cell, half } = theatre, out = [];
  for (const lake of theatre.lakes)
    for (const run of lake.runs) {
      const ax = (run.c0 + 0.5) * cell - half, ay = (run.r0 + 0.5) * cell - half, bx = (run.c1 + 0.5) * cell - half, by = (run.r1 + 0.5) * cell - half;
      const d = Math.hypot(bx - ax, by - ay) || 1;
      out.push({ lake: lake.id, mx: (ax + bx) / 2, my: (ay + by) / 2, ux: (bx - ax) / d, uy: (by - ay) / d, km: run.km });
    }
  return out;
}
export { clampHdg };
