// The planner. For a chosen fire it reads the situation off the forecast, builds up to three
// courses of action, and scores each one by running copies of the world forward under an
// ensemble of winds. The recommendation is the course that loses the least: first the town,
// then structures, then forest.
import { UNBURNED } from "./fire.js";
import { hfiClass } from "./fbp.js";
import { impact, routeTo } from "./forecast.js";
import { planLine } from "./line.js";
import { DROP, LIMIT, RATE } from "./doctrine.js";
import { SPECS } from "./units.js";
import { perturbed } from "./scenario.js";

const COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
export const compass = (dx, dy) => COMPASS[Math.round((((Math.atan2(dx, dy) * 180) / Math.PI + 360) % 360) / 45) % 8];
const STRUCTURE_LOSS = 0.65, PROTECTED_LOSS = 0.15; // share of buildings lost in a burned town cell. Assumption.

/** What a world has lost so far to one fire (or to all of them). */
export function losses(world, fireId = 0) {
  const { fire, theatre } = world;
  let cells = 0, town = 0, structures = 0;
  for (let i = 0; i < fire.arrival.length; i++) {
    if (fire.arrival[i] === UNBURNED || (fireId && fire.owner[i] !== fireId)) continue;
    cells++;
    if (theatre.isTown[i]) {
      town++;
      structures += theatre.buildings[i] * (fire.wet[i] >= fire.arrival[i] ? PROTECTED_LOSS : STRUCTURE_LOSS);
    }
  }
  return { ha: cells * theatre.cell * theatre.cell * 100, townCells: town, structures: Math.round(structures) };
}

/** The situation of every fire: size, behaviour at the edge, and what the forecast says. */
export function incidents(world) {
  const { fire, theatre } = world, forecast = world.getForecast();
  const byId = new Map();
  const get = (id) => byId.get(id) || byId.set(id, { id, cells: 0, edge: 0, active: 0, hfi: 0, ros: 0, classes: [0, 0, 0, 0, 0, 0, 0], sx: 0, sy: 0, hx: 0, hy: 0 }).get(id);
  for (let i = 0; i < fire.arrival.length; i++) if (fire.arrival[i] !== UNBURNED && fire.owner[i]) get(fire.owner[i]).cells++;
  for (const c of fire.front) {
    const f = get(fire.owner[c]);
    const [x, y] = theatre.xy(c);
    f.edge++;
    f.sx += x;
    f.sy += y;
    if (fire.held[c]) continue;
    f.active++;
    f.classes[hfiClass(fire.hfi[c])]++;
    if (fire.hfi[c] > f.hfi) { f.hfi = fire.hfi[c]; f.hx = x; f.hy = y; }
  }
  // rate of the fastest pending advance, and the forecast for the town
  const { heap, tent, arrival, psrc, pros, owner } = fire;
  for (let i = 0; i < heap.n; i++) {
    const v = heap.c[i];
    if (arrival[v] !== UNBURNED || tent[v] !== heap.t[i] || psrc[v] < 0) continue;
    const f = byId.get(owner[psrc[v]]);
    if (f && pros[v] > f.ros) f.ros = pros[v];
  }
  const out = [];
  for (const f of byId.values()) {
    if (!f.id) continue;
    // when this fire, by the forecast, first reaches the town
    let at = UNBURNED, cell = -1;
    for (const c of theatre.town.cells) if (forecast.arrival[c] < at && forecast.sim.owner[c] === f.id) { at = forecast.arrival[c]; cell = c; }
    const x = f.edge ? f.sx / f.edge : 0, y = f.edge ? f.sy / f.edge : 0;
    out.push({
      id: f.id, ha: f.cells * theatre.cell * theatre.cell * 100, edgeKm: f.edge * theatre.cell, activeKm: f.active * theatre.cell, ros: f.ros, hfi: f.hfi, cls: f.active ? hfiClass(f.hfi) : 0,
      classes: f.classes, x, y, head: f.active ? [f.hx, f.hy] : [x, y], km: Math.hypot((f.active ? f.hx : x) - theatre.town.cx, (f.active ? f.hy : y) - theatre.town.cy),
      impact: at === UNBURNED ? null : { at, minutes: Math.max(0, at - world.t), cell }, op: world.ops.get(f.id) || null,
    });
  }
  return out.sort((a, b) => (a.impact?.minutes ?? 1e9) - (b.impact?.minutes ?? 1e9) || b.ha - a.ha);
}

/** The stretches of the town's edge the forecast fire reaches, soonest first. */
function edgeSegments(world, forecast, size = 6) {
  const { theatre, fire } = world, { town, n } = theatre;
  const ordered = town.edge.map((c) => {
    const [x, y] = theatre.xy(c);
    return { c, a: Math.atan2(x - town.cx, y - town.cy), at: forecast.arrival[c], x, y };
  }).sort((p, q) => p.a - q.a);
  const segments = [];
  for (let i = 0; i < ordered.length; i += size) {
    const part = ordered.slice(i, i + size);
    const at = Math.min(...part.map((p) => p.at)), mid = part[part.length >> 1];
    const ring = part.map((p) => {
      const c = p.c % n, r = Math.floor(p.c / n), out = [];
      for (const [dc, dr] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]) {
        const q = (r + dr) * n + c + dc;
        if (c + dc >= 0 && c + dc < n && r + dr >= 0 && r + dr < n && !theatre.isTown[q] && theatre.grid.fuel[q] && fire.arrival[q] === UNBURNED) out.push(q);
      }
      return out;
    });
    segments.push({ cells: part.map((p) => p.c), ring, at, x: mid.x, y: mid.y, label: `${compass(mid.x - town.cx, mid.y - town.cy)} EDGE` });
  }
  // threatened stretches first; the rest by how near the fire is
  const [hx, hy] = forecast.head || [town.cx, town.cy];
  return segments.sort((a, b) => (a.at === b.at ? Math.hypot(a.x - hx, a.y - hy) - Math.hypot(b.x - hx, b.y - hy) : a.at - b.at));
}

/**
 * Read the situation of one fire and lay out the courses of action against it.
 * @returns { situation, coas } — not yet scored
 */
export function plan(world, fireId) {
  const { theatre, fire } = world;
  const forecast = world.getForecast(true);
  const incident = incidents(world).find((f) => f.id === fireId);
  if (!incident) return null;
  forecast.head = incident.head;
  // what this fire, and not its neighbours, is forecast to do to the town
  const hit = incident.impact ? { ...incident.impact } : { cell: -1 };
  const route = hit.cell >= 0 ? routeTo(forecast, hit.cell).map((c) => theatre.xy(c)) : [];
  let atRisk = 0, cellsAtRisk = 0;
  for (const c of theatre.town.cells) if (forecast.arrival[c] !== UNBURNED && forecast.sim.owner[c] === fireId) { cellsAtRisk++; atRisk += theatre.buildings[c]; }
  const threatened = hit.cell >= 0 || impact(forecast, theatre.town.cells).cell >= 0;
  const workable = incident.classes.slice(1, 5).reduce((a, b) => a + b, 0), total = incident.classes.reduce((a, b) => a + b, 0) || 1;
  const situation = {
    fire: fireId, ...incident, hit: hit.cell >= 0 ? hit : null, route, buildingsAtRisk: atRisk, townShare: theatre.town.cells.length ? cellsAtRisk / theatre.town.cells.length : 0,
    workableShare: workable / total, crewShare: (incident.classes[1] + incident.classes[2] + incident.classes[3]) / total, horizonHours: world.forecastHours,
  };

  const free = world.units.filter((u) => !u.op || u.op === fireId);
  const of = (...roles) => free.filter((u) => roles.includes(u.role));
  const ids = (list) => list.map((u) => u.id);
  const segments = edgeSegments(world, forecast);
  const town = of("engine", "spu", "medic", "police");
  const count = (list) => {
    const by = {};
    for (const u of list) by[SPECS[u.type].short] = (by[SPECS[u.type].short] || 0) + 1;
    return Object.entries(by).map(([k, v]) => `${v} ${k}`).join(" · ");
  };
  const coas = [];

  // A. direct attack: put the fire out at its edge
  {
    const units = [...of("tanker", "skimmer", "helo", "lead", "crew"), ...town];
    const notes = [];
    if (incident.cls >= 5) notes.push(`Head is class ${incident.cls}: no action at the head, flanks and rear only`);
    if (situation.workableShare < 0.5) notes.push(`${Math.round((1 - situation.workableShare) * 100)}% of the live edge is above ${fmtKw(LIMIT.skimmer)}: drops will not hold it`);
    if (incident.ha > 2000) notes.push("Past initial attack size: crews cannot follow up this much edge");
    coas.push({
      key: "direct", fire: fireId, name: "DIRECT ATTACK", intent: "EXTINGUISH", units: ids(units), resources: count(units), segments: segments.slice(0, 3), line: null, notes,
      method: ["Water on the fuel the forecast says burns next, where the most fire has to pass; retardant across the same path further out", "Knockdown where the edge is under the aircraft limit; crews mop up behind every drop", "Engines stand by on the town edge"],
    });
  }

  // B and C. hold a line: wide, tied into lakes and roads; or tight round the town
  const tankers = of("tanker"), dozers = of("dozer");
  const lineFor = (reachKm, bufferKm, leads) => {
    for (const leadMin of leads) {
      const line = planLine(theatre, fire, forecast, { reachKm, bufferKm, leadMin });
      if (!line) return null; // the town is not threatened inside the forecast
      if (line.feasible) return { ...line, leadMin };
    }
    return { feasible: false };
  };
  const capacity = (line) => {
    // km of line an hour from tankers (by round trip to the middle of the line) and from dozers
    if (!line?.feasible) return 0;
    const [mx, my] = theatre.xy(line.cells[line.cells.length >> 1]);
    let perHour = dozers.length * RATE.dozer;
    for (const u of tankers) {
      const spec = SPECS[u.type], trip = (2 * Math.hypot(mx - u.home.x, my - u.home.y)) / spec.cruise * 60 + spec.reloadMin + 8;
      perHour += (DROP.heavyLineKm * 60) / trip;
    }
    return perHour;
  };
  const describe = (line) => {
    const perHour = capacity(line), hours = perHour ? line.buildKm / perHour : Infinity;
    const lead = (line.firstArrival - world.t) / 60;
    const notes = [];
    if (!line.anchored[0] || !line.anchored[1]) notes.push("One end of the line ties into nothing: it can be outflanked");
    if (hours > lead) notes.push(`Line needs ${hours.toFixed(1)} h of work; the fire is due on it in ${lead.toFixed(1)} h`);
    return { perHour, hours, lead, notes };
  };
  // a line round the town answers a fire that is coming to it; for one that is not, the question is only whether to put it out
  const wide = hit.cell >= 0 ? lineFor(Math.min(14, Math.max(6, theatre.town.radius + 9)), 1.0, [240, 180, 120, 90]) : null;
  if (wide?.feasible) {
    const units = [...tankers, ...of("lead", "skimmer", "helo", "dozer", "crew"), ...town], d = describe(wide);
    coas.push({
      key: "contain", fire: fireId, name: "CONTROL LINE", intent: "CONTAIN", units: ids(units), resources: count(units), segments: segments.slice(0, 2), line: wide, work: d, notes: d.notes,
      method: [`${wide.km.toFixed(1)} km line: ${wide.naturalKm.toFixed(1)} km water and rock, ${wide.roadKm.toFixed(1)} km road, ${wide.buildKm.toFixed(1)} km to build`, "Retardant where the fire is due first; dozers cut guard; crews burn off from finished line", "Skimmers and buckets wet the strip ahead of the fire and take anything that spots over"],
    });
  }
  const tight = hit.cell >= 0 ? lineFor(theatre.town.radius + 2.6, 0.3, [90, 60, 40]) : null;
  if (tight?.feasible) {
    const units = [...tankers, ...of("lead", "skimmer", "helo", "dozer", "crew"), ...town], d = describe(tight);
    coas.push({
      key: "defend", fire: fireId, name: "POINT PROTECTION", intent: "PROTECT", units: ids(units), resources: count(units), segments, line: tight, work: d, notes: d.notes,
      method: [`${tight.km.toFixed(1)} km line at the edge of town, ${tight.buildKm.toFixed(1)} km of it to build`, "Sprinkler lines and engines on every threatened stretch of the edge", "The forest outside the line is given up"],
    });
  } else if (hit.cell >= 0 && !wide?.feasible) {
    const units = [...of("skimmer", "helo", "lead", "crew"), ...town];
    coas.push({
      key: "defend", fire: fireId, name: "POINT PROTECTION", intent: "PROTECT", units: ids(units), resources: count(units), segments, line: null, notes: ["No line can be finished before the fire arrives"],
      method: ["Sprinkler lines and engines on the town edge", "Aircraft take spot fires in and around the town"],
    });
  }
  if (hit.cell < 0) coas[0].notes.push(threatened ? `The forecast does not bring this fire to ${theatre.town.name}; another fire is coming` : `The forecast does not bring this fire to ${theatre.town.name} in ${world.forecastHours} h`);
  return { situation, coas };
}

const fmtKw = (v) => `${v.toLocaleString("en-CA")} kW/m`;

/**
 * Score the courses of action. Each is run `runs` times on a copy of the world, every run
 * with its own wind (turned and scaled) and its own ember spotting; "no action" is run too.
 * Yields to the page between runs.
 */
export async function evaluate(world, planned, { runs = 5, hours, onProgress, budgetMs = 14, pause = () => new Promise((r) => setTimeout(r)), cancelled = () => false } = {}) {
  let last = performance.now();
  const { situation, coas } = planned;
  // long enough to see the fire reach the town and what happens there, and no longer
  const horizon = hours ?? Math.round(Math.max(8, Math.min(34, situation.hit ? situation.hit.minutes / 60 + 5 : 10)));
  const members = Array.from({ length: runs }, (_, i) => {
    // the same members for every option, so that they are compared on equal terms
    const a = Math.sin((i + 1) * 12.9898) * 43758.5453, b = Math.sin((i + 1) * 78.233) * 12543.123;
    const u = a - Math.floor(a), v = b - Math.floor(b);
    return i === 0 ? { turn: 0, scale: 1, seed: 101 } : { turn: (u - 0.5) * 36, scale: 0.8 + v * 0.5, seed: 101 + i * 17 };
  });
  const options = [{ key: "none", name: "NO ACTION" }, ...coas];
  const total = options.length * runs;
  let done = 0;
  const before = losses(world, situation.fire);
  for (const option of options) {
    const results = [];
    for (const m of members) {
      const copy = world.clone({ seed: m.seed, weather: perturbed(world.weather, m.turn, m.scale) });
      for (const u of copy.units) u.used = false;
      if (option.key !== "none") copy.commit(option);
      let reached = Infinity;
      for (let h = 0; h < horizon * 2; h++) {
        copy.step(30);
        if (reached === Infinity && copy.theatre.town.cells.some((c) => copy.fire.arrival[c] !== UNBURNED && copy.fire.owner[c] === situation.fire)) reached = copy.t - world.t;
        if (performance.now() - last > budgetMs) { await pause(); last = performance.now(); }
      }
      const l = losses(copy, situation.fire);
      const op = copy.ops.get(situation.fire);
      results.push({ ha: l.ha, structures: l.structures, town: l.townCells > 0, reached, held: (op ? op.status === "HELD" : false) || !copy.hasPending(situation.fire), burnedOff: (copy.fire.burnedOut || 0) * copy.theatre.cell * copy.theatre.cell * 100, stats: op?.stats });
      // the run under the forecast wind says which units the plan really uses
      if (m === members[0] && option.key !== "none") option.dispatch = copy.units.filter((u) => u.used).map((u) => u.id);
      onProgress?.(++done, total, option.name);
      if (cancelled()) return null;
    }
    const mean = (f) => results.reduce((a, r) => a + f(r), 0) / results.length;
    const reach = results.filter((r) => r.town).map((r) => r.reached).sort((a, b) => a - b);
    option.score = {
      runs, hours: horizon, ha: mean((r) => r.ha), haMin: Math.min(...results.map((r) => r.ha)), haMax: Math.max(...results.map((r) => r.ha)),
      townRuns: results.filter((r) => r.town).length, structures: mean((r) => r.structures), structuresMax: Math.max(...results.map((r) => r.structures)),
      heldRuns: results.filter((r) => r.held).length, reachedMin: reach.length ? reach[0] : null, burnedOff: mean((r) => r.burnedOff),
      retardant: mean((r) => r.stats?.retardant || 0), water: mean((r) => r.stats?.water || 0), guardKm: mean((r) => r.stats?.guardKm || 0),
    };
  }
  // what each plan has every unit do first: flown once on a copy, so it can be shown before anything moves
  for (const c of coas) {
    if (!c.dispatch?.length) c.dispatch = c.units;
    c.missions = missions(world, c);
    await pause();
  }
  const none = options[0].score;
  for (const c of coas) c.saved = { ha: none.ha - c.score.ha, structures: none.structures - c.score.structures };
  // least loss: the town, then structures, then area; a tie goes to the course that puts the fire out
  const order = ["direct", "contain", "defend"];
  const ranked = [...coas].sort((a, b) => a.score.townRuns - b.score.townRuns || Math.round(a.score.structures / 5) - Math.round(b.score.structures / 5) || Math.round((a.score.ha - b.score.ha) / (0.05 * (none.ha || 1))) || b.score.heldRuns - a.score.heldRuns || order.indexOf(a.key) - order.indexOf(b.key));
  for (const c of coas) c.recommended = c === ranked[0];
  return { situation, coas, none, before, horizon };
}

const QUIET_KINDS = new Set(["ground", "taxi", "climb", "orbit", "giveway"]); // legs that are not part of a route

/**
 * The opening of a plan, unit by unit: the plan is committed on a copy of the world that
 * flies and drives exactly as the live one does, and every leg each unit is given in the
 * first hours is kept. Returns, for each unit that moves, its legs and a one-line task.
 */
export function missions(world, coa, minutes = 150) {
  const copy = world.clone({ seed: world.seed, preview: true });
  copy.commit({ ...coa, units: coa.dispatch || coa.units });
  const seen = new Map(), byUnit = new Map();
  for (let m = 0; m < minutes; m++) {
    copy.step(1);
    for (const u of copy.units) {
      if (u.op !== coa.fire || !u.legs.length || seen.get(u.id) === u.legs) continue;
      seen.set(u.id, u.legs);
      const list = byUnit.get(u.id) || byUnit.set(u.id, []).get(u.id);
      for (const leg of u.legs) list.push({ pts: leg.pts, cum: leg.cum, km: leg.km, t0: leg.t0, t1: leg.t1, kind: leg.kind, status: leg.status, alt0: leg.alt0, alt1: leg.alt1, mark: leg.mark, does: leg.end?.type || "", work: leg.end?.work?.status || "" });
    }
  }
  const out = [];
  for (const u of copy.units) {
    const legs = (byUnit.get(u.id) || []).filter((l) => l.t0 < world.t + minutes);
    if (u.op !== coa.fire) continue;
    const moves = legs.filter((l) => !QUIET_KINDS.has(l.kind) && l.km > 0.05);
    // the first thing it does that changes the fire or the ground, and when
    const act = legs.find((l) => l.does === "retardant" || l.does === "water" || l.does === "work");
    const first = moves[0];
    const drops = legs.filter((l) => l.does === "retardant" || l.does === "water").length;
    let task = "HELD AT BASE", at = null;
    if (act) {
      task = act.does === "work" ? act.work : act.does === "retardant" ? "RETARDANT" : (legs[legs.indexOf(act) - 1]?.status || "").includes("AHEAD") ? "WATER AHEAD OF THE FIRE" : "WATER ON THE EDGE";
      at = act.t1;
    } else if (first) {
      // nothing dropped or built: what it ends up doing once it has got where it is going
      const last = legs[legs.length - 1];
      task = last.status || first.status;
      at = last.kind === "ground" || last.kind === "orbit" ? last.t0 : last.t1;
    }
    // for the map: the way out to the first piece of work and off it again, not every sortie after
    const upTo = act ? Math.min(moves.length, moves.indexOf(act) + 2) : Math.min(moves.length, 6);
    out.push({ id: u.id, call: u.call, type: u.type, cls: u.cls, task, at, drops, legs: moves.slice(0, Math.max(upTo, 1)), held: !first });
  }
  return out;
}
