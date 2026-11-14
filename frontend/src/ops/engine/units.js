// Response units: what each type can do, and how one moves. A unit follows legs: polylines
// with a start and end time. Aircraft fly smoothed tracks; ground units drive the road network.
import { LIMIT, RATE } from "./doctrine.js";

// cruise and work speeds in km/h, loads in litres, altitudes in metres above ground
export const SPECS = {
  ELECTRA: { cls: "air", role: "tanker", name: "L-188 Electra", short: "L188", cruise: 592, work: 250, load: 11365, agent: "retardant", reloadMin: 12, enduranceMin: 240, alt: 1200 },
  Q400: { cls: "air", role: "tanker", name: "Dash 8-400AT", short: "DH8D", cruise: 667, work: 232, load: 10000, agent: "retardant", reloadMin: 7, enduranceMin: 210, alt: 1200 },
  CL415: { cls: "air", role: "skimmer", name: "Canadair CL-415", short: "CL415", cruise: 333, work: 204, load: 6137, agent: "water", limit: LIMIT.skimmer, enduranceMin: 240, alt: 450 },
  CL215T: { cls: "air", role: "skimmer", name: "Canadair CL-215T", short: "CL2T", cruise: 300, work: 200, load: 5400, agent: "water", limit: LIMIT.skimmer, enduranceMin: 240, alt: 450 },
  FIREBOSS: { cls: "air", role: "skimmer", name: "AT-802F Fire Boss", short: "AT8T", cruise: 259, work: 185, load: 3025, agent: "water", limit: LIMIT.skimmer, enduranceMin: 210, alt: 400 },
  BIRDDOG: { cls: "air", role: "lead", name: "Turbo Commander 690", short: "AC90", cruise: 450, work: 300, load: 0, enduranceMin: 270, alt: 1700 },
  B212: { cls: "air", role: "helo", name: "Bell 212", short: "B212", cruise: 185, work: 110, load: 1590, agent: "water", limit: LIMIT.bucket, enduranceMin: 170, alt: 250, heli: true },
  AS350: { cls: "air", role: "helo", name: "Airbus AS350 B3", short: "AS50", cruise: 225, work: 110, load: 681, agent: "water", limit: LIMIT.bucket, enduranceMin: 240, alt: 250, heli: true },
  ENGINE: { cls: "ground", role: "engine", name: "Structure engine", short: "ENG", road: 80 },
  TENDER: { cls: "ground", role: "engine", name: "Water tender", short: "TND", road: 70 },
  SPU: { cls: "ground", role: "spu", name: "Structure protection unit", short: "SPU", road: 80, rate: RATE.sprinklerSetup },
  DOZER: { cls: "ground", role: "dozer", name: "Dozer group", short: "DZR", road: 60, cross: 5, rate: RATE.dozer },
  CREW: { cls: "ground", role: "crew", name: "Unit crew, 20", short: "CRW", road: 80, cross: 3, rate: RATE.crewDirect, lineRate: RATE.crewIndirect },
  IA: { cls: "ground", role: "crew", name: "Initial attack crew, 4", short: "IA", road: 80, cross: 3, rate: RATE.initialAttack, lineRate: RATE.initialAttack * 0.6, heliborne: 185 },
  MEDIC: { cls: "ground", role: "medic", name: "Ambulance", short: "MED", road: 90 },
  POLICE: { cls: "ground", role: "police", name: "RCMP unit", short: "RCMP", road: 100 },
};

export function pathLength(pts) {
  let km = 0;
  for (let i = 1; i < pts.length; i++) km += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
  return km;
}

/** Round the corners of a track (Chaikin), keeping its ends. Collinear runs stay straight. */
export function smooth(pts, rounds = 2) {
  let p = pts;
  for (let k = 0; k < rounds && p.length > 2; k++) {
    const q = [p[0]];
    for (let i = 0; i < p.length - 1; i++) {
      const a = p[i], b = p[i + 1];
      if (i > 0) q.push([a[0] * 0.75 + b[0] * 0.25, a[1] * 0.75 + b[1] * 0.25]);
      if (i < p.length - 2) q.push([a[0] * 0.25 + b[0] * 0.75, a[1] * 0.25 + b[1] * 0.75]);
    }
    q.push(p[p.length - 1]);
    p = q;
  }
  return p;
}

/** Turn radius (km) of an aircraft at a speed (km/h), banked 28 degrees. */
export const turnRadius = (speed) => Math.max(0.12, (speed / 3.6) ** 2 / (9.81 * Math.tan((28 * Math.PI) / 180)) / 1000);

/**
 * A flyable track through waypoints: every corner replaced by an arc of the turn radius
 * (less where the legs are too short for it), and a reversal flown as a turn-back to one side.
 */
export function fillet(points, radius) {
  let pts = points.filter((p, i) => i === 0 || Math.hypot(p[0] - points[i - 1][0], p[1] - points[i - 1][1]) > 1e-6);
  if (pts.length < 3) return pts;
  // a turn of more than 120 degrees is flown out to one side and back
  const wide = [pts[0]];
  for (let i = 1; i < pts.length - 1; i++) {
    const a = pts[i - 1], b = pts[i], c = pts[i + 1];
    const ux = b[0] - a[0], uy = b[1] - a[1], vx = c[0] - b[0], vy = c[1] - b[1];
    const lu = Math.hypot(ux, uy), lv = Math.hypot(vx, vy), turn = Math.acos(Math.max(-1, Math.min(1, (ux * vx + uy * vy) / (lu * lv))));
    if (turn > 2.1) {
      const side = ux * vy - uy * vx >= 0 ? 1 : -1, r = Math.min(radius, lu * 0.6, lv * 0.6);
      wide.push([b[0] + (ux / lu) * r * 0.2, b[1] + (uy / lu) * r * 0.2], [b[0] + (ux / lu) * r - (uy / lu) * side * r * 1.3, b[1] + (uy / lu) * r + (ux / lu) * side * r * 1.3], [b[0] - (uy / lu) * side * r * 2, b[1] + (ux / lu) * side * r * 2]);
    } else wide.push(b);
  }
  wide.push(pts[pts.length - 1]);
  pts = wide;
  const out = [pts[0]];
  for (let i = 1; i < pts.length - 1; i++) {
    const a = out[out.length - 1], b = pts[i], c = pts[i + 1];
    const ux = a[0] - b[0], uy = a[1] - b[1], vx = c[0] - b[0], vy = c[1] - b[1];
    const lu = Math.hypot(ux, uy), lv = Math.hypot(vx, vy);
    if (lu < 1e-6 || lv < 1e-6) continue;
    const cos = Math.max(-1, Math.min(1, (ux * vx + uy * vy) / (lu * lv))), turn = Math.PI - Math.acos(cos);
    if (turn < 0.06) {
      out.push(b);
      continue;
    }
    const half = turn / 2, t = Math.min(radius * Math.tan(half), lu * 0.48, lv * 0.48), r = t / Math.tan(half);
    const p1 = [b[0] + (ux / lu) * t, b[1] + (uy / lu) * t], p2 = [b[0] + (vx / lv) * t, b[1] + (vy / lv) * t];
    const bx = ux / lu + vx / lv, by = uy / lu + vy / lv, bl = Math.hypot(bx, by) || 1, reach = r / Math.cos(half);
    const ox = b[0] + (bx / bl) * reach, oy = b[1] + (by / bl) * reach;
    const a1 = Math.atan2(p1[1] - oy, p1[0] - ox);
    let sweep = Math.atan2(p2[1] - oy, p2[0] - ox) - a1;
    while (sweep > Math.PI) sweep -= 2 * Math.PI;
    while (sweep < -Math.PI) sweep += 2 * Math.PI;
    const steps = Math.max(2, Math.ceil(Math.abs(sweep) / 0.14));
    for (let k = 0; k <= steps; k++) out.push([ox + Math.cos(a1 + (sweep * k) / steps) * r, oy + Math.sin(a1 + (sweep * k) / steps) * r]);
  }
  out.push(pts[pts.length - 1]);
  return out;
}

const TAU = 2 * Math.PI;
const mod = (a) => ((a % TAU) + TAU) % TAU;
const ARC_STEP = 0.12; // radians between points on a turn

/**
 * The shortest flyable track from one position and heading to another, turning no tighter
 * than `r`: a turn, a straight, a turn (a Dubins path). Headings are bearings in radians,
 * clockwise from north. Returns the track as points, starting with `p0` and ending at `p1`.
 */
export function dubins(p0, h0, p1, h1, r) {
  const dx = p1[0] - p0[0], dy = p1[1] - p0[1], D = Math.hypot(dx, dy) / r;
  if (D < 1e-6) return [p0, p1];
  // work in mathematical angles (anticlockwise from east), in units of the turn radius
  const th0 = Math.PI / 2 - h0, th1 = Math.PI / 2 - h1, theta = Math.atan2(dy, dx), a = mod(th0 - theta), b = mod(th1 - theta);
  const sa = Math.sin(a), sb = Math.sin(b), ca = Math.cos(a), cb = Math.cos(b), cab = Math.cos(a - b);
  let best = null;
  const offer = (t, p, q, first, last) => {
    if (!(p >= 0) || !Number.isFinite(t + p + q)) return;
    if (!best || t + p + q < best.t + best.p + best.q) best = { t, p, q, first, last };
  };
  {
    const k = Math.atan2(cb - ca, D + sa - sb);
    offer(mod(-a + k), Math.sqrt(Math.max(0, 2 + D * D - 2 * cab + 2 * D * (sa - sb))), mod(b - k), 1, 1);
  }
  {
    const k = Math.atan2(ca - cb, D - sa + sb);
    offer(mod(a - k), Math.sqrt(Math.max(0, 2 + D * D - 2 * cab + 2 * D * (sb - sa))), mod(-b + k), -1, -1);
  }
  {
    const sq = -2 + D * D + 2 * cab + 2 * D * (sa + sb);
    if (sq >= 0) {
      const p = Math.sqrt(sq), k = Math.atan2(-ca - cb, D + sa + sb) - Math.atan2(-2, p);
      offer(mod(-a + k), p, mod(-mod(b) + k), 1, -1);
    }
  }
  {
    const sq = D * D - 2 + 2 * cab - 2 * D * (sa + sb);
    if (sq >= 0) {
      const p = Math.sqrt(sq), k = Math.atan2(ca + cb, D - sa - sb) - Math.atan2(2, p);
      offer(mod(a - k), p, mod(b - k), -1, 1);
    }
  }
  if (!best) return [p0, p1];
  const out = [p0];
  let x = p0[0], y = p0[1], th = th0;
  const arc = (angle, side) => {
    // side +1 turns left (anticlockwise), -1 right
    const cx = x - side * r * Math.sin(th), cy = y + side * r * Math.cos(th), steps = Math.max(1, Math.ceil(angle / ARC_STEP));
    for (let i = 1; i <= steps; i++) {
      const t = th + (side * angle * i) / steps;
      out.push([cx + side * r * Math.sin(t), cy - side * r * Math.cos(t)]);
    }
    th += side * angle;
    [x, y] = out[out.length - 1];
  };
  if (best.t > 1e-4) arc(best.t, best.first);
  if (best.p > 1e-4) {
    x += best.p * r * Math.cos(th);
    y += best.p * r * Math.sin(th);
    out.push([x, y]);
  }
  if (best.q > 1e-4) arc(best.q, best.last);
  out[out.length - 1] = p1;
  return out;
}

/**
 * A flyable track from a position and heading to a point, with no heading asked for on
 * arrival: one turn, then straight at it. A point inside the turn on its own side is
 * reached by turning the other way round.
 */
export function turnTo(p0, h0, p1, r) {
  let best = null;
  for (const side of [1, -1]) {
    // side +1 turns right (clockwise on the map), -1 left
    const cx = p0[0] + side * r * Math.cos(h0), cy = p0[1] - side * r * Math.sin(h0);
    const d = Math.hypot(p1[0] - cx, p1[1] - cy);
    if (d < r * 1.001) continue;
    const from = Math.atan2(p0[1] - cy, p0[0] - cx), leave = Math.atan2(p1[1] - cy, p1[0] - cx) + side * Math.acos(r / d);
    const sweep = mod(side * (from - leave)), length = sweep * r + Math.sqrt(d * d - r * r);
    if (!best || length < best.length) best = { cx, cy, from, sweep, side, length };
  }
  if (!best) return [p0, p1];
  const out = [p0], steps = Math.ceil(best.sweep / ARC_STEP);
  for (let i = 1; i <= steps; i++) {
    const a = best.from - (best.side * best.sweep * i) / steps;
    out.push([best.cx + r * Math.cos(a), best.cy + r * Math.sin(a)]);
  }
  out.push(p1);
  return out;
}

/**
 * The track of one flown leg: from where the aircraft is and the way it is heading, through
 * each waypoint. It arrives at each pointing at the next, and at the last on `heading` if
 * one is asked for. With `track`, the points after the first are flown exactly as given
 * (an orbit, a holding pattern) and only the way onto them is worked out.
 */
export function flightTrack(from, going, points, speed, { heading = null, track = false } = {}) {
  const r = turnRadius(speed);
  const bearing = (a, b) => Math.atan2(b[0] - a[0], b[1] - a[1]);
  let pos = from, h = going ? Math.atan2(going[0], going[1]) : bearing(from, points[0]);
  const out = [from];
  const goals = track ? [points[0]] : points;
  for (let i = 0; i < goals.length; i++) {
    const to = goals[i];
    if (Math.hypot(to[0] - pos[0], to[1] - pos[1]) < 1e-4) continue;
    const next = track ? points[1] : goals[i + 1];
    const want = next ? bearing(to, next) : heading;
    let leg;
    if (want == null) leg = turnTo(pos, h, to, r);
    else {
      leg = dubins(pos, h, to, want, r);
      // a steeper bank where the gentle one would mean a long way round
      const tight = dubins(pos, h, to, want, r * 0.55);
      if (pathLength(tight) < pathLength(leg) * 0.7) leg = tight;
    }
    for (let k = 1; k < leg.length; k++) out.push(leg[k]);
    const n = out.length;
    h = want ?? bearing(out[n - 2], out[n - 1]);
    pos = to;
  }
  if (track) for (let k = 1; k < points.length; k++) out.push(points[k]);
  return out;
}

/** A circle of points to orbit on. */
export function orbit(cx, cy, radius, startAngle = 0, turns = 1, clockwise = true) {
  const steps = Math.ceil(48 * turns), pts = [];
  for (let i = 0; i <= steps; i++) {
    const a = startAngle + ((clockwise ? 1 : -1) * i * 2 * Math.PI) / 48;
    pts.push([cx + Math.sin(a) * radius, cy + Math.cos(a) * radius]);
  }
  return pts;
}

/** A holding pattern: a racetrack of two straights along a bearing joined by half turns. */
export function racetrack(cx, cy, bearing, length, radius) {
  const ax = Math.sin(bearing), ay = Math.cos(bearing), px = ay, py = -ax, pts = [];
  const put = (along, across) => pts.push([cx + ax * along + px * across, cy + ay * along + py * across]);
  put(-length / 2, -radius);
  put(length / 2, -radius);
  for (let k = 1; k < 12; k++) put(length / 2 + Math.sin((k * Math.PI) / 12) * radius, -Math.cos((k * Math.PI) / 12) * radius);
  put(length / 2, radius);
  put(-length / 2, radius);
  for (let k = 1; k < 12; k++) put(-length / 2 - Math.sin((k * Math.PI) / 12) * radius, Math.cos((k * Math.PI) / 12) * radius);
  put(-length / 2, -radius);
  return pts;
}

let serial = 0;
export function makeUnit(type, call, home, extra = {}) {
  const spec = SPECS[type];
  return {
    id: ++serial, call, type, role: spec.role, cls: spec.cls, x: home.x, y: home.y, alt: 0, hdg: extra.hdg ?? 90, speed: 0,
    home, status: spec.cls === "air" ? "AT BASE" : "AT BASE", state: "idle", // idle | busy | committed
    load: spec.load || 0, fuel: spec.enduranceMin || 0, legs: [], li: 0, work: null, op: 0, job: null,
    sorties: 0, delivered: 0, trail: [], nextIdle: 0, bank: 0, climb: 0, ...extra,
  };
}

/**
 * Give a unit legs to follow from time t. Each: { pts } or { hold: minutes }, with
 * speed (km/h), status, kind, alt: [from, to], end (an event raised when the leg completes).
 * A leg marked `smooth` is flown: it starts on the heading the aircraft has and turns no
 * tighter than its speed allows. `heading` (a bearing, radians) is the way it must be
 * pointing at the end; `track` says the points are a pattern to join and then follow.
 */
export function schedule(unit, t, specs, { append = false, round = true } = {}) {
  let at = append && unit.legs.length ? unit.legs[unit.legs.length - 1].t1 : t;
  let from = append && unit.legs.length ? unit.legs[unit.legs.length - 1].pts.at(-1) : [unit.x, unit.y];
  let alt = append && unit.legs.length ? unit.legs[unit.legs.length - 1].alt1 : unit.alt;
  // the direction the unit is travelling in as each leg begins, as a unit vector (or null at rest)
  const lastOf = (leg) => {
    const p = leg.pts, n = p.length;
    for (let i = n - 1; i > 0; i--) {
      const dx = p[i][0] - p[i - 1][0], dy = p[i][1] - p[i - 1][1], d = Math.hypot(dx, dy);
      if (d > 1e-6) return [dx / d, dy / d];
    }
    return null;
  };
  const rad = (unit.hdg * Math.PI) / 180;
  let going = append && unit.legs.length ? lastOf(unit.legs[unit.legs.length - 1]) : unit.cls === "air" && unit.alt > 0 ? [Math.sin(rad), Math.cos(rad)] : null;
  if (!append) {
    unit.legs = [];
    unit.li = 0;
  }
  for (const s of specs) {
    let pts = s.pts ? (s.pts[0][0] === from[0] && s.pts[0][1] === from[1] ? s.pts : [from, ...s.pts]) : [from, from];
    if (round && s.smooth && unit.cls === "air") pts = flightTrack(from, going, pts.slice(1), s.speed, { heading: s.heading, track: s.track });
    else if (round && s.smooth) pts = smooth(pts);
    const cum = [0];
    for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    const km = cum[cum.length - 1];
    const minutes = s.hold != null ? s.hold : (km / s.speed) * 60;
    const a1 = s.alt ? s.alt[1] : alt;
    unit.legs.push({ pts, cum, km, t0: at, t1: at + minutes, kind: s.kind || "move", status: s.status || "", alt0: s.alt ? s.alt[0] ?? alt : alt, alt1: a1, speed: s.hold != null ? 0 : s.speed, end: s.end || null, mark: s.mark || null });
    at += minutes;
    from = pts[pts.length - 1];
    alt = a1;
    going = lastOf(unit.legs[unit.legs.length - 1]) || going;
  }
  return at;
}

/** Move a unit to time t, collecting the events of the legs it completed. */
export function advanceUnit(unit, t, events) {
  const legs = unit.legs;
  while (unit.li < legs.length && t >= legs[unit.li].t1) {
    const leg = legs[unit.li++], end = leg.pts[leg.pts.length - 1];
    unit.x = end[0];
    unit.y = end[1];
    unit.alt = leg.alt1;
    if (leg.end) events.push({ unit, at: leg.t1, ...leg.end });
  }
  if (unit.li >= legs.length) {
    if (legs.length) {
      unit.legs = [];
      unit.li = 0;
      unit.speed = 0;
    }
    return;
  }
  const leg = legs[unit.li];
  if (t <= leg.t0) return;
  const f = (t - leg.t0) / (leg.t1 - leg.t0), d = f * leg.km;
  unit.status = leg.status || unit.status;
  unit.speed = leg.speed;
  unit.alt = leg.alt0 + (leg.alt1 - leg.alt0) * f;
  if (leg.km <= 0) return;
  let i = 1;
  while (i < leg.cum.length - 1 && leg.cum[i] < d) i++;
  const a = leg.pts[i - 1], b = leg.pts[i], span = leg.cum[i] - leg.cum[i - 1] || 1, k = (d - leg.cum[i - 1]) / span;
  unit.x = a[0] + (b[0] - a[0]) * k;
  unit.y = a[1] + (b[1] - a[1]) * k;
  if (unit.cls !== "air") {
    if (b[0] !== a[0] || b[1] !== a[1]) unit.hdg = (Math.atan2(b[0] - a[0], b[1] - a[1]) * 180) / Math.PI;
    return;
  }
  // an aircraft points where its track is about to go, and banks with the turn it is in
  const look = Math.min(0.35, leg.km * 0.25), p1 = pointAt(leg, d + look), p2 = pointAt(leg, d + 2 * look);
  const h1 = Math.atan2(p1[0] - unit.x, p1[1] - unit.y), h2 = Math.atan2(p2[0] - p1[0], p2[1] - p1[1]);
  if (Math.hypot(p1[0] - unit.x, p1[1] - unit.y) > 1e-4) {
    unit.hdg = (h1 * 180) / Math.PI;
    let turn = h2 - h1;
    while (turn > Math.PI) turn -= 2 * Math.PI;
    while (turn < -Math.PI) turn += 2 * Math.PI;
    const v = leg.speed / 3.6, curvature = look > 0 && Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) > 1e-4 ? turn / (look * 1000) : 0;
    unit.bank = Math.max(-40, Math.min(40, (Math.atan((v * v * curvature) / 9.81) * 180) / Math.PI));
  } else unit.bank = 0;
  unit.climb = leg.km > 0 ? (leg.alt1 - leg.alt0) / (leg.km * 1000) : 0;
}

/** The point a distance along a leg (clamped to its ends). */
export function pointAt(leg, d) {
  const { pts, cum } = leg;
  if (d <= 0) return pts[0];
  if (d >= leg.km) return pts[pts.length - 1];
  let i = 1;
  while (i < cum.length - 1 && cum[i] < d) i++;
  const a = pts[i - 1], b = pts[i], k = (d - cum[i - 1]) / (cum[i] - cum[i - 1] || 1);
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k];
}

/** The rest of a unit's planned track at time t, from where it is: for drawing its plan. */
export function remainingTrack(unit, t) {
  const out = [];
  for (let i = unit.li; i < unit.legs.length; i++) {
    const leg = unit.legs[i];
    let pts = leg.pts;
    if (i === unit.li && t > leg.t0) {
      const d = ((t - leg.t0) / (leg.t1 - leg.t0)) * leg.km;
      pts = [[unit.x, unit.y], ...leg.pts.filter((_, k) => leg.cum[k] > d)];
    }
    out.push({ kind: leg.kind, pts, t0: leg.t0, t1: leg.t1, mark: leg.mark, status: leg.status, alt0: leg.alt0, alt1: leg.alt1, cum: leg.cum, km: leg.km, first: i === unit.li });
  }
  return out;
}

/** Where a unit will be at a time, by the legs it has now: [x, y, altitude]. */
export function positionAt(unit, time) {
  for (let i = unit.li; i < unit.legs.length; i++) {
    const leg = unit.legs[i];
    if (time > leg.t1) continue;
    const f = Math.max(0, Math.min(1, (time - leg.t0) / (leg.t1 - leg.t0 || 1))), p = pointAt(leg, f * leg.km);
    return [p[0], p[1], leg.alt0 + (leg.alt1 - leg.alt0) * f];
  }
  const last = unit.legs[unit.legs.length - 1];
  return last ? [...last.pts[last.pts.length - 1], last.alt1] : [unit.x, unit.y, unit.alt];
}

/**
 * Put a full turn into what a unit is flying, where it is now: the rest of its legs follow
 * on after it, each later by the time the turn takes. Returns that time in minutes.
 */
export function insertTurn(unit, t, status) {
  const leg = unit.legs[unit.li];
  if (!leg || leg.km <= 0) return 0;
  const speed = leg.speed || 200, r = turnRadius(speed), h = (unit.hdg * Math.PI) / 180;
  // a turn to the right, back to this point and this heading
  const cx = unit.x + r * Math.cos(h), cy = unit.y - r * Math.sin(h), from = Math.atan2(unit.y - cy, unit.x - cx);
  const circle = [];
  for (let i = 0; i <= 48; i++) circle.push([cx + r * Math.cos(from - (i * TAU) / 48), cy + r * Math.sin(from - (i * TAU) / 48)]);
  circle[0] = [unit.x, unit.y];
  circle[48] = [unit.x, unit.y];
  const build = (pts, t0, minutes, fields) => {
    const cum = [0];
    for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    return { ...fields, pts, cum, km: cum[cum.length - 1], t0, t1: t0 + minutes };
  };
  const minutes = ((TAU * r) / speed) * 60;
  const done = ((t - leg.t0) / (leg.t1 - leg.t0)) * leg.km;
  const rest = [[unit.x, unit.y], ...leg.pts.filter((_, k) => leg.cum[k] > done)];
  const legs = [
    build(circle, t, minutes, { kind: "giveway", status, alt0: unit.alt, alt1: unit.alt, speed, end: null, mark: null }),
    build(rest, t + minutes, Math.max(0.01, leg.t1 - t), { ...leg, alt0: unit.alt }),
  ];
  for (let i = unit.li + 1; i < unit.legs.length; i++) legs.push({ ...unit.legs[i], t0: unit.legs[i].t0 + minutes, t1: unit.legs[i].t1 + minutes });
  unit.legs = legs;
  unit.li = 0;
  return minutes;
}
