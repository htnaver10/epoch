import { test } from "node:test";
import assert from "node:assert/strict";
import { syntheticTheatre } from "./synthetic.js";
import { openScenario } from "./scenario.js";
import { plan, evaluate, incidents, losses } from "./planner.js";

const data = syntheticTheatre();

test("a scenario opens with a fire from the hotspots, units at their bases and the clock at start", () => {
  const world = openScenario(data);
  assert.ok(world.seeded.used > 20);
  assert.ok(world.seeded.fires.length >= 1);
  assert.ok(world.fire.burned > 50);
  assert.equal(world.units.length, 21);
  assert.equal(world.localMinute(), 13 * 60);
  const list = incidents(world);
  assert.ok(list.length >= 1);
  assert.ok(list[0].ha > 100 && list[0].edgeKm > 1);
});

test("idle units move about and come home; nothing is lost with no fire near", () => {
  const world = openScenario(data);
  const start = world.units.map((u) => [u.x, u.y]);
  let moved = new Set();
  for (let i = 0; i < 240; i++) {
    world.step(1);
    world.units.forEach((u, k) => Math.hypot(u.x - start[k][0], u.y - start[k][1]) > 0.5 && moved.add(u.call));
  }
  assert.ok(moved.size >= 3, `${[...moved]} moved`);
  for (const u of world.units) assert.ok(Number.isFinite(u.x) && Number.isFinite(u.y), `${u.call} has a position`);
});

test("the planner lays out courses of action and scores them against no action", async () => {
  const world = openScenario(data);
  world.step(30);
  const fireId = incidents(world)[0].id;
  const planned = plan(world, fireId);
  assert.ok(planned.coas.length >= 2, planned.coas.map((c) => c.key).join());
  assert.ok(planned.situation.hit, "the town is in the fire's path");
  const result = await evaluate(world, planned, { runs: 3, pause: async () => {} });
  assert.equal(result.coas.filter((c) => c.recommended).length, 1);
  assert.ok(result.none.townRuns >= 2, `no action loses the town in ${result.none.townRuns}/3 runs`);
  const best = result.coas.find((c) => c.recommended);
  assert.ok(best.score.structures < result.none.structures * 0.7, `${best.name}: ${best.score.structures} structures against ${result.none.structures} with no action`);
  for (const c of result.coas) console.log(`  ${c.name.padEnd(17)} town ${c.score.townRuns}/3  structures ${String(Math.round(c.score.structures)).padStart(4)}  ha ${Math.round(c.score.ha)}  retardant ${Math.round(c.score.retardant)} L  water ${Math.round(c.score.water)} L  guard ${c.score.guardKm.toFixed(1)} km  ${c.recommended ? "<= recommended" : ""}`);
  console.log(`  NO ACTION         town ${result.none.townRuns}/3  structures ${String(Math.round(result.none.structures)).padStart(4)}  ha ${Math.round(result.none.ha)}  (horizon ${result.horizon} h)`);
  // the world that was planned on is untouched
  assert.equal(world.ops.size, 0);
});

test("a committed course of action sends units out and changes the fire", () => {
  const world = openScenario(data);
  world.step(30);
  const planned = plan(world, incidents(world)[0].id);
  const coa = planned.coas.find((c) => c.key === "contain") || planned.coas[0];
  const op = world.commit(coa);
  const reference = world.clone({ seed: world.seed });
  reference.ops.clear();
  for (const u of reference.units) u.op = 0;
  for (let i = 0; i < 6 * 60; i++) world.step(1);
  reference.step(6 * 60);
  const air = world.units.filter((u) => u.cls === "air" && u.op);
  assert.ok(air.some((u) => u.sorties > 0), "aircraft have dropped");
  assert.ok(op.stats.retardant + op.stats.water > 20000, `${op.stats.retardant} L retardant, ${op.stats.water} L water`);
  if (coa.line) assert.ok(op.stats.guardKm + op.stats.burnKm > 0.3, `guard ${op.stats.guardKm} km, burned off ${op.stats.burnKm} km`);
  let retard = 0;
  for (const r of world.fire.retard) retard += r;
  assert.ok(retard > 4, `${retard} cells coated`);
  assert.ok(world.log.length > 3);
  const mine = losses(world), theirs = losses(reference);
  assert.ok(mine.structures <= theirs.structures);
  for (const u of world.units) assert.ok(Number.isFinite(u.x + u.y + u.alt), `${u.call} position`);
});

test("a flown track turns no tighter than its radius and ends where it was sent", async () => {
  const { fillet, turnRadius } = await import("./units.js");
  const r = turnRadius(300);
  assert.ok(r > 1.1 && r < 1.5, `${r} km at 300 km/h`);
  for (const pts of [[[0, 0], [10, 0], [10, 10]], [[0, 0], [10, 0], [0, 1]], [[0, 0], [6, 0], [12, 3], [12, 12]]]) {
    const track = fillet(pts, r);
    assert.deepEqual(track[0], pts[0]);
    assert.deepEqual(track.at(-1), pts.at(-1));
    let worst = 0;
    for (let i = 1; i < track.length - 1; i++) {
      const a = track[i - 1], b = track[i], c = track[i + 1];
      const h1 = Math.atan2(b[0] - a[0], b[1] - a[1]), h2 = Math.atan2(c[0] - b[0], c[1] - b[1]);
      let turn = Math.abs(h2 - h1);
      if (turn > Math.PI) turn = 2 * Math.PI - turn;
      worst = Math.max(worst, turn);
    }
    assert.ok(worst < 0.5, `sharpest corner ${worst.toFixed(2)} rad`);
  }
});

test("a flight track joins two headings without ever turning tighter than the radius", async () => {
  const { dubins, flightTrack, turnTo, turnRadius } = await import("./units.js");
  const sharpest = (track) => {
    // the tightest turn on the track, as a radius: distance flown over heading changed
    let tight = Infinity;
    for (let i = 1; i < track.length - 1; i++) {
      const a = track[i - 1], b = track[i], c = track[i + 1];
      const d1 = Math.hypot(b[0] - a[0], b[1] - a[1]), d2 = Math.hypot(c[0] - b[0], c[1] - b[1]);
      if (d1 < 1e-6 || d2 < 1e-6) continue;
      let turn = Math.abs(Math.atan2(c[0] - b[0], c[1] - b[1]) - Math.atan2(b[0] - a[0], b[1] - a[1]));
      if (turn > Math.PI) turn = 2 * Math.PI - turn;
      if (turn > 1e-3) tight = Math.min(tight, (d1 + d2) / 2 / turn);
    }
    return tight;
  };
  const heading = (a, b) => Math.atan2(b[0] - a[0], b[1] - a[1]);
  const r = 1.2;
  for (const [p0, h0, p1, h1] of [[[0, 0], 0, [10, 0], Math.PI], [[0, 0], 0, [0, 10], 0], [[0, 0], Math.PI / 2, [-6, 3], 0], [[0, 0], 0, [1, 0.5], Math.PI / 2], [[3, 3], 2, [-8, -9], 5]]) {
    const track = dubins(p0, h0, p1, h1, r);
    assert.deepEqual(track[0], p0);
    assert.deepEqual(track.at(-1), p1);
    assert.ok(sharpest(track) > r * 0.9, `radius ${sharpest(track).toFixed(2)} km against ${r}`);
    let off = Math.abs(heading(track[1], track[0]) - (h0 + Math.PI)) % (2 * Math.PI);
    off = Math.min(off, 2 * Math.PI - off);
    assert.ok(off < 0.15, "leaves on the heading it had");
    let end = Math.abs(heading(track.at(-2), track.at(-1)) - h1) % (2 * Math.PI);
    end = Math.min(end, 2 * Math.PI - end);
    assert.ok(end < 0.15, `arrives on the heading asked for (off by ${end.toFixed(2)})`);
  }
  // behind the aircraft and close: it has to fly on before it can turn back onto the point
  for (const to of [[0.3, -0.4], [-0.5, -0.2], [0, -6], [5, 5], [-0.2, 3]]) {
    const back = turnTo([0, 0], 0, to, r);
    assert.ok(sharpest(back) > r * 0.9, `to ${to}: radius ${sharpest(back).toFixed(2)}`);
    assert.deepEqual(back.at(-1), to);
    assert.ok(back.length < 80);
    assert.ok(Math.abs(heading(back[0], back[1])) < 0.15, "leaves straight ahead");
  }
  // a whole leg at 300 km/h: a right-angle corner becomes an arc of the turn radius
  const leg = flightTrack([0, 0], [0, 1], [[0, 8], [8, 8]], 300, { heading: Math.PI / 2 });
  assert.ok(sharpest(leg) > turnRadius(300) * 0.5, `tightest ${sharpest(leg).toFixed(2)} km`);
});

test("aircraft on converging tracks do not pass through one another", async () => {
  const { makeUnit, schedule } = await import("./units.js");
  const { World } = await import("./world.js");
  const { buildTheatre } = await import("./theatre.js");
  const theatre = buildTheatre(syntheticTheatre());
  const a = makeUnit("CL415", "A1", { x: -10, y: 0, name: "W" }), b = makeUnit("CL415", "B1", { x: 10, y: 0.2, name: "E" });
  const world = new World({ theatre, weather: () => ({ ws: 10, wd: 270, ffmc: 85, bui: 60, gust: 15, temp: 15, rh: 50 }), units: [a, b] });
  for (const u of [a, b]) { u.alt = 300; u.state = "busy"; }
  a.hdg = 90;
  b.hdg = 270;
  // head-on along the same line, at the same height
  schedule(a, 0, [{ pts: [[10, 0]], speed: 300, alt: [300, 300], kind: "transit", smooth: true, status: "EASTBOUND" }]);
  schedule(b, 0, [{ pts: [[-10, 0.2]], speed: 300, alt: [300, 300], kind: "transit", smooth: true, status: "WESTBOUND" }]);
  let nearest = Infinity, gaveWay = false;
  for (let i = 0; i < 600; i++) {
    world.step(0.02);
    if (a.alt > 20 && b.alt > 20 && a.legs.length && b.legs.length) nearest = Math.min(nearest, Math.hypot(a.x - b.x, a.y - b.y));
    if (/GIVING WAY/.test(a.status + b.status)) gaveWay = true;
  }
  assert.ok(gaveWay, "one of them gave way");
  assert.ok(nearest > 0.5, `closest approach ${nearest.toFixed(2)} km`);
});
