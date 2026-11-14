import { test } from "node:test";
import assert from "node:assert/strict";
import { FireSim, UNBURNED, lineCells } from "./fire.js";
import { FUEL, behaviour } from "./fbp.js";

const N = 240, CELL = 0.16;
function grid(fill = FUEL.C2) {
  return { n: N, cell: CELL, fuel: new Uint8Array(N * N).fill(fill), barrier: new Uint8Array(N * N), elev: new Float32Array(N * N) };
}
const WX = { ws: 20, wd: 270, ffmc: 90, bui: 80 }; // wind from the west: the fire runs east
const steady = () => WX;
const at = (c, r) => r * N + c;
/** Farthest burned cell from the origin along +x, -x and across, in metres. */
function extent(sim, c0, r0) {
  let east = 0, west = 0, north = 0;
  for (let i = 0; i < N * N; i++) {
    if (sim.arrival[i] === UNBURNED) continue;
    const c = i % N, r = Math.floor(i / N);
    east = Math.max(east, c - c0);
    west = Math.max(west, c0 - c);
    north = Math.max(north, r - r0);
  }
  return { east: east * 160, west: west * 160, north: north * 160 };
}

test("a fire in uniform fuel grows as the FBP ellipse", () => {
  const sim = new FireSim(grid(), steady, { spotting: false });
  sim.ignite(at(40, 120));
  sim.advance(180);
  const b = behaviour(WX).fuels[FUEL.C2];
  const e = extent(sim, 40, 120);
  const within = (got, want, tol) => assert.ok(Math.abs(got - want) <= tol * want + 240, `${got} m against ${want} m`);
  within(e.east, b.head * 180, 0.08);
  within(e.west, b.back * 180, 0.25);
  within(e.north, ((b.head + b.back) / 2 / b.lb) * 180, 0.12);
  // symmetric about the wind axis
  let south = 0;
  for (let i = 0; i < N * N; i++) if (sim.arrival[i] !== UNBURNED) south = Math.max(south, 120 - Math.floor(i / N));
  assert.ok(Math.abs(south * 160 - e.north) <= 160);
});

test("arrival times do not depend on how the run is stepped", () => {
  const a = new FireSim(grid(), steady, { spotting: false }), b = new FireSim(grid(), steady, { spotting: false });
  a.ignite(at(40, 120));
  b.ignite(at(40, 120));
  a.advance(150);
  for (let t = 1; t <= 150; t += 1) b.advance(t);
  let worst = 0, differ = 0;
  for (let i = 0; i < N * N; i++) {
    if ((a.arrival[i] === UNBURNED) !== (b.arrival[i] === UNBURNED)) differ++;
    else if (a.arrival[i] !== UNBURNED) worst = Math.max(worst, Math.abs(a.arrival[i] - b.arrival[i]));
  }
  assert.equal(differ, 0);
  assert.ok(worst < 1e-3, `worst ${worst}`);
});

test("water and rock do not burn, and a fire goes round them", () => {
  const g = grid();
  for (let r = 112; r <= 128; r++) for (let c = 70; c < 76; c++) g.fuel[at(c, r)] = FUEL.NF; // a lake across the fire's path
  const sim = new FireSim(g, steady, { spotting: false });
  sim.ignite(at(40, 120));
  sim.advance(1500);
  for (let r = 112; r <= 128; r++) assert.equal(sim.arrival[at(72, r)], UNBURNED);
  assert.notEqual(sim.arrival[at(80, 120)], UNBURNED, "the fire should wrap round the lake");
  const open = new FireSim(grid(), steady, { spotting: false });
  open.ignite(at(40, 120));
  open.advance(1500);
  // behind the lake the fire has to come in from the side, at the flank's pace
  assert.ok(sim.arrival[at(80, 120)] > open.arrival[at(80, 120)] + 300, "and arrive hours later than over open ground");
});

test("a retardant line holds a flank but not a crowning head", () => {
  const flank = new FireSim(grid(), steady, { spotting: false });
  flank.layRetardant(lineCells(N, 0, 126, N - 1, 126));
  flank.ignite(at(40, 120));
  flank.advance(900);
  let reached = 0, crossed = 0;
  for (let c = 0; c < N; c++) {
    if (flank.arrival[at(c, 125)] !== UNBURNED) reached++;
    if (flank.arrival[at(c, 127)] !== UNBURNED || flank.arrival[at(c, 126)] !== UNBURNED) crossed++;
  }
  assert.ok(reached > 20, "the flank reached the line");
  assert.equal(crossed, 0, "and did not cross: flank intensity is under the retardant limit");
  const head = new FireSim(grid(), steady, { spotting: false });
  head.layRetardant(lineCells(N, 70, 0, 70, N - 1));
  head.ignite(at(40, 120));
  head.advance(900);
  assert.notEqual(head.arrival[at(80, 120)], UNBURNED, "a class 6 head burns through");
});

test("a knocked-down edge comes back unless a crew mops it up", () => {
  const calm = () => ({ ws: 15, wd: 270, ffmc: 91, bui: 60 }); // aspen: a class 3 head, workable by crews
  const run = (mop) => {
    const sim = new FireSim(grid(FUEL.D1), calm, { spotting: false });
    sim.ignite(at(120, 120));
    sim.advance(240);
    const edge = sim.perimeter();
    assert.ok(edge.length > 4, `edge of ${edge.length}`);
    assert.equal(sim.dropWater(edge, { minutes: 45, limit: 2000 }), edge.length, "every edge cell knocked down");
    if (mop) assert.equal(sim.mopUp(edge), edge.length, "every edge cell mopped up");
    const before = sim.burned;
    sim.advance(160);
    const paused = sim.burned - before;
    sim.advance(400);
    return { paused, grew: sim.burned - before };
  };
  const left = run(false), mopped = run(true);
  assert.equal(left.paused, 0);
  assert.ok(left.grew > 0, "rekindles");
  assert.equal(mopped.grew, 0, "stays out");
});

test("a clone runs on its own, and the same seed repeats", () => {
  const sim = new FireSim(grid(), () => ({ ws: 30, wd: 270, ffmc: 93, bui: 100 }), { seed: 7 });
  sim.ignite(at(30, 120));
  sim.advance(60);
  const a = sim.clone(), b = sim.clone();
  const burnedThen = sim.burned;
  a.advance(240);
  b.advance(240);
  assert.equal(sim.burned, burnedThen);
  assert.equal(a.burned, b.burned);
  assert.ok(a.burned > burnedThen);
  const c = sim.clone({ seed: 99 });
  c.advance(240);
  assert.notEqual(c.burned, a.burned, "spotting differs with the seed");
});

test("lines are four-connected", () => {
  for (const [c0, r0, c1, r1] of [[0, 0, 9, 3], [5, 5, 5, 20], [10, 2, 1, 30], [3, 3, 3, 3], [20, 20, 2, 19]]) {
    const cells = lineCells(N, c0, r0, c1, r1);
    assert.equal(cells[0], at(c0, r0));
    assert.equal(cells.at(-1), at(c1, r1));
    for (let i = 1; i < cells.length; i++) {
      const d = Math.abs((cells[i] % N) - (cells[i - 1] % N)) + Math.abs(Math.floor(cells[i] / N) - Math.floor(cells[i - 1] / N));
      assert.equal(d, 1);
    }
  }
});
