import { test } from "node:test";
import assert from "node:assert/strict";
import { buildTheatre, COVER, fromLatLon, toLatLon } from "./theatre.js";
import { syntheticTheatre } from "./synthetic.js";
import { FUEL } from "./fbp.js";

const theatre = buildTheatre(syntheticTheatre());

test("the grid matches the map data", () => {
  assert.equal(theatre.n, 400);
  assert.equal(theatre.grid.fuel.length, 160000);
  const [x, y] = theatre.xy(theatre.toCell(3.3, -7.1));
  assert.ok(Math.abs(x - 3.3) <= 0.08 && Math.abs(y + 7.1) <= 0.08);
  let water = 0;
  for (let i = 0; i < 160000; i++) {
    if (theatre.cover[i] === COVER.water) { water++; assert.equal(theatre.grid.fuel[i], FUEL.NF); }
    if (theatre.cover[i] === COVER.built) assert.equal(theatre.grid.fuel[i], FUEL.URB);
  }
  assert.ok(water > 5000);
});

test("the town is found, with an edge and its buildings", () => {
  assert.ok(theatre.town.cells.length > 300);
  assert.ok(theatre.town.edge.length > 30 && theatre.town.edge.length < theatre.town.cells.length);
  assert.ok(theatre.town.buildings > 1000);
  assert.ok(Math.hypot(theatre.town.cx - 12, theatre.town.cy + 2) < 1.5);
  assert.ok(theatre.town.radius > 1.5 && theatre.town.radius < 5);
});

test("lakes have scooping runs, each one over water for its whole length", () => {
  const big = theatre.lakes.filter((l) => l.runs.length);
  assert.ok(big.length >= 2, `${big.length} scoopable lakes`);
  for (const lake of big)
    for (const run of lake.runs) {
      assert.ok(run.km >= 1.4);
      const steps = Math.max(Math.abs(run.c1 - run.c0), Math.abs(run.r1 - run.r0));
      for (let i = 0; i <= steps; i++) {
        const c = run.c0 + Math.round(((run.c1 - run.c0) * i) / steps), r = run.r0 + Math.round(((run.r1 - run.r0) * i) / steps);
        assert.equal(theatre.cover[r * 400 + c], COVER.water);
      }
    }
});

test("roads route along the network", () => {
  const route = theatre.roads.route([12.2, -1.8], [-30, -2]);
  assert.ok(route, "a route exists");
  assert.ok(route.km > 41 && route.km < 48, `${route.km} km`);
  assert.ok(route.min > 25 && route.min < 50, `${route.min} min`);
  for (let i = 1; i < route.path.length; i++) assert.ok(Math.hypot(route.path[i][0] - route.path[i - 1][0], route.path[i][1] - route.path[i - 1][1]) < 1.6);
});

test("roads are breaks in the fuel", () => {
  let breaks = 0;
  for (let c = 0; c < 400; c++) if (theatre.grid.barrier[theatre.toCell(-32 + (c + 0.5) * 0.16, -2)]) breaks++;
  assert.ok(breaks > 150, `${breaks} highway cells are breaks`);
});

test("local kilometres and coordinates convert both ways", () => {
  const [lat, lon] = toLatLon(theatre, 10, -5);
  const [x, y] = fromLatLon(theatre, lat, lon);
  assert.ok(Math.abs(x - 10) < 1e-9 && Math.abs(y + 5) < 1e-9);
});
