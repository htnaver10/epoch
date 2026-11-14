import { test } from "node:test";
import assert from "node:assert/strict";
import { buildTheatre } from "./theatre.js";
import { syntheticTheatre } from "./synthetic.js";
import { FireSim, UNBURNED } from "./fire.js";
import { project, impact, routeTo, threat } from "./forecast.js";
import { planLine, KIND } from "./line.js";

const theatre = buildTheatre(syntheticTheatre());
const weather = () => ({ ws: 22, wd: 265, ffmc: 91, bui: 100 });
function burning() {
  const fire = new FireSim(theatre.grid, weather, { seed: 4 });
  fire.ignite(theatre.nearestFuel(-6, -3));
  fire.advance(120);
  return fire;
}

test("the forecast says when and by which route the town is reached", () => {
  const fire = burning();
  const forecast = project(fire, 24);
  const hit = impact(forecast, theatre.town.cells);
  assert.ok(hit.minutes > 120 && hit.minutes < 24 * 60, `town reached in ${hit.minutes} min`);
  const path = routeTo(forecast, hit.cell);
  assert.ok(path.length > 20);
  assert.ok(fire.arrival[path[0]] !== UNBURNED, "the route starts on today's perimeter");
  assert.equal(path.at(-1), hit.cell);
  const weight = threat(forecast, theatre.isTown);
  let worst = 0, cellsOn = 0;
  for (const u of fire.front) if (weight[u] > 0) { cellsOn++; worst = Math.max(worst, weight[u]); }
  assert.ok(cellsOn > 0 && worst > 100, "the head of the fire carries the threat");
  assert.ok(weight[path[0]] > 0, "and the route to town starts at a cell that carries some");
});

test("the control line stands between the fire and the town, and can be built in time", () => {
  const fire = burning();
  const forecast = project(fire, 30);
  const line = planLine(theatre, fire, forecast, { reachKm: 12, leadMin: 90 });
  assert.ok(line && line.feasible, line?.reason);
  assert.ok(line.km > 3 && line.km < 60, `${line.km} km`);
  assert.ok(line.naturalKm + line.roadKm > 0, "it uses the lakes and the highway");
  line.cells.forEach((k, i) => {
    assert.equal(fire.arrival[k] <= fire.t && line.kinds[i] !== KIND.natural, false, "not through the black");
    if (line.kinds[i] !== KIND.natural) assert.ok(forecast.arrival[k] - fire.t >= 90, "built before the fire arrives");
    assert.equal(theatre.isTown[k], 0);
  });
  // guard, retardant and a burned-off strip: the projected fire no longer takes the town,
  // though the head arriving is a crown fire no line would hold on its own
  const lostWithout = theatre.town.cells.filter((c) => forecast.arrival[c] !== UNBURNED).length;
  assert.ok(lostWithout > 50, `${lostWithout} town cells burn with no action`);
  const work = line.cells.filter((_, i) => line.kinds[i] !== KIND.natural);
  const bare = fire.clone({ spotting: false });
  bare.buildGuard(work, 2);
  bare.layRetardant(work);
  bare.advance(fire.t + 30 * 60);
  assert.ok(theatre.town.cells.filter((c) => bare.arrival[c] !== UNBURNED).length > lostWithout * 0.5, "a coated guard alone does not stop a crown fire");
  const held = fire.clone({ spotting: false });
  held.buildGuard(work, 2);
  held.layRetardant(work);
  assert.ok(held.burnOut(line.band.flat()) > work.length, "a strip is burned off");
  held.advance(fire.t + 30 * 60);
  const lost = theatre.town.cells.filter((c) => held.arrival[c] !== UNBURNED).length;
  assert.ok(lost < lostWithout * 0.1, `${lost} town cells burn behind the line against ${lostWithout}`);
  const [tx, ty] = [theatre.town.cx, theatre.town.cy];
  assert.equal(line.townSide(tx, ty), true);
  assert.equal(line.townSide(-6, -3), false);
});

test("no line is asked for when the town is not threatened", () => {
  const fire = new FireSim(theatre.grid, () => ({ ws: 25, wd: 90, ffmc: 91, bui: 100 }), { seed: 4 }); // wind from the east: away from town
  fire.ignite(theatre.nearestFuel(-20, -3));
  fire.advance(60);
  assert.equal(planLine(theatre, fire, project(fire, 6), { reachKm: 8 }), null);
});
