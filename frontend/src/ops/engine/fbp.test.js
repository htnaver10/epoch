import { test } from "node:test";
import assert from "node:assert/strict";
import { FUELS, FUEL, behaviour, ellipseRate, hfiClass, hourlyFfmc, intensity, isiOf, lengthToBreadth, moistureOf, ffmcOf, slopeFactor } from "./fbp.js";

const near = (a, b, tol) => assert.ok(Math.abs(a - b) <= tol, `${a} is not within ${tol} of ${b}`);

test("ISI matches the published FWI tables", () => {
  near(isiOf(90, 20), 11.7, 0.1); // FFMC 90, 20 km/h
  near(isiOf(85, 10), 3.5, 0.1);
});

test("C-2 at ISI 12.5, BUI 64 spreads at the published rate", () => {
  const b = behaviour({ ws: 20, ffmc: 90, bui: 64 });
  near(b.fuels[FUEL.C2].head, 110 * (1 - Math.exp(-0.0282 * isiOf(90, 20))) ** 1.5, 1e-9);
  assert.ok(b.fuels[FUEL.C2].head > 15 && b.fuels[FUEL.C2].head < 22);
  assert.ok(b.fuels[FUEL.C2].back < 2 && b.fuels[FUEL.C2].back > 0);
  assert.ok(b.fuels[FUEL.D1].head < b.fuels[FUEL.M2].head && b.fuels[FUEL.M2].head < b.fuels[FUEL.C2].head);
  assert.equal(b.fuels[FUEL.NF].head, 0);
});

test("a crowning spruce fire is a class 6 fire; a backing one is not", () => {
  const b = behaviour({ ws: 30, ffmc: 92, bui: 90 });
  const c2 = b.fuels[FUEL.C2];
  assert.equal(hfiClass(intensity(c2, c2.head)), 6);
  assert.ok(hfiClass(intensity(c2, c2.back)) <= 4);
});

test("the ellipse has the head, back and flank rates it was built from", () => {
  const h = 20, b = 1, lb = 3;
  near(ellipseRate(h, b, lb, 0), h, 1e-9);
  near(ellipseRate(h, b, lb, Math.PI), b, 1e-9);
  // widest point: semi-minor axis (h + b) / (2 lb)
  let widest = 0;
  for (let t = 0; t < Math.PI; t += 0.001) widest = Math.max(widest, ellipseRate(h, b, lb, t) * Math.sin(t));
  near(widest, (h + b) / (2 * lb), 1e-3);
  near(lengthToBreadth(0), 1, 1e-9);
  assert.ok(lengthToBreadth(30) > 2.5 && lengthToBreadth(30) < 4);
});

test("fine fuels dry by day and recover at night", () => {
  near(ffmcOf(moistureOf(88)), 88, 0.1); // the two published conversions are not exact inverses
  let day = 85, night = 92;
  for (let i = 0; i < 6; i++) day = hourlyFfmc(day, 28, 20, 20);
  for (let i = 0; i < 8; i++) night = hourlyFfmc(night, 8, 90, 5);
  assert.ok(day > 90, `day ${day}`);
  assert.ok(night < 86, `night ${night}`);
});

test("slope speeds a fire uphill and never stops it downhill", () => {
  near(slopeFactor(0), 1, 1e-9);
  assert.ok(slopeFactor(0.3) > 2 && slopeFactor(0.3) < 2.6);
  assert.equal(slopeFactor(2), slopeFactor(0.7));
  assert.ok(slopeFactor(-1) >= 0.6);
  assert.equal(FUELS.length, Object.keys(FUEL).length);
});
