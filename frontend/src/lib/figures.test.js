import test from "node:test";
import assert from "node:assert/strict";
import { num, signed, pad2, classOf, seasonFigures, criticalPeriods } from "./figures.js";

const seasons = { season: ["06-01", "08-31"], season_days: 92 };
const base = { kind: "high", record: "high", third: "upper", likeForLike: 4.02, naive: 53.1, naiveLike: 4.02, naiveWindow: 92, window: 92, partial: false, crossPct: 1 };

test("figures are worded one way", () => {
  assert.equal(num(4.02), "4.0");
  assert.equal(num(53.1), "53");
  assert.equal(num(0.184), "0.18");
  assert.equal(num(null), "–");
  assert.equal(signed(2.04), "+2.0");
  assert.equal(signed(-2.5), "−2.5");
  assert.equal(pad2(4), "04");
});

test("a season's class names the record seasons and the thirds of the normal range", () => {
  assert.equal(classOf(base), "Record high");
  assert.equal(classOf({ ...base, record: null }), "Unusually high");
  assert.equal(classOf({ kind: "low", record: "low" }), "Record low");
  assert.equal(classOf({ kind: "typical", third: "upper" }), "Above median, in range");
  assert.equal(classOf({ kind: "typical", third: "middle" }), "Typical");
});

test("the sensor factor divides like by like when VIIRS stops early", () => {
  const whole = seasonFigures(seasons, base, 2023);
  assert.equal(whole.window, "");
  assert.ok(Math.abs(whole.factor - 53.1 / 4.02) < 1e-9);
  assert.equal(whole.cross, "ok · P100");
  // 2026: MODIS has all 92 days, VIIRS is archived through Jul 31 (61 days)
  const short = seasonFigures(seasons, { ...base, kind: "typical", likeForLike: 1.8, naive: 12.4, naiveLike: 2.35, naiveWindow: 61, crossPct: 0.64 }, 2026);
  assert.equal(short.window, "thru Jul 31");
  assert.ok(Math.abs(short.factor - 12.4 / 2.35) < 1e-9); // not 12.4 / 1.8
  assert.equal(short.cross, "ok · P64");
  assert.equal(seasonFigures(seasons, { ...base, naive: null, crossPct: null }, 2005).hasNaive, false);
  assert.equal(seasonFigures(seasons, { ...base, kind: "mixed", crossPct: 0.3 }, 2023).cross, "disagrees · P30");
});

test("critical periods: the usual peak weeks and a season's busiest week", () => {
  // three seasons, four weeks: the usual (median) week is [1, 10, 8, 2]
  const calendar = [
    { year: 2021, cells: [0, 9, 8, 2] },
    { year: 2022, cells: [1, 10, 7, 3] },
    { year: 2023, cells: [2, 12, 30, 1] },
  ];
  const { usual, peak, busiest } = criticalPeriods(seasons, calendar, 2023);
  assert.deepEqual(usual, [1, 10, 8, 2]);
  // weeks within 75% of the busiest usual week (10): weeks 1 and 2
  assert.deepEqual(peak.weeks, [1, 2]);
  assert.equal(peak.from, "2001-06-08");
  assert.equal(peak.to, "2001-06-21");
  assert.equal(busiest.week, 2);
  assert.equal(busiest.from, "2023-06-15");
  assert.ok(Math.abs(busiest.vsUsual - 30 / 8) < 1e-9);
});
