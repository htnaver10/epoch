import test from "node:test";
import assert from "node:assert/strict";
import {
  linear, log, niceTicks, linePath, areaPath, bandPath, compact, times, ordinal, signedPercent, formatDate, clamp,
} from "./chart.js";

test("scales map their domain onto the pixel range", () => {
  const x = linear(0, 10, 100, 200);
  assert.equal(x(0), 100);
  assert.equal(x(5), 150);
  assert.equal(linear(3, 3, 0, 50)(3), 0); // degenerate domain does not divide by zero
  const y = log(1, 100, 0, 200);
  assert.ok(Math.abs(y(10) - 100) < 1e-9);
  assert.equal(y(0), 0);
  assert.equal(clamp(12, 0, 10), 10);
});

test("ticks are round, start at zero and cover the maximum", () => {
  assert.deepEqual(niceTicks(7258, 4), [0, 2000, 4000, 6000, 8000]);
  assert.deepEqual(niceTicks(100, 4), [0, 50, 100]);
  assert.deepEqual(niceTicks(0.9, 3), [0, 0.5, 1]);
  assert.deepEqual(niceTicks(0), [0, 1]);
  for (const max of [3, 17, 260, 999, 12345, 4.2e6]) {
    const t = niceTicks(max);
    assert.ok(t.at(-1) >= max && t.at(-2) < max, `${max}: ${t}`);
    assert.ok(t.length >= 3 && t.length <= 7, `${max}: ${t.length} ticks`);
  }
});

test("lines break at gaps and areas close on the baseline", () => {
  assert.equal(linePath([[0, 1], [10, 2], [20, null], [30, 4], [40, 5]]), "M0.0 1.0L10.0 2.0M30.0 4.0L40.0 5.0");
  assert.equal(areaPath([[0, 1], [10, 2]], 9), "M0.0 9.0L0.0 1.0L10.0 2.0L10.0 9.0Z");
  assert.equal(areaPath([[0, 1], [10, null], [20, 3]], 9), ""); // single points enclose nothing
  assert.equal(bandPath([0, 10], [8, 9], [2, 3]), "M0.0 2.0L10.0 3.0L10.0 9.0L0.0 8.0Z");
  assert.equal(bandPath([], [], []), "");
});

test("numbers read the way the interface prints them", () => {
  assert.equal(compact(950), "950");
  assert.equal(compact(1500), "1.5k");
  assert.equal(compact(2000), "2k");
  assert.equal(compact(18100), "18k");
  assert.equal(compact(1123524), "1.1M");
  assert.equal(compact(null), "–");
  assert.equal(times(52.9), "53×");
  assert.equal(times(4.0), "4.0×");
  assert.equal(times(0.18), "0.18×");
  assert.equal(ordinal(1), "1st");
  assert.equal(ordinal(22), "22nd");
  assert.equal(ordinal(13), "13th");
  assert.equal(ordinal(96), "96th");
  assert.equal(signedPercent(0.54), "+54%");
  assert.equal(signedPercent(-0.04), "−4%");
});

test("dates format from the ISO string without timezone drift", () => {
  assert.equal(formatDate("2023-07-04"), "Jul 4");
  assert.equal(formatDate("2023-07-04", "long"), "Jul 4, 2023");
  assert.equal(formatDate("2023-06-01", "month"), "June 2023");
});
