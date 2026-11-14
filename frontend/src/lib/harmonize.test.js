import test from "node:test";
import assert from "node:assert/strict";
import { staticKeys, withoutStatic, fillGaps } from "./harmonize.js";

test("static heat sources are dropped from the records, everything else kept", () => {
  const keys = staticKeys({ static_sources: [[57.125, -111.625]] });
  const records = [["2023-06-01", 57.125, -111.625, 2, 9], ["2023-06-01", 57.375, -111.625, 1, 3]];
  assert.deepEqual(withoutStatic(records, keys), [records[1]]);
  assert.equal(withoutStatic(records, new Set()), records);
});

test("a missing VIIRS day is filled from MODIS that day, at the neighbouring days' ratio", () => {
  const series = (a) => ({ cells: [...a], w: [...a], raw: [...a] });
  // VIIRS sees twice what MODIS does, except on day 3, when its archive is empty
  const row = { modis: series([10, 10, 10, 30, 10, 10, 10]), viirs: series([20, 20, 20, 0, 20, 20, 20]) };
  fillGaps(row, { modis: {}, viirs: { 2024: [3] } }, 2024);
  assert.equal(row.viirs.cells[3], 60);
  assert.equal(row.viirs.w[3], 60);
  assert.deepEqual(row.modis.cells, [10, 10, 10, 30, 10, 10, 10]); // the other sensor is untouched
});

test("a gap is never lowered, and other years are left alone", () => {
  const row = { modis: { cells: [1, 1, 1], w: [1, 1, 1], raw: [1, 1, 1] }, viirs: { cells: [9, 9, 9], w: [9, 9, 9], raw: [9, 9, 9] } };
  fillGaps(row, { modis: {}, viirs: { 2024: [1] } }, 2023);
  assert.deepEqual(row.viirs.cells, [9, 9, 9]);
  fillGaps(row, { modis: {}, viirs: { 2023: [1] } }, 2023);
  assert.equal(row.viirs.cells[1], 9);
});
