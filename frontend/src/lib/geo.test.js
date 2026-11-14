import test from "node:test";
import assert from "node:assert/strict";
import { haversineKm, snapCell, cellSizeKm, nearestCell, zoneReport, coverageOf, formatCoords } from "./geo.js";

const rows = [
  ["2023-06-01", 60.125, -110.125, 2, 8],
  ["2023-06-01", 60.125, -109.875, 0, 3],
  ["2023-06-02", 60.125, -110.125, 1, 0],
  ["2023-06-05", 61.375, -110.125, 4, 4],
  ["2023-06-05", 52.125, -70.125, 9, 9],
];

test("distances and cell geometry", () => {
  assert.ok(Math.abs(haversineKm(60, -110, 61, -110) - 111.19) < 0.1);
  assert.ok(Math.abs(haversineKm(60, -110, 60, -109) - 55.6) < 0.1);
  assert.equal(haversineKm(55, -100, 55, -100), 0);
  assert.deepEqual(snapCell(60.2, -110.01), [60.125, -110.125]);
  assert.deepEqual(snapCell(50, -60.25), [50.125, -60.125]);
  const [ew, ns] = cellSizeKm(60);
  assert.ok(Math.abs(ns - 27.8) < 0.1 && Math.abs(ew - 13.9) < 0.1);
});

test("nearest cell respects the tolerance", () => {
  const cells = [[60.125, -110.125, 3, 8, 1, 2], [61.375, -110.125, 4, 4, 1, 1]];
  assert.equal(nearestCell(cells, 60.2, -110.2, 30), cells[0]);
  assert.equal(nearestCell(cells, 61.3, -110.0, 30), cells[1]);
  assert.equal(nearestCell(cells, 65, -110, 30), null);
});

test("a single cell reports only its own cell-days", () => {
  const z = zoneReport(rows, 60.125, -110.125);
  assert.equal(z.cells, 1);
  assert.equal(z.modis, 3);
  assert.equal(z.viirs, 8);
  assert.equal(z.cellDays, 2);
  assert.equal(z.sharedCellDays, 1);
  assert.equal(z.modisCellDays, 2); // on the common grid the sensors are comparable
  assert.equal(z.viirsCellDays, 1);
  assert.equal(z.activeDays, 2);
  assert.equal(z.first, "2023-06-01");
  assert.equal(z.last, "2023-06-02");
  assert.equal(z.peak, "2023-06-01");
});

test("a radius gathers neighbouring cells and nothing far away", () => {
  const z = zoneReport(rows, 60.2, -110, 50);
  assert.equal(z.cells, 2);
  assert.equal(z.viirs, 11);
  assert.equal(z.activeDays, 2);
  assert.deepEqual(z.daily[0], { date: "2023-06-01", modis: 2, viirs: 11, cells: 2 });
  const wide = zoneReport(rows, 60.2, -110, 200);
  assert.equal(wide.cells, 3);
  assert.equal(wide.last, "2023-06-05");
  const empty = zoneReport(rows, 68, -130, 100);
  assert.equal(empty.cells, 0);
  assert.equal(empty.peak, null);
  assert.deepEqual(empty.daily, []);
});

test("coverage separates no data from no fire", () => {
  assert.equal(coverageOf(60, -110, 100), "inside");
  assert.equal(coverageOf(60.125, -110.125), "inside");
  assert.equal(coverageOf(50.3, -110, 100), "partial"); // reaches south of 50N
  assert.equal(coverageOf(49.6, -110, 100), "partial"); // centred outside, but overlaps
  assert.equal(coverageOf(34.05, -118.25, 100), "outside"); // Los Angeles
  assert.equal(coverageOf(60, -30, 100), "outside");
  assert.equal(coverageOf(49.9, -110), "outside"); // a single cell just south of the edge
});

test("coordinates print with hemispheres", () => {
  assert.equal(formatCoords(58.375, -112.125), "58.375°N 112.125°W");
  assert.equal(formatCoords(-12.5, 30, 1), "12.5°S 30.0°E");
});
