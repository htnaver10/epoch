import test from "node:test";
import assert from "node:assert/strict";
import { summarize, groupByDate, dateRange, dailySeries, mapCells, readView, viewSearch, writeTarget, rawWeight, RAW_REFERENCE, RAW_CEILING } from "./analysis.js";

const rows = [
  ["2025-06-01", 50.125, -100.125, 2, 8],
  ["2025-06-01", 51.125, -101.125, 0, 3],
  ["2025-06-02", 51.125, -101.125, 1, 0],
];

test("raw ratio and grid overlap use distinct denominators", () => {
  const s = summarize(rows);
  assert.equal(s.modis, 3);
  assert.equal(s.viirs, 11);
  assert.equal(s.rawRatio, 11 / 3);
  assert.equal(s.gridRatio, 1);
  assert.equal(s.overlap, 1 / 3);
  assert.equal(s.shared, 1);
});

test("empty and VIIRS-only samples do not manufacture a ratio", () => {
  assert.equal(summarize([]).overlap, null);
  assert.equal(summarize([rows[1]]).rawRatio, null);
});

test("days VIIRS never archived are not counted as VIIRS misses", () => {
  const s = summarize(rows, "2025-06-01"); // VIIRS archive stops after June 1
  assert.equal(s.modis, 3); // totals still cover the whole selection
  assert.equal(s.comparable, 2);
  assert.equal(s.overlap, 1 / 2);
  assert.equal(s.rawRatio, 11 / 2);
  const none = summarize(rows, null); // a pre-VIIRS season
  assert.equal(none.overlap, null);
  assert.equal(none.gridRatio, null);
  assert.equal(none.modisCells, 2);
});

test("different-day sensor detections remain visible as disagreement", () => {
  const points = mapCells(rows, "uncertain");
  assert.equal(points.length, 1);
  assert.deepEqual(points[0], [51.125, -101.125, 1, 3, 0, 2]);
  assert.equal(mapCells(rows, "all").find((p) => p[0] === 50.125)[4], 1);
});

test("the disagreement layer ignores days outside VIIRS coverage", () => {
  assert.deepEqual(mapCells(rows, "uncertain", "2025-06-01"), [[51.125, -101.125, 0, 3, 0, 1]]);
  assert.deepEqual(mapCells(rows, "uncertain", null), []);
  assert.equal(mapCells(rows, "modis", null).length, 2);
});

test("cells keyed by position never collide across the region", () => {
  const grid = [];
  for (let lat = 50.125; lat < 70; lat += 0.25) for (let lon = -139.875; lon < -60; lon += 0.25) grid.push(["2025-06-01", lat, lon, 1, 0]);
  assert.equal(mapCells(grid, "all").length, grid.length);
});

test("daily series keeps quiet days on the axis", () => {
  const dates = dateRange("2025-05-31", "2025-06-02");
  assert.deepEqual(dates, ["2025-05-31", "2025-06-01", "2025-06-02"]);
  const daily = dailySeries(groupByDate(rows), dates, "grid");
  assert.deepEqual(daily[0], { date: "2025-05-31", day: "31", modis: 0, viirs: 0, occupied: 0 });
  assert.deepEqual(daily[1], { date: "2025-06-01", day: "01", modis: 1, viirs: 2, occupied: 2 });
  assert.equal(dailySeries(groupByDate(rows), dates, "raw")[1].viirs, 11);
});

test("shared views round trip and stay short", () => {
  const view = { region: "central", year: 2023, date: "2023-07-14", metric: "raw", layer: "uncertain", optical: true, panel: "record" };
  assert.deepEqual(readView(viewSearch(view)), { ...view, view: "corrected", period: null, at: null, scale: 1 });
  assert.deepEqual(readView(viewSearch({ ...view, period: "30-61" })).period, { from: 30, to: 61 });
  for (const bad of ["", "61-30", "5-8", "a-b", "0-400"]) assert.equal(readView(`?period=${bad}`).period, null, bad);
  assert.equal(readView(viewSearch({ ...view, view: "raw" })).view, "raw");
  assert.ok(!viewSearch({ ...view, view: "corrected" }).includes("view=")); // the default stays out of the link
  assert.equal(readView("?view=inflated").view, "corrected");
  assert.equal(readView(viewSearch({ ...view, scale: 1.3 })).scale, 1.3);
  for (const bad of ["9", "0.2", "big", ""]) assert.equal(readView(`?scale=${bad}`).scale, 1, bad);
  const target = { lat: 62.45, lon: -114.37, km: 150 };
  assert.deepEqual(readView(viewSearch({ ...view, at: writeTarget(target) })).at, target);
  assert.equal(writeTarget(null), null);
  for (const bad of ["", "1,2", "a,b,c", "95,0,10", "60,-200,10", "60,-110,5000"]) assert.equal(readView(`?at=${bad}`).at, null, bad);
  assert.equal(viewSearch({ region: "canada", year: 2023, date: "all", metric: "grid", layer: "all", optical: false, panel: null }), "year=2023");
});

test("invalid links fall back to the default view", () => {
  assert.deepEqual(readView("?region=unknown&year=abc&date=2025-06-31&metric=hack&layer=x&optical=yes&panel=nope"), {
    region: "canada", year: null, date: "all", metric: "grid", layer: "all", view: "corrected", period: null, optical: false, scale: 1, at: null, panel: null,
  });
  assert.equal(readView("?year=2023&date=2024-07-01").date, "all"); // date from another season
  assert.equal(readView("?year=2023&date=2023-06-31").date, "all"); // not a real day
  assert.equal(readView("?year=2023&date=2023-07-01").date, "2023-07-01");
  assert.equal(readView("?dataset=season&year=2021").year, 2021); // links from the earlier version
});

test("the uncorrected view sizes a cell by its raw count against the MODIS reference", () => {
  assert.equal(rawWeight(RAW_REFERENCE), 1);
  assert.ok(rawWeight(1) > 0 && rawWeight(1) < rawWeight(50)); // even one detection is drawn
  assert.ok(rawWeight(4 * RAW_REFERENCE) === 2); // area, not width, follows the count
  assert.equal(rawWeight(RAW_CEILING * 10), rawWeight(RAW_CEILING * 100)); // it stops growing
  assert.ok(RAW_CEILING > RAW_REFERENCE * 6);
});
