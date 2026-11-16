import test from "node:test";
import assert from "node:assert/strict";
import { COLORS } from "../ml/data.js";
import { historicalCellsCsv, summarizeHistoricalScenes } from "./historicalMLP.js";

const CANADA = [[50, -140], [70, -60]];

test("compact model cells retain scored-query and alert totals on the archive grid", () => {
  const result = summarizeHistoricalScenes([
    { date: "2003-06-01", cells: [
      { lat: 60.125, lon: -110.125, scored_queries: 4, plswork_alerts: 1 },
      { lat: 60.375, lon: -110.375, scored_queries: 3, plswork_alerts: 0 },
    ] },
    { date: "2003-06-03", cells: [
      { lat: 60.125, lon: -110.125, scored_queries: 6, plswork_alerts: 0 },
    ] },
  ], { date: "all", firstDay: null, lastDay: null, bounds: CANADA });

  assert.equal(result.queries, 13);
  assert.equal(result.alerts, 1);
  assert.deepEqual(result.dates, ["2003-06-01", "2003-06-03"]);
  assert.equal(result.cellDegrees, 0.25);
  assert.deepEqual(result.points, [
    { lat: 60.125, lon: -110.125, scoredQueries: 10, alerts: 1, color: COLORS.mlp, weight: 0.65 },
    { lat: 60.375, lon: -110.375, scoredQueries: 3, alerts: 0, color: COLORS.coverage, weight: 0.12 },
  ]);
  assert.equal(result.sceneDates, 2);
  assert.equal(result.records.length, 3);
  assert.equal(result.records[2].scoredQueries, 6);
});

test("date and target-area selectors filter the compact counts", () => {
  const scenes = [
    { date: "2003-06-01", cells: [{ lat: 60.125, lon: -110.125, scored_queries: 100, plswork_alerts: 10 }] },
    { date: "2003-06-03", cells: [
      { lat: 60.125, lon: -110.125, scored_queries: 6, plswork_alerts: 2 },
      { lat: 60.375, lon: -110.375, scored_queries: 9, plswork_alerts: 3 },
    ] },
  ];
  const result = summarizeHistoricalScenes(scenes, {
    date: "all",
    firstDay: "2003-06-02",
    lastDay: "2003-06-04",
    bounds: CANADA,
    target: { lat: 60.125, lon: -110.125, km: 0 },
  });

  assert.deepEqual(result.dates, ["2003-06-03"]);
  assert.equal(result.sceneDates, 1);
  assert.equal(result.queries, 6);
  assert.equal(result.alerts, 2);
  assert.deepEqual(result.points.map(({ lat, lon }) => [lat, lon]), [[60.125, -110.125]]);
});

test("legacy precise points remain count-compatible and group into 0.25-degree cells", () => {
  const result = summarizeHistoricalScenes([{
    date: "2003-08-17",
    coverage: [[-110.126, 60.124], [-110.12, 60.12], [-110.6, 60.4]],
    alerts: [[-110.2, 60.2]],
  }], { date: "2003-08-17", firstDay: null, lastDay: null, bounds: CANADA });

  assert.equal(result.queries, 3);
  assert.equal(result.alerts, 1);
  assert.equal(result.points.length, 2);
  assert.deepEqual(result.points.map(({ lat, lon }) => [lat, lon]), [[60.125, -110.125], [60.375, -110.625]]);
  assert.equal(result.points[0].color, COLORS.mlp);
  assert.equal(result.points[1].color, COLORS.coverage);
  assert.equal(result.records[0].scoredQueries, 2);
  assert.equal(result.records[0].alerts, 1);
});

test("compact totals take precedence when a scene also retains the precise pilot points", () => {
  const result = summarizeHistoricalScenes([{
    date: "2003-08-17",
    coverage: [[-110.126, 60.124], [-110.12, 60.12]],
    alerts: [[-110.2, 60.2]],
    cells: [{ lat: 60.125, lon: -110.125, scored_queries: 2, plswork_alerts: 1 }],
  }], { date: "2003-08-17", firstDay: null, lastDay: null, bounds: CANADA });

  assert.equal(result.queries, 2);
  assert.equal(result.alerts, 1);
  assert.equal(result.points.length, 1);
});

test("an overpass with no cells in the selected area remains distinguishable from no overpass", () => {
  const result = summarizeHistoricalScenes([
    { date: "2003-06-01", cells: [{ lat: 60.125, lon: -110.125, scored_queries: 2, plswork_alerts: 0 }] },
  ], {
    date: "all",
    firstDay: null,
    lastDay: null,
    bounds: CANADA,
    target: { lat: 56.125, lon: -110.125, km: 0 },
  });

  assert.equal(result.sceneDates, 1);
  assert.equal(result.queries, 0);
  assert.equal(result.points.length, 0);
});

test("model-cell CSV exports one selected-date grid cell with genuine counts", () => {
  const result = summarizeHistoricalScenes([
    { date: "2003-06-01", cells: [{ lat: 60.125, lon: -110.125, scored_queries: 4, plswork_alerts: 1 }] },
  ], { date: "all", firstDay: null, lastDay: null, bounds: CANADA });

  assert.equal(historicalCellsCsv(result.records),
    "date,latitude,longitude,scored_queries,plswork_alerts,grid_degrees\r\n2003-06-01,60.125,-110.125,4,1,0.25\r\n");
});
