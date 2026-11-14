import test from "node:test";
import assert from "node:assert/strict";
import { areaSeries, withArea, inArea, cellWeight, readPeriod, writePeriod, monthPeriods } from "./area.js";
import { yearTotals, assessSeason } from "./seasons.js";
import { calendarCsv, cellDaysCsv } from "./export.js";
import { weeklyCalendar } from "./seasons.js";

// A fourteen-day "season" starting Jun 1, two sensors from 2012, twelve years of record.
const DAYS = 14;
const years = Array.from({ length: 12 }, (_, i) => 2010 + i);
const published = Object.fromEntries(years.map((y) => [y, y >= 2012
  ? { modis: { cells: [], w: [], raw: [], days: DAYS }, viirs: { cells: [], w: [], raw: [], days: DAYS }, shared: [] }
  : { modis: { cells: [], w: [], raw: [], days: DAYS } }]));
const seasons = { season: ["06-01", "06-14"], season_days: DAYS, series: { canada: published } };
const HERE = { lat: 60.125, lon: -110.125, km: 60 }; // a zone around one cell
const NEAR = [60.125, -110.125], ALSO = [60.375, -110.125], FAR = [64.125, -100.125];
// year y burns its first (y - 2009) days in the near cell, so no two years tie; the far cell
// burns every day of every year
const records = Object.fromEntries(years.map((y) => [y, [
  ...Array.from({ length: Math.min(DAYS, y - 2009) }, (_, d) => [`${y}-06-${String(d + 1).padStart(2, "0")}`, ...NEAR, 2, y >= 2012 ? 9 : 0]),
  [`${y}-06-03`, ...ALSO, 0, y >= 2012 ? 4 : 0],
  ...Array.from({ length: DAYS }, (_, d) => [`${y}-06-${String(d + 1).padStart(2, "0")}`, ...FAR, 5, 5]),
]]));

test("a circle takes the cells whose centres fall inside it; radius 0 is one cell", () => {
  assert.ok(inArea(HERE, ...NEAR));
  assert.ok(inArea(HERE, ...ALSO)); // 28 km north
  assert.ok(!inArea(HERE, ...FAR));
  assert.ok(inArea({ lat: 60.125, lon: -110.125, km: 0 }, ...NEAR));
  assert.ok(!inArea({ lat: 60.125, lon: -110.125, km: 0 }, ...ALSO));
});

test("an area's series has the shape and the arithmetic of a published sector", () => {
  const series = areaSeries(records, seasons, HERE);
  const y = series[2015];
  assert.deepEqual(Object.keys(y).sort(), ["modis", "shared", "viirs"]);
  assert.equal(y.modis.days, DAYS);
  const quiet = new Array(8).fill(0);
  assert.deepEqual(y.modis.cells, [1, 1, 1, 1, 1, 1, ...quiet]); // six days in the near cell, never the far one
  assert.deepEqual(y.modis.raw, [2, 2, 2, 2, 2, 2, ...quiet]);
  assert.deepEqual(y.viirs.cells, [1, 1, 2, 1, 1, 1, ...quiet]); // the second cell is VIIRS only, on day 3
  assert.equal(y.viirs.raw[2], 13);
  assert.deepEqual(y.shared, [1, 1, 1, 1, 1, 1, ...quiet]);
  assert.ok(Math.abs(y.modis.w[0] - cellWeight(60.125)) < 1e-12); // ground-area weight of that cell
  assert.ok(Math.abs(cellWeight(60) - 1) < 1e-12 && cellWeight(70) < cellWeight(50));
  assert.equal(series[2010].viirs, undefined); // before VIIRS, as published
  assert.equal(series[2010].shared, undefined);
});

test("an area is judged like a sector: against its own other seasons", () => {
  const record = withArea(seasons, "aoi", areaSeries(records, seasons, HERE));
  assert.ok(record.series.canada === seasons.series.canada); // the published regions are untouched
  const totals = yearTotals(record, "aoi");
  assert.equal(totals.length, years.length);
  assert.ok(totals[11].modis > totals[0].modis); // later years burned more days here
  const latest = assessSeason(record, "aoi", 2021);
  assert.equal(latest.rank, 1);
  assert.equal(latest.kind, "high");
  const first = assessSeason(record, "aoi", 2010);
  assert.equal(first.rank, 12);
  // an area that never burned cannot be ranked
  const nowhere = withArea(seasons, "aoi", areaSeries(records, seasons, { lat: 55.125, lon: -120.125, km: 50 }));
  assert.equal(assessSeason(nowhere, "aoi", 2021).kind, "insufficient");
});

test("a period judges the same days of every season", () => {
  const record = withArea(seasons, "aoi", areaSeries(records, seasons, HERE));
  const early = yearTotals(record, "aoi", 3, 0), late = yearTotals(record, "aoi", 14, 7);
  const y2012 = (list) => list.find((t) => t.year === 2012).modisCells;
  assert.equal(y2012(early), 3); // 2012 burns its first three days
  assert.equal(y2012(late), 0); // and none of the second week
  // in the second week only the last five years burn at all, each a day more than the one before
  const span = assessSeason(record, "aoi", 2021, { from: 7, to: 14 });
  assert.equal(span.from, 7);
  assert.equal(span.window, 14);
  assert.equal(span.rank, 1);
  assert.equal(span.kind, "high");
  assert.equal(assessSeason(record, "aoi", 2021, { from: 7, to: 14 }).target.modisCells, 5);
});

test("periods round trip through a link and reject nonsense", () => {
  assert.deepEqual(readPeriod("30-61", 92), { from: 30, to: 61 });
  assert.equal(writePeriod({ from: 30, to: 61 }), "30-61");
  assert.equal(writePeriod(null), null);
  assert.equal(readPeriod("0-92", 92), null); // the whole season is no period
  assert.equal(readPeriod("40-44", 92), null); // shorter than a week
  assert.equal(readPeriod("80-99", 92), null);
  assert.deepEqual(monthPeriods({ season: ["06-01", "08-31"], season_days: 92 }), [
    { key: "Jun", from: 0, to: 30 }, { key: "Jul", from: 30, to: 61 }, { key: "Aug", from: 61, to: 92 },
  ]);
});

test("the calendar and the cell-days export as plain CSV", () => {
  const record = withArea(seasons, "aoi", areaSeries(records, seasons, HERE));
  const analysis = { calendar: weeklyCalendar(record, "aoi"), verdicts: new Map(years.map((y) => [y, assessSeason(record, "aoi", y)])) };
  const lines = calendarCsv(record, analysis, { area: "Test, area", period: null }).split("\n");
  assert.equal(lines.length, years.length + 1);
  assert.ok(lines[0].startsWith("area,season,basis,week_of_06-01,week_of_06-08,period_first_day"));
  const last = lines[lines.length - 1].split(",");
  assert.equal(last[0], '"Test'); // a comma in the area name is quoted
  assert.ok(lines[lines.length - 1].includes(",2021,modis,"));
  assert.ok(lines[lines.length - 1].endsWith(",high"));
  const cells = cellDaysCsv(records[2012].slice(0, 2)).split("\n");
  assert.equal(cells.length, 3);
  assert.equal(cells[1], "2012-06-01,60.125,-110.125,2,9,1,1,1,0.25");
});
