import test from "node:test";
import assert from "node:assert/strict";
import {
  median, quantile, percentileRank, robustLogZ, yearTotals, fitBridge, validateBridge, verdict,
  assessSeason, weeklyCalendar, harmonizedSeries, mannKendall, theilSen, trend, splicedTrend,
  seasonCurves, seasonDate, medianCrossing, recordDailyPeak,
} from "./seasons.js";

const days = 28;
const flat = (v, n = days) => Array(n).fill(v);
const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

/** VIIRS sees exactly twice the cells MODIS does; 2015 is a genuinely big season. */
function fixture() {
  const series = {};
  for (let y = 2003; y <= 2016; y++) {
    const m = y === 2015 ? 30 : 10;
    const s = { modis: { cells: flat(m), raw: flat(m * 3), days } };
    if (y >= 2012) {
      s.viirs = { cells: flat(m * 2), raw: flat(m * 20), days };
      s.shared = flat(m);
    }
    series[y] = s;
  }
  series[2017] = { viirs: { cells: flat(20), raw: flat(200), days } }; // MODIS gone, ordinary year
  return { season: ["06-01", "06-28"], season_days: days, series: { canada: series } };
}

/** Overlap seasons that follow MODIS = 2 * VIIRS^0.8 exactly, with varied activity. */
function powerFixture() {
  const series = {};
  const levels = [40, 90, 150, 220, 300, 420, 560, 700];
  levels.forEach((v, i) => {
    const m = 2 * v ** 0.8;
    series[2012 + i] = {
      modis: { cells: flat(m), raw: flat(m), days },
      viirs: { cells: flat(v), raw: flat(v * 10), days },
    };
  });
  return { season: ["06-01", "06-28"], season_days: days, series: { canada: series } };
}

test("median, quantile and mid-rank percentile", () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([1, 2, 3, 4]), 2.5);
  assert.equal(median([]), null);
  assert.equal(quantile([1, 2, 3, 4, 5], 0.25), 2);
  assert.equal(percentileRank(5, [1, 2, 5, 9]), 0.625);
  assert.equal(percentileRank(1, []), null);
});

test("robust log z is zero at the median and scale-free", () => {
  near(robustLogZ(4, [1, 2, 4, 8, 16]), 0);
  near(robustLogZ(16, [1, 2, 4, 8, 16]), robustLogZ(1600, [100, 200, 400, 800, 1600]));
  assert.ok(robustLogZ(16, [1, 2, 4, 8, 16]) > 0);
  assert.equal(robustLogZ(5, [3, 3, 3, 3]), null); // no spread, no score
});

test("year totals keep missing sensors null, never zero", () => {
  const t = yearTotals(fixture(), "canada");
  assert.equal(t.find((r) => r.year === 2005).viirs, null);
  assert.equal(t.find((r) => r.year === 2017).modis, null);
  assert.equal(t.find((r) => r.year === 2014).viirs, 20 * days);
  assert.deepEqual(yearTotals(fixture(), "nowhere"), []);
});

test("area-weighted activity is preferred over plain cell counts when present", () => {
  const f = fixture();
  f.series.canada[2014].modis.w = flat(7.5);
  const row = yearTotals(f, "canada").find((r) => r.year === 2014);
  assert.equal(row.modis, 7.5 * days);
  assert.equal(row.modisCells, 10 * days);
});

test("partial season is excluded at full window but comparable on its own window", () => {
  const f = fixture();
  f.series.canada[2016].modis.days = 14;
  assert.equal(yearTotals(f, "canada").find((r) => r.year === 2016).modis, null);
  assert.equal(yearTotals(f, "canada", 14).find((r) => r.year === 2016).modis, 140);
});

test("bridge falls back to the constant ratio without spread, and leaves the target out", () => {
  const t = yearTotals(fixture(), "canada");
  const b = fitBridge(t);
  assert.equal(b.ratio, 2);
  near(b.toModis(400), 200);
  assert.equal(fitBridge(t, 2015).model, "ratio"); // remaining seasons are identical
  near(fitBridge(t, 2015).toModis(400), 200);
  assert.equal(fitBridge(t.filter((r) => r.year < 2014)), null);
  const v = validateBridge(t);
  assert.equal(v.results.length, 5);
  near(v.mdape, 0);
});

test("bridge recovers a power law and beats the constant ratio on held-out seasons", () => {
  const t = yearTotals(powerFixture(), "canada");
  const b = fitBridge(t);
  assert.equal(b.model, "power");
  near(b.b, 0.8, 1e-9);
  near(b.toModis(1000 * days), 2 * 1000 ** 0.8 * days, 1e-6);
  const [lo, hi] = b.interval(1000 * days);
  assert.ok(lo <= b.toModis(1000 * days) && b.toModis(1000 * days) <= hi);
  const v = validateBridge(t);
  near(v.mdape, 0, 1e-9);
  assert.ok(v.ratioMdape > 0.05, `ratio bridge should miss: ${v.ratioMdape}`);
});

test("a big MODIS year is high and VIIRS agrees", () => {
  const v = verdict(yearTotals(fixture(), "canada"), 2015);
  assert.equal(v.kind, "high");
  assert.equal(v.rank, 1);
  assert.equal(v.record, "high");
  assert.equal(v.likeForLike, 3);
  assert.equal(v.crossPct, 1);
});

test("a sensor on the wrong side of its median turns a strong call into a disagreement", () => {
  const f = fixture();
  f.series.canada[2015].viirs.cells = flat(5); // VIIRS saw a quiet season
  const v = verdict(yearTotals(f, "canada"), 2015);
  assert.equal(v.strong, "high");
  assert.equal(v.kind, "mixed");
});

test("an ordinary VIIRS-era year looks like a jump naively but is typical like-for-like", () => {
  const v = verdict(yearTotals(fixture(), "canada"), 2014);
  assert.equal(v.kind, "typical");
  assert.ok(v.naive > 6, `naive ${v.naive}`);
  assert.equal(v.likeForLike, 1);
  near(v.detectionInflation, v.naive);
});

test("pre-VIIRS seasons have no naive comparison to make", () => {
  assert.equal(verdict(yearTotals(fixture(), "canada"), 2005).naive, null);
});

test("a VIIRS-only year is judged through the held-out bridge", () => {
  const t = yearTotals(fixture(), "canada");
  const h = harmonizedSeries(t).find((r) => r.year === 2017);
  assert.equal(h.basis, "viirs");
  near(h.equivalent, 10 * days);
  assert.equal(verdict(t, 2017).kind, "typical");
});

test("too little history is reported, not guessed", () => {
  const t = yearTotals(fixture(), "canada").filter((r) => r.year >= 2012);
  assert.equal(verdict(t, 2014).kind, "insufficient");
});

test("a season in progress is ranked over the same calendar window", () => {
  const f = fixture();
  f.series.canada[2016].modis = { cells: flat(30, 14), raw: flat(90, 14), days: 14 };
  f.series.canada[2016].viirs = { cells: flat(60, 14), raw: flat(600, 14), days: 14 };
  const v = assessSeason(f, "canada", 2016);
  assert.equal(v.window, 14);
  assert.equal(v.partial, true);
  assert.equal(v.likeForLike, 3); // 30/day against 10/day over the same 14 days
  f.series.canada[2016].modis.days = 5;
  f.series.canada[2016].viirs.days = 5;
  assert.equal(assessSeason(f, "canada", 2016).kind, "insufficient");
  assert.equal(assessSeason(f, "canada", 1999).kind, "insufficient");
});

test("VIIRS figures use VIIRS's own window when its archive stops early", () => {
  const f = fixture();
  f.series.canada[2016].viirs.days = 14; // MODIS complete, VIIRS two weeks only
  const v = assessSeason(f, "canada", 2016);
  assert.equal(v.window, days);
  assert.equal(v.crossWindow, 14);
  assert.equal(v.naiveWindow, 14);
  assert.ok(v.crossPct !== null && v.naive !== null);
});

test("Mann-Kendall and Theil-Sen find a monotonic trend and ignore noise-free flats", () => {
  const up = mannKendall([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  assert.equal(up.s, 45);
  assert.ok(up.p < 0.001);
  assert.equal(mannKendall([5, 5, 5, 5, 5]).p, 1);
  assert.equal(mannKendall([1, 2]), null);
  const line = theilSen([0, 1, 2, 3, 4], [1, 3, 5, 7, 100]); // one wild outlier
  assert.equal(line.slope, 2);
});

test("trend reads as percent per decade and the spliced record overstates it", () => {
  const series = {};
  for (let y = 2003; y <= 2022; y++) {
    const m = 100 * 1.05 ** (y - 2003); // +5% a year on the consistent record
    series[y] = { modis: { cells: flat(m), raw: flat(m), days } };
    if (y >= 2012) series[y].viirs = { cells: flat(m * 2), raw: flat(m * 12), days };
  }
  const t = yearTotals({ season_days: days, series: { canada: series } }, "canada");
  const real = trend(t);
  near(real.perDecade, 1.05 ** 10 - 1, 1e-9);
  assert.ok(real.p < 0.001);
  near(real.at(2003), 100 * days, 1e-6);
  assert.ok(splicedTrend(t).perDecade > 3 * real.perDecade);
  assert.equal(trend(t.slice(0, 5)), null);
});

test("season curves accumulate and the envelope leaves the target out", () => {
  const c = seasonCurves(fixture(), "canada", 2015);
  assert.equal(c.target.values[days - 1], 30 * days);
  assert.equal(c.band[0].mid, 10);
  assert.equal(c.band[days - 1].hi, 10 * days); // 2015 itself is not in its own envelope
  assert.equal(c.reference, 13);
  assert.equal(seasonDate(fixture(), 2015, 0), "2015-06-01");
  assert.equal(seasonDate(fixture(), 2015, 30), "2015-07-01");
});

test("a big season passes the median season's total early; an ordinary one only at the end", () => {
  const f = fixture();
  // 2015 burns 30 a day against a median of 10: it passes 10 x 28 on day 10 (index 9)
  assert.equal(medianCrossing(seasonCurves(f, "canada", 2015)), 9);
  assert.equal(medianCrossing(seasonCurves(f, "canada", 2014)), days - 1);
  f.series.canada[2014].modis.cells = flat(4);
  assert.equal(medianCrossing(seasonCurves(f, "canada", 2014)), null);
  assert.equal(recordDailyPeak(fixture(), "canada"), 60); // VIIRS in 2015
  assert.equal(recordDailyPeak(fixture(), "nowhere"), 0);
});

test("weekly calendar marks unavailable weeks as null and rescales VIIRS-only years", () => {
  const f = fixture();
  f.series.canada[2016].modis.days = 7;
  const cal = weeklyCalendar(f, "canada");
  const y16 = cal.find((r) => r.year === 2016);
  assert.equal(y16.cells[0], 70);
  assert.equal(y16.cells[1], null);
  const y17 = cal.find((r) => r.year === 2017);
  assert.equal(y17.basis, "viirs");
  near(y17.cells[0], 70);
});
