// How the console words a season's figures. One place, so the stage, the panel and the
// season table can never disagree about a number or its label.
import { VERDICTS, seasonDate, median } from "./seasons.js";
import { formatDate } from "./chart.js";

/** A multiple without its sign ("4.0", "53", "0.18"): the unit sits in its own column. */
export const num = (v) => (v === null || v === undefined ? "–" : v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));
export const signed = (z) => `${z >= 0 ? "+" : "−"}${Math.abs(z).toFixed(1)}`;
export const pad2 = (n) => String(n).padStart(2, "0");

export function classOf(v) {
  if (v.kind === "high") return v.record === "high" ? "Record high" : VERDICTS.high;
  if (v.kind === "low") return v.record === "low" ? "Record low" : VERDICTS.low;
  if (v.kind === "mixed") return VERDICTS.mixed;
  return v.third === "upper" ? "Above median, in range" : v.third === "lower" ? "Below median, in range" : "Typical";
}

/**
 * The parts of a verdict that need wording. VIIRS's archive can stop before MODIS's; the
 * uncorrected figure and the sensor factor are then judged on VIIRS's shorter window, and
 * `window` says so.
 */
export function seasonFigures(seasons, v, year) {
  const through = (days) => formatDate(seasonDate(seasons, year, days - 1));
  const hasNaive = v.naive !== null && v.naive !== undefined;
  const short = hasNaive && Boolean(v.naiveWindow) && v.naiveWindow < v.window;
  return {
    hasNaive,
    window: short ? `thru ${through(v.naiveWindow)}` : "",
    factor: hasNaive ? v.naive / (v.naiveLike || v.likeForLike) : null,
    partial: v.partial ? `thru ${through(v.window)}` : "",
    cross: v.crossPct === null ? "n/a" : `${v.kind === "mixed" ? "disagrees" : "ok"} · P${Math.round(v.crossPct * 100)}`,
  };
}

/** The usual activity in each week: the median of every season's week, for the area in scope. */
export function usualWeeks(calendar) {
  const weeks = calendar[0]?.cells.length || 0;
  return Array.from({ length: weeks }, (_, w) => median(calendar.map((r) => r.cells[w]).filter((c) => c !== null && c !== undefined)));
}

/**
 * Critical periods for the area in scope. `peakWeeks` are the weeks whose usual activity is
 * within 75% of the busiest usual week: when burning normally concentrates. `busiest` is the
 * selected season's busiest week, with its multiple of the usual for that week.
 */
export function criticalPeriods(seasons, calendar, year) {
  const usual = usualWeeks(calendar);
  const top = Math.max(0, ...usual.map((u) => u || 0));
  const peakWeeks = top > 0 ? usual.map((u, w) => ((u || 0) >= 0.75 * top ? w : -1)).filter((w) => w >= 0) : [];
  const days = seasons.season_days;
  const start = (y, w) => seasonDate(seasons, y, Math.min(days - 1, w * 7));
  const end = (y, w) => seasonDate(seasons, y, Math.min(days - 1, w * 7 + 6));
  const peak = peakWeeks.length ? { from: start(2001, peakWeeks[0]), to: end(2001, peakWeeks[peakWeeks.length - 1]), weeks: peakWeeks } : null;
  const row = calendar.find((r) => r.year === year);
  let busiest = null;
  if (row) {
    row.cells.forEach((c, w) => {
      if (c !== null && c !== undefined && c > 0 && (!busiest || c > busiest.value)) busiest = { week: w, value: c, from: start(year, w), vsUsual: usual[w] ? c / usual[w] : null };
    });
  }
  return { usual, peak, busiest };
}
