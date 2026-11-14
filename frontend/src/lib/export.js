// CSV exports. Plain text builders, so they can be tested without a browser.
import { seasonDate } from "./seasons.js";

const line = (cells) => cells.map((c) => (c === null || c === undefined ? "" : typeof c === "number" ? (Number.isInteger(c) ? c : Number(c.toFixed(3))) : /[",\n]/.test(c) ? `"${String(c).replace(/"/g, '""')}"` : c)).join(",");

/** The hotspot cell-days in a selection: one row per grid cell and day. */
export function cellDaysCsv(rows) {
  return [
    "date,grid_center_latitude,grid_center_longitude,modis_detections,viirs_detections,modis_occupied,viirs_occupied,both_sensors,grid_degrees",
    ...rows.map((r) => [...r, Number(r[3] > 0), Number(r[4] > 0), Number(r[3] > 0 && r[4] > 0), 0.25].join(",")),
  ].join("\n");
}

/**
 * The calendar as a table: one row per season, a column per week of harmonized activity
 * (MODIS-scale, area-weighted cell-days), then the season's figures for the period in scope.
 */
export function calendarCsv(seasons, analysis, { area, period }) {
  const { calendar, verdicts } = analysis;
  const weeks = calendar[0]?.cells.length || 0;
  const from = period?.from ?? 0, to = period?.to ?? seasons.season_days;
  const head = [
    "area", "season", "basis",
    ...Array.from({ length: weeks }, (_, w) => `week_of_${seasonDate(seasons, 2001, w * 7).slice(5)}`),
    "period_first_day", "period_last_day", "period_activity", "usual_activity", "index", "rank", "of", "percentile", "class",
  ];
  const body = calendar.map((row) => {
    const v = verdicts.get(row.year);
    const known = v && v.kind !== "insufficient";
    return line([
      area, row.year, row.basis,
      ...row.cells,
      seasonDate(seasons, row.year, from), seasonDate(seasons, row.year, (known ? v.window : to) - 1),
      known ? v.likeForLike * v.base : null, known ? v.base : null, known ? v.likeForLike : null,
      known ? v.rank : null, known ? v.of : null, known ? v.pct : null, v ? v.kind : null,
    ]);
  });
  return [line(head), ...body].join("\n");
}
