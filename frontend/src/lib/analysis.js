import { readPeriod } from "./area.js";

/** Observations are [UTC date, cell latitude, cell longitude, MODIS, VIIRS]. */

/**
 * Totals for a selection. `viirsThrough` is the last date VIIRS covers: `undefined` when it
 * covers everything, `null` when VIIRS has no record for the season at all. Sensor
 * comparisons only use the days both instruments were archived, so a day VIIRS never
 * reported is not counted as a day VIIRS saw nothing.
 */
export function summarize(rows, viirsThrough) {
  let modis = 0, viirs = 0, modisCells = 0, viirsCells = 0;
  let both = 0, modisBoth = 0, viirsBoth = 0, modisCellsBoth = 0, viirsCellsBoth = 0, shared = 0;
  for (const row of rows) {
    const m = row[3], v = row[4];
    modis += m;
    viirs += v;
    modisCells += Number(m > 0);
    viirsCells += Number(v > 0);
    if (viirsThrough === null || (viirsThrough !== undefined && row[0] > viirsThrough)) continue;
    both += 1;
    modisBoth += m;
    viirsBoth += v;
    modisCellsBoth += Number(m > 0);
    viirsCellsBoth += Number(v > 0);
    shared += Number(m > 0 && v > 0);
  }
  return {
    modis, viirs, modisCells, viirsCells, shared,
    union: rows.length,
    comparable: both,
    rawRatio: modisBoth ? viirsBoth / modisBoth : null,
    gridRatio: modisCellsBoth ? viirsCellsBoth / modisCellsBoth : null,
    overlap: both ? shared / both : null,
  };
}

/** Group rows by date once so stepping through days never rescans the season. */
export function groupByDate(rows) {
  const groups = new Map();
  for (const row of rows) {
    const g = groups.get(row[0]);
    if (g) g.push(row);
    else groups.set(row[0], [row]);
  }
  return groups;
}

/** Every calendar day from `first` to `last` inclusive, so quiet days stay on the axis. */
export function dateRange(first, last) {
  const out = [];
  for (let t = Date.parse(`${first}T00:00:00Z`), end = Date.parse(`${last}T00:00:00Z`); t <= end; t += 864e5)
    out.push(new Date(t).toISOString().slice(0, 10));
  return out;
}

export function dailySeries(byDate, dates, metric) {
  return dates.map((date) => {
    const s = summarize(byDate.get(date) || []);
    return {
      date,
      day: date.slice(8),
      modis: metric === "grid" ? s.modisCells : s.modis,
      viirs: metric === "grid" ? s.viirsCells : s.viirs,
      occupied: s.union,
    };
  });
}

/**
 * Merge cell-days into one marker per cell: [lat, lon, MODIS, VIIRS, same-day days, active days].
 * Agreement is decided per cell-day before merging, so sensors that saw a place on
 * different dates are not reported as agreeing. The "uncertain" layer keeps only cell-days
 * one sensor reported while both were archived.
 */
export function mapCells(rows, layer, viirsThrough) {
  const cells = new Map();
  for (const row of rows) {
    const m = row[3], v = row[4];
    if (layer === "modis" && !m) continue;
    if (layer === "viirs" && !v) continue;
    if (layer === "uncertain") {
      if (m && v) continue;
      if (viirsThrough === null || (viirsThrough !== undefined && row[0] > viirsThrough)) continue;
    }
    const key = row[1] * 1e4 + row[2];
    let cell = cells.get(key);
    if (!cell) cells.set(key, (cell = [row[1], row[2], 0, 0, 0, 0]));
    cell[2] += m;
    cell[3] += v;
    cell[4] += Number(m > 0 && v > 0);
    cell[5] += 1;
  }
  return [...cells.values()];
}

/**
 * The uncorrected view draws a cell by its raw detection count, the way a naive map would.
 * A weight of 1 is the count 99% of MODIS cells stay under across the whole record; VIIRS
 * runs far past it, which is the inflation the view exists to show.
 */
export const RAW_REFERENCE = 520;
const RAW_MAX_WEIGHT = 2.6;
/** The count at which a mark stops growing. */
export const RAW_CEILING = Math.round((RAW_REFERENCE * RAW_MAX_WEIGHT ** 2) / 100) * 100;
export const rawWeight = (count) => Math.max(0.3, Math.min(RAW_MAX_WEIGHT, Math.sqrt(count / RAW_REFERENCE)));

export const REGION_KEYS = ["canada", "west", "central", "east"];
export const LAYER_KEYS = ["all", "modis", "viirs", "uncertain"];
export const PANEL_KEYS = ["", "brief", "record", "season", "sensors", "method"];

/** Presentation scale: a multiplier on the whole interface, for projectors and big rooms. */
function readScale(text) {
  const v = Number(text);
  return text && Number.isFinite(v) && v >= 0.8 && v <= 1.8 ? Math.round(v * 10) / 10 : 1;
}

/** "62.45,-114.37,150" -> { lat, lon, km }: the target a link was focused on. */
function readTarget(text) {
  const parts = (text || "").split(",").map(Number);
  const [lat, lon, km] = parts;
  const ok = parts.length === 3 && parts.every(Number.isFinite) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180 && km >= 0 && km <= 1000;
  return ok ? { lat, lon, km } : null;
}
export const writeTarget = (t) => (t ? `${Number(t.lat.toFixed(3))},${Number(t.lon.toFixed(3))},${t.km}` : null);

/** Parse a shared link. Anything unrecognised falls back to the default view. */
export function readView(search) {
  const params = new URLSearchParams(search);
  const choose = (key, values) => (values.includes(params.get(key)) ? params.get(key) : values[0]);
  const y = Number(params.get("year"));
  const year = Number.isInteger(y) && y >= 2000 && y <= 2100 ? y : null;
  const date = params.get("date") || "";
  const inSeason = year !== null && date.startsWith(`${year}-`) && /^\d{4}-\d{2}-\d{2}$/.test(date) &&
    !Number.isNaN(Date.parse(`${date}T00:00:00Z`)) && new Date(`${date}T00:00:00Z`).toISOString().slice(0, 10) === date;
  return {
    region: choose("region", REGION_KEYS),
    year,
    date: inSeason ? date : "all",
    metric: choose("metric", ["grid", "raw"]),
    layer: choose("layer", LAYER_KEYS),
    view: choose("view", ["corrected", "raw"]),
    period: readPeriod(params.get("period"), 366),
    optical: params.get("optical") === "1",
    scale: readScale(params.get("scale")),
    at: readTarget(params.get("at")),
    panel: choose("panel", PANEL_KEYS) || null,
  };
}

/** Serialise a view, leaving defaults out so shared links stay short. */
export function viewSearch(view) {
  const defaults = { region: "canada", date: "all", metric: "grid", layer: "all", view: "corrected", optical: false, scale: 1 };
  return new URLSearchParams(
    Object.entries(view)
      .filter(([k, v]) => v !== null && v !== undefined && v !== defaults[k])
      .map(([k, v]) => [k, v === true ? "1" : v]),
  ).toString();
}
