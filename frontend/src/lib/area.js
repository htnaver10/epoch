// An area of interest is treated exactly like a built-in sector: its daily series is built in
// the browser from the cell-day records, the same way the data pipeline builds a sector's
// (scripts/prepare_seasons.py, summarise). Everything downstream then works on any area.
import { haversineKm, GRID } from "./geo.js";
import { fillGaps } from "./harmonize.js";

const REFERENCE_LATITUDE = 60;
const RAD = Math.PI / 180;
/** Ground area of a grid cell centred at `lat`, relative to one at 60N. */
export const cellWeight = (lat) => Math.cos(lat * RAD) / Math.cos(REFERENCE_LATITUDE * RAD);

/** Whether the grid cell centred at (lat, lon) belongs to a circular area. `km` 0 is one cell. */
export function inArea(area, lat, lon) {
  if (!area.km) return lat === area.lat && lon === area.lon;
  if (Math.abs(lat - area.lat) > area.km / 111 + GRID) return false; // cheap reject first
  return haversineKm(area.lat, area.lon, lat, lon) <= area.km;
}

/**
 * Daily series for one area, per year, in the shape of `seasons.series[region]`:
 * { [year]: { modis: { cells, w, raw, days }, viirs?: {...}, shared?: [...] } }.
 * `records` is { [year]: [[iso date, lat, lon, modis detections, viirs detections], ...] }.
 * Which sensors a year has, and how many days each archived, come from the published record.
 */
export function areaSeries(records, seasons, area) {
  const days = seasons.season_days;
  const [month, day] = seasons.season[0].split("-").map(Number);
  const reference = seasons.series.canada || Object.values(seasons.series)[0];
  const out = {};
  for (const [year, published] of Object.entries(reference)) {
    const sensors = ["modis", "viirs"].filter((s) => published[s]);
    const row = {};
    for (const s of sensors) row[s] = { cells: new Array(days).fill(0), w: new Array(days).fill(0), raw: new Array(days).fill(0), days: published[s].days };
    if (sensors.length === 2) row.shared = new Array(days).fill(0);
    const first = Date.UTC(Number(year), month - 1, day);
    const index = new Map(); // iso date -> day of season: records come grouped by date
    for (const [iso, lat, lon, m, v] of records[year] || []) {
      if (!inArea(area, lat, lon)) continue;
      let i = index.get(iso);
      if (i === undefined) index.set(iso, (i = Math.round((Date.parse(`${iso}T00:00:00Z`) - first) / 86400000)));
      if (i < 0 || i >= days) continue;
      const w = cellWeight(lat);
      if (m && row.modis) {
        row.modis.cells[i] += 1;
        row.modis.w[i] += w;
        row.modis.raw[i] += m;
      }
      if (v && row.viirs) {
        row.viirs.cells[i] += 1;
        row.viirs.w[i] += w;
        row.viirs.raw[i] += v;
      }
      if (m && v && row.shared) row.shared[i] += 1;
    }
    // days missing from one archive are filled as the pipeline fills them for the sectors
    out[year] = fillGaps(row, seasons.gaps, year);
  }
  return out;
}

/** The published record with one more region in it. */
export const withArea = (seasons, key, series) => ({ ...seasons, series: { ...seasons.series, [key]: series } });

// ---- periods: a span of days of the season, [from, to) ----------------------------------

/** "30-61" -> { from: 30, to: 61 } if it is a real span of at least a week, else null. */
export function readPeriod(text, days) {
  const m = /^(\d{1,3})-(\d{1,3})$/.exec(text || "");
  if (!m) return null;
  const from = Number(m[1]), to = Number(m[2]);
  return from >= 0 && to <= days && to - from >= 7 && !(from === 0 && to === days) ? { from, to } : null;
}
export const writePeriod = (period) => (period ? `${period.from}-${period.to}` : null);

/** The calendar months the season covers, each as a period. */
export function monthPeriods(seasons) {
  const [month, day] = seasons.season[0].split("-").map(Number);
  const out = [];
  for (let i = 0; i < seasons.season_days; ) {
    const d = new Date(Date.UTC(2001, month - 1, day + i));
    const m = d.getUTCMonth();
    let j = i;
    while (j < seasons.season_days && new Date(Date.UTC(2001, month - 1, day + j)).getUTCMonth() === m) j++;
    out.push({ key: d.toLocaleString("en-CA", { month: "short", timeZone: "UTC" }), from: i, to: j });
    i = j;
  }
  return out;
}
