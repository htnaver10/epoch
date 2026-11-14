// The command bar: a small gazetteer of the theatre and a parser that turns what was typed
// into something to do (fly to a place or coordinates, pick a season, a day, a sector...).

import { haversineKm } from "./geo.js";

/** [name, kind, latitude, longitude, radius of interest in km] */
const TABLE = [
  ["British Columbia", "province", 55.5, -124.5, 420],
  ["Alberta", "province", 56.0, -114.5, 380],
  ["Saskatchewan", "province", 56.0, -106.0, 380],
  ["Manitoba", "province", 56.0, -97.5, 380],
  ["Ontario", "province", 52.0, -87.0, 380],
  ["Quebec", "province", 54.0, -72.0, 480],
  ["Labrador", "region", 54.0, -62.0, 300],
  ["Yukon", "territory", 63.5, -135.5, 380],
  ["Northwest Territories", "territory", 64.0, -119.0, 520],
  ["Nunavut", "territory", 65.0, -95.0, 520],
  ["Yellowknife", "town", 62.45, -114.37, 150],
  ["Hay River", "town", 60.82, -115.79, 120],
  ["Fort Smith", "town", 60.0, -111.88, 120],
  ["Fort Simpson", "town", 61.86, -121.35, 120],
  ["Norman Wells", "town", 65.28, -126.83, 150],
  ["Inuvik", "town", 68.36, -133.72, 150],
  ["Whitehorse", "town", 60.72, -135.06, 150],
  ["Dawson City", "town", 64.06, -139.43, 150],
  ["Fort Nelson", "town", 58.81, -122.7, 150],
  ["Fort St. John", "town", 56.25, -120.85, 120],
  ["Prince George", "town", 53.92, -122.75, 150],
  ["Fort McMurray", "town", 56.73, -111.38, 150],
  ["High Level", "town", 58.52, -117.14, 150],
  ["Grande Prairie", "town", 55.17, -118.8, 120],
  ["Slave Lake", "town", 55.28, -114.77, 120],
  ["Edmonton", "city", 53.55, -113.49, 150],
  ["La Ronge", "town", 55.1, -105.28, 150],
  ["Saskatoon", "city", 52.13, -106.67, 150],
  ["Flin Flon", "town", 54.77, -101.88, 120],
  ["Lynn Lake", "town", 56.85, -101.05, 120],
  ["Thompson", "town", 55.74, -97.86, 150],
  ["Churchill", "town", 58.77, -94.17, 150],
  ["Red Lake", "town", 51.02, -93.83, 120],
  ["Moosonee", "town", 51.27, -80.65, 120],
  ["Radisson", "town", 53.79, -77.62, 150],
  ["Eastmain", "town", 52.23, -78.52, 150],
  ["Kuujjuaq", "town", 58.1, -68.42, 150],
  ["Schefferville", "town", 54.8, -66.82, 150],
  ["Labrador City", "town", 52.94, -66.91, 120],
  ["Goose Bay", "town", 53.3, -60.33, 120],
  ["Jasper", "town", 52.87, -118.08, 120],
  ["Hinton", "town", 53.4, -117.58, 120],
  ["Whitecourt", "town", 54.14, -115.68, 120],
  ["Peace River", "town", 56.23, -117.29, 120],
  ["Fort Vermilion", "town", 58.39, -116.0, 120],
  ["Fort Chipewyan", "town", 58.72, -111.15, 120],
  ["Wabasca", "town", 55.96, -113.83, 120],
  ["Tumbler Ridge", "town", 55.13, -121.0, 120],
  ["Lytton", "town", 50.23, -121.58, 120],
  ["Enterprise", "town", 60.55, -116.14, 120],
  ["Behchoko", "town", 62.8, -116.05, 120],
  ["Fort Good Hope", "town", 66.26, -128.63, 150],
  ["Old Crow", "town", 67.57, -139.83, 150],
  ["Uranium City", "town", 59.57, -108.61, 120],
  ["Stony Rapids", "town", 59.25, -105.83, 120],
  ["Cranberry Portage", "town", 54.59, -101.38, 120],
  ["Norway House", "town", 53.98, -97.84, 120],
  ["Gillam", "town", 56.35, -94.71, 120],
  ["Pickle Lake", "town", 51.47, -90.2, 120],
  ["Chisasibi", "town", 53.78, -78.9, 120],
  ["Mistissini", "town", 50.42, -73.87, 120],
  ["Sept-Iles", "town", 50.22, -66.38, 120],
  ["Fermont", "town", 52.78, -67.08, 120],
  // larger centres inside the analysis area
  ["Calgary", "city", 51.05, -114.07, 100],
  ["Regina", "city", 50.45, -104.62, 100],
  ["Kamloops", "city", 50.67, -120.33, 100],
  ["Red Deer", "city", 52.27, -113.81, 100],
  ["Medicine Hat", "town", 50.04, -110.68, 100],
  ["Banff", "town", 51.18, -115.57, 100],
  ["Revelstoke", "town", 51.0, -118.2, 100],
  ["Whistler", "town", 50.12, -122.95, 100],
  ["Williams Lake", "town", 52.13, -122.14, 100],
  ["Quesnel", "town", 52.98, -122.49, 100],
  ["Prince Rupert", "town", 54.31, -130.32, 100],
  ["Terrace", "town", 54.52, -128.6, 100],
  ["Smithers", "town", 54.78, -127.17, 100],
  ["Dawson Creek", "town", 55.76, -120.24, 100],
  ["Lloydminster", "town", 53.28, -110.0, 100],
  ["Cold Lake", "town", 54.46, -110.18, 100],
  ["Prince Albert", "town", 53.2, -105.75, 100],
  ["The Pas", "town", 53.82, -101.25, 100],
  ["Dauphin", "town", 51.15, -100.05, 100],
  ["Sioux Lookout", "town", 50.1, -91.92, 100],
  ["Rankin Inlet", "town", 62.81, -92.09, 100],
  ["Iqaluit", "town", 63.75, -68.52, 100],
  // well-known centres outside the analysis area: reachable, and their report says so
  ["Vancouver", "city", 49.28, -123.12, 100],
  ["Victoria", "city", 48.43, -123.37, 100],
  ["Kelowna", "city", 49.89, -119.5, 100],
  ["Lethbridge", "city", 49.69, -112.84, 100],
  ["Winnipeg", "city", 49.9, -97.14, 100],
  ["Thunder Bay", "city", 48.38, -89.25, 100],
  ["Timmins", "town", 48.48, -81.33, 100],
  ["Sudbury", "city", 46.49, -81.0, 100],
  ["Toronto", "city", 43.65, -79.38, 100],
  ["Ottawa", "city", 45.42, -75.7, 100],
  ["Montreal", "city", 45.5, -73.57, 100],
  ["Quebec City", "city", 46.81, -71.21, 100],
  ["Chibougamau", "town", 49.91, -74.37, 100],
  ["Val-d'Or", "town", 48.1, -77.78, 100],
  ["Fredericton", "city", 45.96, -66.64, 100],
  ["Halifax", "city", 44.65, -63.58, 100],
  ["St. John's", "city", 47.56, -52.71, 100],
  ["Great Slave Lake", "lake", 61.5, -114.0, 200],
  ["Great Bear Lake", "lake", 66.0, -121.0, 220],
  ["Lake Athabasca", "lake", 59.2, -109.3, 180],
  ["Wood Buffalo", "park", 59.4, -112.9, 180],
  ["James Bay", "bay", 53.0, -80.0, 220],
];
export const PLACES = TABLE.map(([name, kind, lat, lon, km]) => ({ name, kind, lat, lon, km }));

const REGION_WORDS = { canada: "canada", all: "canada", theatre: "canada", west: "west", central: "central", centre: "central", center: "central", east: "east" };
const PANEL_WORDS = { brief: "brief", verdict: "brief", record: "record", history: "record", trend: "record", season: "season", daily: "season", sensors: "sensors", bridge: "sensors", method: "method", help: "method" };
const MONTHS = { jan: 1, feb: 2, mar: 3, apr: 4, may: 5, jun: 6, jul: 7, aug: 8, sep: 9, oct: 10, nov: 11, dec: 12 };
const pad = (n) => String(n).padStart(2, "0");

/**
 * "58.4 -112.1", "58.4N 112.1W", "58.4, 112.1": latitude then longitude. A longitude given
 * without a sign or hemisphere is taken as west, because the theatre is North America.
 */
export function parseCoords(text) {
  const m = text.trim().match(/^(-?\d{1,2}(?:\.\d+)?)\s*°?\s*([ns])?[\s,;/]+(-?\d{1,3}(?:\.\d+)?)\s*°?\s*([ew])?$/i);
  if (!m) return null;
  let lat = Number(m[1]), lon = Number(m[3]);
  if (m[2]?.toLowerCase() === "s") lat = -Math.abs(lat);
  if (m[4]) lon = m[4].toLowerCase() === "w" ? -Math.abs(lon) : Math.abs(lon);
  else if (lon > 0) lon = -lon;
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return { lat, lon };
}

/** "jul 14", "14 jul", "july 14", "2023-07-14" -> ISO date in `year` (or the year given). */
export function parseDay(text, year) {
  const t = text.trim().toLowerCase();
  const iso = t.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
  let y = year, mo, d;
  if (iso) [y, mo, d] = [Number(iso[1]), Number(iso[2]), Number(iso[3])];
  else {
    const a = t.match(/^([a-z]{3})[a-z]*\.?\s+(\d{1,2})$/), b = t.match(/^(\d{1,2})\s+([a-z]{3})[a-z]*\.?$/);
    if (a && MONTHS[a[1]]) [mo, d] = [MONTHS[a[1]], Number(a[2])];
    else if (b && MONTHS[b[2]]) [mo, d] = [MONTHS[b[2]], Number(b[1])];
    else return null;
  }
  const date = `${y}-${pad(mo)}-${pad(d)}`;
  return new Date(`${date}T00:00:00Z`).toISOString().slice(0, 10) === date ? date : null;
}

/** The nearest town or city within `maxKm` of a point, or null: a name for an unnamed click. */
export function nearestPlace(lat, lon, maxKm = 120) {
  let best = null, bestKm = maxKm;
  for (const p of PLACES) {
    if (p.kind !== "town" && p.kind !== "city") continue;
    const km = haversineKm(lat, lon, p.lat, p.lon);
    if (km <= bestKm) [best, bestKm] = [p, km];
  }
  return best;
}

/**
 * Everything a query could mean, best first. `ctx` is { year, years, dates }.
 * Each result is { type, label, hint, ... } and is safe to run as is. At most `limit` come back.
 */
export function interpret(query, ctx, limit = 7) {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  const out = [];
  const coords = parseCoords(q);
  if (coords) out.push({ type: "coords", ...coords, km: 100, label: `${coords.lat.toFixed(3)}, ${coords.lon.toFixed(3)}`, hint: "fly to coordinates" });
  if (/^\d{4}$/.test(q) && ctx.years.includes(Number(q))) out.push({ type: "year", year: Number(q), label: q, hint: "open season" });
  const day = parseDay(q, ctx.year);
  if (day) {
    const y = Number(day.slice(0, 4));
    if (y === ctx.year ? ctx.dates.includes(day) : ctx.years.includes(y)) out.push({ type: "date", date: day, year: y, label: day, hint: "show this day" });
  }
  if (q === "peak" || q === "peak day") out.push({ type: "peak", label: "PEAK DAY", hint: "busiest day of the season" });
  if (q === "all" || q === "all days" || q === "season" || q === "whole season") out.push({ type: "date", date: "all", year: ctx.year, label: "WHOLE SEASON", hint: "show every day" });
  if (REGION_WORDS[q]) out.push({ type: "region", key: REGION_WORDS[q], label: REGION_WORDS[q].toUpperCase(), hint: "select region" });
  if (PANEL_WORDS[q]) out.push({ type: "panel", key: PANEL_WORDS[q], label: PANEL_WORDS[q].toUpperCase(), hint: "open panel" });
  if (q.length >= 2) {
    const starts = PLACES.filter((p) => p.name.toLowerCase().startsWith(q));
    const contains = PLACES.filter((p) => !p.name.toLowerCase().startsWith(q) && p.name.toLowerCase().includes(q));
    for (const p of [...starts, ...contains]) out.push({ type: "place", ...p, label: p.name, hint: p.kind });
  }
  return out.slice(0, limit);
}
