// Geography on the 0.25 degree analysis grid: distances, cell snapping, and what happened
// in a zone (one cell, or everything within a radius) over the loaded season.

export const GRID = 0.25;
const R_KM = 6371;
const RAD = Math.PI / 180;

export function haversineKm(lat1, lon1, lat2, lon2) {
  const dLat = (lat2 - lat1) * RAD, dLon = (lon2 - lon1) * RAD;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * RAD) * Math.cos(lat2 * RAD) * Math.sin(dLon / 2) ** 2;
  return 2 * R_KM * Math.asin(Math.min(1, Math.sqrt(a)));
}

/** Centre of the grid cell containing a point. */
export const snapCell = (lat, lon) => [
  Math.floor(lat / GRID) * GRID + GRID / 2,
  Math.floor(lon / GRID) * GRID + GRID / 2,
];

/** Ground size of a cell at a latitude, in km: [east-west, north-south]. */
export const cellSizeKm = (lat) => [GRID * RAD * R_KM * Math.cos(lat * RAD), GRID * RAD * R_KM];

/** Nearest map cell ([lat, lon, ...]) to a point within `maxKm`, or null. */
export function nearestCell(cells, lat, lon, maxKm) {
  let best = null, bestKm = maxKm;
  const band = maxKm / 111 + GRID;
  for (const c of cells) {
    if (Math.abs(c[0] - lat) > band) continue;
    const km = haversineKm(lat, lon, c[0], c[1]);
    if (km <= bestKm) {
      best = c;
      bestKm = km;
    }
  }
  return best;
}

/**
 * What the season's records say about a zone. `radiusKm` 0 means the single grid cell
 * centred on (lat, lon); otherwise every cell whose centre is within the radius.
 * Rows are [date, lat, lon, modis, viirs].
 */
export function zoneReport(rows, lat, lon, radiusKm = 0) {
  const band = radiusKm / 111 + GRID;
  const days = new Map();
  const cells = new Set();
  let modis = 0, viirs = 0, cellDays = 0, sharedCellDays = 0, modisCellDays = 0, viirsCellDays = 0;
  for (const row of rows) {
    if (radiusKm === 0) {
      if (row[1] !== lat || row[2] !== lon) continue;
    } else {
      if (Math.abs(row[1] - lat) > band) continue;
      if (haversineKm(lat, lon, row[1], row[2]) > radiusKm) continue;
    }
    const m = row[3], v = row[4];
    modis += m;
    viirs += v;
    cellDays += 1;
    sharedCellDays += Number(m > 0 && v > 0);
    modisCellDays += Number(m > 0);
    viirsCellDays += Number(v > 0);
    cells.add(row[1] * 1e4 + row[2]);
    let d = days.get(row[0]);
    if (!d) days.set(row[0], (d = { date: row[0], modis: 0, viirs: 0, cells: 0 }));
    d.modis += m;
    d.viirs += v;
    d.cells += 1;
  }
  const daily = [...days.values()].sort((a, b) => (a.date < b.date ? -1 : 1));
  const peak = daily.reduce((best, d) => (!best || d.cells > best.cells || (d.cells === best.cells && d.modis + d.viirs > best.modis + best.viirs) ? d : best), null);
  return {
    cells: cells.size,
    modis, viirs, cellDays, sharedCellDays, modisCellDays, viirsCellDays,
    activeDays: daily.length,
    first: daily[0]?.date ?? null,
    last: daily[daily.length - 1]?.date ?? null,
    peak: peak?.date ?? null,
    daily,
  };
}

/** The rectangle the archive covers: [[south, west], [north, east]]. */
export const THEATRE = [[50, -140], [70, -60]];

/**
 * How much of a zone the archive covers: "inside", "partial" (it reaches past an edge) or
 * "outside". Outside means no data, which is not the same as no fire.
 */
export function coverageOf(lat, lon, radiusKm = 0) {
  const [[s, w], [n, e]] = THEATRE;
  const within = lat >= s && lat < n && lon >= w && lon < e;
  if (radiusKm === 0) return within ? "inside" : "outside";
  const nearLat = Math.max(s, Math.min(n, lat)), nearLon = Math.max(w, Math.min(e, lon));
  if (!within) return haversineKm(lat, lon, nearLat, nearLon) < radiusKm ? "partial" : "outside";
  const toEdge = Math.min(
    haversineKm(lat, lon, s, lon), haversineKm(lat, lon, n, lon),
    haversineKm(lat, lon, lat, w), haversineKm(lat, lon, lat, e),
  );
  return toEdge >= radiusKm ? "inside" : "partial";
}

/** "58.375°N 112.125°W" */
export function formatCoords(lat, lon, digits = 3) {
  return `${Math.abs(lat).toFixed(digits)}°${lat >= 0 ? "N" : "S"} ${Math.abs(lon).toFixed(digits)}°${lon >= 0 ? "E" : "W"}`;
}
