// Two corrections to the hotspot record, made the same way here and in the data pipeline
// (scripts/prepare_seasons.py), so a built-in sector and any area of interest are cleaned
// alike. Which cells are static heat sources, and which days are missing from an archive,
// is decided once by the pipeline and published in seasons.json.

/** "lat,lon" keys of the static heat sources (refineries, upgraders, gas plants, a cement kiln). */
export const staticKeys = (seasons) => new Set((seasons?.static_sources || []).map(([lat, lon]) => `${lat},${lon}`));

/** Cell-day records [iso, lat, lon, modis, viirs] without the static heat sources. */
export function withoutStatic(records, keys) {
  if (!keys || !keys.size) return records;
  return records.filter((r) => !keys.has(`${r[1]},${r[2]}`));
}

/**
 * Fill the days missing from one sensor's archive in one year's series (the shape of
 * `seasons.series[region][year]`), in place: each gap day becomes what the other sensor saw
 * that day, scaled by the ratio between the two over the neighbouring days that are not gaps.
 * With nothing from the other sensor nearby it is the mean of the sensor's own neighbours.
 * A gap day is never lowered. Mirrors fill_gaps() in prepare_seasons.py.
 */
export function fillGaps(row, gaps, year, reach = 3) {
  if (!gaps) return row;
  for (const [sensor, other] of [["modis", "viirs"], ["viirs", "modis"]]) {
    const days = gaps[sensor]?.[year] || [];
    if (!days.length || !row[sensor] || !row[other]) continue;
    const skip = new Set([...days, ...(gaps[other]?.[year] || [])]);
    const n = row[sensor].cells.length;
    for (const i of days) {
      const near = [];
      for (let j = i - reach; j <= i + reach; j++) if (j >= 0 && j < n && !skip.has(j)) near.push(j);
      for (const k of ["cells", "w", "raw"]) {
        const mine = near.reduce((a, j) => a + row[sensor][k][j], 0);
        const theirs = near.reduce((a, j) => a + row[other][k][j], 0);
        const estimate = theirs ? (row[other][k][i] * mine) / theirs : near.length ? mine / near.length : 0;
        row[sensor][k][i] = Math.max(row[sensor][k][i], Math.round(estimate * 100) / 100);
      }
    }
  }
  return row;
}
