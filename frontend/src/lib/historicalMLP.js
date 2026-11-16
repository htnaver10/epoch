import { useEffect, useMemo, useState } from "react";
import { COLORS } from "../ml/data.js";
import { inArea } from "./area.js";
import { GRID, snapCell } from "./geo.js";

const DATA = `${import.meta.env?.BASE_URL || "/"}data/previirs-mlp/`;
const EMPTY = [];
const validCoordinate = (lat, lon) => Number.isFinite(lat) && Number.isFinite(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180;
const validCount = (count) => Number.isSafeInteger(count) && count >= 0;

function insideScope(lat, lon, bounds, target) {
  return lat >= bounds[0][0] && lat < bounds[1][0] && lon >= bounds[0][1] && lon < bounds[1][1]
    && (!target || inArea(target, lat, lon));
}

/** Convert compact cell counts or legacy query points into the selected archive-grid layer. */
export function summarizeHistoricalScenes(scenes, { date, firstDay, lastDay, bounds, target = null }) {
  const matches = scenes.filter((scene) =>
    date !== "all" ? scene.date === date : !firstDay || (scene.date >= firstDay && scene.date <= lastDay)
  );
  const byCell = new Map();
  const byDateCell = new Map();
  let queries = 0;
  let alerts = 0;

  const add = (isoDate, lat, lon, scoredQueries, plsworkAlerts) => {
    if (!validCoordinate(lat, lon) || !validCount(scoredQueries) || !validCount(plsworkAlerts) || (!scoredQueries && !plsworkAlerts)) return;
    const [cellLat, cellLon] = snapCell(lat, lon);
    if (!insideScope(cellLat, cellLon, bounds, target)) return;
    const key = `${cellLat},${cellLon}`;
    const cell = byCell.get(key) || { lat: cellLat, lon: cellLon, scoredQueries: 0, alerts: 0 };
    cell.scoredQueries += scoredQueries;
    cell.alerts += plsworkAlerts;
    byCell.set(key, cell);
    const recordKey = `${isoDate}:${key}`;
    const record = byDateCell.get(recordKey) || { date: isoDate, lat: cellLat, lon: cellLon, scoredQueries: 0, alerts: 0 };
    record.scoredQueries += scoredQueries;
    record.alerts += plsworkAlerts;
    byDateCell.set(recordKey, record);
    queries += scoredQueries;
    alerts += plsworkAlerts;
  };

  for (const scene of matches) {
    // Prefer genuine per-cell totals when present. Precise 375 m query coordinates remain a
    // fallback for the original pilot files and are counted before being grouped to the grid.
    if (Array.isArray(scene.cells) && scene.cells.length) {
      for (const cell of scene.cells) {
        add(scene.date, cell.lat, cell.lon, cell.scored_queries, cell.plswork_alerts);
      }
      continue;
    }
    for (const point of Array.isArray(scene.coverage) ? scene.coverage : EMPTY) {
      if (Array.isArray(point) && point.length >= 2) add(scene.date, point[1], point[0], 1, 0);
    }
    for (const point of Array.isArray(scene.alerts) ? scene.alerts : EMPTY) {
      if (Array.isArray(point) && point.length >= 2) add(scene.date, point[1], point[0], 0, 1);
    }
  }

  const cells = [...byCell.values()];
  const points = cells.map((cell) => ({
    lat: cell.lat,
    lon: cell.lon,
    scoredQueries: cell.scoredQueries,
    alerts: cell.alerts,
    color: cell.alerts > 0 ? COLORS.mlp : COLORS.coverage,
    weight: cell.alerts > 0 ? 0.65 : 0.12,
  }));
  const dates = [...new Set(matches.map((scene) => scene.date))].sort();
  return {
    dates,
    sceneDates: dates.length,
    alerts,
    queries,
    points,
    records: [...byDateCell.values()].sort((a, b) => a.date.localeCompare(b.date) || a.lat - b.lat || a.lon - b.lon),
    cellDegrees: GRID,
  };
}

export function historicalCellsCsv(records) {
  const rows = ["date,latitude,longitude,scored_queries,plswork_alerts,grid_degrees"];
  for (const record of records) {
    rows.push([
      record.date,
      record.lat,
      record.lon,
      record.scoredQueries,
      record.alerts,
      GRID,
    ].join(","));
  }
  return `${rows.join("\r\n")}\r\n`;
}

export function useHistoricalMLP(year, enabled, date, firstDay, lastDay, bounds, target = null) {
  const [loaded, setLoaded] = useState(null);
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    async function read(file) {
      const response = await fetch(DATA + file, { signal: controller.signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    }
    async function load() {
      const index = await read("index.json");
      const years = Array.isArray(index.years) ? index.years.map(Number) : EMPTY;
      const scenes = years.includes(year) ? (await read(`${year}.json`)).scenes : EMPTY;
      if (!controller.signal.aborted) setLoaded({ year, scenes: Array.isArray(scenes) ? scenes : EMPTY });
    }
    load().catch(() => {
      if (!controller.signal.aborted) setLoaded({ year, scenes: EMPTY, error: true });
    });
    return () => controller.abort();
  }, [enabled, year]);

  return useMemo(() => {
    if (!enabled) return null;
    if (loaded?.year !== year) return { loading: true };
    if (loaded.error) return { error: true };
    const result = summarizeHistoricalScenes(loaded.scenes, { date, firstDay, lastDay, bounds, target });
    if (!result.queries && !result.alerts) return { ...result, missing: true };
    return result;
  }, [enabled, year, loaded, date, firstDay, lastDay, bounds, target]);
}
