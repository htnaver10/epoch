import { COLORS } from "./data.js";

export const OVERVIEW_LAYERS = [["all", "Compare"], ["mlp", "50.6M MLP"], ["modis", "MODIS"], ["viirs", "VIIRS reference"], ["difference", "Difference"]];
export const CANADA_BOUNDS = [[38, -141], [83, -52]];
export function overviewView(search) {
  const q = new URLSearchParams(search);
  return { date: q.get("date") || "", split: ["all", "train", "validation", "test", "additional"].includes(q.get("split")) ? q.get("split") : "all",
    region: q.get("region") || "", layer: OVERVIEW_LAYERS.some(([id]) => id === q.get("layer")) ? q.get("layer") : "all", coverage: q.get("coverage") !== "0",
    tile: q.get("tile") || "", outlines: q.get("outlines") !== "0", labels: q.get("labels") !== "0" };
}
export function overviewURL(view) {
  const q = new URLSearchParams({ mode: "ml", view: "overview" });
  for (const key of ["date", "region", "tile"]) if (view[key]) q.set(key, view[key]);
  if (view.split !== "all") q.set("split", view.split);
  if (view.layer !== "all") q.set("layer", view.layer);
  if (!view.coverage) q.set("coverage", "0");
  if (view.outlines === false) q.set("outlines", "0");
  if (view.labels === false) q.set("labels", "0");
  return `?${q}`;
}
export function aggregateCells(records, { date = "", split = "all", region = "" } = {}) {
  const cells = new Map();
  for (const record of records) {
    if ((date && record.date !== date) || (split !== "all" && record.split !== split) || (region && record.group !== region)) continue;
    for (const [lat, lon, queries, mlp, modis, viirs, mlpOnly, modisOnly, score, sceneSamples] of record.cells) {
      const key = `${lat}:${lon}`;
      const cell = cells.get(key) || { key, lat, lon, queries: 0, mlp: 0, modis: 0, viirs: 0, mlpOnly: 0, modisOnly: 0, score: 0, sceneSamples: 0, dates: new Set() };
      for (const [field, value] of Object.entries({ queries, mlp, modis, viirs, mlpOnly, modisOnly, sceneSamples })) cell[field] += value;
      cell.score = Math.max(cell.score, score); cell.dates.add(record.date); cells.set(key, cell);
    }
  }
  return [...cells.values()].map((c) => ({ ...c, dates: [...c.dates].sort() }));
}
export const alertCell = (c, layer) => layer === "mlp" ? c.mlp > 0 : layer === "modis" ? c.modis > 0 : layer === "viirs" ? c.viirs > 0 : layer === "difference" ? c.mlpOnly + c.modisOnly > 0 : c.mlp + c.modis + c.viirs > 0;
export function cellColor(c, layer) {
  if (layer === "mlp") return COLORS.mlp;
  if (layer === "modis") return COLORS.modis;
  if (layer === "viirs") return COLORS.viirs;
  const m = layer === "difference" ? c.mlpOnly : c.mlp, s = layer === "difference" ? c.modisOnly : c.modis;
  return m && s ? COLORS.both : m ? COLORS.mlp : s ? COLORS.modis : COLORS.viirs;
}
export function overviewPoints(cells, layer, coverage) {
  return [...(coverage ? cells.map((c) => ({ lat: c.lat, lon: c.lon, color: COLORS.coverage, weight: 0.08 })) : []),
    ...cells.filter((c) => alertCell(c, layer)).map((c) => ({ lat: c.lat, lon: c.lon, color: cellColor(c, layer), weight: 0.65 }))];
}
export function cellAt(cells, lat, lon, grid) {
  const y = Math.floor((lat + 90) / grid) * grid - 90 + grid / 2;
  const x = Math.floor((lon + 180) / grid) * grid - 180 + grid / 2;
  return cells.find((c) => c.lat === y && c.lon === x) || null;
}
export function overviewExport(cells, catalogue, view) {
  return { type: "FeatureCollection", model_version: catalogue.model.version, grid_degrees: catalogue.grid_degrees,
    filters: view, scope: catalogue.scope, count_semantics: catalogue.count_semantics,
    features: cells.filter((c) => alertCell(c, view.layer)).map(({ lat, lon, key, ...properties }) => ({ type: "Feature", id: key, geometry: { type: "Point", coordinates: [lon, lat] }, properties })) };
}
