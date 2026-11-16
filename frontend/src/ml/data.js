// Research points stay independent of historical 0.25-degree cell-day analysis.
export const LAYERS = [
  ["all", "Compare"], ["mlp", "MLP"], ["modis", "MODIS"], ["viirs", "VIIRS reference"], ["difference", "MLP / MODIS difference"],
];
export const COLORS = { coverage: "#34495c", mlp: "#55ccff", modis: "#ffb224", both: "#e6edf5", viirs: "#b794ff" };
export const layerValid = (value) => LAYERS.some(([key]) => key === value);
export function visibleFeatures(features, layer) {
  return features.filter(({ properties: p }) => layer === "all" || (layer === "mlp" ? p.mlp_detected : layer === "modis" ? p.modis_detected : layer === "viirs" ? p.viirs_reference : p.mlp_detected !== p.modis_detected));
}
export function featureColor(feature, layer) {
  const p = feature.properties;
  if (layer === "mlp") return COLORS.mlp;
  if (layer === "modis") return COLORS.modis;
  if (layer === "viirs") return COLORS.viirs;
  return p.mlp_detected && p.modis_detected ? COLORS.both : p.mlp_detected ? COLORS.mlp : p.modis_detected ? COLORS.modis : COLORS.viirs;
}
export function researchPoints(features, coverage, layer, showCoverage) {
  const base = showCoverage ? coverage.map(([lon, lat]) => ({ lat, lon, color: COLORS.coverage, weight: 0.03 })) : [];
  return [...base, ...visibleFeatures(features, layer).map((f) => ({ lat: f.geometry.coordinates[1], lon: f.geometry.coordinates[0], color: featureColor(f, layer), weight: 0.65 }))];
}
export function readMLView(search) {
  const q = new URLSearchParams(search);
  return { scene: q.get("scene") || "", layer: layerValid(q.get("layer")) ? q.get("layer") : "all", coverage: q.get("coverage") !== "0" };
}
export function writeMLView(scene, layer, coverage) {
  const q = new URLSearchParams({ mode: "ml" });
  if (scene) q.set("scene", scene);
  if (layer !== "all") q.set("layer", layer);
  if (!coverage) q.set("coverage", "0");
  return `?${q}`;
}
export function nearestFeature(features, lat, lon, maxKm = 1.5) {
  let best = null, distance = maxKm;
  for (const f of features) {
    const [x, y] = f.geometry.coordinates;
    const km = 111.195 * Math.hypot(y - lat, (x - lon) * Math.cos(lat * Math.PI / 180));
    if (km < distance) { best = f; distance = km; }
  }
  return best;
}
export function validateScene(data, coverage, expectedId, version) {
  if (data.type !== "FeatureCollection" || data.scene?.id !== expectedId || data.scene.model_version !== version || !Array.isArray(data.features)) throw Error("The scene does not match this model version.");
  if (coverage.geometry?.type !== "MultiPoint" || coverage.properties?.scene_id !== expectedId || coverage.properties.count !== coverage.geometry.coordinates.length) throw Error("The scene coverage is invalid.");
  const good = (p) => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite) && Math.abs(p[0]) <= 180 && Math.abs(p[1]) <= 90;
  if (!coverage.geometry.coordinates.every(good) || !data.features.every((f) => good(f.geometry?.coordinates) && Number.isFinite(f.properties?.mlp_score))) throw Error("The scene contains invalid geographic predictions.");
  return { ...data, coverage: coverage.geometry.coordinates };
}
export function geoJSONDownload(data, layer) {
  return { type: "FeatureCollection", model_version: data.scene.model_version, observed_at: data.scene.observed_at,
    coverage_kind: data.scene.coverage_kind, layer, features: visibleFeatures(data.features, layer),
    note: "Experimental cached predictions. Unrepresented locations are unavailable; scores are not guaranteed fire probabilities." };
}
