import { validateScene } from "./data.js";

async function get(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw Error(`HTTP ${response.status}`);
  return response.json();
}
export async function loadCatalogue(signal) {
  const api = (import.meta.env.VITE_ML_API_URL || "/api/ml").replace(/\/$/, "");
  try {
    const [model, result] = await Promise.all([get(`${api}/model`, signal), get(`${api}/scenes`, signal)]);
    if (!model.version || model.version !== result.model_version || !result.scenes?.length) throw Error("Model catalogue mismatch");
    return { model, scenes: result.scenes, source: "api", base: api, message: "Backend cache" };
  } catch (error) {
    if (signal.aborted || error.name === "AbortError") throw error;
    const base = `${import.meta.env.BASE_URL}data/ml`;
    const catalogue = await get(`${base}/catalog.json`, signal);
    if (catalogue.schema !== "epoch_mlp_cache_v1" || !catalogue.scenes?.length) throw Error("The bundled MLP catalogue is unavailable.");
    return { ...catalogue, source: "snapshot", base, message: "Bundled snapshot · API unavailable" };
  }
}
export async function loadScene(catalogue, id, signal) {
  const api = catalogue.source === "api";
  const [data, coverage] = await Promise.all([
    get(`${catalogue.base}/scenes/${id}${api ? "/predictions" : ".json"}`, signal),
    get(api ? `${catalogue.base}/scenes/${id}/coverage` : `${catalogue.base}/coverage/${id}.json`, signal),
  ]);
  return validateScene(data, coverage, id, catalogue.model.version);
}
