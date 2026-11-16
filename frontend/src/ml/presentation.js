export const SPLITS = [["test", "Benchmark"], ["validation", "Validation"], ["train", "Training"], ["additional", "Additional windows"], ["all", "All scenes"]];
export const VIEWS = [["compare", "Compare all"], ["mlp", "50.6M MLP"], ["modis", "MODIS"], ["viirs", "VIIRS reference"], ["difference", "Difference"]];
export const splitLabel = (key) => SPLITS.find(([id]) => id === key)?.[1] || key;
export function presentationView(search) {
  const q = new URLSearchParams(search);
  return { scene: q.get("scene") || "", split: SPLITS.some(([id]) => id === q.get("split")) ? q.get("split") : "test",
    layer: VIEWS.some(([id]) => id === q.get("layer")) ? q.get("layer") : "compare" };
}
export function filterScenes(scenes, split, date, search, sort) {
  const term = search.trim().toLowerCase();
  return scenes.filter((s) => (split === "all" || s.split === split) && (!date || s.date === date)
    && (!term || `${s.id} ${s.group} ${s.date}`.toLowerCase().includes(term)))
    .sort((a, b) => (sort === "date" ? b.observed_at.localeCompare(a.observed_at) : b.counts[sort === "alerts" ? "mlp" : "viirs"] - a.counts[sort === "alerts" ? "mlp" : "viirs"]) || a.id.localeCompare(b.id));
}
export function selectScene(scenes, preferred) {
  return scenes.find((scene) => scene.id === preferred) || scenes[0] || null;
}
export function viewURL(scene, split, layer) {
  const params = new URLSearchParams({ mode: "ml", view: "presentation", split, layer });
  if (scene) params.set("scene", scene);
  return `?${params}`;
}
