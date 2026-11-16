const REGIONS = { bc: "British Columbia", qc: "Quebec", alberta: "Alberta", canada: "Canada", nwt: "Northwest Territories" };
export const tileCode = (tile) => tile.id.slice(0, 8);
export function tileRegion(tile) {
  return tile.group.split("_").map((part) => REGIONS[part] || part.charAt(0).toUpperCase() + part.slice(1)).join(" ");
}
export function tileTitle(tile) { return `${tileRegion(tile)} · ${tile.date} · Tile ${tileCode(tile)}`; }
export function filterTiles(tiles, { date = "", split = "all", region = "" } = {}, search = "") {
  const term = search.trim().toLowerCase();
  return tiles.filter((tile) => (!date || tile.date === date) && (split === "all" || tile.split === split)
    && (!region || tile.group === region) && (!term || `${tile.id} ${tileTitle(tile)} ${tile.group}`.toLowerCase().includes(term)))
    .sort((a, b) => b.counts.mlp - a.counts.mlp || a.id.localeCompare(b.id));
}
export function tilesAt(tiles, lat, lon) {
  return tiles.filter(({ bounds: [w,s,e,n] }) => lon >= w && lon <= e && lat >= s && lat <= n);
}
export const tileViewLink = (tile) => `?${new URLSearchParams({ mode: "ml", view: "presentation", split: tile.split, scene: tile.id })}`;
export const tileBounds = (tile) => [[tile.bounds[1],tile.bounds[0]], [tile.bounds[3],tile.bounds[2]]];
