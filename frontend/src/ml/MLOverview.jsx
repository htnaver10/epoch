import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { COLORS } from "./data.js";
import { filterTiles, tileBounds, tileCode, tileRegion, tileTitle, tilesAt, tileViewLink } from "./tiles.js";
import { aggregateCells, alertCell, CANADA_BOUNDS, cellAt, overviewExport, overviewPoints, overviewURL, overviewView } from "./overview.js";
import "./ml.css";
import "./overview.css";

const Globe = lazy(() => import("../components/Globe.jsx"));
const EMPTY = [], SECTORS = {};
const LAYERS = [["all", "Compare"], ["mlp", "50.6M MLP"], ["modis", "MODIS"], ["viirs", "VIIRS reference"], ["difference", "Difference"]];
const number = (n) => Number(n).toLocaleString();
const names = (s) => s.replaceAll("_", " ");

export default function MLOverview({ onMode }) {
  const [view, setView] = useState(() => overviewView(window.location.search));
  const [result, setResult] = useState(null), [attempt, setAttempt] = useState(0);
  const [picked, setPicked] = useState(null), [frameTarget, setFrameTarget] = useState(() => overviewView(window.location.search).tile ? "tile" : "canada"), [fit, setFit] = useState(0);
  const [tileSearch, setTileSearch] = useState("");
  const [markerScale, setMarkerScale] = useState(1);
  const catalogue = result?.attempt === attempt ? result.value : null;
  const tileIndex = result?.attempt === attempt ? result.tiles : null;
  const error = result?.attempt === attempt ? result.error : "";
  useEffect(() => {
    const controller = new AbortController();
    Promise.all(["ml-overview.json", "ml-tiles.json"].map(async (path) => {
      const response = await fetch(`${import.meta.env.BASE_URL}data/${path}`, { signal: controller.signal });
      if (!response.ok) throw Error(`Overview HTTP ${response.status}`);
      return response.json();
    })).then(([value, tiles]) => {
      if (value.schema !== "epoch_ml_overview_v1" || !value.records?.length || !value.model?.version) throw Error("The bundled overview is incomplete.");
      if (tiles.schema !== "epoch_ml_tiles_v1" || tiles.model_version !== value.model.version || tiles.source_catalogue_sha256 !== value.source_catalogue_sha256) throw Error("Tile identities do not match this overview.");
      if (!controller.signal.aborted) setResult({ attempt, value, tiles });
    }).catch((e) => { if (!controller.signal.aborted) setResult({ attempt, error: e.message }); });
    return () => controller.abort();
  }, [attempt]);
  useEffect(() => { window.history.replaceState(null, "", overviewURL(view)); }, [view]);
  const cells = useMemo(() => aggregateCells(catalogue?.records || [], view), [catalogue, view]);
  const tiles = useMemo(() => filterTiles(tileIndex?.tiles || [], view), [tileIndex, view]);
  const tileMatches = useMemo(() => filterTiles(tiles, {}, tileSearch), [tiles, tileSearch]);
  const selectedTile = tiles.find((t) => t.id === view.tile);
  const shown = useMemo(() => cells.filter((c) => alertCell(c, view.layer)), [cells, view.layer]);
  const points = useMemo(() => overviewPoints(cells, view.layer, view.coverage), [cells, view.layer, view.coverage]);
  const bounds = useMemo(() => {
    if (frameTarget === "tile" && selectedTile) return tileBounds(selectedTile);
    if (frameTarget === "canada" || !cells.length) return CANADA_BOUNDS;
    return [[Math.min(...cells.map((c) => c.lat)) - 0.3, Math.min(...cells.map((c) => c.lon)) - 0.3],
      [Math.max(...cells.map((c) => c.lat)) + 0.3, Math.max(...cells.map((c) => c.lon)) + 0.3]];
  }, [cells, frameTarget, selectedTile]);
  const selected = cells.find((c) => c.key === picked?.key);
  const nearbyTiles = picked ? tilesAt(tiles, picked.lat, picked.lon) : [];
  const counts = useMemo(() => ({ mlp: cells.filter((c) => c.mlp).length, modis: cells.filter((c) => c.modis).length, viirs: cells.filter((c) => c.viirs).length }), [cells]);
  const change = (key, value) => { setPicked(null); setView((v) => ({ ...v, [key]: value, ...(["date","split","region"].includes(key) ? { tile: "" } : {}) })); };
  const chooseTile = (tile) => { setPicked(null); setView((v) => ({ ...v, tile: tile.id })); setFrameTarget("tile"); setFit((n) => n + 1); };
  const download = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(overviewExport(cells, catalogue, view))], { type: "application/geo+json" }));
    const a = document.createElement("a"); a.href = url; a.download = `epoch-overview-${view.date || "historical"}-${view.layer}.geojson`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <div className="ml-console overview-console">
    <header className="ml-header"><button className="ml-brand" onClick={() => onMode("analysis")}>EPOCH</button><nav className="seg mode" aria-label="Mode"><button onClick={() => onMode("analysis")}>Calendar</button><button onClick={() => onMode("ops")}>Scenarios</button><button className="on" aria-pressed="true">MLP research</button></nav><div className="ml-heading"><strong>Canada overview</strong><span>Sparse historical observations</span></div><a className="overview-atlas" href="?mode=ml&view=presentation">Scene atlas</a><span className="ml-source api">Bundled data · no backend</span></header>
    <main className="ml-layout"><aside className="ml-sidebar" aria-label="Overview controls">
      <section><span className="ml-eyebrow">50.6M MLP / Geographic view</span><h1>Canada overview</h1><p>Explore the saved observations on a 3D globe. Each point summarizes a 0.25° display cell.</p><p className="ml-small">Only locally processed patches are shown. This is not continuous coverage of Canada.</p></section>
      {catalogue && <section className="ml-fields">
        <label>Date (UTC)<select aria-label="Overview date" value={view.date} onChange={(e) => change("date", e.target.value)}><option value="">All cached dates · historical composite</option>{catalogue.dates.map((d) => <option key={d}>{d}</option>)}</select></label>
        <label>Dataset<select aria-label="Overview dataset" value={view.split} onChange={(e) => change("split", e.target.value)}><option value="all">All cached scenes</option><option value="test">Reused benchmark</option><option value="validation">Validation</option><option value="train">Training</option><option value="additional">Additional windows</option></select></label>
        <label>Region<select aria-label="Overview region" value={view.region} onChange={(e) => { change("region", e.target.value); setFrameTarget(e.target.value ? "coverage" : "canada"); }}><option value="">Canada and surrounding regions</option>{Object.keys(catalogue.regions).sort().map((r) => <option key={r} value={r}>{names(r)}</option>)}</select></label>
        <p className="ml-small">{view.date ? `${view.date} UTC observations.` : `${catalogue.dates[0]} to ${catalogue.dates.at(-1)}. Different years and days are combined; this is not one fire situation.`}</p>
      </section>}
      <section><h2>Map layer</h2><div className="ml-layers">{LAYERS.map(([id, label]) => <button key={id} aria-pressed={view.layer === id} className={view.layer === id ? "on" : ""} onClick={() => change("layer", id)}>{label}</button>)}</div><label className="ml-check"><input type="checkbox" checked={view.coverage} onChange={(e) => change("coverage", e.target.checked)} /> Show cells containing cached samples</label><label className="overview-marker-size">Detection marker size <output>{markerScale.toFixed(1)}x</output><input aria-label="Detection marker size" type="range" min="1" max="2" step="0.2" value={markerScale} onChange={(e) => setMarkerScale(Number(e.target.value))} /></label><div className="ml-actions"><button onClick={() => { setFrameTarget("canada"); setFit((v) => v + 1); }}>Fit Canada</button><button disabled={!cells.length} onClick={() => { setFrameTarget("coverage"); setFit((v) => v + 1); }}>Fit samples</button></div><p className="ml-small">Scroll to zoom. Drag to rotate. Use 3D or right-drag for a tilted view when zoomed in.</p></section>
      {catalogue && <section className="overview-tile-browser"><h2>Find a tile</h2><input aria-label="Find tile" placeholder="Tile ID, region or date" value={tileSearch} onChange={(e) => setTileSearch(e.target.value)} /><p className="ml-small">{number(tileMatches.length)} tiles match these filters. Select one to highlight and zoom.</p><div className="overview-tile-list" aria-label="Matching tiles">{tileMatches.slice(0,40).map((tile) => <button key={tile.id} aria-label={`Focus tile ${tile.id}`} aria-pressed={tile.id === selectedTile?.id} className={tile.id === selectedTile?.id ? "on" : ""} onClick={() => chooseTile(tile)}><strong>Tile {tileCode(tile)}</strong><span>{tileRegion(tile)} · {tile.date}</span><small>{tile.counts.mlp} MLP · {tile.counts.modis} MODIS · {tile.counts.viirs} VIIRS</small></button>)}{!tileMatches.length && <p className="ml-small">No tiles match. Clear the search or change the filters.</p>}</div>{tileMatches.length > 40 && <p className="ml-small">Showing 40 tiles. Search by ID, region or date to narrow the list.</p>}<label className="ml-check"><input type="checkbox" checked={view.outlines} onChange={(e) => change("outlines", e.target.checked)} /> Show tile boundaries</label><label className="ml-check"><input type="checkbox" checked={view.labels} onChange={(e) => change("labels", e.target.checked)} /> Show tile IDs</label><p className="ml-small">Labels separate as you zoom in. Boundaries enclose sampled queries, which may contain gaps.</p></section>}
      {catalogue && <section><h2>Cells in this view</h2><dl className="ml-counts"><div><dt>Contain MLP alerts</dt><dd>{number(counts.mlp)}</dd></div><div><dt>Contain MODIS detections</dt><dd>{number(counts.modis)}</dd></div><div><dt>Contain VIIRS positives</dt><dd>{number(counts.viirs)}</dd></div><div><dt>Contain cached samples</dt><dd>{number(cells.length)}</dd></div></dl><div className="ml-actions"><button disabled={!cells.length} onClick={download}>Download overview GeoJSON</button></div></section>}
      <section className="ml-caution"><h2>Read this as sampled history</h2><p>These dots aggregate existing query samples. They do not change MODIS resolution or identify independent fires. Unmarked places are unavailable, not fire-free.</p><p className="ml-small">{view.split === "all" || view.split === "train" ? "Training scenes are included; this view does not measure performance on new events." : view.split === "validation" ? "Validation selected the model and policy; these scenes are not an independent test." : view.split === "test" ? "This benchmark has already been examined; it is not a new independent test." : "Additional windows share training acquisitions; they are not an independent incident test."}</p>{catalogue && <p className="ml-small">{catalogue.model.version}<br />{number(catalogue.scene_count)} patches · {number(catalogue.source_totals.queries)} query samples reduced to {number(catalogue.display_cell_count)} display locations. Overlapping patches can repeat samples.</p>}</section>
    </aside><div className="ml-map-area">
      <div className="ml-map-status" role="status">{error ? <div className="ml-error"><strong>Overview unavailable</strong><p>{error}</p><button onClick={() => setAttempt((n) => n + 1)}>Retry</button></div> : !catalogue ? <span>Loading bundled overview…</span> : <span>{view.date || "Historical composite"} · {number(shown.length)} cells with displayed alerts · {number(cells.length)} sampled cells{!cells.length && " · No cached samples match these filters"}</span>}</div>
      {catalogue && <Suspense fallback={<div className="ml-map-placeholder">Opening geographic view…</div>}><Globe bounds={bounds} sectors={SECTORS} region="ml-overview" points={EMPTY} researchPoints={points} researchMarkerScale={markerScale} patches={tiles} selectedPatch={selectedTile?.id} patchOutlines={view.outlines} patchLabels={view.labels} onPatchPick={chooseTile} centerBounds playing={false} optical={false} allDays={false} fit={fit} onPick={({ lat, lon }) => setPicked({ lat, lon, key: cellAt(cells, lat, lon, catalogue.grid_degrees)?.key })} /></Suspense>}
      {selectedTile && !picked && <section className="ml-inspector overview-tile-inspector" aria-label="Selected tile"><button className="ml-close" aria-label="Close selected tile" onClick={() => change("tile", "")}>×</button><span className="overview-tile-chip">SELECTED TILE</span><h2>Tile {tileCode(selectedTile)}</h2><p>{tileRegion(selectedTile)}<br />{selectedTile.observed_at.replace("T", " ").replace(/(?:\.\d+)?Z$/, " UTC")}</p><code>{selectedTile.id}</code><dl className="ml-counts"><div><dt>MLP alerts</dt><dd>{selectedTile.counts.mlp}</dd></div><div><dt>MODIS detections</dt><dd>{selectedTile.counts.modis}</dd></div><div><dt>VIIRS positives</dt><dd>{selectedTile.counts.viirs}</dd></div></dl><a href={tileViewLink(selectedTile)}>Open this tile in the scene atlas →</a><p className="ml-small">{view.outlines ? "Yellow outline = selected tile. " : "Tile outlines are hidden. "}Sampled envelope: {selectedTile.bounds.map((n) => n.toFixed(3)).join(", ")} (west, south, east, north).</p></section>}
      <div className="ml-legend" aria-label="Overview legend">{(view.layer === "mlp" ? [[COLORS.mlp,"MLP"]] : view.layer === "modis" ? [[COLORS.modis,"MODIS"]] : view.layer === "viirs" ? [[COLORS.viirs,"VIIRS reference"]] : [[COLORS.both,view.layer === "difference" ? "Both disagreement types" : "Both methods in cell"],[COLORS.mlp,"MLP"],[COLORS.modis,"MODIS"],...(view.layer === "all" ? [[COLORS.viirs,"VIIRS reference only"]] : [])]).map(([color, label]) => <span key={label}><i style={{ background: color }} />{label}</span>)}<span><i style={{ background: COLORS.coverage }} />Sampled cell</span><small>0.25° display grid. Cell centers summarize samples; entire cells may not be observed.</small></div>
      {picked && <section className="ml-inspector" aria-label="Selected overview cell"><button className="ml-close" aria-label="Close overview selection" onClick={() => setPicked(null)}>×</button><h2>{selected ? "Sampled display cell" : "No cached samples in this cell"}</h2>{selected ? <><p>{selected.lat.toFixed(3)}°, {selected.lon.toFixed(3)}°</p><dl className="ml-counts"><div><dt>MLP alert samples</dt><dd>{number(selected.mlp)}</dd></div><div><dt>MODIS positive samples</dt><dd>{number(selected.modis)}</dd></div><div><dt>VIIRS reference samples</dt><dd>{number(selected.viirs)}</dd></div><div><dt>All query samples</dt><dd>{number(selected.queries)}</dd></div><div><dt>Maximum MLP score</dt><dd>{selected.score.toFixed(4)}</dd></div></dl><p className="ml-small">Dates: {[...selected.dates].sort().join(", ")}. Samples can repeat across overlapping patches; scores are not guaranteed fire probabilities.</p></> : <p>Other locations may lie outside the processed sample. This does not establish that no fire was present.</p>}</section>}
      {picked && nearbyTiles.length > 0 && <div className="overview-nearby" aria-label="Tiles at selected location"><span>Tiles containing this location</span>{nearbyTiles.slice(0,8).map((tile) => <button key={tile.id} title={tileTitle(tile)} onClick={() => chooseTile(tile)}>Tile {tileCode(tile)} · {tile.date}</button>)}{nearbyTiles.length > 8 && <small>{nearbyTiles.length - 8} more; narrow by date to separate overpasses.</small>}</div>}
      {catalogue && <details className="ml-accessible"><summary>Accessible overview table ({number(shown.length)})</summary><div><table><thead><tr><th>Cell</th><th>Latitude</th><th>Longitude</th><th>MLP samples</th><th>MODIS samples</th><th>VIIRS samples</th></tr></thead><tbody>{shown.map((c) => <tr key={c.key}><td><button aria-label={`Inspect cell ${c.key}`} onClick={() => setPicked(c)}>Inspect</button></td><td>{c.lat}</td><td>{c.lon}</td><td>{c.mlp}</td><td>{c.modis}</td><td>{c.viirs}</td></tr>)}</tbody></table>{!shown.length && <p>No displayed alerts in the selected cached samples.</p>}</div></details>}
    </div></main>
  </div>;
}
