import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { COLORS } from "./data.js";
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
  const [picked, setPicked] = useState(null), [frameTarget, setFrameTarget] = useState("canada"), [fit, setFit] = useState(0);
  const catalogue = result?.attempt === attempt ? result.value : null;
  const error = result?.attempt === attempt ? result.error : "";
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${import.meta.env.BASE_URL}data/ml-overview.json`, { signal: controller.signal }).then(async (response) => {
      if (!response.ok) throw Error(`Overview HTTP ${response.status}`);
      const value = await response.json();
      if (value.schema !== "epoch_ml_overview_v1" || !value.records?.length || !value.model?.version) throw Error("The bundled overview is incomplete.");
      if (!controller.signal.aborted) setResult({ attempt, value });
    }).catch((e) => { if (!controller.signal.aborted) setResult({ attempt, error: e.message }); });
    return () => controller.abort();
  }, [attempt]);
  useEffect(() => { window.history.replaceState(null, "", overviewURL(view)); }, [view]);
  const cells = useMemo(() => aggregateCells(catalogue?.records || [], view), [catalogue, view]);
  const shown = useMemo(() => cells.filter((c) => alertCell(c, view.layer)), [cells, view.layer]);
  const points = useMemo(() => overviewPoints(cells, view.layer, view.coverage), [cells, view.layer, view.coverage]);
  const bounds = useMemo(() => {
    if (frameTarget === "canada" || !cells.length) return CANADA_BOUNDS;
    return [[Math.min(...cells.map((c) => c.lat)) - 0.3, Math.min(...cells.map((c) => c.lon)) - 0.3],
      [Math.max(...cells.map((c) => c.lat)) + 0.3, Math.max(...cells.map((c) => c.lon)) + 0.3]];
  }, [cells, frameTarget]);
  const selected = cells.find((c) => c.key === picked?.key);
  const counts = useMemo(() => ({ mlp: cells.filter((c) => c.mlp).length, modis: cells.filter((c) => c.modis).length, viirs: cells.filter((c) => c.viirs).length }), [cells]);
  const change = (key, value) => { setPicked(null); setView((v) => ({ ...v, [key]: value })); };
  const download = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(overviewExport(cells, catalogue, view))], { type: "application/geo+json" }));
    const a = document.createElement("a"); a.href = url; a.download = `epoch-overview-${view.date || "historical"}-${view.layer}.geojson`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <div className="ml-console overview-console">
    <header className="ml-header"><button className="ml-brand" onClick={() => onMode("analysis")}>EPOCH</button><nav className="seg mode" aria-label="Mode"><button onClick={() => onMode("analysis")}>Calendar</button><button onClick={() => onMode("ops")}>Scenarios</button><button className="on" aria-pressed="true">MLP research</button></nav><div className="ml-heading"><strong>Canada overview</strong><span>Sparse historical observations</span></div><a className="overview-atlas" href="?mode=ml">Scene atlas</a><span className="ml-source api">Bundled data · no backend</span></header>
    <main className="ml-layout"><aside className="ml-sidebar" aria-label="Overview controls">
      <section><span className="ml-eyebrow">50.6M MLP / Geographic view</span><h1>Canada overview</h1><p>Explore the saved observations on a 3D globe. Each point summarizes a 0.25° display cell.</p><p className="ml-small">Only locally processed patches are shown. This is not continuous coverage of Canada.</p></section>
      {catalogue && <section className="ml-fields">
        <label>Date (UTC)<select aria-label="Overview date" value={view.date} onChange={(e) => change("date", e.target.value)}><option value="">All cached dates · historical composite</option>{catalogue.dates.map((d) => <option key={d}>{d}</option>)}</select></label>
        <label>Dataset<select aria-label="Overview dataset" value={view.split} onChange={(e) => change("split", e.target.value)}><option value="all">All cached scenes</option><option value="test">Reused benchmark</option><option value="validation">Validation</option><option value="train">Training</option><option value="additional">Additional windows</option></select></label>
        <label>Region<select aria-label="Overview region" value={view.region} onChange={(e) => { change("region", e.target.value); setFrameTarget(e.target.value ? "coverage" : "canada"); }}><option value="">Canada and surrounding regions</option>{Object.keys(catalogue.regions).sort().map((r) => <option key={r} value={r}>{names(r)}</option>)}</select></label>
        <p className="ml-small">{view.date ? `${view.date} UTC observations.` : `${catalogue.dates[0]} to ${catalogue.dates.at(-1)}. Different years and days are combined; this is not one fire situation.`}</p>
      </section>}
      <section><h2>Map layer</h2><div className="ml-layers">{LAYERS.map(([id, label]) => <button key={id} aria-pressed={view.layer === id} className={view.layer === id ? "on" : ""} onClick={() => change("layer", id)}>{label}</button>)}</div><label className="ml-check"><input type="checkbox" checked={view.coverage} onChange={(e) => change("coverage", e.target.checked)} /> Show cells containing cached samples</label><div className="ml-actions"><button onClick={() => { setFrameTarget("canada"); setFit((v) => v + 1); }}>Fit Canada</button><button disabled={!cells.length} onClick={() => { setFrameTarget("coverage"); setFit((v) => v + 1); }}>Fit samples</button></div><p className="ml-small">Scroll to zoom. Drag to rotate. Use 3D or right-drag for a tilted view when zoomed in.</p></section>
      {catalogue && <section><h2>Cells in this view</h2><dl className="ml-counts"><div><dt>Contain MLP alerts</dt><dd>{number(counts.mlp)}</dd></div><div><dt>Contain MODIS detections</dt><dd>{number(counts.modis)}</dd></div><div><dt>Contain VIIRS positives</dt><dd>{number(counts.viirs)}</dd></div><div><dt>Contain cached samples</dt><dd>{number(cells.length)}</dd></div></dl><div className="ml-actions"><button disabled={!cells.length} onClick={download}>Download overview GeoJSON</button></div></section>}
      <section className="ml-caution"><h2>Read this as sampled history</h2><p>These dots aggregate existing query samples. They do not change MODIS resolution or identify independent fires. Unmarked places are unavailable, not fire-free.</p><p className="ml-small">{view.split === "all" || view.split === "train" ? "Training scenes are included; this view does not measure performance on new events." : view.split === "validation" ? "Validation selected the model and policy; these scenes are not an independent test." : view.split === "test" ? "This benchmark has already been examined; it is not a new independent test." : "Additional windows share training acquisitions; they are not an independent incident test."}</p>{catalogue && <p className="ml-small">{catalogue.model.version}<br />{number(catalogue.scene_count)} patches · {number(catalogue.source_totals.queries)} query samples reduced to {number(catalogue.display_cell_count)} display locations. Overlapping patches can repeat samples.</p>}</section>
    </aside><div className="ml-map-area">
      <div className="ml-map-status" role="status">{error ? <div className="ml-error"><strong>Overview unavailable</strong><p>{error}</p><button onClick={() => setAttempt((n) => n + 1)}>Retry</button></div> : !catalogue ? <span>Loading bundled overview…</span> : <span>{view.date || "Historical composite"} · {number(shown.length)} cells with displayed alerts · {number(cells.length)} sampled cells{!cells.length && " · No cached samples match these filters"}</span>}</div>
      {catalogue && <Suspense fallback={<div className="ml-map-placeholder">Opening geographic view…</div>}><Globe bounds={bounds} sectors={SECTORS} region="ml-overview" points={EMPTY} researchPoints={points} centerBounds playing={false} optical={false} allDays={false} fit={fit} onPick={({ lat, lon }) => setPicked({ lat, lon, key: cellAt(cells, lat, lon, catalogue.grid_degrees)?.key })} /></Suspense>}
      <div className="ml-legend" aria-label="Overview legend">{(view.layer === "mlp" ? [[COLORS.mlp,"MLP"]] : view.layer === "modis" ? [[COLORS.modis,"MODIS"]] : view.layer === "viirs" ? [[COLORS.viirs,"VIIRS reference"]] : [[COLORS.both,view.layer === "difference" ? "Both disagreement types" : "Both methods in cell"],[COLORS.mlp,"MLP"],[COLORS.modis,"MODIS"],...(view.layer === "all" ? [[COLORS.viirs,"VIIRS reference only"]] : [])]).map(([color, label]) => <span key={label}><i style={{ background: color }} />{label}</span>)}<span><i style={{ background: COLORS.coverage }} />Sampled cell</span><small>0.25° display grid. Cell centers summarize samples; entire cells may not be observed.</small></div>
      {picked && <section className="ml-inspector" aria-label="Selected overview cell"><button className="ml-close" aria-label="Close overview selection" onClick={() => setPicked(null)}>×</button><h2>{selected ? "Sampled display cell" : "No cached samples in this cell"}</h2>{selected ? <><p>{selected.lat.toFixed(3)}°, {selected.lon.toFixed(3)}°</p><dl className="ml-counts"><div><dt>MLP alert samples</dt><dd>{number(selected.mlp)}</dd></div><div><dt>MODIS positive samples</dt><dd>{number(selected.modis)}</dd></div><div><dt>VIIRS reference samples</dt><dd>{number(selected.viirs)}</dd></div><div><dt>All query samples</dt><dd>{number(selected.queries)}</dd></div><div><dt>Maximum MLP score</dt><dd>{selected.score.toFixed(4)}</dd></div></dl><p className="ml-small">Dates: {[...selected.dates].sort().join(", ")}. Samples can repeat across overlapping patches; scores are not guaranteed fire probabilities.</p></> : <p>Other locations may lie outside the processed sample. This does not establish that no fire was present.</p>}</section>}
      {catalogue && <details className="ml-accessible"><summary>Accessible overview table ({number(shown.length)})</summary><div><table><thead><tr><th>Cell</th><th>Latitude</th><th>Longitude</th><th>MLP samples</th><th>MODIS samples</th><th>VIIRS samples</th></tr></thead><tbody>{shown.map((c) => <tr key={c.key}><td><button aria-label={`Inspect cell ${c.key}`} onClick={() => setPicked(c)}>Inspect</button></td><td>{c.lat}</td><td>{c.lon}</td><td>{c.mlp}</td><td>{c.modis}</td><td>{c.viirs}</td></tr>)}</tbody></table>{!shown.length && <p>No displayed alerts in the selected cached samples.</p>}</div></details>}
    </div></main>
  </div>;
}
