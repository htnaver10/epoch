import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { loadCatalogue, loadScene } from "./api.js";
import { COLORS, LAYERS, geoJSONDownload, nearestFeature, readMLView, researchPoints, visibleFeatures, writeMLView } from "./data.js";
import "./ml.css";
import MLPresentation from "./MLPresentation.jsx";

const Globe = lazy(() => import("../components/Globe.jsx"));
const MLOverview = lazy(() => import("./MLOverview.jsx"));
const EMPTY = [], SECTORS = {};
const percent = (n) => `${(100 * n).toFixed(1)}%`;
const integer = (n) => Number(n).toLocaleString();
const utc = (s) => s.replace("T", " ").replace(/(?:\.\d+)?Z$/, " UTC");
const yes = (value) => value ? "Yes" : "No";

export default function MLViewer({ onMode }) {
  const query = new URLSearchParams(window.location.search);
  const view = query.get("view");
  if (view === "inspect") return <OriginalMLViewer onMode={onMode} />;
  // Keep previously shared atlas scene links working alongside the explicit route.
  if (view === "presentation" || (!view && query.has("scene"))) return <MLPresentation onMode={onMode} />;
  return <MLOverview onMode={onMode} />;
}

function OriginalMLViewer({ onMode }) {
  const [initial] = useState(() => readMLView(window.location.search));
  const [catalogueResult, setCatalogueResult] = useState(null), [sceneId, setSceneId] = useState(initial.scene);
  const [sceneResult, setSceneResult] = useState(null), [attempt, setAttempt] = useState(0);
  const catalogue = catalogueResult?.attempt === attempt ? catalogueResult.value : null;
  const sceneKey = catalogue ? `${catalogue.source}:${catalogue.model.version}:${sceneId}` : "";
  const data = sceneResult?.key === sceneKey ? sceneResult.value : null;
  const error = (catalogueResult?.attempt === attempt && catalogueResult.error) || (sceneResult?.key === sceneKey && sceneResult.error) || "";
  const [layer, setLayer] = useState(initial.layer), [showCoverage, setCoverage] = useState(initial.coverage);
  const [picked, setPicked] = useState(null), [fit, setFit] = useState(0), [optical, setOptical] = useState(false);
  const [image, setImage] = useState("off");
  useEffect(() => {
    const controller = new AbortController();
    loadCatalogue(controller.signal).then((result) => {
      setCatalogueResult({ attempt, value: result });
      setSceneId((id) => result.scenes.some((s) => s.id === id) ? id : [...result.scenes].sort((a, b) => b.counts.mlp - a.counts.mlp)[0].id);
    }).catch((e) => { if (!controller.signal.aborted) setCatalogueResult({ attempt, error: `MLP scenes could not be loaded. ${e.message}` }); });
    return () => controller.abort();
  }, [attempt]);
  useEffect(() => {
    if (!catalogue || !sceneId) return;
    const controller = new AbortController();
    loadScene(catalogue, sceneId, controller.signal).then((value) => setSceneResult({ key: sceneKey, value })).catch((e) => { if (!controller.signal.aborted) setSceneResult({ key: sceneKey, error: `This scene could not be loaded. ${e.message}` }); });
    return () => controller.abort();
  }, [catalogue, sceneId, sceneKey]);
  useEffect(() => { window.history.replaceState(null, "", writeMLView(sceneId, layer, showCoverage)); }, [sceneId, layer, showCoverage]);
  const scene = data?.scene;
  const dates = useMemo(() => [...new Set(catalogue?.scenes.map((s) => s.date) || [])], [catalogue]);
  const selected = catalogue?.scenes.find((s) => s.id === sceneId);
  const bounds = useMemo(() => scene ? [[scene.bounds[1], scene.bounds[0]], [scene.bounds[3], scene.bounds[2]]] : [[50, -80], [55, -70]], [scene]);
  const points = useMemo(() => data ? researchPoints(data.features, data.coverage, layer, showCoverage) : EMPTY, [data, layer, showCoverage]);
  const shown = useMemo(() => data ? visibleFeatures(data.features, layer) : EMPTY, [data, layer]);
  const chooseDate = (date) => {
    const found = catalogue.scenes.filter((s) => s.date === date).sort((a, b) => b.counts.mlp - a.counts.mlp);
    setPicked(null); setSceneId(found[0].id);
  };
  const download = () => {
    if (!data) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(geoJSONDownload(data, layer), null, 2)], { type: "application/geo+json" }));
    const a = document.createElement("a"); a.href = url; a.download = `epoch-mlp-${sceneId}-${layer}.geojson`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const pickedFeature = picked?.feature;
  return <div className="ml-console">
    <header className="ml-header">
      <button className="ml-brand" onClick={() => onMode("analysis")}>EPOCH</button>
      <nav className="seg mode" aria-label="Mode">
        <button onClick={() => onMode("analysis")}>Calendar</button>
        <button onClick={() => onMode("ops")}>Scenarios</button>
        <button className="on" aria-pressed="true">MLP research</button>
      </nav>
      <div className="ml-heading"><strong>MODIS → MLP</strong><span>Experimental scene comparison</span></div>
      <a href="?mode=ml&view=presentation">50.6M presentation</a>
      <span className={`ml-source ${catalogue?.source === "api" ? "api" : "snapshot"}`} role="status">{catalogue?.message || "Opening catalogue…"}</span>
    </header>
    <main className="ml-layout">
      <aside className="ml-sidebar" aria-label="MLP scene controls">
        <section><span className="ml-eyebrow">Processed observations</span><h1>Inspect the model</h1><p>Compare the MLP’s saved predictions with MODIS detections and VIIRS reference observations.</p></section>
        {catalogue && <section className="ml-fields">
          <label>Date (UTC)<select aria-label="Date (UTC)" value={selected?.date || dates[0]} onChange={(e) => chooseDate(e.target.value)}>{dates.map((date) => <option key={date}>{date}</option>)}</select></label>
          <label>Scene patch<select aria-label="Scene patch" value={sceneId} onChange={(e) => { setPicked(null); setSceneId(e.target.value); }}>{catalogue.scenes.filter((s) => s.date === selected?.date).map((s) => <option key={s.id} value={s.id}>{s.id.slice(0, 8)} · {s.counts.mlp} MLP alerts · {s.counts.modis} MODIS</option>)}</select></label>
          <p className="ml-small">{catalogue.scenes.length} cached patches across {dates.length} dates. Other dates and areas are unavailable.</p>
        </section>}
        <section><h2>Map layer</h2><div className="ml-layers">{LAYERS.map(([key, label]) => <button key={key} className={layer === key ? "on" : ""} aria-pressed={layer === key} onClick={() => { setLayer(key); setPicked(null); }}>{label}</button>)}</div>
          <label className="ml-check"><input type="checkbox" checked={showCoverage} onChange={(e) => setCoverage(e.target.checked)} /> Show cached query coverage</label>
          <label className="ml-check"><input type="checkbox" checked={optical} onChange={(e) => setOptical(e.target.checked)} /> NASA true-colour image</label>
          {optical && <p className="ml-small" aria-live="polite">Image: {image}. Satellite imagery is background context, separate from predictions.</p>}
        </section>
        {scene && <section><h2>This patch</h2><p className="ml-time">{utc(scene.observed_at)}</p><dl className="ml-counts">
          <div><dt>MLP alerts</dt><dd>{scene.counts.mlp}</dd></div><div><dt>MODIS detections</dt><dd>{scene.counts.modis}</dd></div>
          <div><dt>VIIRS reference positives</dt><dd>{scene.counts.viirs}</dd></div><div><dt>Cached query centers</dt><dd>{integer(scene.counts.queries)}</dd></div>
        </dl><div className="ml-actions"><button onClick={() => { setPicked(null); setFit((v) => v + 1); }}>Fit patch</button><button onClick={download}>Download GeoJSON</button></div></section>}
        {catalogue && <section className="ml-benchmark"><h2>Existing benchmark</h2><p className="ml-small">95 positive query labels in 84 patches. These are not counts of independent fires.</p><table><thead><tr><th>Method</th><th>Precision</th><th>Recall</th></tr></thead><tbody>
          <tr><th>MLP</th><td>{percent(catalogue.model.metrics.precision)}</td><td>{percent(catalogue.model.metrics.recall)}</td></tr>
          <tr><th>MODIS</th><td>{percent(catalogue.model.baseline_metrics.precision)}</td><td>{percent(catalogue.model.baseline_metrics.recall)}</td></tr>
        </tbody></table><p className="ml-small">Previously examined scenes; this is not an independent test. VIIRS supplies the reference labels.</p></section>}
        <section className="ml-caution"><h2>Research use</h2><p>This is cached research data, not live monitoring. Empty areas outside the coverage marks are unavailable, not fire-free. Model scores are not guaranteed fire probabilities.</p>
          {catalogue && <details><summary>Model and sources</summary><p>{catalogue.model.version}</p><p>96 MODIS features · frozen detection policy · saved CUDA BF16 predictions.</p><p>Only eligible reference-matched query centers are included.</p>{scene && <><p>Pair timing bound: {scene.pair_time_bound_seconds} seconds.</p><ul>{scene.source_granules.map((s) => <li key={s}>{s}</li>)}</ul></>}</details>}
        </section>
      </aside>
      <div className="ml-map-area">
        <div className="ml-map-status" role="status">{error ? <div className="ml-error"><strong>Scene unavailable</strong><p>{error}</p><button onClick={() => setAttempt((v) => v + 1)}>Retry</button></div> : !data ? <span>Loading processed scene…</span> : <span>{utc(scene.observed_at)} · {shown.length} displayed marks · click a mark to inspect</span>}</div>
        {data && !error && <Suspense fallback={<div className="ml-map-placeholder">Opening geographic view…</div>}>
          <Globe bounds={bounds} sectors={SECTORS} region="ml" points={EMPTY} researchPoints={points} centerBounds optical={optical} imageryDate={scene.date}
            playing={false} allDays={false} fit={fit} onImage={({ state }) => setImage(state)}
            onPick={({ lat, lon }) => setPicked({ lat, lon, feature: nearestFeature(shown, lat, lon) })} />
        </Suspense>}
        <div className="ml-legend" aria-label="Map legend">{[[COLORS.both,"MLP + MODIS"],[COLORS.mlp,"MLP"],[COLORS.modis,"MODIS"],[COLORS.viirs,"VIIRS reference"],[COLORS.coverage,"Cached coverage"]].map(([c,label]) => <span key={label}><i style={{ background: c }} />{label}</span>)}<small>Point locations are query centers, not exact fire footprints.</small></div>
        {picked && data && <section className="ml-inspector" aria-label="Selected prediction"><button className="ml-close" aria-label="Close selected prediction" onClick={() => setPicked(null)}>×</button>
          <h2>{pickedFeature ? "Selected query" : "No displayed alert here"}</h2>
          {pickedFeature ? <><p>{pickedFeature.geometry.coordinates[1].toFixed(5)}°, {pickedFeature.geometry.coordinates[0].toFixed(5)}°</p>
            <dl className="ml-counts"><div><dt>MLP score</dt><dd>{pickedFeature.properties.mlp_score.toFixed(4)}</dd></div><div><dt>MLP alert</dt><dd>{yes(pickedFeature.properties.mlp_detected)}</dd></div><div><dt>MODIS detection</dt><dd>{yes(pickedFeature.properties.modis_detected)}</dd></div><div><dt>VIIRS reference</dt><dd>{yes(pickedFeature.properties.viirs_reference)}</dd></div></dl>
            <p className="ml-small">{utc(scene.observed_at)}<br />{scene.model_version}</p></> : <p>Choose a visible mark to inspect its score and sensor agreement. An empty location may be outside the cached sample; it does not establish that no fire was present.</p>}
        </section>}
        {data && <details className="ml-accessible"><summary>Accessible prediction table ({shown.length})</summary><div><table><thead><tr><th>Query</th><th>Latitude</th><th>Longitude</th><th>MLP score</th><th>MLP</th><th>MODIS</th><th>VIIRS ref.</th></tr></thead><tbody>{shown.map((f) => <tr key={f.id}><td><button aria-label={`Inspect query ${f.id}`} onClick={() => setPicked({ lat: f.geometry.coordinates[1], lon: f.geometry.coordinates[0], feature: f })}>Inspect</button></td><td>{f.geometry.coordinates[1].toFixed(5)}</td><td>{f.geometry.coordinates[0].toFixed(5)}</td><td>{f.properties.mlp_score.toFixed(4)}</td><td>{yes(f.properties.mlp_detected)}</td><td>{yes(f.properties.modis_detected)}</td><td>{yes(f.properties.viirs_reference)}</td></tr>)}</tbody></table>{!shown.length && <p>No alerts for this layer in this cached patch.</p>}</div></details>}
      </div>
    </main>
  </div>;
}
