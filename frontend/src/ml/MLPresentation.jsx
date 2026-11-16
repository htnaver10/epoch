import { useEffect, useMemo, useState } from "react";
import { filterScenes, presentationView, selectScene, splitLabel, SPLITS, viewURL, VIEWS } from "./presentation.js";
import { tileCode, tileTitle } from "./tiles.js";
import ResearchHeader from "./ResearchHeader.jsx";
import "./presentation.css";

const BASE = `${import.meta.env.BASE_URL}data/ml-presentation`;
const number = (n) => Number(n).toLocaleString();
const percent = (n) => `${(100 * n).toFixed(1)}%`;

function Benchmark({ benchmark }) {
  const methods = [["50.6M MLP", benchmark.wide_frozen_policy, "var(--accent)"], ["Original MLP", benchmark.parent_mlp_frozen_policy, "var(--muted)"], ["MODIS", benchmark.regular_modis, "var(--modis)"]];
  return <div className="pres-benchmark"><div className="pres-section-title"><div><span className="pres-kicker">Previously examined 2024 benchmark</span><h2>Frozen model performance</h2></div><span>84 patches · 95 positive query labels</span></div>
    <div className="pres-results"><table><thead><tr><th>Method</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP / FP / FN</th></tr></thead><tbody>{methods.map(([label, m, color]) => <tr key={label}><th><i style={{ background: color }} />{label}</th><td>{percent(m.precision)}</td><td>{percent(m.recall)}</td><td>{m.f1.toFixed(3)}</td><td>{m.true_positives} / {m.false_positives} / {m.false_negatives}</td></tr>)}</tbody></table>
      <div className="pres-bars" aria-label="Benchmark F1 comparison">{methods.map(([label, m, color]) => <div key={label}><span>{label}</span><div><i style={{ width: percent(m.f1), background: color }} /></div><b>{m.f1.toFixed(3)}</b></div>)}</div></div>
    <p className="pres-note">VIIRS supplies the reference labels. These correlated query labels are not independent fires. This reused benchmark does not establish performance on new incidents.</p>
  </div>;
}

function TrainingChart({ history }) {
  if (!history?.length) return null;
  const points = history.map((row, i) => `${25 + i / Math.max(history.length - 1, 1) * 550},${135 - row.validation_average_precision * 220}`).join(" ");
  return <div className="pres-training"><div><span className="pres-kicker">Completed training</span><h3>Selection on validation only</h3><p>{history.length} epochs completed. Epoch 10 was selected by validation average precision. The operating policy was frozen before benchmark scoring.</p></div><svg viewBox="0 0 600 170" role="img" aria-label={`Validation average precision over ${history.length} training epochs`}><line x1="25" x2="575" y1="135" y2="135" stroke="#34495c" /><line x1="25" x2="575" y1="25" y2="25" stroke="#243442" /><text x="25" y="18">Validation AP</text><text x="25" y="160">Epoch 1</text><text x="515" y="160">Epoch {history.length}</text><polyline fill="none" stroke="#55ccff" strokeWidth="3" points={points} />{history.map((row, i) => row.epoch === 10 && <circle key={i} cx={25 + i / Math.max(history.length - 1, 1) * 550} cy={135 - row.validation_average_precision * 220} r="5" fill="#edf5fc" />)}</svg></div>;
}

export default function MLPresentation({ onMode }) {
  const [initial] = useState(() => presentationView(window.location.search));
  const [catalogue, setCatalogue] = useState(null), [error, setError] = useState(""), [attempt, setAttempt] = useState(0);
  const [split, setSplit] = useState(initial.split), [date, setDate] = useState(""), [search, setSearch] = useState(""), [sort, setSort] = useState("reference");
  const [sceneId, setSceneId] = useState(initial.scene), [layer, setLayer] = useState(initial.layer);
  const [presenting, setPresenting] = useState(false), [playing, setPlaying] = useState(false), [imageState, setImageState] = useState({});
  const [evidence, setEvidence] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    let ignore = false;
    fetch(`${BASE}/catalog.json`, { signal: controller.signal }).then(async (response) => {
      if (!response.ok) throw Error(`Catalogue HTTP ${response.status}`);
      const value = await response.json();
      if (value.schema !== "epoch_ml_presentation_v1" || !value.complete || !value.scenes?.length) throw Error("The presentation build is incomplete.");
      if (!ignore) { setCatalogue(value); setError(""); }
    }).catch((e) => { if (!ignore && e.name !== "AbortError") setError(e.message); });
    return () => { ignore = true; controller.abort(); };
  }, [attempt]);
  const scenes = useMemo(() => filterScenes(catalogue?.scenes || [], split, date, search, sort), [catalogue, split, date, search, sort]);
  const scene = selectScene(scenes, sceneId);
  const index = scenes.findIndex((item) => item.id === scene?.id);
  const imageURL = scene ? `${BASE}/scenes/${scene.id}/${layer}.png` : "";
  const image = imageState.url === imageURL ? imageState.status : "loading";
  const dates = useMemo(() => [...new Set((catalogue?.scenes || []).filter((s) => split === "all" || s.split === split).map((s) => s.date))].sort().reverse(), [catalogue, split]);
  useEffect(() => { window.history.replaceState(null, "", viewURL(scene?.id, split, layer)); }, [scene?.id, split, layer]);
  useEffect(() => {
    const next = scenes[(index + 1) % scenes.length];
    if (!next || !scene) return;
    const preload = new Image(); preload.src = `${BASE}/scenes/${next.id}/${layer}.png`;
  }, [scenes, index, scene, layer]);
  useEffect(() => {
    if (!playing || scenes.length < 2 || image !== "ready") return;
    const timer = setTimeout(() => setSceneId(scenes[(index + 1) % scenes.length].id), 6500);
    return () => clearTimeout(timer);
  }, [playing, scenes, index, image]);
  useEffect(() => {
    const key = (e) => {
      if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName) || e.altKey || e.metaKey || e.ctrlKey) return;
      if (e.key === "Escape") { setPresenting(false); setPlaying(false); setEvidence(false); }
      if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && scenes.length) {
        e.preventDefault(); setSceneId(scenes[(index + (e.key === "ArrowRight" ? 1 : -1) + scenes.length) % scenes.length].id);
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [scenes, index]);
  const move = (delta) => { if (scenes.length) setSceneId(scenes[(index + delta + scenes.length) % scenes.length].id); };
  const chooseSplit = (value) => { setSplit(value); setDate(""); setSearch(""); setSceneId(""); setPlaying(false); };
  const togglePresentation = () => { setPresenting((value) => !value); setPlaying(false); document.querySelector(".ml-presentation")?.scrollTo(0, 0); };
  return <div className={`ml-presentation ${presenting ? "is-presenting" : ""}`}>
    <ResearchHeader onMode={onMode} view="presentation" title="Scene atlas · 50.6M MLP" status={catalogue ? `${number(catalogue.scene_count)} cached patches · frozen research model` : "Opening scene archive"} evidence={evidence} onEvidence={() => setEvidence((v) => !v)} />
    <div className="scope pres-splits" role="group" aria-label="Dataset split"><div className="scope-group"><span className="scope-key">Dataset</span>{SPLITS.filter(([id]) => id === "all" || catalogue?.splits[id]).map(([id, label]) => <button key={id} disabled={!catalogue} aria-label={`${label} ${number(id === "all" ? catalogue?.scene_count || 0 : catalogue?.splits[id].scenes || 0)}`} className={split === id ? "on" : ""} aria-pressed={split === id} onClick={() => chooseSplit(id)}>{label}<span>{number(id === "all" ? catalogue?.scene_count || 0 : catalogue?.splits[id].scenes || 0)}</span></button>)}</div><span className="scope-count">Precomputed observations · no live inference</span></div>
    {error ? <main className="pres-failure" role="alert"><h1>Presentation unavailable</h1><p>{error}</p><button onClick={() => { setError(""); setAttempt((v) => v + 1); }}>Retry</button></main> : !catalogue ? <main className="pres-failure"><p role="status">Loading the precomputed model presentation…</p></main> : <>
      <main className="pres-workspace"><section className="pres-explorer" aria-label="Precomputed scene explorer"><div className="pres-explorer-top"><h2>Scene comparison</h2><div className="pres-top-actions"><a href={`${BASE}/scene-summary.csv`} download>Summary CSV ↓</a><button aria-pressed={presenting} onClick={togglePresentation}>{presenting ? "Exit presentation" : "Present scenes"}</button></div></div>
        {split !== "test" && <p className="pres-split-note">{split === "train" ? "These patches were used to train this model. Their detections do not measure generalization." : split === "validation" ? "These patches selected the checkpoint and operating policy. They are not an independent test." : split === "additional" ? "Additional prepared windows from shared training acquisitions. These were not used to fit the selected 50.6M model and are not an independent incident test." : "This view includes training, validation, reused benchmark and additional windows. Keep these roles separate when discussing performance."}</p>}
        <div className="pres-explorer-body"><aside className="pres-browser"><div className="pres-filters"><label>Date (UTC)<select aria-label="Presentation date" value={date} onChange={(e) => { setDate(e.target.value); setSceneId(""); }}><option value="">All dates</option>{dates.map((value) => <option key={value}>{value}</option>)}</select></label><label>Find a patch<input aria-label="Find a patch" value={search} onChange={(e) => { setSearch(e.target.value); setSceneId(""); }} placeholder="Patch ID or region" /></label><label>Order by<select aria-label="Scene order" value={sort} onChange={(e) => setSort(e.target.value)}><option value="reference">Most reference positives</option><option value="alerts">Most MLP alerts</option><option value="date">Newest observation</option></select></label></div><p className="pres-list-count" role="status">{number(scenes.length)} patches in this view</p><div className="pres-scene-list" aria-label="Scene patches">{scenes.slice(0, 180).map((item) => <button key={item.id} title={tileTitle(item)} aria-pressed={scene?.id === item.id} className={scene?.id === item.id ? "on" : ""} onClick={() => { setSceneId(item.id); setPlaying(false); }}><span>{item.date}<small>{item.group.replaceAll("_", " ")}</small></span><b>{item.counts.mlp}<small>MLP alerts</small></b><code>Tile {tileCode(item)}</code></button>)}{scenes.length > 180 && <p className="pres-note">Showing the first 180 results. Narrow by date or search, or use Next to visit all {number(scenes.length)} patches.</p>}</div></aside>
          <div className="pres-stage">{scene ? <><div className="pres-scene-heading"><div><span className="pres-kicker">{splitLabel(scene.split)} / {scene.group.replaceAll("_", " ")}</span><h3>{scene.observed_at.replace("T", " · ").replace(".000Z", " UTC")}</h3><code title={scene.id}>Tile {tileCode(scene)} · {scene.id}</code></div><div className="pres-stepper"><button aria-label="Previous scene" onClick={() => move(-1)}>←</button><span>{index + 1} / {number(scenes.length)}</span><button aria-label="Next scene" onClick={() => move(1)}>→</button></div></div>
            <div className="pres-view-tabs" aria-label="Rendered layer">{VIEWS.map(([id, label]) => <button key={id} aria-pressed={layer === id} className={layer === id ? "on" : ""} onClick={() => setLayer(id)}>{label}</button>)}</div>
            <div className={`pres-image ${layer === "compare" ? "comparison" : "single"}`}>
              {image !== "ready" && <p role="status">{image === "error" ? "This rendered view could not load." : "Opening rendered view…"}{image === "error" && <button onClick={() => { setImageState({}); setAttempt((n) => n + 1); }}>Retry image</button>}</p>}
              <img key={`${imageURL}:${attempt}`} src={imageURL} alt={`${VIEWS.find(([id]) => id === layer)[1]} for ${scene.date}, patch ${scene.id}. Gray marks show eligible coverage.`} onLoad={() => setImageState({ url: imageURL, status: "ready" })} onError={() => setImageState({ url: imageURL, status: "error" })} />
            </div><div className="pres-patch-stats"><span><b>{number(scene.counts.mlp)}</b> MLP alerts</span><span><b>{number(scene.counts.modis)}</b> MODIS</span><span><b>{number(scene.counts.viirs)}</b> VIIRS positives</span><span><b>{number(scene.counts.queries)}</b> eligible queries</span></div>
            <div className="pres-downloads"><button aria-pressed={playing} onClick={() => setPlaying((v) => !v)}>{playing ? "Pause tour" : "Play scene tour"}</button><a href={imageURL} download={`${scene.id}-${layer}.png`}>Image PNG ↓</a><a href={`${BASE}/scenes/${scene.id}/predictions.geojson`} download>GeoJSON ↓</a><a href={`${BASE}/scenes/${scene.id}/queries.npz`} download>All query scores ↓</a></div><details className="pres-scene-detail"><summary>Scene provenance and query details</summary><p>Observation timing bound: {scene.pair_time_bound_seconds} seconds. Execution: {scene.precision}. Model: {scene.model_version}. Source tile SHA-256: <code>{scene.source_sha256}</code>.</p><p>The GeoJSON contains the union of MLP, MODIS and VIIRS positives. The NPZ includes every eligible query center, raw logits, adjusted scores and flags, including all nonalerts.</p><ul>{scene.source_granules.map((value) => <li key={value}>{value}</li>)}</ul></details></> : <div className="pres-empty"><h3>No matching patches</h3><p>Try another date or clear the search.</p><button onClick={() => { setDate(""); setSearch(""); }}>Clear filters</button></div>}</div></div>
      </section>
      {evidence && <aside className="research-evidence" aria-label="Method and evidence"><header><h2>Method and evidence</h2><button className="x" aria-label="Close method and evidence" onClick={() => setEvidence(false)}>×</button></header>
      <section className="pres-archive"><span className="pres-kicker">Frozen research model</span><h2>50.6M parameter MLP</h2><dl className="kv small"><div className="row"><dt>Processed patches</dt><dd>{number(catalogue.scene_count)}</dd></div><div className="row"><dt>Eligible query centers</dt><dd>{number(catalogue.total_queries)}</dd></div><div className="row"><dt>Rendered views</dt><dd>{number(catalogue.render_count)}</dd></div><div className="row"><dt>Selected epoch</dt><dd>{catalogue.model.selected_epoch}</dd></div></dl><a href={`${BASE}/gallery.html`} target="_blank" rel="noreferrer">Open the full rendered atlas ↗</a></section>
      <Benchmark benchmark={catalogue.benchmark} /><TrainingChart history={catalogue.training_history} />
      <section className="pres-method"><div><span className="pres-kicker">Model inputs and reference</span><h2>96 MODIS features</h2><p>Thermal radiance, local background context, reflectance and viewing geometry form 96 explicit features. The network predicts at query centers with a training-only normalizer and a frozen detection threshold. VIIRS labels are used for comparison, never as model inputs.</p><a href="?mode=ml&view=inspect">Open the original 8,321-parameter MLP globe ↗</a></div><div><span className="pres-kicker">Research limitations</span><ul>{catalogue.limitations.map((value) => <li key={value}>{value}</li>)}</ul></div></section>
      <footer className="pres-footer"><span>{catalogue.model.version} · Input snapshot {catalogue.created_utc.slice(0, 16).replace("T", " ")} UTC</span><a href={`${BASE}/catalog.json`} download>Download complete provenance</a></footer></aside>}
      </main><footer className="pres-status"><span>50.6M MLP · frozen policy · VIIRS reference</span><span>Research data · sampled patches · empty areas may have no data</span></footer>
    </>}
  </div>;
}
