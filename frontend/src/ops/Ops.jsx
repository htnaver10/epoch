import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./ops.css";
import Icon from "../components/Icon";
import { useRem } from "../lib/hooks";
import { TacScene } from "./tacscene";
import { openScenario } from "./engine/scenario";
import { syntheticTheatre } from "./engine/synthetic";
import { evaluate, incidents as readIncidents, plan as makePlan } from "./engine/planner";
import { startProjection } from "./engine/forecast";
import { toLatLon } from "./engine/theatre";
import { isiOf } from "./engine/fbp";
import { IncidentBoard, Roster, PlanBoard, OperationBoard, UnitStrip, EventLog, Sources, MapKey } from "./panels";
import { clockText } from "./format";

const SPEEDS = [1, 10, 30, 120, 600]; // simulated seconds per second
const FORECAST_EVERY = 30; // simulated minutes between forecast renewals
const PLACE_KINDS = new Set(["city", "town", "village", "hamlet"]);
const ORDER = ["jasper", "yellowknife", "hayriver"];
const LEFT_REM = 34.7, RIGHT_REM = 34.3; // room the boards take on each side of the stage

const load = (path, signal) =>
  fetch(`${import.meta.env.BASE_URL}data/ops/${path}`, { signal }).then((r) => {
    if (!r.ok) throw Error(`HTTP ${r.status}`);
    return r.json();
  });
const isTyping = (el) => el && /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName);

/**
 * OPS: the response console. A theatre round a town, the fire from the satellite record,
 * a simulated roster of units, and a planner that lays out and scores courses of action.
 */
export default function Ops({ onMode, initial = {} }) {
  const [index, setIndex] = useState(null);
  const [theatreId, setTheatreId] = useState(initial.theatre || "");
  const [world, setWorld] = useState(null);
  const [error, setError] = useState("");
  const [, setTick] = useState(0);
  const [speed, setSpeed] = useState(initial.speed ?? 30);
  const [running, setRunning] = useState(initial.paused ? false : true);
  const [fireId, setFireId] = useState(0);
  const [unitId, setUnitId] = useState(0);
  const [planning, setPlanning] = useState(null); // { fire, situation, coas, none, progress, chosen }
  const [layers, setLayers] = useState({ forecast: true, hotspots: false, actual: false });
  const [tool, setTool] = useState("select");
  const [follow, setFollow] = useState(false);
  const [showSources, setShowSources] = useState(false);
  const { rem } = useRem(1);
  const mount = useRef(null), labels = useRef(null), cursor = useRef(null), compass = useRef(null), altitude = useRef(null);
  const scene = useRef(null);
  const live = useRef({});
  const job = useRef(null);
  const evaluation = useRef(0);
  const held = useRef(false); // the clock was stopped for planning, not by the user
  const opened = useRef(false);

  // the list of theatres; with none bundled, a synthetic one so the console still runs
  useEffect(() => {
    const controller = new AbortController();
    load("index.json", controller.signal)
      .then((found) => {
        // the theatre whose fire reaches its town comes first
        const list = [...found].sort((a, b) => ORDER.indexOf(a.id) - ORDER.indexOf(b.id));
        setIndex(list);
        setTheatreId((id) => (list.some((t) => t.id === id) ? id : list[0].id));
      })
      .catch((e) => {
        if (e.name === "AbortError") return;
        setIndex([{ id: "synthetic", name: "Test scenario", region: "SYN", date: "2023-08-15" }]);
        setTheatreId("synthetic");
      });
    return () => controller.abort();
  }, []);

  // open a theatre
  useEffect(() => {
    if (!theatreId || !index) return undefined;
    const controller = new AbortController();
    evaluation.current++;
    const entry = index.find((t) => t.id === theatreId);
    (theatreId === "synthetic" ? Promise.resolve(syntheticTheatre()) : load(entry?.file || `${theatreId}.json`, controller.signal))
      .then((data) => {
        const w = openScenario(data);
        if (initial.advance) w.step(initial.advance);
        w.getForecast(true);
        setWorld(w);
        if (initial.fire && !opened.current) setTimeout(() => live.current.selectFire?.(initial.fire.charCodeAt(0) - 64), 4200);
        opened.current = true;
      })
      .catch((e) => e.name !== "AbortError" && setError(`Could not load the ${entry?.name || theatreId} scenario (${e.message}).`));
    return () => controller.abort();
  }, [theatreId, index, initial.advance, initial.fire]);

  // the incident list is re-read every two simulated minutes, and whenever a fire or an operation is added
  const beat = Math.floor((world?.t || 0) / 2), forecastAt = world?.forecastAt, fires = world?.fire.fires, operations = world?.ops.size;
  const incidents = useMemo(() => (world ? readIncidents(world) : []), [world, beat, forecastAt, fires, operations]); // eslint-disable-line react-hooks/exhaustive-deps
  // a patch that has gone out and was never worked is not an incident any more
  const shown = useMemo(() => incidents.filter((f) => f.cls > 0 || f.op || f.id === fireId), [incidents, fireId]);
  const places = useMemo(() => (world ? (world.theatre.data.places || []).filter((p) => PLACE_KINDS.has(p.k)).slice(0, 14) : []), [world]);
  const op = world && fireId ? world.ops.get(fireId) : null;
  const activeOp = op && op.status === "ACTIVE" ? op : null;
  const chosen = planning?.coas?.find((c) => c.key === planning.chosen) || null;
  const unit = world && unitId ? world.units.find((u) => u.id === unitId) : null;

  // what the map draws of the plan: the previewed course, or the committed one
  const mapPlan = useMemo(() => {
    if (chosen && planning) {
      // the plan as it would be flown and driven: every unit's way to its first piece of work
      const missions = (chosen.missions || []).filter((m) => !m.held).map((m) => ({ ...m, label: m.at != null && m.cls === "air" && m.drops ? `${m.call} · ${clockText(world.localMinute(m.at), true)}` : "" }));
      return { key: chosen.key, line: chosen.line, segments: chosen.segments, missions, route: planning.situation.route, routeLabel: planning.situation.hit ? `PROJECTED RUN · ${clockText(planning.situation.hit.minutes)} TO TOWN` : "" };
    }
    if (activeOp) return { line: activeOp.line, segments: activeOp.segments };
    return null;
  }, [chosen, planning, activeOp, world]);

  const panel = unit ? "unit" : planning ? "plan" : op ? "op" : "none";
  useEffect(() => {
    live.current = { ...live.current, theatreId, world, running, speed, fireId, unitId, layers, tool, follow, incidents: shown, places, mapPlan, rem, insets: [LEFT_REM * rem, panel === "none" ? 0 : RIGHT_REM * rem] };
  });

  // the scene, one per theatre
  useEffect(() => {
    if (!world || !mount.current) return undefined;
    const theatre = world.theatre;
    let s;
    try {
      s = new TacScene(mount.current, labels.current, theatre, {
        onPick: (hit) => live.current.pick?.(hit),
        onCursor: (p) => {
          if (!cursor.current) return;
          if (!p) return void (cursor.current.textContent = "");
          const [lat, lon] = toLatLon(theatre, p.x, p.y);
          cursor.current.textContent = `${Math.abs(lat).toFixed(4)}°${lat >= 0 ? "N" : "S"} ${Math.abs(lon).toFixed(4)}°${lon >= 0 ? "E" : "W"}   ELEV ${Math.round(theatre.elevAt(p.x, p.y))} M`;
        },
        onCamera: (cam) => {
          if (compass.current) compass.current.style.transform = `rotate(${(cam.az * 180) / Math.PI}deg)`;
          if (altitude.current) altitude.current.textContent = "";
        },
      });
    } catch {
      queueMicrotask(() => setError("This browser could not start WebGL, so the map cannot be drawn."));
      return undefined;
    }
    scene.current = s;
    s.setHotspots(world.opens);
    // open on the town and the fire together, tilted
    const first = readIncidents(world).find((f) => f.cls > 0);
    const focus = first ? [(theatre.town.cx + first.x) / 2, (theatre.town.cy + first.y) / 2] : [theatre.town.cx, theatre.town.cy];
    const span = first ? Math.hypot(first.x - theatre.town.cx, first.y - theatre.town.cy) : 20;
    s.flyTo({ x: focus[0], y: focus[1], dist: Math.max(28, Math.min(95, span * 1.9 + 14)), polar: 0.82, az: -0.35 }, 3.2);
    let raf = 0, last = performance.now(), sinceTick = 0;
    const loop = (now) => {
      raf = requestAnimationFrame(loop);
      const state = live.current, w = state.world;
      if (!w) return;
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      if (state.running) w.step(Math.min(20, (dt * state.speed) / 60));
      // renew the forecast a slice at a time
      if (!job.current && w.t - w.forecastAt >= FORECAST_EVERY) job.current = startProjection(w.fire, w.forecastHours);
      if (job.current) {
        const done = job.current.step(5);
        if (done) { w.setForecast(done); job.current = null; }
      }
      if (state.follow && state.unitId) {
        const u = w.units.find((x) => x.id === state.unitId);
        if (u && !s.flight) s.setView({ ...s.currentView(), x: u.x, y: u.y });
      }
      s.setInset(state.insets[0], state.insets[1]);
      s.frame(w, { unit: state.unitId, fire: state.fireId, layers: state.layers, plan: state.mapPlan, incidents: state.incidents, places: state.places, rem: state.rem }, now);
      sinceTick += dt;
      if (sinceTick > 0.25) { setTick((k) => k + 1); sinceTick = 0; }
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      s.dispose();
      scene.current = null;
      job.current = null;
    };
  }, [world]);

  // ---- actions

  const chooseTheatre = useCallback((id) => {
    if (id === live.current.theatreId) return;
    setWorld(null);
    setFireId(0);
    setUnitId(0);
    setPlanning(null);
    setError("");
    setTheatreId(id);
  }, []);

  const flyTo = useCallback((x, y, dist) => scene.current?.flyTo({ x, y, dist: dist ?? Math.min(scene.current.currentView().dist, 30) }), []);

  const selectFire = useCallback((id, { fly = true } = {}) => {
    const w = live.current.world;
    if (!w) return;
    setFireId(id);
    setUnitId(0);
    setFollow(false);
    evaluation.current++;
    const incident = readIncidents(w).find((f) => f.id === id);
    if (fly && incident) {
      const town = w.theatre.town, span = Math.hypot(incident.head[0] - town.cx, incident.head[1] - town.cy);
      // near enough to see the fire and the town in one picture, if they are that close
      if (span < 45) scene.current?.flyTo({ x: (incident.head[0] + town.cx) / 2, y: (incident.head[1] + town.cy) / 2, dist: Math.max(16, span * 1.7 + 8), polar: 0.8 });
      else flyTo(incident.head[0], incident.head[1], 40);
    }
    const existing = w.ops.get(id);
    if (existing && existing.status === "ACTIVE") return setPlanning(null);
    // plan with the clock stopped: the options are worked out for this moment
    if (live.current.running) { setRunning(false); held.current = true; }
    setPlanning({ fire: id, progress: { done: 0, total: 1, name: "READING THE SITUATION" } });
    const run = evaluation.current;
    setTimeout(async () => {
      if (run !== evaluation.current) return;
      const planned = makePlan(w, id);
      if (!planned) return setPlanning(null);
      setPlanning({ fire: id, ...planned, chosen: planned.coas[0]?.key, progress: { done: 0, total: 1, name: "ENSEMBLE" } });
      const result = await evaluate(w, planned, {
        cancelled: () => run !== evaluation.current,
        onProgress: (done, total, name) => run === evaluation.current && setPlanning((p) => (p && p.fire === id ? { ...p, progress: { done, total, name } } : p)),
      });
      if (!result || run !== evaluation.current) return;
      setPlanning((p) => (p && p.fire === id ? { ...p, ...result, chosen: result.coas.find((c) => c.recommended)?.key || p.chosen, progress: null } : p));
    }, 30);
  }, [flyTo]);

  useEffect(() => {
    live.current.selectFire = selectFire;
  }, [selectFire]);

  const closePlan = useCallback(() => {
    evaluation.current++;
    setPlanning(null);
    setFireId(0);
    if (held.current) { setRunning(true); held.current = false; }
  }, []);

  const commit = useCallback((coa) => {
    const w = live.current.world;
    if (!w || !coa) return;
    evaluation.current++;
    // only the units the plan was seen to use go out; the rest stay free for the next fire
    w.commit({ ...coa, units: coa.dispatch?.length ? coa.dispatch : coa.units });
    setPlanning(null);
    setRunning(true);
    held.current = false;
  }, []);

  const standDown = useCallback(() => {
    const { world: w, fireId: id } = live.current;
    if (w && id) w.standDown(id);
  }, []);

  const selectUnit = useCallback((id, { fly = false } = {}) => {
    const w = live.current.world;
    setUnitId(id);
    setFollow(false);
    const u = w?.units.find((x) => x.id === id);
    // a unit beyond the mapped square (a tanker base off the map) is looked at from farther back
    if (fly && u) flyTo(u.x, u.y, w.theatre.inside(u.x, u.y) ? 14 : 60);
  }, [flyTo]);

  // a click on the map: a unit, a fire, a new ignition, or nothing
  useEffect(() => {
    live.current.pick = (hit) => {
      const { world: w, tool: mode } = live.current;
      if (!w) return;
      if (hit.unit) return selectUnit(hit.unit);
      if (hit.x === undefined) return;
      if (mode === "ignite") {
        const id = w.igniteAt(hit.x, hit.y);
        setTool("select");
        if (id) {
          w.step(0.01);
          w.getForecast(true);
          selectFire(id, { fly: false });
        }
        return;
      }
      const id = w.fireAt(hit.x, hit.y);
      if (id) return selectFire(id, { fly: false });
      setUnitId(0);
    };
  });

  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || isTyping(e.target)) return;
      const k = e.key.toLowerCase();
      if (k === " ") { e.preventDefault(); setRunning((r) => !r); held.current = false; }
      else if (k === "]" || k === "=") setSpeed((s) => SPEEDS[Math.min(SPEEDS.length - 1, SPEEDS.indexOf(s) + 1)]);
      else if (k === "[" || k === "-") setSpeed((s) => SPEEDS[Math.max(0, SPEEDS.indexOf(s) - 1)]);
      else if (k === "f") setLayers((l) => ({ ...l, forecast: !l.forecast }));
      else if (k === "h") setLayers((l) => ({ ...l, hotspots: !l.hotspots }));
      else if (k === "a") setLayers((l) => ({ ...l, actual: !l.actual }));
      else if (k === "i") setTool((t) => (t === "ignite" ? "select" : "ignite"));
      else if (k === "t") scene.current?.flyTo({ polar: scene.current.currentView().polar > 0.3 ? 0.02 : 0.85 });
      else if (k === "escape") {
        if (live.current.tool === "ignite") setTool("select");
        else if (live.current.unitId) setUnitId(0);
        else closePlan();
      } else return;
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [closePlan]);

  // ---- render

  const entry = index?.find((t) => t.id === theatreId);
  const wx = world ? world.weather(world.t) : null;
  const committed = world ? world.units.filter((u) => u.op).length : 0;

  return (
    <div className={`console ops${tool === "ignite" ? " igniting" : ""}`}>
      <header className="bar">
        <h1 className="brand"><button onClick={() => onMode("analysis")} title="EPOCH: back to the burning calendar">EPOCH</button></h1>
        <div className="seg mode" role="group" aria-label="Mode">
          <button onClick={() => onMode("analysis")}>Calendar</button>
          <button className="on" aria-pressed="true">Scenarios</button>
        </div>
        <p className="sysline">
          <b>Response scenario · {entry ? `${entry.name} ${entry.region}` : "loading"}</b>
          <span>{world ? `Simulation · fire from CWFIS hotspots to ${world.dateText(0)} · crews and aircraft are illustrative` : "opening the scenario"}</span>
        </p>
        <nav className="tabs" aria-label="Scenarios">
          {(index || []).map((t, i) => (
            <button key={t.id} className={t.id === theatreId ? "on" : undefined} aria-pressed={t.id === theatreId} onClick={() => chooseTheatre(t.id)} title={`${t.name}, ${t.date}`}>
              <b>{i + 1}</b>
              {t.name}
            </button>
          ))}
        </nav>
        <div className="actions">
          <button onClick={() => setShowSources((s) => !s)} aria-pressed={showSources} title="Method and sources" aria-label="Method and sources"><Icon name="info" size="1.15rem" /></button>
        </div>
      </header>

      <div className="scope">
        <div className="scope-group">
          <span className="scope-key">Wind</span>
          <span className="scope-value">{wx ? `${String(Math.round(((wx.wd % 360) + 360) % 360)).padStart(3, "0")}° ${Math.round(wx.ws)} G${Math.round(wx.gust)} km/h` : "–"}</span>
        </div>
        <div className="scope-group">
          <span className="scope-key">RH</span><span className="scope-value">{wx ? `${Math.round(wx.rh)}%` : "–"}</span>
          <span className="scope-key gap">FFMC</span><span className="scope-value">{wx ? wx.ffmc.toFixed(1) : "–"}</span>
          <span className="scope-key gap">ISI</span><span className="scope-value">{wx ? isiOf(wx.ffmc, wx.ws).toFixed(1) : "–"}</span>
          <span className="scope-key gap">BUI</span><span className="scope-value">{wx ? Math.round(wx.bui) : "–"}</span>
        </div>
        <div className="scope-group">
          <span className="scope-key">Show</span>
          <button className={layers.forecast ? "on" : undefined} aria-pressed={layers.forecast} onClick={() => setLayers((l) => ({ ...l, forecast: !l.forecast }))} title="Forecast isochrones: where the fire is expected in 1 to 24 hours (F)">Forecast</button>
          <button className={layers.hotspots ? "on" : undefined} aria-pressed={layers.hotspots} onClick={() => setLayers((l) => ({ ...l, hotspots: !l.hotspots }))} title="Satellite detections up to the opening time: what the fire was started from (H)">Hotspots</button>
          <button className={layers.actual ? "on" : undefined} aria-pressed={layers.actual} onClick={() => setLayers((l) => ({ ...l, actual: !l.actual }))} title="What happened: later detections and the final burn perimeter (A)">Actual</button>
        </div>
        <div className={`scope-group${tool === "ignite" ? " scope-alert" : ""}`}>
          <span className="scope-key">Tool</span>
          <button className={tool === "ignite" ? "on" : undefined} aria-pressed={tool === "ignite"} onClick={() => setTool((t) => (t === "ignite" ? "select" : "ignite"))} title="Light a new fire: click the map (I)">Ignite</button>
        </div>
        <span className="scope-count">{world ? `${world.units.length} crews and aircraft · ${committed} assigned` : "loading"}</span>
      </div>

      <main className="stage ops-stage">
        <div className="tac-stage">
          <div ref={mount} className="tac-canvas" role="img" aria-label="Map of the scenario: terrain, the fire, forecast lines, crews and aircraft." />
          <div ref={labels} className="tac-labels" />
          <div className="tac-hud">
            <div className="rose" aria-hidden="true">
              <i ref={compass}><b>N</b></i>
              {wx && <em style={{ transform: `rotate(${wx.wd + 180}deg)` }} />}
            </div>
            <span ref={altitude} />
            <span ref={cursor} />
          </div>
        </div>

        <aside className="ops-left" aria-label="Incidents and units">
          <IncidentBoard incidents={shown} fire={fireId} world={world} onSelect={selectFire} />
          <Roster world={world} unit={unitId} onSelect={(id) => selectUnit(id, { fly: true })} />
        </aside>

        {panel !== "none" && (
          <aside className="ops-right" aria-label="Detail">
            {panel === "unit" && <UnitStrip world={world} unit={unit} follow={follow} onFollow={() => setFollow((f) => !f)} onClose={() => setUnitId(0)} onRelease={() => (world.release(unit), setTick((k) => k + 1))} />}
            {panel === "plan" && <PlanBoard world={world} planning={planning} onChoose={(key) => setPlanning((p) => ({ ...p, chosen: key }))} onCommit={() => commit(chosen)} onClose={closePlan} />}
            {panel === "op" && <OperationBoard world={world} op={op} incident={shown.find((f) => f.id === fireId)} onStandDown={standDown} onSelectUnit={(id) => selectUnit(id, { fly: true })} onClose={() => setFireId(0)} />}
          </aside>
        )}
        {showSources && <Sources world={world} onClose={() => setShowSources(false)} />}
        <MapKey />

        {error ? (
          <div className="status status-error" role="alert">{error}</div>
        ) : !world ? (
          <div className="status status-busy" role="status"><i className="spinner" aria-hidden="true" />Opening the scenario</div>
        ) : tool === "ignite" ? (
          <p className="status status-alert">Ignite: click the map to light a fire · Esc to cancel</p>
        ) : panel === "none" && !incidents.some((f) => f.op) ? (
          <p className="status status-prompt">Select a fire: click it on the map or in the list of fires</p>
        ) : null}
      </main>

      <footer className="dock ops-dock">
        <section className="clock" aria-label="Simulation clock">
          <button className="play" onClick={() => (setRunning((r) => !r), (held.current = false))} aria-label={running ? "Hold the clock" : "Run the clock"} title="Run or hold the clock (Space)"><Icon name={running ? "pause" : "play"} size="1.1rem" /></button>
          <div className="clock-time">
            <strong>{world ? clockText(world.localMinute(), true) : "--:--"}</strong>
            <span>{world ? `${world.dateText()} · ${clockText(world.t)} elapsed · ${world.isDay() ? "day" : "night: aircraft grounded"}` : ""}</span>
          </div>
          <div className="seg" role="group" aria-label="Speed">
            {SPEEDS.map((s) => (
              <button key={s} className={s === speed ? "on" : undefined} aria-pressed={s === speed} onClick={() => setSpeed(s)} title={`${s} simulated seconds each second`}>{s}×</button>
            ))}
          </div>
          <span className={`clock-state${running ? "" : " held"}`}>{running ? "RUNNING" : planning ? "HELD FOR PLANNING" : "HELD"}</span>
        </section>
        <EventLog world={world} />
      </footer>
    </div>
  );
}

