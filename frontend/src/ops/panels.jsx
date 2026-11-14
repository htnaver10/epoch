import { useEffect, useRef } from "react";
import Icon from "../components/Icon";
import { C, fireRamp } from "../lib/theme";
import { SPECS } from "./engine/units";
import { LIMIT, SOURCES } from "./engine/doctrine";
import { KIND } from "./engine/line";
import { fireName } from "./engine/world";
import { SILHOUETTES, SYMBOL, DESIGNATOR, ROTOR } from "./icons";
import { clockText, hectares, integer, kilolitres } from "./format";

const CLASS_TONE = ["", C.typical, C.typical, C.warn, C.warn, C.high, C.high];
const ROLE_LABEL = { tanker: "Airtanker", skimmer: "Skimmer", helo: "Helicopter", lead: "Bird dog", engine: "Engine", spu: "Sprinklers", dozer: "Dozer", crew: "Crew", medic: "Medical", police: "Police" };

/** A unit's symbol, as on the map. */
export function UnitSymbol({ type, size = "1.5rem" }) {
  const key = SYMBOL[type];
  if (!key)
    return (
      <svg className="usym" width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
        <rect x="6.5" y="16.5" width="51" height="31" fill="none" stroke="currentColor" strokeWidth="3.5" />
        <text x="32" y="34" textAnchor="middle" dominantBaseline="middle" fontSize="20" fontWeight="600" fill="currentColor" fontFamily="var(--mono)">{DESIGNATOR[type]}</text>
      </svg>
    );
  return (
    <svg className="usym" width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
      <path d={SILHOUETTES[key]} fill="currentColor" stroke="currentColor" strokeWidth="2.6" strokeLinejoin="round" strokeLinecap="round" />
      {key === "helo" && <path d={ROTOR} fill="none" stroke="currentColor" strokeWidth="1.8" />}
    </svg>
  );
}

/** Every fire in the theatre: size, behaviour, and when the forecast brings it to the town. */
export function IncidentBoard({ incidents, fire, world, onSelect }) {
  return (
    <section className="board incidents">
      <header>
        <h2>Fires</h2>
        <span>{world ? `${incidents.length} · ${world.theatre.town.name}` : ""}</span>
      </header>
      <table className="grid">
        <thead>
          <tr><th>Fire</th><th>ha</th><th>Cls</th><th title="Fastest advance, metres a minute">m/min</th><th title="From the fire's hottest edge to the town">km</th><th title="When the forecast brings it to the town">Arrives</th><th>Status</th></tr>
        </thead>
        <tbody>
          {incidents.map((f) => (
            <tr key={f.id} className={f.id === fire ? "on" : undefined} onClick={() => onSelect(f.id)}>
              <th><button onClick={(e) => (e.stopPropagation(), onSelect(f.id))}>{fireName(f.id)}</button></th>
              <td>{hectares(f.ha)}</td>
              <td style={{ color: CLASS_TONE[f.cls] }}>{f.cls || "–"}</td>
              <td>{f.ros >= 0.1 ? f.ros.toFixed(f.ros < 10 ? 1 : 0) : "–"}</td>
              <td>{f.km.toFixed(f.km < 10 ? 1 : 0)}</td>
              <td className={f.impact ? "hot" : "dim"}>{f.impact ? `in ${clockText(f.impact.minutes)}` : "none"}</td>
              <td className={f.op?.status === "ACTIVE" ? "task" : f.op?.status === "HELD" ? "good" : "dim"}>{f.op ? (f.op.status === "ACTIVE" ? f.op.name : f.op.status) : f.cls ? "unassigned" : "out"}</td>
            </tr>
          ))}
          {!incidents.length && <tr className="none"><td colSpan="7">{world ? "No fire in this scenario" : "Loading"}</td></tr>}
        </tbody>
      </table>
    </section>
  );
}

/** Every unit: what it is, what it is doing, what it carries. */
export function Roster({ world, unit, onSelect }) {
  const units = world?.units || [];
  const row = (u) => {
    const spec = SPECS[u.type];
    const load = spec.load ? Math.round((u.load / spec.load) * 100) : null;
    return (
      <tr key={u.id} className={`${u.id === unit ? "on" : ""}${u.op ? " task" : ""}`} onClick={() => onSelect(u.id)}>
        <th><button onClick={(e) => (e.stopPropagation(), onSelect(u.id))}><UnitSymbol type={u.type} size="1.25rem" />{u.call}</button></th>
        <td className="dim">{spec.short}</td>
        <td className="status-cell">{u.status}</td>
        <td>{load === null ? "" : <i className="meter" title={`${integer(u.load)} of ${integer(spec.load)} L`}><b style={{ width: `${load}%` }} /></i>}</td>
        <td className="dim">{u.op ? fireName(u.op) : ""}</td>
      </tr>
    );
  };
  const air = units.filter((u) => u.cls === "air"), ground = units.filter((u) => u.cls !== "air");
  return (
    <section className="board roster">
      <header>
        <h2>Crews and aircraft</h2>
        <span>simulated</span>
      </header>
      <div className="scroll">
        <table className="grid">
          <thead><tr><th>Air · {air.length}</th><th>Type</th><th className="l">Status</th><th>Load</th><th>Fire</th></tr></thead>
          <tbody>{air.map(row)}</tbody>
          <thead><tr><th>Ground · {ground.length}</th><th>Type</th><th className="l">Status</th><th /><th>Fire</th></tr></thead>
          <tbody>{ground.map(row)}</tbody>
        </table>
      </div>
    </section>
  );
}

const Row = ({ k, v, unit, tone, title }) => (
  <div className="row" title={title}>
    <dt>{k}</dt>
    <dd style={tone ? { color: tone } : undefined}>{v}</dd>
    <dd className="unit">{unit}</dd>
  </div>
);

function Situation({ s, world }) {
  const town = world.theatre.town;
  return (
    <dl className="kv small">
      <Row k="Size" v={hectares(s.ha)} unit="ha" />
      <Row k="Live edge" v={s.activeKm.toFixed(s.activeKm < 10 ? 1 : 0)} unit={`km · ${Math.round(s.workableShare * 100)}% under ${integer(LIMIT.skimmer)} kW/m`} title="Share of the live edge an airtanker can work" />
      <Row k="Head" v={s.ros.toFixed(s.ros < 10 ? 1 : 0)} unit={`m/min · class ${s.cls} · ${integer(s.hfi)} kW/m`} tone={CLASS_TONE[s.cls]} />
      <Row k={`To ${town.name}`} v={s.hit ? `in ${clockText(s.hit.minutes)}` : "none"} unit={s.hit ? `${s.km.toFixed(0)} km · arrives ${clockText(world.localMinute(s.hit.at), true)}` : `not in ${s.horizonHours} h · ${s.km.toFixed(0)} km`} tone={s.hit ? C.high : undefined} />
      <Row k="Structures in path" v={integer(s.buildingsAtRisk)} unit={`of ${integer(town.buildings)} · no action`} tone={s.buildingsAtRisk ? C.high : undefined} />
    </dl>
  );
}

/** The plan for a fire: the situation, the courses of action side by side, and the one in view. */
export function PlanBoard({ world, planning, onChoose, onCommit, onClose }) {
  const { situation, coas, none, progress, chosen } = planning;
  const selected = coas?.find((c) => c.key === chosen);
  const options = coas ? [{ key: "none", name: "NO ACTION", score: none }, ...coas] : [];
  const runs = none?.runs;
  return (
    <section className="board plan">
      <header>
        <h2>Fire {fireName(planning.fire)} · plan</h2>
        <button className="x" onClick={onClose} aria-label="Close the plan" title="Close (Esc)"><Icon name="close" size="1rem" /></button>
      </header>
      <div className="scroll">
        {situation ? <Situation s={situation} world={world} /> : <div className="shimmer" style={{ height: "9rem" }} />}
        {progress && (
          <p className="progress" role="status">
            <span>{progress.name}{progress.total > 1 ? ` · run ${progress.done} of ${progress.total}` : ""}</span>
            <i><b style={{ width: `${(progress.done / progress.total) * 100}%` }} /></i>
          </p>
        )}
        {coas && (
          <table className="grid coas">
            <thead>
              <tr><th>Plan</th><th title="Runs of the ensemble in which this fire entered the town">Town hit</th><th title="Mean structures lost to this fire">Struct.</th><th title="Mean area this fire has burned at the end of the run">ha</th><th title="Runs in which this fire had stopped spreading by the end">Held</th></tr>
            </thead>
            <tbody>
              {options.map((c, i) => (
                <tr key={c.key} className={`${c.key === chosen ? "on" : ""}${c.recommended ? " rec" : ""}${c.key === "none" ? " base" : ""}`} onClick={c.key === "none" ? undefined : () => onChoose(c.key)}>
                  <th>{c.key === "none" ? c.name : <button onClick={(e) => (e.stopPropagation(), onChoose(c.key))}><b>{String.fromCharCode(64 + i)}</b>{c.name}</button>}{c.recommended && <em>REC</em>}</th>
                  <td className={c.score ? (c.score.townRuns ? "hot" : "good") : "dim"}>{c.score ? `${c.score.townRuns}/${c.score.runs}` : "…"}</td>
                  <td className={c.score?.structures >= 1 ? "hot" : undefined}>{c.score ? integer(c.score.structures) : "…"}</td>
                  <td>{c.score ? hectares(c.score.ha) : "…"}</td>
                  <td className={c.score?.heldRuns ? "good" : "dim"}>{c.score ? `${c.score.heldRuns}/${c.score.runs}` : "…"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {runs && <p className="fine">{runs} runs each over {none.hours} h: wind turned up to ±18° and scaled 0.8 to 1.3, embers on. Means shown.</p>}
        {selected && (
          <div className="coa">
            <h3><span>{selected.intent}</span>{selected.name}</h3>
            <ol>{selected.method.map((m) => <li key={m}>{m}</li>)}</ol>
            {selected.line?.feasible && (
              <dl className="kv small">
                <Row k="Line" v={selected.line.km.toFixed(1)} unit={`km · build ${selected.line.buildKm.toFixed(1)} · road ${selected.line.roadKm.toFixed(1)} · water ${selected.line.naturalKm.toFixed(1)}`} />
                <Row k="Work" v={Number.isFinite(selected.work.hours) ? selected.work.hours.toFixed(1) : "–"} unit={`h at ${selected.work.perHour.toFixed(2)} km/h · fire due ${Number.isFinite(selected.work.lead) ? `in ${selected.work.lead.toFixed(1)} h` : "beyond the forecast"}`} tone={selected.work.hours > selected.work.lead ? C.warn : undefined} />
                {selected.score && <Row k="Used" v={kilolitres(selected.score.retardant)} unit={`retardant · ${kilolitres(selected.score.water)} water · ${selected.score.guardKm.toFixed(1)} km guard`} />}
              </dl>
            )}
            <p className="resources">{selected.resources}</p>
            {selected.notes.map((n) => <p key={n} className="caution">{n}</p>)}
            {selected.missions && (
              <table className="grid tasking">
                <thead>
                  <tr><th>{selected.dispatch.length} out{selected.units.length > selected.dispatch.length ? ` · ${selected.units.length - selected.dispatch.length} kept back` : ""}</th><th className="l">First task</th><th>At</th><th title="Drops in the first two and a half hours">Drops</th></tr>
                </thead>
                <tbody>
                  {selected.missions.map((m) => (
                    <tr key={m.id} className={m.held ? "held" : undefined}>
                      <th><UnitSymbol type={m.type} size="1.2rem" />{m.call}</th>
                      <td className="status-cell">{m.task}</td>
                      <td>{m.at != null ? clockText(world.localMinute(m.at), true) : ""}</td>
                      <td className="dim">{m.drops || ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {selected.saved && <p className="fine">Against no action: {integer(Math.max(0, selected.saved.structures))} structures and {hectares(Math.max(0, selected.saved.ha))} ha saved on average{selected.score.burnedOff >= 1 ? `; ${hectares(selected.score.burnedOff)} ha burned off on purpose` : ""}.</p>}
          </div>
        )}
      </div>
      <footer>
        <button className="commit" disabled={!selected || !selected.missions} onClick={onCommit}>{selected?.missions ? `Dispatch ${selected.name.toLowerCase()} · ${selected.dispatch.length} crews and aircraft` : "Working out the options"}</button>
      </footer>
    </section>
  );
}

/** An operation under way: how far the work has got, and who is on it. */
export function OperationBoard({ world, op, incident, onStandDown, onSelectUnit, onClose }) {
  const units = world.units.filter((u) => u.op === op.fire);
  const line = op.line;
  let built = 0, toBuild = 0, coated = 0, burned = 0;
  if (line)
    line.kinds.forEach((k, i) => {
      if (k === KIND.natural) return;
      if (k === KIND.build) { toBuild++; if (op.guard[i] >= 2) built++; }
      if (op.ret[i] === 2) coated++;
      if (op.burn[i] === 2) burned++;
    });
  const work = line ? line.kinds.filter((k) => k !== KIND.natural).length : 0;
  const pct = (a, b) => (b ? `${Math.round((a / b) * 100)}%` : "–");
  const active = op.status === "ACTIVE";
  return (
    <section className="board plan">
      <header>
        <h2>Fire {fireName(op.fire)} · {op.name.toLowerCase()}</h2>
        <button className="x" onClick={onClose} aria-label="Close" title="Close (Esc)"><Icon name="close" size="1rem" /></button>
      </header>
      <div className="scroll">
        <dl className="kv small">
          <Row k="Status" v={op.status} unit={`since ${clockText(op.since)} elapsed`} tone={op.status === "HELD" ? C.typical : active ? C.accent : undefined} />
          {incident && <Row k="Fire" v={hectares(incident.ha)} unit={`ha · class ${incident.cls || "–"} · ${incident.activeKm.toFixed(1)} km live`} tone={CLASS_TONE[incident.cls]} />}
          {incident && <Row k="To town" v={incident.impact ? `in ${clockText(incident.impact.minutes)}` : "none"} unit={incident.impact ? "by the forecast" : "in the forecast"} tone={incident.impact ? C.high : C.typical} />}
          {line && <Row k="Retardant" v={pct(coated, work)} unit={`of ${(work * world.theatre.cell).toFixed(1)} km · ${kilolitres(op.stats.retardant)}`} />}
          {line && <Row k="Guard" v={pct(built, toBuild)} unit={`of ${(toBuild * world.theatre.cell).toFixed(1)} km · ${op.stats.guardKm.toFixed(1)} km cut`} />}
          {line && <Row k="Burned off" v={pct(burned, work)} unit={`${op.stats.burnKm.toFixed(1)} km of line`} />}
          <Row k="Water" v={kilolitres(op.stats.water)} unit={`${op.stats.drops} drops`} />
          <Row k="Treated ahead" v={op.stats.aheadKm.toFixed(1)} unit="km wetted or coated before the fire got there" />
          <Row k="Edge mopped up" v={op.stats.mopKm.toFixed(1)} unit="km" />
          {op.stats.sprinklerKm > 0 && <Row k="Sprinklers" v={op.stats.sprinklerKm.toFixed(1)} unit="km running" />}
        </dl>
        <table className="grid">
          <thead><tr><th>Assigned · {units.length}</th><th className="l">Status</th><th title="Trips flown">Trips</th></tr></thead>
          <tbody>
            {units.map((u) => (
              <tr key={u.id} onClick={() => onSelectUnit(u.id)}>
                <th><button onClick={(e) => (e.stopPropagation(), onSelectUnit(u.id))}><UnitSymbol type={u.type} size="1.25rem" />{u.call}</button></th>
                <td className="status-cell">{u.status}</td>
                <td>{u.sorties || ""}</td>
              </tr>
            ))}
            {!units.length && <tr className="none"><td colSpan="3">Everyone released</td></tr>}
          </tbody>
        </table>
      </div>
      {active && (
        <footer>
          <button className="commit quiet" onClick={onStandDown}>End plan · release {units.length}</button>
        </footer>
      )}
    </section>
  );
}

/** One unit: where it is, what it carries, and the legs it has planned. */
export function UnitStrip({ world, unit, follow, onFollow, onRelease, onClose }) {
  if (!unit) return null;
  const spec = SPECS[unit.type], air = spec.cls === "air";
  const legs = unit.legs.slice(unit.li, unit.li + 7);
  return (
    <section className="board plan unitcard">
      <header>
        <h2><UnitSymbol type={unit.type} size="1.5rem" />{unit.call} · {spec.name}</h2>
        <button className="x" onClick={onClose} aria-label="Close" title="Close (Esc)"><Icon name="close" size="1rem" /></button>
      </header>
      <div className="scroll">
        <dl className="kv small">
          <Row k="Status" v={unit.status} unit={unit.op ? `fire ${fireName(unit.op)}` : ROLE_LABEL[unit.role]} tone={unit.op ? C.accent : undefined} />
          {air && <Row k="Altitude" v={integer(Math.round(unit.alt / 10) * 10)} unit="m above ground" />}
          <Row k="Speed" v={integer(unit.speed)} unit={`km/h · heading ${String(Math.round(((unit.hdg % 360) + 360) % 360)).padStart(3, "0")}°`} />
          {spec.load > 0 && <Row k="Load" v={integer(unit.load)} unit={`of ${integer(spec.load)} L ${spec.agent}`} />}
          {air && <Row k="Endurance" v={clockText(Math.max(0, unit.fuel))} unit="left" tone={unit.fuel < 40 ? C.warn : undefined} />}
          {spec.load > 0 && <Row k="Delivered" v={kilolitres(unit.delivered)} unit={`${unit.sorties} sorties`} />}
          <Row k="Base" v={unit.home.name} unit={unit.home.off ? "off the map" : ""} />
        </dl>
        <table className="grid legs">
          <thead><tr><th>{air ? "Flight plan" : "Movement"}</th><th>At</th><th>Elapsed</th></tr></thead>
          <tbody>
            {legs.map((leg, i) => (
              <tr key={`${leg.t1}:${i}`} className={i === 0 ? "now" : undefined}>
                <th>{leg.status || leg.kind}{leg.mark ? ` → ${leg.mark}` : ""}</th>
                <td>{clockText(world.localMinute(leg.t1), true)}</td>
                <td className="dim">{clockText(Math.max(0, leg.t1 - world.t))}</td>
              </tr>
            ))}
            {unit.work && <tr className="now"><th>{unit.work.status || unit.status}</th><td colSpan="2" className="dim">working</td></tr>}
            {!legs.length && !unit.work && <tr className="none"><td colSpan="3">Nothing planned</td></tr>}
          </tbody>
        </table>
      </div>
      <footer className="pair">
        <button className={`commit quiet${follow ? " on" : ""}`} aria-pressed={follow} onClick={onFollow}>{follow ? "Following" : "Follow"}</button>
        <button className="commit quiet" disabled={!unit.op} onClick={onRelease} title="Take this crew or aircraft off the plan and send it home">Release</button>
      </footer>
    </section>
  );
}

/** The log: the last things that happened, newest at the bottom. */
export function EventLog({ world }) {
  const box = useRef(null);
  const lines = world ? world.log.slice(-40) : [];
  const last = lines.length ? lines[lines.length - 1].t : 0;
  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [last, lines.length]);
  return (
    <section className="log" aria-label="Event log">
      <h2>Log</h2>
      <ol ref={box}>
        {lines.map((l, i) => (
          <li key={`${l.t}:${i}`} className={l.tone || undefined}>
            <time>{clockText(world.localMinute(l.t), true)}</time>
            <b>{l.call}</b>
            <span>{l.text}</span>
          </li>
        ))}
        {!lines.length && <li className="none"><span>{world ? world.brief : ""}</span></li>}
      </ol>
    </section>
  );
}

/** What the marks on the map mean. */
export function MapKey() {
  const ramp = `linear-gradient(90deg, ${fireRamp(0.3)}, ${fireRamp(0.55)}, ${fireRamp(0.8)}, ${fireRamp(1)})`;
  return (
    <div className="mapkey" aria-label="Map key">
      <span><i style={{ background: ramp }} />flame, by intensity</span>
      <span><i className="k-scar" />burned</span>
      <span><i className="k-iso" />forecast, hours ahead</span>
      <span><i className="k-ret" />retardant</span>
      <span><i className="k-guard" />guard</span>
      <span><i className="k-wet" />wetted</span>
      <span><i className="k-line" />control line</span>
      <span><i className="k-town" />town edge</span>
    </div>
  );
}

/** Where the numbers come from, and what is and is not real. */
export function Sources({ world, onClose }) {
  return (
    <section className="board sources" aria-label="Method and sources">
      <header>
        <h2>Method and sources</h2>
        <button className="x" onClick={onClose} aria-label="Close"><Icon name="close" size="1rem" /></button>
      </header>
      <div className="scroll">
        <dl className="kv small facts">
          <div className="row"><dt>Real</dt><dd className="unit">Terrain, water, roads, buildings; satellite hotspots to the opening time; hourly wind, temperature and humidity (reanalysis, with a share of the gust)</dd></div>
          <div className="row"><dt>Modelled</dt><dd className="unit">Fuel types inside mapped forest; fire growth (Canadian FBP System on a 160 m grid); what suppression does to it</dd></div>
          {world?.adjusted && <div className="row"><dt>Adjusted</dt><dd className="unit">{world.adjusted}</dd></div>}
          <div className="row"><dt>Illustrative</dt><dd className="unit">Which crews and aircraft are on hand, and everything they do. Aircraft types, loads, speeds and bases are real</dd></div>
        </dl>
        <table className="grid">
          <tbody>
            {SOURCES.map(([what, who, url]) => (
              <tr key={what}><th>{what}</th><td className="wrap"><a href={url} target="_blank" rel="noreferrer">{who}</a></td></tr>
            ))}
            {(world?.theatre.data.sources || []).map((s) => (
              <tr key={s}><th>Data</th><td className="wrap">{s}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
