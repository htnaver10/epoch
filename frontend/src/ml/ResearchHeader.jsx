import Icon from "../components/Icon.jsx";
import "./research.css";

// Use the same console chrome as Calendar and Scenarios in every research view.
export default function ResearchHeader({ onMode, view, title, status, evidence, onEvidence }) {
  return <header className="bar research-header">
    <h1 className="brand"><button onClick={() => onMode("analysis")} title="EPOCH: back to the burning calendar">EPOCH</button></h1>
    <nav className="seg mode" aria-label="Mode">
      <button onClick={() => onMode("analysis")}>Calendar</button>
      <button onClick={() => onMode("ops")}>Scenarios</button>
      <button className="on" aria-pressed="true">MLP research</button>
    </nav>
    <p className="sysline"><b>{title}</b><span>{status}</span></p>
    <nav className="tabs research-views" aria-label="Research views">
      <a aria-label="3D globe" className={view === "overview" ? "on" : undefined} aria-current={view === "overview" ? "page" : undefined} href="?mode=ml&view=overview"><b>1</b>3D globe</a>
      <a aria-label="Scene atlas" className={view === "presentation" ? "on" : undefined} aria-current={view === "presentation" ? "page" : undefined} href="?mode=ml&view=presentation"><b>2</b>Scene atlas</a>
    </nav>
    {onEvidence && <div className="actions"><button aria-label="Method and evidence" title="Method and evidence" aria-pressed={evidence} onClick={onEvidence}><Icon name="info" size="1.15rem" /></button></div>}
  </header>;
}
