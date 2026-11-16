import { C, mapRamp } from "../lib/theme";
import { integer, formatDate } from "../lib/chart";
import { RAW_REFERENCE, RAW_CEILING } from "../lib/analysis";
import { COLORS } from "../ml/data";

const LAYERS = [
  ["all", "All", "Every cell with a hotspot"],
  ["modis", "MODIS", "Cells MODIS detected"],
  ["viirs", "VIIRS", "Cells VIIRS detected"],
  ["uncertain", "Diff", "Only the cell-days one sensor reported and the other did not (D)"],
];

/**
 * One line that states exactly what is on screen: area, season, period, day, layer, view,
 * satellite image, targets. Scope is read and set in the same place. `target` names the
 * area of interest when one is in scope; `months` are the season's months as periods.
 * `image` is { state: "off" | "loading" | "on" | "held" | "paused" | "error", date }.
 */
export function ScopeBar({ regions, region, onRegion, target, year, months, period, periodLabel, onPeriod, date, layer, onLayer, historicalMLP, onHistoricalMLP, raw, onRaw, optical, onOptical, image, onRetryImage, targets, onClearTargets, scale, cells, modelAlerts, modelState, loading }) {
  const isMonth = (m) => period && period.from === m.from && period.to === m.to;
  const imageText = !optical
    ? "Off"
    : image.state === "on" || image.state === "held"
      ? `${formatDate(image.date)}${image.state === "held" ? " · held" : ""}`
      : image.state === "error"
        ? "Unavailable"
        : image.state === "paused"
          ? "Paused"
          : "Loading";
  return (
    <div className="scope" role="toolbar" aria-label="What is on screen">
      <div className="scope-group">
        <span className="scope-key">Area</span>
        {/* with a target in scope the line reads as a breadcrumb: the sector it sits in, then the target */}
        {Object.entries(regions)
          .filter(([key]) => !target || key === region)
          .map(([key, r]) => (
            <button key={key} aria-pressed={!target && region === key} className={!target && region === key ? "on" : undefined} title={target ? `Back to ${r.name}` : r.name} onClick={() => onRegion(key)}>
              {target ? `${r.short} ›` : r.short}
            </button>
          ))}
        {target && <span className="scope-chip">{target}</span>}
      </div>
      <div className="scope-group scope-season">
        <span className="scope-key">Season</span>
        <span className="scope-value">{year ?? "–"}</span>
      </div>
      <div className="scope-group">
        <span className="scope-key">Period</span>
        <button aria-pressed={!period} className={period ? undefined : "on"} title="The whole season" onClick={() => onPeriod(null)}>
          Full
        </button>
        {months.map((m) => (
          <button key={m.key} aria-pressed={Boolean(isMonth(m))} className={isMonth(m) ? "on" : undefined} title={`Compare ${m.key} of every season`} onClick={() => onPeriod({ from: m.from, to: m.to })}>
            {m.key}
          </button>
        ))}
        {period && !months.some(isMonth) && <span className="scope-chip">{periodLabel}</span>}
      </div>
      {date !== "all" && (
        <div className="scope-group scope-day">
          <span className="scope-key">Day</span>
          <span className="scope-value">{formatDate(date)}</span>
        </div>
      )}
      <div className="scope-group">
        <span className="scope-key">Show</span>
        {LAYERS.map(([key, text, title]) => (
          <button key={key} aria-pressed={!historicalMLP && layer === key} className={!historicalMLP && layer === key ? "on" : undefined} title={title} onClick={() => onLayer(key)}>
            {text}
          </button>
        ))}
        {historicalMLP !== undefined && (
          <button aria-pressed={historicalMLP} className={historicalMLP ? "on" : undefined} onClick={onHistoricalMLP} title="Show model-generated hotspot alerts on processed pre-VIIRS Aqua MODIS scenes">PLSWORK</button>
        )}
      </div>
      <div className={`scope-group${raw ? " scope-alert" : ""}`}>
        <span className="scope-key">View</span>
        <button aria-pressed={raw} className={raw ? "on" : undefined} title="Corrected: days burning per cell, the same for either sensor. Uncorrected: raw detection counts, what a naive comparison plots. Click to switch, or hold R." onClick={() => onRaw(!raw)}>
          {raw ? "Uncorrected" : "Corrected"}
        </button>
      </div>
      <div className={`scope-group${optical && image.state === "error" ? " scope-alert" : ""}`} aria-live="polite">
        <span className="scope-key">Sat image</span>
        <button aria-pressed={optical} className={optical ? "on" : undefined} title="Drape that day's NASA satellite image over the analysis area (S)" onClick={() => onOptical(!optical)}>
          {imageText}
        </button>
        {optical && image.state === "error" && (
          <button title="Ask NASA for the image again" onClick={onRetryImage}>
            Retry
          </button>
        )}
      </div>
      {targets > 1 && (
        <div className="scope-group">
          <span className="scope-key">Areas</span>
          <span className="scope-value">{targets}</span>
          <button title="Remove every area marker (C)" onClick={onClearTargets}>
            Clear
          </button>
        </div>
      )}
      {scale !== 1 && (
        <div className="scope-group">
          <span className="scope-key">Size</span>
          <span className="scope-value">{scale.toFixed(1)}×</span>
        </div>
      )}
      <span className="scope-count">
        {loading ? "loading" : modelState === "loading" ? "loading model" : modelState === "error" ? "model unavailable" : modelState === "missing" ? "no model coverage" : modelState === "ready" ? `${integer(modelAlerts)} model alerts` : `${integer(cells)} cells`}
      </span>
    </div>
  );
}

const Key = ({ shape, color, children }) => (
  <span>
    <i className={`g-${shape}`} style={{ "--c": color }} />
    {children}
  </span>
);

/**
 * What the marks on the globe mean. Shape is always the sensor. Colour and size are how
 * long a cell burned; on the Diff layer colour is the sensor instead, and in the
 * uncorrected view size and colour are raw detection counts.
 */
export function Legend({ layer, noViirs, allDays, raw, historicalMLP }) {
  if (historicalMLP) {
    return (
      <div className="legend-row" aria-label="Legend">
        <Key shape="plain" color={COLORS.mlp}>0.25° cell with PLSWORK alerts</Key>
        <Key shape="plain" color={COLORS.coverage}>Scored 0.25° cell, no alerts</Key>
        <span>Only processed Aqua MODIS scene dates are scored; no mark means unavailable or unscored</span>
      </div>
    );
  }
  const diff = layer === "uncertain" && !noViirs;
  return (
    <div className="legend-row" aria-label="Legend">
      {noViirs && diff ? (
        <Key shape="modis" color={C.soft}>MODIS (VIIRS not yet launched)</Key>
      ) : diff ? (
        <>
          <Key shape="modis" color={C.modis}>only MODIS reported it</Key>
          <Key shape="viirs" color={C.viirs}>only VIIRS reported it</Key>
          <Key shape="split" color={C.split}>each alone, on different days</Key>
        </>
      ) : (
        <Key shape="plain" color={mapRamp(0.35)}>{noViirs ? "0.25° cell with a hotspot (MODIS)" : "0.25° cell with a hotspot"}</Key>
      )}
      {raw && !diff ? (
        <>
          <span className="ramp">
            raw detections
            <i style={{ background: `linear-gradient(90deg, ${mapRamp(0.4)}, ${mapRamp(0.7)}, ${mapRamp(1)})` }} />
            1 → {integer(RAW_CEILING)}+
          </span>
          <span className="ramp">99% of MODIS cells: under {integer(RAW_REFERENCE)}</span>
        </>
      ) : (
        !diff &&
        allDays && (
          <span className="ramp">
            days burning
            <i style={{ background: `linear-gradient(90deg, ${mapRamp(0.3)}, ${mapRamp(0.65)}, ${mapRamp(1)})` }} />
            1 → 45+
          </span>
        )
      )}
    </div>
  );
}
