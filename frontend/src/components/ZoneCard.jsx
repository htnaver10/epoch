import { useEffect, useMemo, useRef, useState } from "react";
import Icon from "./Icon";
import { KV, SensorMatrix } from "./Readout";
import { fireRamp, firmsUrl } from "../lib/theme";
import { zoneReport, coverageOf, cellSizeKm, formatCoords } from "../lib/geo";
import { integer, formatDate } from "../lib/chart";

const KIND = { cell: "Grid cell", zone: "Area of interest", place: "Area of interest", coords: "Area of interest" };

/**
 * Daily activity in the zone across the season. One surface to hover and click rather than
 * a button per day: the bars are a few pixels wide, too small to aim at individually. The
 * arrow keys step the same days from anywhere, so this strip is not a tab stop.
 */
function ZoneTimeline({ dates, report, date, onDate }) {
  const [hover, setHover] = useState(null);
  const byDate = new Map(report.daily.map((d) => [d.date, d]));
  const max = Math.max(1, ...report.daily.map((d) => d.cells));
  const at = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    return dates[Math.max(0, Math.min(dates.length - 1, Math.floor(((e.clientX - box.left) / Math.max(1, box.width)) * dates.length)))];
  };
  const shown = hover || (date !== "all" ? date : null);
  const v = shown ? byDate.get(shown) : null;
  return (
    <>
      <div
        className="zone-timeline"
        role="img"
        aria-label="Daily activity in this zone through the season"
        onMouseMove={(e) => setHover(at(e))}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => {
          const d = at(e);
          onDate(d === date ? "all" : d);
        }}
      >
        {dates.map((d) => {
          const day = byDate.get(d);
          return (
            <span key={d} className={d === date ? "on" : d === hover ? "hover" : undefined}>
              {day && <i style={{ height: `${Math.max(12, (day.cells / max) * 100)}%`, background: fireRamp(0.45 + 0.55 * (day.cells / max)) }} />}
            </span>
          );
        })}
      </div>
      <p className="zone-readout">
        {shown
          ? `${formatDate(shown)}: ${v ? `${v.cells} cell${v.cells === 1 ? "" : "s"} burning` : "nothing detected"}`
          : `${formatDate(dates[0])} – ${formatDate(dates[dates.length - 1])} · click a day · ← → step`}
      </p>
    </>
  );
}

/**
 * One hotspot cell, day by day, in the manner of NASA FIRMS's detection table: when it burned
 * and what each sensor reported. A row is a day: click it to show that day on the map. The
 * archive aggregates hold counts per cell and day, not each pixel's brightness or FRP, so the
 * link opens FIRMS itself at this spot on that day for the individual detections.
 */
function Detections({ lat, lon, daily, date, onDate, noViirs }) {
  const day = date !== "all" ? date : daily.reduce((a, d) => (!a || d.modis + d.viirs > a.modis + a.viirs ? d : a), null)?.date;
  return (
    <div className="detections">
      <p className="det-cell">
        <span>Lat <b>{lat.toFixed(3)}</b></span>
        <span>Lon <b>{lon.toFixed(3)}</b></span>
        <span>Cell <b>0.25°</b></span>
      </p>
      <div className="det-scroll">
        <table>
          <thead>
            <tr><th>Date</th><th>MODIS</th>{!noViirs && <th>VIIRS</th>}<th>Seen by</th></tr>
          </thead>
          <tbody>
            {daily.map((d) => (
              <tr key={d.date} className={d.date === date ? "on" : undefined} onClick={() => onDate(d.date === date ? "all" : d.date)} title="Show this day on the map">
                <td>{formatDate(d.date)}</td>
                <td>{d.modis || "–"}</td>
                {!noViirs && <td>{d.viirs || "–"}</td>}
                <td className="by">{d.modis && d.viirs ? "both" : d.modis ? "MODIS" : "VIIRS"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {day && (
        <a className="det-link" href={firmsUrl(lat, lon, day)} target="_blank" rel="noreferrer">
          Each detection on {formatDate(day)} in NASA FIRMS ↗
        </a>
      )}
    </div>
  );
}

/**
 * The dossier that pops out at a focused target: what the season's archive says about one
 * grid cell, or about everything within the zone's radius, as a table of properties. The
 * two sensors are compared on the common grid; the ratio of raw detections is listed after
 * it, dimmed and labelled uncorrected, because it is the comparison the product corrects.
 */
export default function ZoneCard({ focus, index, rows, dates, year, date, noViirs, onDate, onRadius, onClose }) {
  const report = useMemo(() => zoneReport(rows, focus.lat, focus.lon, focus.km), [rows, focus]);
  const coverage = coverageOf(focus.lat, focus.lon, focus.km);
  const [ew, ns] = cellSizeKm(focus.lat);
  const quiet = report.cellDays === 0;
  const id = String(index);
  // The card takes the focus when it opens: Tab then starts inside it, and because the card
  // itself is not a control, Space and the other keys still reach the console.
  const card = useRef(null);
  useEffect(() => {
    card.current?.focus({ preventScroll: true });
  }, []);
  return (
    <section ref={card} tabIndex={-1} className="callout" aria-live="polite" aria-label={`Area of interest ${id}`}>
      <header>
        <span className="tid">{id}</span>
        <span className="kind">{KIND[focus.kind] || "Area of interest"}</span>
        <button className="x" onClick={onClose} aria-label="Close report" title="Close (Esc)">
          <Icon name="close" size={14} />
        </button>
      </header>
      <h3>{focus.label || formatCoords(focus.lat, focus.lon)}</h3>
      <p className="where">
        {focus.label ? `${formatCoords(focus.lat, focus.lon)} · ` : ""}
        {focus.km ? `${focus.km} km` : `${ew.toFixed(0)} × ${ns.toFixed(0)} km cell`}
        <span className="radius">
          <button onClick={() => onRadius(-1)} disabled={!focus.km} aria-label="Smaller area" title="Smaller area">−</button>
          <button onClick={() => onRadius(1)} aria-label="Larger area" title="Larger area">+</button>
        </span>
      </p>

      {coverage === "outside" ? (
        <p className="quiet warn">Outside the analysis area (50–70°N, 140–60°W). No data here is not the same as no fire.</p>
      ) : quiet ? (
        <p className="quiet">No hotspots here in Jun–Aug {year}.</p>
      ) : (
        <>
          <ZoneTimeline dates={dates} report={report} date={date} onDate={onDate} />
          <KV
            className="small"
            rows={[
              { k: "Active days", v: report.activeDays, unit: `of ${dates.length}`, lead: true },
              focus.km > 0 && { k: "Cells burning", v: integer(report.cells) },
              { k: "First det.", v: formatDate(report.first), unit: report.first === report.last ? "" : `to ${formatDate(report.last)}` },
            ]}
          />
          <SensorMatrix cellDays={[report.modisCellDays, report.viirsCellDays]} detections={[report.modis, report.viirs]} noViirs={noViirs} />
          {!focus.km && <Detections lat={focus.lat} lon={focus.lon} daily={report.daily} date={date} onDate={onDate} noViirs={noViirs} />}
        </>
      )}
      {coverage === "partial" && <p className="edge-note">Part of this zone is outside the analysis area; counts cover only the part inside.</p>}

      <footer>
        {!quiet && coverage !== "outside" && report.peak && (
          <button onClick={() => onDate(report.peak)} disabled={date === report.peak}>
            Peak day · {formatDate(report.peak)}
          </button>
        )}
        {date !== "all" && <button onClick={() => onDate("all")}>All days</button>}
      </footer>
      <p className="source">NASA FIRMS · MODIS{noViirs ? "" : " + VIIRS"} · {year}</p>
    </section>
  );
}
