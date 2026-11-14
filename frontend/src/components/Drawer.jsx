import Icon from "./Icon";
import { KV, SensorMatrix } from "./Readout";
import { BriefPanel, BridgeScatter } from "./Verdict";
import { TrendLine, SeasonBars } from "./SeasonRecord";
import ActivityCalendar from "./ActivityCalendar";
import { CumulativeChart, DailyChart, Calendar } from "./DailyDetail";
import { C } from "../lib/theme";
import { integer, formatDate } from "../lib/chart";
import { medianReached } from "../lib/milestones";

const TITLES = { brief: "Summary", record: "History", season: "Season", sensors: "Sensors", method: "Method" };
const pct = (v) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`);

/** A figure in a panel: one line saying what it is, then the thing itself. No box. */
function Fig({ say, children, aside }) {
  return (
    <figure className="fig">
      <figcaption>
        <span>{say}</span>
        {aside}
      </figcaption>
      {children}
    </figure>
  );
}

function Method({ analysis, seasons }) {
  const b = analysis.bridge, val = analysis.validation;
  const statics = seasons?.static_sources?.length || 0;
  const gapDays = Object.values(seasons?.gaps?.viirs || {}).reduce((n, d) => n + d.length, 0) + Object.values(seasons?.gaps?.modis || {}).reduce((n, d) => n + d.length, 0);
  return (
    <div className="panel-flow method">
      <p className="say">
        Every season is counted the same way on the one sensor that has flown the whole time. A second, sharper sensor checks it. Every figure is computed in the browser from NASA's public archive.
      </p>
      <details open>
        <summary>Data and analysis area</summary>
        <p>
          NASA FIRMS standard-processing archives for June to August: MODIS Collection 6.1 (Terra and Aqua, 1 km pixels) from 2003 and VIIRS on Suomi NPP (375 m pixels) from 2012. The analysis area is a rectangle, 50–70°N by 140–60°W, not the national border: southern Ontario and Quebec, the Maritimes and Newfoundland are outside it. Hotspots are thermal detections. They are not individual fires and not burned area.
        </p>
      </details>
      <details open>
        <summary>Not corrected</summary>
        <p>
          Cloud cover, overpass timing, day and night sampling, and Terra's orbit drift after 2020. Two dozen seasons is a short record: “top tenth” means the top two or three.
        </p>
      </details>
      <details open>
        <summary>Areas, periods and the calendar</summary>
        <p>
          Any place clicked or searched becomes the area in scope: a circle whose history is rebuilt from the same cell-day records and judged against its own other seasons, exactly as the built-in sectors are. A period (a month, or a span of weeks dragged across the calendar) compares the same days of every season. The calendar shows activity per week on the MODIS scale; a season MODIS did not observe is shown from VIIRS, translated, and hatched. Both the calendar and the hotspot cell-days of the area and period in scope download as CSV.
        </p>
      </details>
      <details>
        <summary>One grid, weighted by ground area</summary>
        <p>
          Detections are binned into 0.25° cells by UTC date, and each sensor counts a cell at most once a day. That removes repeat detections of the same fire, which is most of VIIRS's raw-count advantage. A cell at 70°N covers about half the ground of one at 50°N, so each cell-day is weighted by cos(latitude), scaled to 1 at 60°N.
        </p>
      </details>
      <details>
        <summary>Static heat sources and missing days</summary>
        <p>
          {statics} cells are hot almost every day of almost every summer whatever the fire season does: oil sands upgraders, refineries, a cement plant, other industrial sites. VIIRS sees these small steady sources far more often than MODIS, so they are left out of both sensors everywhere, including the map. A cell is static when VIIRS saw it in at least 80% of its full seasons, on a median of at least 5 days of each. On {gapDays} days one sensor's archive is all but empty while the other sees a normal day; those days are filled from the other sensor at the ratio of the neighbouring days. Before VIIRS, one sensor alone cannot tell a missing day from a cloudy one, and nothing is filled.
        </p>
      </details>
      <details>
        <summary>Ranking a season</summary>
        <p>
          A season's MODIS total is ranked against every other MODIS season. The top tenth is classed unusually high, the bottom tenth unusually quiet. Robust z is a robust z-score on the logarithm of the total. A season still being archived is compared over the same calendar days of every other year.
        </p>
      </details>
      <details>
        <summary>The VIIRS cross-check and the sensor bridge</summary>
        <p>
          VIIRS ranks each season on its own record; if it lands on the other side of its median from MODIS's call, the class is shown as a disagreement. To show VIIRS on the MODIS scale, EPOCH fits ln(MODIS) = a + b·ln(VIIRS) on the seasons both flew
          {b ? ` (${b.years} here${b.model === "power" ? `, b = ${b.b.toFixed(2)}` : ""})` : ""}, always without the season being judged.
          {val?.mdape != null && ` Held out this way the median error is ${pct(val.mdape)}, against ${pct(val.ratioMdape)} for a constant ratio; the worst season misses by ${pct(val.worst)}.`}
        </p>
      </details>
      <details>
        <summary>Uncorrected figures and trend</summary>
        <p>
          “Uncorrected” is what a comparison of raw counts gives: a VIIRS-era season's raw VIIRS detections against the raw MODIS counts of the years before VIIRS. The sensor factor is uncorrected divided by corrected over the same days. The trend is a Theil–Sen slope on the logarithm of MODIS season totals, with a Mann–Kendall test; the uncorrected trend repeats it on raw VIIRS counts joined to the earlier MODIS record.
        </p>
      </details>
      <details>
        <summary>Keys</summary>
        <dl className="shortcuts">
          <dt><kbd>/</kbd></dt><dd>Search: place, coordinates, season, day, region</dd>
          <dt><kbd>←</kbd><kbd>→</kbd></dt><dd>Step one day</dd>
          <dt><kbd>Space</kbd></dt><dd>Play or pause the season</dd>
          <dt><kbd>[</kbd><kbd>]</kbd></dt><dd>Previous or next season</dd>
          <dt><kbd>1</kbd>–<kbd>5</kbd></dt><dd>Open a panel</dd>
          <dt><kbd>D</kbd> <kbd>S</kbd></dt><dd>Sensor disagreement layer · satellite image</dd>
          <dt><kbd>R</kbd></dt><dd>Hold for the uncorrected view</dd>
          <dt><kbd>C</kbd></dt><dd>Clear area markers</dd>
          <dt>Scroll</dt><dd>Zoom toward the spot under the pointer</dd>
          <dt>Right-drag</dt><dd>Tilt the view toward the horizon (or the 3D button)</dd>
          <dt><kbd>0</kbd></dt><dd>Summary: the record season with the Summary panel</dd>
          <dt><kbd>+</kbd> <kbd>−</kbd></dt><dd>Make everything larger or smaller</dd>
          <dt><kbd>H</kbd> <kbd>Esc</kbd></dt><dd>Home · close whatever was opened last</dd>
        </dl>
      </details>
    </div>
  );
}

/** The side panel. One subject at a time; nothing here is on screen until asked for. */
export default function Drawer({ panel, onClose, ctx }) {
  const { seasons, analysis, curves, year, regionName, date, period, span, onPeriod, onExportCalendar, onYear, onDate, tone, daily, hasRecords, metric, onMetric, viirsThrough, noViirs, viirsEnded, summary, peak } = ctx;
  const { bridge, validation, totals } = analysis;
  const reached = curves ? medianReached(seasons, curves, year) : null;
  const busiest = daily.reduce((best, d) => (d.occupied > (best?.occupied ?? 0) ? d : best), null);
  const active = daily.filter((d) => d.occupied > 0).length;
  return (
    <aside className="drawer" aria-label={TITLES[panel]}>
      <header className="drawer-head">
        <h2>{TITLES[panel]}</h2>
        <span className="label">
          {panel === "record" || panel === "method" ? regionName : `${year} · ${regionName}`}
        </span>
        <button className="x" onClick={onClose} aria-label="Close panel" title="Close (Esc)"><Icon name="close" size={14} /></button>
      </header>
      <div className="drawer-body">
        {panel === "brief" && <BriefPanel seasons={seasons} analysis={analysis} year={year} onYear={onYear} />}

        {panel === "record" && (
          <div className="panel-flow">
            <TrendLine analysis={analysis} />
            <Fig say={`Activity index per season, ${analysis.first}–${analysis.last}. Bars: MODIS. Rings: VIIRS on the same scale. Click a season to select it.`}>
              <SeasonBars analysis={analysis} year={year} onYear={onYear} />
            </Fig>
            <ActivityCalendar seasons={seasons} analysis={analysis} year={year} onYear={onYear} period={span} onPeriod={onPeriod} areaName={regionName} onExport={onExportCalendar} />
          </div>
        )}

        {panel === "season" && (
          <div className="panel-flow">
            {hasRecords && daily.length > 0 && (
              <KV
                className="small"
                rows={[
                  reached && { k: "Median total reached", v: reached.v, unit: reached.unit, lead: true },
                  busiest && { k: "Peak day", v: formatDate(busiest.date), unit: `${integer(busiest.occupied)} cells` },
                  { k: "Active days", v: `${active}/${daily.length}`, unit: "any hotspot" },
                ]}
              />
            )}
            <Fig say={`${year}'s running total against every other season. Dashed: the median season. Click a day to show it on the globe.`}>
              {curves && <CumulativeChart seasons={seasons} curves={curves} year={year} date={date} onDate={onDate} tone={tone} />}
            </Fig>
            <Fig
              say={metric === "grid" ? "Cells each sensor lit per day, on the common grid." : "Raw detections per day. Not comparable across sensors."}
              aside={
                <span className="seg" role="group" aria-label="Counting unit">
                  <button className={metric === "grid" ? "on" : undefined} aria-pressed={metric === "grid"} onClick={() => onMetric("grid")}>Grid</button>
                  <button className={metric === "raw" ? "on" : undefined} aria-pressed={metric === "raw"} onClick={() => onMetric("raw")}>Raw</button>
                </span>
              }
            >
              <p className="keys">
                <span><i style={{ background: C.modis }} /> MODIS</span>
                {!noViirs && <span><i style={{ background: C.viirs }} /> VIIRS</span>}
              </p>
              {hasRecords && daily.length > 0 ? <DailyChart daily={daily} date={date} onDate={onDate} viirsThrough={viirsThrough} /> : <div className="shimmer" style={{ height: "14rem" }} />}
            </Fig>
            <Fig say="Cells burning per day.">
              {hasRecords && daily.length > 0 ? <Calendar daily={daily} date={date} onDate={onDate} peak={peak} /> : <div className="shimmer" style={{ height: "12rem" }} />}
            </Fig>
          </div>
        )}

        {panel === "sensors" && (
          <div className="panel-flow">
            {bridge ? (
              <>
                <KV
                  className="small"
                  rows={[
                    { k: "Bridge error", v: pct(validation.mdape), unit: "median", flag: bridge.model === "power" ? "power law" : "ratio", lead: true },
                    { k: "Constant-ratio error", v: pct(validation.ratioMdape), unit: "median", dim: true },
                    { k: "Worst season", v: pct(validation.worst), unit: `ratio ${pct(validation.ratioWorst)}` },
                    { k: "Seasons fitted", v: bridge.years, unit: "held out" },
                  ]}
                />
                <Fig say={`Season totals, VIIRS against MODIS, log axes. Line: the fitted bridge. Dashed: a constant ${bridge.ratio.toFixed(2)}× ratio.`}>
                  <BridgeScatter totals={totals} bridge={bridge} year={year} />
                </Fig>
              </>
            ) : (
              <p className="note">Fewer than three seasons were seen by both sensors, so no bridge is fitted.</p>
            )}
            <Fig say={`On the globe now: ${date === "all" ? period : formatDate(date, "long")}.`}>
              <SensorMatrix cellDays={[summary.modisCells, summary.viirsCells]} detections={[summary.modis, summary.viirs]} noViirs={noViirs || viirsEnded} unit="Cells" />
              <KV
                className="small"
                rows={[
                  { k: "Seen by both", v: summary.overlap === null ? "–" : `${Math.round(summary.overlap * 100)}%`, unit: "of cell-days" },
                  noViirs && { k: "VIIRS", v: "–", unit: "not yet launched" },
                  !noViirs && viirsEnded && { k: "VIIRS", v: "–", unit: "not archived for this day" },
                ]}
              />
            </Fig>
          </div>
        )}

        {panel === "method" && <Method analysis={analysis} seasons={seasons} />}
      </div>
    </aside>
  );
}
