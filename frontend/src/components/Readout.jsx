import { VERDICTS } from "../lib/seasons";
import { verdictColor } from "../lib/theme";
import { num, signed, pad2, classOf, seasonFigures, criticalPeriods } from "../lib/figures";
import { formatDate } from "../lib/chart";

const FLAG = { high: "High ▲", low: "Low ▼", typical: "Normal", mixed: "Mixed" };

/**
 * Rows of labelled values in shared columns: the one way this console states a number.
 * A row is { k, v, unit, flag, lead, dim, hot, tone, title }; false entries are skipped.
 */
export function KV({ rows, className = "" }) {
  return (
    <dl className={`kv ${className}`.trim()}>
      {rows.filter(Boolean).map((r) => (
        <div key={r.k} className={`row${r.lead ? " lead" : ""}${r.dim ? " dim" : ""}${r.hot ? " hot" : ""}`} style={r.tone ? { "--tone": r.tone } : undefined}>
          <dt title={r.title}>{r.k}</dt>
          <dd>{r.v}</dd>
          <dd className="unit">{r.unit}</dd>
          <dd className="flag">{r.flag}</dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * Two measures across the two sensors, with VIIRS over MODIS for each. The common grid is
 * the comparison that holds; raw detections are listed under it, dimmed and labelled.
 */
export function SensorMatrix({ cellDays, detections, noViirs, unit = "cell-days" }) {
  const ratio = ([m, v]) => (noViirs || !m ? "–" : `${(v / m).toFixed(1)}×`);
  const n = (x) => (x === null || x === undefined ? "–" : Math.round(x).toLocaleString("en-CA"));
  return (
    <table className="matrix">
      <thead>
        <tr>
          <th />
          <th scope="col">MODIS</th>
          <th scope="col">VIIRS</th>
          <th scope="col">V/M</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th scope="row">{unit}</th>
          <td>{n(cellDays[0])}</td>
          <td>{noViirs ? "–" : n(cellDays[1])}</td>
          <td className="ratio">{ratio(cellDays)}</td>
        </tr>
        <tr className="dim">
          <th scope="row">
            Detections<small>uncorr.</small>
          </th>
          <td>{n(detections[0])}</td>
          <td>{noViirs ? "–" : n(detections[1])}</td>
          <td className="ratio">{ratio(detections)}</td>
        </tr>
      </tbody>
    </table>
  );
}

/**
 * The assessment of one season, for the area and period in scope, as an instrument readout. The corrected index and the uncorrected
 * figure are two rows of one table, and the gap between them is a row of its own. `compact`
 * folds it to the title line with the index in it, for when a target and a panel need the room.
 */
export function Readout({ seasons, analysis, year, areaName, periodLabel, raw, compact }) {
  const v = analysis.verdicts.get(year);
  const known = v && v.kind !== "insufficient";
  const f = known ? seasonFigures(seasons, v, year) : null;
  const tone = verdictColor(v?.kind);
  const periods = known && analysis.calendar?.length ? criticalPeriods(seasons, analysis.calendar, year) : null;
  return (
    <section className={`readout${raw ? " is-raw" : ""}${compact ? " compact" : ""}`} style={{ "--tone": tone }} aria-label="Season assessment" aria-live="polite">
      <header>
        <h2 title={areaName}>{areaName.replace(/ · (\d+) km$/, " $1 km")}</h2>
        <span>
          {year}
          {compact && /^[A-Z][a-z]{2}–[A-Z][a-z]{2}$/.test(periodLabel) ? "" : ` · ${periodLabel}`}
          {known && f.partial ? ` · ${f.partial}` : ""}
          {known && compact ? (
            <>
              {" · "}
              <b>{num(v.likeForLike)}×</b> median
            </>
          ) : known ? (
            ` · Rank ${pad2(v.rank)}/${v.of}`
          ) : (
            ""
          )}
        </span>
      </header>
      {compact ? null : known ? (
        <>
          <KV
            rows={[
              { k: "Activity index", v: num(v.likeForLike), unit: "× median", flag: FLAG[v.kind], lead: true },
              f.hasNaive && { k: "Uncorrected", v: num(v.naive), unit: "× median", flag: f.window || "raw VIIRS", dim: !raw, hot: raw },
              f.hasNaive && { k: "Sensor factor", v: num(f.factor), unit: "×", flag: f.window },
              { k: "Robust z", v: v.z === null ? "–" : signed(v.z), unit: <span className="keep">σ</span>, title: "Robust z-score of the season total, in log space" },
              { k: "Percentile", v: Math.round(v.pct * 100) },
            ]}
          />
          {periods && (periods.peak || periods.busiest) && (
            <footer className="periods" aria-label="Critical periods">
              {periods.peak && (
                <span title="Weeks whose usual activity is within 75% of the busiest usual week, for this area">
                  Usual peak <b>{formatDate(periods.peak.from)}–{formatDate(periods.peak.to)}</b>
                </span>
              )}
              {periods.busiest && (
                <span title="This season's busiest week, against the usual for that week">
                  Peak wk <b>{formatDate(periods.busiest.from)}{periods.busiest.vsUsual ? ` · ${num(periods.busiest.vsUsual)}×` : ""}</b>
                </span>
              )}
            </footer>
          )}
          <footer>
            <span>
              Class <b>{classOf(v)}</b>
            </span>
            <span>
              VIIRS check <b>{f.cross}</b>
            </span>
          </footer>
        </>
      ) : (
        <footer>
          <span>
            Class <b>{VERDICTS.insufficient}</b>
          </span>
          <span>{v?.reason || "No observations."}</span>
        </footer>
      )}
    </section>
  );
}
