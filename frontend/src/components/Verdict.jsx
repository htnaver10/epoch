import { useId } from "react";
import { seasonDate } from "../lib/seasons";
import { C, verdictColor } from "../lib/theme";
import { log, linePath, times, compact, formatDate } from "../lib/chart";
import { KV } from "./Readout";
import { num, signed, pad2, classOf, seasonFigures } from "../lib/figures";
import { useWidth, useRem } from "../lib/hooks";

const FRACTIONS = { 0.125: "⅛", 0.25: "¼", 0.5: "½" };

/** Every season on one log axis, in multiples of the median, with the unusual tails marked. */
function StripPlot({ v, year, onYear }) {
  const values = v.field.map((f) => f.value / v.base);
  const lo = Math.min(...values) * 0.8, hi = Math.max(...values) * 1.25;
  const scale = log(lo, hi, 0, 100);
  const at = (r) => `${scale(r).toFixed(2)}%`;
  const marks = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32].filter((m) => m > lo * 1.08 && m < hi / 1.08);
  return (
    <div className="strip-plot">
      <div className="strip-track">
        <i className="strip-band" style={{ left: at(v.lowCut / v.base), width: `${(scale(v.highCut / v.base) - scale(v.lowCut / v.base)).toFixed(2)}%` }} />
        <i className="strip-median" style={{ left: at(1) }} />
        {v.field.map((f) =>
          f.year === year ? null : (
            <button
              key={f.year}
              className="strip-tick"
              style={{ left: at(f.value / v.base) }}
              title={`${f.year} · ${times(f.value / v.base)} the median`}
              aria-label={`Select ${f.year}, ${times(f.value / v.base)} the median`}
              onClick={() => onYear(f.year)}
            />
          ),
        )}
        <span className="strip-target" style={{ left: at(v.likeForLike) }}>
          <b>{year}</b>
        </span>
      </div>
      <div className="strip-axis" aria-hidden="true">
        {marks.map((m) => (
          <span key={m} style={{ left: at(m) }}>{FRACTIONS[m] || m}×</span>
        ))}
      </div>
    </div>
  );
}

/**
 * The Brief panel: the whole assessment as one table, where the season sits among the
 * others, and the corrected and uncorrected comparisons drawn to one scale.
 */
export function BriefPanel({ seasons, analysis, year, onYear }) {
  const v = analysis.verdicts.get(year);
  if (!v || v.kind === "insufficient") return <p className="note">{v?.reason || "No observations for this season."}</p>;
  const { first, viirsFirst } = analysis;
  const f = seasonFigures(seasons, v, year);
  const max = f.hasNaive ? Math.max(v.naive, v.naiveLike) : 1;
  const width = (x) => `${Math.max(1.5, (x / max) * 100).toFixed(1)}%`;
  const crossShort = v.crossPct !== null && v.crossWindow < v.window;
  return (
    <div style={{ "--tone": verdictColor(v.kind) }} className="panel-flow">
      <KV
        className="small"
        rows={[
          { k: "Corrected index", v: num(v.likeForLike), unit: "× median", flag: classOf(v), lead: true },
          f.hasNaive && { k: "Uncorrected", v: num(v.naive), unit: "× median", flag: f.window || "raw VIIRS", dim: true },
          f.hasNaive && { k: "Sensor factor", v: num(f.factor), unit: "×", flag: f.window },
          { k: "Rank", v: `${pad2(v.rank)}/${v.of}`, unit: v.basis === "viirs" ? "MODIS scale, from VIIRS" : "MODIS record", flag: f.partial },
          { k: "Percentile", v: Math.round(v.pct * 100) },
          { k: "Robust z", v: v.z === null ? "–" : signed(v.z), unit: <span className="keep">σ, log scale</span> },
          {
            k: "VIIRS own record",
            v: v.crossPct === null ? "–" : `P${Math.round(v.crossPct * 100)}`,
            unit: v.crossPct === null ? "not observing" : v.kind === "mixed" ? "disagrees" : "agrees",
            flag: crossShort ? `thru ${formatDate(seasonDate(seasons, year, v.crossWindow - 1))}` : "",
          },
        ]}
      />

      <figure className="fig">
        <figcaption>All seasons, × median, log scale. Band: the middle 80%. Click a line to select that season.</figcaption>
        <StripPlot v={v} year={year} onYear={onYear} />
      </figure>

      {f.hasNaive ? (
        <figure className="fig">
          <figcaption>Uncorrected against corrected, on one scale{f.window ? `, ${f.window}` : ""}.</figcaption>
          <div className="cmp">
            <div className="cmp-row cmp-naive">
              <div className="cmp-head">
                <span>
                  Uncorrected: raw VIIRS count against the MODIS record of {first}–{viirsFirst - 1}
                </span>
                <b>{num(v.naive)}×</b>
              </div>
              <div className="cmp-track"><i style={{ width: width(v.naive) }} /></div>
            </div>
            <div className="cmp-row cmp-real">
              <div className="cmp-head">
                <span>Corrected: same sensor scale, same grid, same days</span>
                <b>{num(v.naiveLike)}×</b>
              </div>
              <div className="cmp-track"><i style={{ width: width(v.naiveLike) }} /></div>
            </div>
          </div>
        </figure>
      ) : (
        <p className="note">
          VIIRS began observing in {viirsFirst ?? "2012"}. {year} was seen by MODIS alone, so there is no uncorrected figure.
        </p>
      )}
    </div>
  );
}

const LOG_TICKS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000];

/** Overlap seasons on log axes with the fitted bridge and the constant-ratio line it replaces. */
export function BridgeScatter({ totals, bridge, year }) {
  const [ref, width] = useWidth(360);
  const { k } = useRem();
  const clip = useId();
  const H = 200 * k, m = { t: 10 * k, r: 12 * k, b: 30 * k, l: 40 * k };
  const pts = totals.filter((t) => t.modis > 0 && t.viirs > 0);
  const x0 = Math.min(...pts.map((t) => t.viirs)) * 0.75, x1 = Math.max(...pts.map((t) => t.viirs)) * 1.3;
  const y0 = Math.min(...pts.map((t) => t.modis)) * 0.75, y1 = Math.max(...pts.map((t) => t.modis)) * 1.3;
  const x = log(x0, x1, m.l, width - m.r), y = log(y0, y1, H - m.b, m.t);
  const sweep = Array.from({ length: 25 }, (_, i) => x0 * (x1 / x0) ** (i / 24));
  const xt = LOG_TICKS.filter((t) => t > x0 && t < x1), yt = LOG_TICKS.filter((t) => t > y0 && t < y1);
  const selected = pts.find((t) => t.year === year);
  return (
    <div ref={ref} className="chart">
      <svg width={width} height={H} role="img" aria-label="VIIRS against MODIS season totals for the seasons both sensors flew, with the fitted bridge">
        <defs>
          <clipPath id={clip}>
            <rect x={m.l} y={m.t} width={Math.max(0, width - m.l - m.r)} height={H - m.t - m.b} />
          </clipPath>
        </defs>
        {yt.map((t) => (
          <g key={`y${t}`}>
            <line x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} stroke={C.grid} />
            <text x={m.l - 6 * k} y={y(t) + 3 * k} textAnchor="end" className="tick">{compact(t)}</text>
          </g>
        ))}
        {xt.map((t) => (
          <g key={`x${t}`}>
            <line x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} stroke={C.grid} />
            <text x={x(t)} y={H - m.b + 13 * k} textAnchor="middle" className="tick">{compact(t)}</text>
          </g>
        ))}
        <text x={width - m.r} y={H - 3 * k} textAnchor="end" className="axis-title">VIIRS cell-days →</text>
        <text x={m.l + 4 * k} y={m.t + 9 * k} className="axis-title">↑ MODIS</text>
        <g clipPath={`url(#${clip})`}>
          <path d={linePath(sweep.map((v) => [x(v), y(v / bridge.ratio)]))} fill="none" stroke={C.faint} strokeWidth="1" strokeDasharray="3 4" />
          <path d={linePath(sweep.map((v) => [x(v), y(bridge.toModis(v))]))} fill="none" stroke={C.accent} strokeWidth="1.5" />
        </g>
        {pts.map((t) => (
          <circle key={t.year} cx={x(t.viirs)} cy={y(t.modis)} r={(t.year === year ? 5 : 3.2) * k}
            fill={t.year === year ? C.text : C.viirs} fillOpacity={t.year === year ? 1 : 0.8} stroke={C.surface} strokeWidth="1">
            <title>{`${t.year}: VIIRS ${compact(t.viirs)}, MODIS ${compact(t.modis)}`}</title>
          </circle>
        ))}
        {selected && (
          <text x={x(selected.viirs)} y={y(selected.modis) - 9 * k} textAnchor={x(selected.viirs) > width - 60 ? "end" : "middle"} className="point-label">
            {year}
          </text>
        )}
      </svg>
    </div>
  );
}
