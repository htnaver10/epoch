import { useId, useState } from "react";
import { C, verdictColor } from "../lib/theme";
import { linear, niceTicks, linePath, clamp, compact, integer, times, signedPercent } from "../lib/chart";
import { useWidth, useRem } from "../lib/hooks";
import { KV } from "./Readout";

const pValue = (p) => (p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(p < 0.1 ? 3 : 2)}`);
// Only the unusual seasons carry a verdict colour; the rest stay neutral so they recede.
const barColor = (kind) => (kind === "high" || kind === "low" || kind === "mixed" ? verdictColor(kind) : "#8fa2b6");

/**
 * The record's headline: the trend on the consistent sensor, and under it the trend a chart
 * of spliced raw counts would show.
 */
export function TrendLine({ analysis }) {
  const { trend, spliced, first, last } = analysis;
  if (!trend) return null;
  return (
    <KV
      className="small"
      rows={[
        { k: "Trend", v: signedPercent(trend.perDecade), unit: "per decade", flag: `MODIS ${first}–${last}`, lead: true },
        spliced && { k: "Uncorr. trend", v: signedPercent(spliced.perDecade), unit: "per decade", flag: "spliced", dim: true },
        { k: "Mann–Kendall", v: <span className="keep">{pValue(trend.p)}</span>, unit: trend.p < 0.05 ? "detectable" : "not detectable", flag: "Theil–Sen" },
      ]}
    />
  );
}

/**
 * Season totals on the consistent record. The selected season also carries a hatched ghost
 * bar: how tall it would stand if raw VIIRS counts were read against the old MODIS record.
 */
export function SeasonBars({ analysis, year, onYear, height = 300 }) {
  const { series, band, base, trend, verdicts } = analysis;
  const [ref, width] = useWidth(420);
  const { k } = useRem();
  const [hover, setHover] = useState(null);
  const clip = useId(), hatch = useId();
  const H = height * k, m = { t: 26 * k, r: 50 * k, b: 28 * k, l: 40 * k };
  const n = series.length, step = (width - m.l - m.r) / Math.max(1, n);
  const peak = Math.max(1, ...series.map((t) => Math.max(t.modis ?? 0, t.fromViirsHigh ?? 0)));
  const ticks = niceTicks(peak, 5), yMax = ticks[ticks.length - 1];
  const y = linear(0, yMax, H - m.b, m.t);
  const cx = (i) => m.l + step * (i + 0.5);
  const bar = clamp(step * 0.62, 3, 28 * k);
  const every = step >= 34 * k ? 1 : step >= 17 * k ? 2 : 4;
  const h = hover === null ? null : series[hover];
  const hv = h ? verdicts.get(h.year) : null;
  const sel = series.findIndex((t) => t.year === year);
  const sv = verdicts.get(year);
  // what the selected season would measure if the naive comparison were true
  const ghost = sv && sv.kind !== "insufficient" && sv.naive !== null && sv.detectionInflation > 1.2 ? sv.naive * sv.base : null;
  return (
    <div ref={ref} className="chart" onMouseLeave={() => setHover(null)}>
      <svg width={width} height={H} role="img" aria-label="Fire activity per season on the consistent MODIS record, with VIIRS translated onto the same scale">
        <defs>
          <clipPath id={clip}>
            <rect x={m.l} y={m.t - 6} width={Math.max(0, width - m.l - m.r)} height={H - m.t - m.b + 6} />
          </clipPath>
          <pattern id={hatch} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke={C.soft} strokeWidth="2" strokeOpacity="0.5" />
          </pattern>
        </defs>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} stroke={C.grid} />
            <text x={m.l - 8 * k} y={y(t) + 3.5 * k} textAnchor="end" className="tick">{compact(t)}</text>
          </g>
        ))}
        {band && <rect x={m.l} width={Math.max(0, width - m.l - m.r)} y={y(band[1])} height={Math.max(0, y(band[0]) - y(band[1]))} fill="#ffffff" fillOpacity="0.045" />}
        {base !== null && (
          <>
            <line x1={m.l} x2={width - m.r} y1={y(base)} y2={y(base)} stroke={C.muted} strokeDasharray="4 4" />
            <text x={width - m.r + 8 * k} y={y(base) + 3.5 * k} className="axis-title">median</text>
          </>
        )}
        {ghost !== null && sel >= 0 && (
          <g>
            <rect x={cx(sel) - bar / 2} y={m.t - 6} width={bar} height={Math.max(0, y(0) - (m.t - 6))} fill={`url(#${hatch})`} clipPath={`url(#${clip})`} />
            <text x={clamp(cx(sel), m.l + 70 * k, width - 76 * k)} y={m.t - 12 * k} textAnchor="middle" className="ghost-label">
              uncorrected {times(sv.naive)}: off the chart ↑
            </text>
          </g>
        )}
        {series.map((t, i) => {
          const v = verdicts.get(t.year);
          return (
            <g key={t.year}>
              {hover === i && <rect x={m.l + step * i} y={m.t} width={step} height={H - m.t - m.b} fill="#ffffff" fillOpacity="0.05" />}
              {t.modis !== null && (
                <rect x={cx(i) - bar / 2} y={y(t.modis)} width={bar} height={Math.max(1, y(0) - y(t.modis))} fill={barColor(v?.kind)} fillOpacity={t.year === year ? 1 : hover === i ? 0.8 : 0.5} />
              )}
              {t.fromViirs !== null && (
                <g clipPath={`url(#${clip})`}>
                  <line x1={cx(i)} x2={cx(i)} y1={y(t.fromViirsHigh)} y2={y(t.fromViirsLow)} stroke={C.viirs} strokeWidth="1.5" />
                  <circle cx={cx(i)} cy={y(t.fromViirs)} r={Math.min(3.5 * k, bar / 2.2 + 1)} fill={C.surface} stroke={C.viirs} strokeWidth="1.5" />
                </g>
              )}
              {(t.year === year || (t.year % every === 0 && Math.abs(t.year - year) >= every)) && (
                <text x={cx(i)} y={H - m.b + 16 * k} textAnchor="middle" className={t.year === year ? "tick tick-on" : "tick"}>
                  {every > 1 && t.year !== year ? `’${String(t.year).slice(2)}` : t.year}
                </text>
              )}
            </g>
          );
        })}
        {trend && (
          <path d={linePath(series.map((t, i) => [cx(i), y(trend.at(t.year))]))} fill="none" stroke={C.accent} strokeWidth="1.5" strokeDasharray="1 5" strokeLinecap="round" clipPath={`url(#${clip})`} />
        )}
        {series.map((t, i) => (
          <rect key={t.year} x={m.l + step * i} y={m.t} width={step} height={H - m.t - m.b} fill="transparent" style={{ cursor: "pointer" }} onMouseEnter={() => setHover(i)} onClick={() => onYear(t.year)} />
        ))}
      </svg>
      {h && (
        // sits in the corner away from the pointer, so it never covers the bar it describes
        <div className={`tip ${hover < n / 2 ? "tip-right" : "tip-left"}`}>
          <strong>Jun–Aug {h.year}</strong>
          {h.modis !== null && <span><i style={{ background: barColor(hv?.kind) }} /> MODIS <b>{integer(h.modis)}</b></span>}
          {h.fromViirs !== null && <span><i className="ring" style={{ borderColor: C.viirs }} /> VIIRS, translated <b>{integer(h.fromViirs)}</b></span>}
          {hv && hv.kind !== "insufficient" && <span className="tip-foot">{times(hv.likeForLike)} the median · rank {hv.rank} of {hv.of}</span>}
        </div>
      )}
    </div>
  );
}
