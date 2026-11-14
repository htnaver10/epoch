import { useId, useState } from "react";
import { seasonDate, medianCrossing } from "../lib/seasons";
import { C, fireRamp, inkOn } from "../lib/theme";
import { linear, niceTicks, linePath, areaPath, bandPath, clamp, compact, integer, times, formatDate } from "../lib/chart";
import { useWidth, useRem } from "../lib/hooks";

/** Index under the pointer for a plot spanning `count` evenly spaced points. */
function indexAt(event, count) {
  const box = event.currentTarget.getBoundingClientRect();
  return clamp(Math.round(((event.clientX - box.left) / Math.max(1, box.width)) * (count - 1)), 0, count - 1);
}

function MonthGrid({ dates, x, top, bottom, k }) {
  return dates.map((d, i) =>
    i === 0 || d.endsWith("-01") ? (
      <g key={d}>
        {i > 0 && <line x1={x(i)} x2={x(i)} y1={top} y2={bottom} stroke={C.grid} />}
        <text x={x(i) + (i === 0 ? 0 : 5 * k)} y={bottom + 16 * k} className="tick">{formatDate(d, "monthShort")} 1</text>
      </g>
    ) : null,
  );
}

/**
 * Running total of the season against the envelope of every other season: shows when a
 * season pulled away from the pack, not only that it did.
 */
export function CumulativeChart({ seasons, curves, year, date, onDate, tone, height = 260 }) {
  const [ref, width] = useWidth(420);
  const { k } = useRem();
  const [hover, setHover] = useState(null);
  const clip = useId();
  const { days, band, target, reference } = curves;
  const H = height * k, m = { t: 16 * k, r: 54 * k, b: 28 * k, l: 40 * k };
  const dates = Array.from({ length: days }, (_, i) => seasonDate(seasons, year, i));
  const hasBand = reference >= 3;
  const reached = target ? target.values.length : 0;
  const peak = Math.max(1, reached ? target.values[reached - 1] : 0, hasBand ? band[days - 1].hi : 0);
  const ticks = niceTicks(peak * 1.03, 4), yMax = ticks[ticks.length - 1];
  const x = linear(0, days - 1, m.l, width - m.r), y = linear(0, yMax, H - m.b, m.t);
  const xs = dates.map((_, i) => x(i));
  const endY = reached ? y(target.values[reached - 1]) : null;
  const medianY = hasBand ? y(band[days - 1].mid) : null;
  // keep the two end labels from sitting on top of each other
  const medianLabelY = medianY === null ? null : endY !== null && reached === days && Math.abs(medianY - endY) < 13 * k ? endY + 13 * k : medianY;
  const selected = date === "all" ? -1 : dates.indexOf(date);
  const cross = medianCrossing(curves);
  const hv = hover === null ? null : { date: dates[hover], value: hover < reached ? target.values[hover] : null, mid: hasBand ? band[hover].mid : null };
  return (
    <div ref={ref} className="chart" onMouseLeave={() => setHover(null)}>
      <svg width={width} height={H} role="img" aria-label={`Cumulative fire activity through the ${year} season against the range of other seasons`}>
        <defs>
          <clipPath id={clip}>
            <rect x={m.l} y={m.t} width={Math.max(0, width - m.l - m.r)} height={H - m.t - m.b} />
          </clipPath>
        </defs>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} stroke={C.grid} />
            <text x={m.l - 8 * k} y={y(t) + 3.5 * k} textAnchor="end" className="tick">{compact(t)}</text>
          </g>
        ))}
        <MonthGrid dates={dates} x={x} top={m.t} bottom={H - m.b} k={k} />
        <g clipPath={`url(#${clip})`}>
          {hasBand && <path d={bandPath(xs, band.map((b) => y(b.lo)), band.map((b) => y(b.hi)))} fill="#ffffff" fillOpacity="0.06" />}
          {curves.curves.map((c) =>
            c.year === year ? null : (
              <path key={c.year} d={linePath(c.values.map((v, i) => [xs[i], y(v)]))} fill="none" stroke={C.soft} strokeOpacity="0.16" strokeWidth="1" />
            ),
          )}
          {hasBand && <path d={linePath(band.map((b, i) => [xs[i], y(b.mid)]))} fill="none" stroke={C.muted} strokeWidth="1.25" strokeDasharray="4 4" />}
          {reached > 1 && <path d={linePath(target.values.map((v, i) => [xs[i], y(v)]))} fill="none" stroke={tone} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />}
        </g>
        {cross !== null && cross < days - 3 && hasBand && (
          // the moment this season passed a median season's whole total
          <g>
            <line x1={xs[cross]} x2={xs[cross]} y1={y(band[days - 1].mid)} y2={H - m.b} stroke={tone} strokeWidth="1" />
            <circle cx={xs[cross]} cy={y(band[days - 1].mid)} r={3.5 * k} fill={C.surface} stroke={tone} strokeWidth="1.5" />
            <text x={xs[cross] + 6 * k} y={H - m.b - 6 * k} className="point-label">{formatDate(dates[cross])}</text>
          </g>
        )}
        {selected >= 0 && <line x1={xs[selected]} x2={xs[selected]} y1={m.t} y2={H - m.b} stroke={C.text} strokeOpacity="0.55" strokeDasharray="3 3" />}
        {hover !== null && <line x1={xs[hover]} x2={xs[hover]} y1={m.t} y2={H - m.b} stroke={C.text} strokeOpacity="0.3" />}
        {hv && hv.value !== null && <circle cx={xs[hover]} cy={y(hv.value)} r={4 * k} fill={tone} stroke={C.surface} strokeWidth="1.5" />}
        {reached > 0 && (
          <>
            <circle cx={xs[reached - 1]} cy={endY} r={3.5 * k} fill={tone} />
            <text x={xs[reached - 1] + 8 * k} y={clamp(endY + 4 * k, m.t + 8 * k, H - m.b)} className="point-label">{year}</text>
          </>
        )}
        {medianLabelY !== null && <text x={width - m.r + 8 * k} y={clamp(medianLabelY + 4 * k, m.t + 8 * k, H - m.b)} className="axis-title">median</text>}
        <rect
          x={m.l} y={m.t} width={Math.max(0, width - m.l - m.r)} height={H - m.t - m.b}
          fill="transparent" style={{ cursor: "pointer" }}
          onMouseMove={(e) => setHover(indexAt(e, days))}
          onClick={(e) => {
            const i = indexAt(e, days);
            if (i < reached) onDate(dates[i] === date ? "all" : dates[i]);
          }}
        />
      </svg>
      {hv && (
        <div className={`tip ${hover < days / 2 ? "tip-right tip-low" : "tip-left"}`}>
          <strong>Through {formatDate(hv.date, "long")}</strong>
          <span><i style={{ background: tone }} /> {year} <b>{hv.value === null ? "not archived" : integer(hv.value)}</b></span>
          {hv.mid !== null && <span><i className="dash" /> Median season <b>{integer(hv.mid)}</b></span>}
          {hv.value !== null && hv.mid > 0 && <span className="tip-foot">{times(hv.value / hv.mid)} the median to date</span>}
        </div>
      )}
    </div>
  );
}

/** MODIS and VIIRS day by day, in occupied cells or raw detections. */
export function DailyChart({ daily, date, onDate, viirsThrough, height = 190 }) {
  const [ref, width] = useWidth(420);
  const { k } = useRem();
  const [hover, setHover] = useState(null);
  const H = height * k, m = { t: 12 * k, r: 10 * k, b: 28 * k, l: 40 * k };
  const n = daily.length;
  const covered = (d) => viirsThrough !== null && (viirsThrough === undefined || d.date <= viirsThrough);
  const peak = Math.max(1, ...daily.map((d) => Math.max(d.modis, covered(d) ? d.viirs : 0)));
  const ticks = niceTicks(peak, 4), yMax = ticks[ticks.length - 1];
  const x = linear(0, Math.max(1, n - 1), m.l, width - m.r), y = linear(0, yMax, H - m.b, m.t);
  const viirs = daily.map((d, i) => [x(i), covered(d) ? y(d.viirs) : null]);
  const modis = daily.map((d, i) => [x(i), y(d.modis)]);
  const selected = date === "all" ? -1 : daily.findIndex((d) => d.date === date);
  const h = hover === null ? null : daily[hover];
  return (
    <div ref={ref} className="chart" onMouseLeave={() => setHover(null)}>
      <svg width={width} height={H} role="img" aria-label="Daily activity seen by MODIS and VIIRS">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} stroke={C.grid} />
            <text x={m.l - 8 * k} y={y(t) + 3.5 * k} textAnchor="end" className="tick">{compact(t)}</text>
          </g>
        ))}
        <MonthGrid dates={daily.map((d) => d.date)} x={x} top={m.t} bottom={H - m.b} k={k} />
        <path d={areaPath(viirs, y(0))} fill={C.viirs} fillOpacity="0.12" />
        <path d={linePath(viirs)} fill="none" stroke={C.viirs} strokeWidth="1.5" strokeLinejoin="round" />
        <path d={linePath(modis)} fill="none" stroke={C.modis} strokeWidth="1.5" strokeLinejoin="round" />
        {selected >= 0 && <line x1={x(selected)} x2={x(selected)} y1={m.t} y2={H - m.b} stroke={C.text} strokeOpacity="0.55" strokeDasharray="3 3" />}
        {h && (
          <>
            <line x1={x(hover)} x2={x(hover)} y1={m.t} y2={H - m.b} stroke={C.text} strokeOpacity="0.3" />
            {covered(h) && <circle cx={x(hover)} cy={y(h.viirs)} r={3.5 * k} fill={C.viirs} stroke={C.surface} strokeWidth="1.5" />}
            <circle cx={x(hover)} cy={y(h.modis)} r={3.5 * k} fill={C.modis} stroke={C.surface} strokeWidth="1.5" />
          </>
        )}
        <rect
          x={m.l} y={m.t} width={Math.max(0, width - m.l - m.r)} height={H - m.t - m.b}
          fill="transparent" style={{ cursor: "pointer" }}
          onMouseMove={(e) => setHover(indexAt(e, n))}
          onClick={(e) => {
            const d = daily[indexAt(e, n)];
            if (d) onDate(d.date === date ? "all" : d.date);
          }}
        />
      </svg>
      {h && (
        <div className={`tip ${hover < n / 2 ? "tip-right" : "tip-left"}`}>
          <strong>{formatDate(h.date, "long")}</strong>
          <span><i style={{ background: C.modis }} /> MODIS <b>{integer(h.modis)}</b></span>
          <span><i style={{ background: C.viirs }} /> VIIRS <b>{covered(h) ? integer(h.viirs) : "not archived"}</b></span>
        </div>
      )}
    </div>
  );
}

/** The season as calendar months, each day shaded by how much was burning. */
export function Calendar({ daily, date, onDate, peak }) {
  const max = Math.max(1, peak || 0, ...daily.map((d) => d.occupied));
  const months = [...new Set(daily.map((d) => d.date.slice(0, 7)))];
  return (
    <div className="months">
      {months.map((month) => {
        const days = daily.filter((d) => d.date.startsWith(month));
        const lead = (new Date(`${days[0].date}T12:00:00Z`).getUTCDay() + 6) % 7;
        return (
          <div key={month}>
            <span className="month-label">{formatDate(`${month}-01`, "month")}</span>
            <div className="days">
              {Array.from({ length: lead }, (_, i) => <span key={`b${i}`} />)}
              {days.map((d) => {
                const t = Math.sqrt(d.occupied / max), fill = fireRamp(t);
                return (
                  <button
                    key={d.date}
                    aria-pressed={date === d.date}
                    aria-label={`${formatDate(d.date, "long")}: ${d.occupied} active cells`}
                    className={date === d.date ? "on" : undefined}
                    title={`${formatDate(d.date)} · ${integer(d.occupied)} active cells`}
                    style={{ background: fill, color: inkOn(fill) }}
                    onClick={() => onDate(date === d.date ? "all" : d.date)}
                  >
                    {Number(d.day)}
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
