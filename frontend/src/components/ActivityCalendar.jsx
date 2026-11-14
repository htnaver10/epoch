import { useEffect, useMemo, useState } from "react";
import Icon from "./Icon";
import { fireRamp, verdictColor } from "../lib/theme";
import { seasonDate, median } from "../lib/seasons";
import { formatDate, integer } from "../lib/chart";
import { num, pad2, criticalPeriods } from "../lib/figures";

// Dark for a quiet week, bright for the busiest week in this area's record. The exponent
// spreads the many ordinary weeks apart instead of crowding them into one orange.
const heat = (value, max) => fireRamp(0.1 + 0.9 * Math.min(1, value / (max || 1)) ** 0.6);

/**
 * The burning-activity calendar for the area in scope: one row per season, one column per
 * week, on the harmonized (MODIS-scale) record, with the usual week under it. It is also the
 * ranked season table: every row carries that season's index and rank for the period in
 * scope, and selects the season. Dragging across the week headings sets the period.
 */
export default function ActivityCalendar({ seasons, analysis, year, onYear, period, onPeriod, areaName, onExport, busy, docked = false }) {
  const [order, setOrder] = useState("year");
  const [drag, setDrag] = useState(null); // week span being dragged: { a, b }
  const [hover, setHover] = useState(null);
  const { calendar, verdicts } = analysis;
  const days = seasons.season_days;
  const weeks = calendar[0]?.cells.length || 0;
  const max = Math.max(1, ...calendar.flatMap((r) => r.cells.filter((c) => c !== null)));
  const usual = useMemo(
    () => Array.from({ length: weeks }, (_, w) => median(calendar.map((r) => r.cells[w]).filter((c) => c !== null))),
    [calendar, weeks],
  );
  // the weeks that usually burn most here: the critical period, ticked under the headings
  const peakWeeks = useMemo(() => new Set(criticalPeriods(seasons, calendar, year).peak?.weeks || []), [seasons, calendar, year]);
  const rows = useMemo(() => {
    const list = calendar.map((r) => ({ ...r, v: verdicts.get(r.year) }));
    const known = (r) => r.v && r.v.kind !== "insufficient";
    return order === "rank"
      ? list.sort((a, b) => Number(known(b)) - Number(known(a)) || (known(a) ? a.v.rank - b.v.rank : 0) || b.year - a.year)
      : list.sort((a, b) => b.year - a.year);
  }, [calendar, verdicts, order]);

  // label a week when it is the first to start in a new month
  const labels = Array.from({ length: weeks }, (_, w) => {
    const d = seasonDate(seasons, 2001, w * 7), prev = w ? seasonDate(seasons, 2001, (w - 1) * 7) : null;
    return !prev || prev.slice(5, 7) !== d.slice(5, 7) ? formatDate(d, "monthShort") : "";
  });
  const weekStart = (y, w) => formatDate(seasonDate(seasons, y, w * 7));
  // a thin rule where a new month begins, so a week can be read as early or late in its month
  const ms = (w) => (w > 0 && labels[w] ? " ms" : "");
  // the weeks the period in scope touches; a drag in progress shows what it would select
  const span = drag ? { from: Math.min(drag.a, drag.b) * 7, to: Math.min(days, (Math.max(drag.a, drag.b) + 1) * 7) } : period;
  const inSpan = (w) => !span || (w * 7 < span.to && Math.min(days, w * 7 + 7) > span.from);

  useEffect(() => {
    if (!drag) return undefined;
    const commit = () => {
      const a = Math.min(drag.a, drag.b), b = Math.max(drag.a, drag.b);
      // the whole season is no period at all
      onPeriod(a === 0 && b === weeks - 1 ? null : { from: a * 7, to: Math.min(days, (b + 1) * 7) });
      setDrag(null);
    };
    window.addEventListener("pointerup", commit);
    return () => window.removeEventListener("pointerup", commit);
  }, [drag, weeks, days, onPeriod]);

  return (
    <section className={`calendar${docked ? " docked" : ""}`} aria-label={`Burning-activity calendar, ${areaName}`}>
      <header>
        <h2 title={`Calendar · ${areaName}`}>
          {/* a target's name is long enough on its own; a sector keeps the word */}
          {/ km$/.test(areaName) ? areaName.replace(/ · (\d+) km$/, " $1 km") : `Calendar · ${areaName}`}
        </h2>
        <span className="seg" role="group" aria-label="Order of seasons">
          <button className={order === "year" ? "on" : undefined} aria-pressed={order === "year"} onClick={() => setOrder("year")}>Year</button>
          <button className={order === "rank" ? "on" : undefined} aria-pressed={order === "rank"} onClick={() => setOrder("rank")}>Rank</button>
        </span>
      </header>
      <div className={`cal${span ? " has-span" : ""}`} style={{ "--weeks": weeks }} onMouseLeave={() => setHover(null)}>
        <div className="cal-row cal-head">
          <span>Week</span>
          {labels.map((label, w) => (
            <button
              key={w}
              tabIndex={-1}
              className={[inSpan(w) && "in", peakWeeks.has(w) && "peak-week", ms(w).trim()].filter(Boolean).join(" ") || undefined}
              aria-label={`Week of ${weekStart(2001, w)}`}
              title={`Week of ${weekStart(2001, w)}${peakWeeks.has(w) ? " (usual peak)" : ""}: click, or drag across weeks, to set the period`}
              onPointerDown={(e) => { e.preventDefault(); setDrag({ a: w, b: w }); }}
              onPointerEnter={() => drag && setDrag({ a: drag.a, b: w })}
            >
              {label}
            </button>
          ))}
          <b>Idx</b>
          <b>Rnk</b>
        </div>
        <ol>
          {rows.map((row) => {
            const known = row.v && row.v.kind !== "insufficient";
            // only the unusual seasons carry a verdict colour, so they stand out of the column
            const tone = !known || row.v.kind === "typical" ? "var(--soft)" : verdictColor(row.v.kind);
            return (
              <li key={row.year}>
                <button
                  className={`cal-row${row.year === year ? " on" : ""}`}
                  aria-pressed={row.year === year}
                  aria-label={known ? `${row.year}: ${num(row.v.likeForLike)} times the usual, rank ${row.v.rank} of ${row.v.of}` : `${row.year}: not enough archived to rank`}
                  style={{ "--tone": tone }}
                  onClick={() => onYear(row.year)}
                >
                  <span>{row.year}</span>
                  {row.cells.map((c, w) => (
                    <i
                      key={w}
                      className={`${c === null ? "missing" : row.basis === "viirs" ? "translated" : ""}${inSpan(w) ? "" : " out"}${ms(w)}`}
                      style={c ? { background: heat(c, max) } : undefined}
                      onMouseEnter={() => setHover({ year: row.year, w, c })}
                    />
                  ))}
                  <b className="idx">{known ? num(row.v.likeForLike) : "–"}</b>
                  <b className="rnk">{known ? pad2(row.v.rank) : "–"}</b>
                </button>
              </li>
            );
          })}
        </ol>
        <div className="cal-row cal-usual">
          <span>Usual</span>
          {usual.map((c, w) => (
            <i key={w} className={`${inSpan(w) ? "" : "out"}${ms(w)}`.trim() || undefined} style={c ? { background: heat(c, max) } : undefined} onMouseEnter={() => setHover({ year: null, w, c })} />
          ))}
          <b className="idx">1.0</b>
          <b className="rnk" />
        </div>
      </div>
      <footer>
        <p className="cal-note" aria-live="off">
          {busy ||
            (hover
              ? `${hover.year ?? "Usual"}, week of ${weekStart(hover.year ?? 2001, hover.w)}: ${hover.c === null ? "not archived" : `${integer(hover.c)} cell-days${hover.year && usual[hover.w] ? `, ${num(hover.c / usual[hover.w])}× usual` : ""}`}`
              : "Brighter = more fire. Amber = usual peak.")}
        </p>
        <button onClick={onExport} title="Download this calendar as CSV: every season and week, with index and rank" aria-label="Export the calendar as CSV">
          <Icon name="download" size="1rem" />
          CSV
        </button>
      </footer>
    </section>
  );
}
