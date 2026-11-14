import { useState } from "react";
import Icon from "./Icon";
import { fireRamp, verdictColor } from "../lib/theme";
import { times, formatDate } from "../lib/chart";
import { VERDICTS } from "../lib/seasons";

const barColor = (kind) => (kind === "high" || kind === "low" || kind === "mixed" ? verdictColor(kind) : "#8fa2b6");

/** Every season as a bar: the whole record, and the year selector. */
function SeasonStrip({ analysis, year, onYear }) {
  const { series, verdicts, years } = analysis;
  const [hover, setHover] = useState(null);
  const max = Math.max(1, ...series.map((t) => t.equivalent ?? 0));
  const i = years.indexOf(year);
  const shown = hover ?? year;
  const v = verdicts.get(shown);
  const known = v && v.kind !== "insufficient";
  return (
    <div className="dock-block">
      <div className="dock-head">
        <span className="label">Season</span>
        <button className="step" onClick={() => onYear(years[i - 1])} disabled={i <= 0} aria-label="Previous season" title="Previous season ( [ )"><Icon name="left" size={14} /></button>
        <strong>{year}</strong>
        <button className="step" onClick={() => onYear(years[i + 1])} disabled={i < 0 || i >= years.length - 1} aria-label="Next season" title="Next season ( ] )"><Icon name="right" size={14} /></button>
        <span className="sub" aria-live="off">
          {hover !== null && hover !== year ? `${hover}: ` : ""}
          {known ? `${times(v.likeForLike)} the median, ${VERDICTS[v.kind].toLowerCase()}` : "not enough data"}
        </span>
      </div>
      <div className="strip" role="group" aria-label="Seasons by fire activity" onMouseLeave={() => setHover(null)}>
        {series.map((t) => {
          const sv = verdicts.get(t.year);
          return (
            <button
              key={t.year}
              className={t.year === year ? "on" : undefined}
              aria-pressed={t.year === year}
              aria-label={`${t.year}: ${sv && sv.kind !== "insufficient" ? `${times(sv.likeForLike)} the median` : "not enough data"}`}
              onMouseEnter={() => setHover(t.year)}
              onFocus={() => setHover(t.year)}
              onBlur={() => setHover(null)}
              onClick={() => onYear(t.year)}
              style={{ "--h": `${Math.max(4, ((t.equivalent ?? 0) / max) * 100)}%`, "--c": barColor(sv?.kind) }}
            >
              <i />
            </button>
          );
        })}
      </div>
      <div className="strip-years" aria-hidden="true">
        {series.map((t) => (
          <span key={t.year} className={t.year === year ? "on" : undefined}>{t.year === year || t.year % 5 === 0 ? (t.year === year ? t.year : `’${String(t.year).slice(2)}`) : ""}</span>
        ))}
      </div>
    </div>
  );
}

/**
 * Day scrubber: play, date readout, an activity strip to drag across. The strip is scaled
 * to the busiest day in the whole record, so a quiet season looks quiet.
 */
function DayScrubber({ dates, daily, date, playing, onDate, onPlay, peak }) {
  const n = dates.length;
  const i = date === "all" ? -1 : dates.indexOf(date);
  const max = Math.max(1, peak || 0, ...daily.map((d) => d.occupied));
  const months = dates.map((d, k) => [d, k]).filter(([d, k]) => k === 0 || d.endsWith("-01"));
  return (
    <div className="dock-block">
      <div className="dock-head">
        <span className="label">Day</span>
        <button className="play" onClick={onPlay} disabled={!n} aria-label={playing ? "Pause" : "Play through the season"} title="Play / pause (space)">
          <Icon name={playing ? "pause" : "play"} size={14} />
        </button>
        <strong aria-live="polite">{i < 0 ? "All days" : formatDate(date)}</strong>
        <span className="sub">{!n ? "loading" : i < 0 ? `${n} days, Jun 1 – Aug 31` : `day ${i + 1} of ${n}`}</span>
        {i >= 0 && <button className="text" onClick={() => onDate("all")} title="Show the whole season (Esc)">All days</button>}
      </div>
      <div className={`scrubber${i < 0 ? " is-all" : ""}`}>
        <div className="activity" aria-hidden="true">
          {daily.map((d, k) => {
            const t = Math.sqrt(d.occupied / max);
            return <i key={d.date} className={i >= 0 && k > i ? "ahead" : undefined} style={{ height: `${Math.max(3, t * 100)}%`, background: fireRamp(0.2 + t * 0.8) }} />;
          })}
        </div>
        <input
          type="range"
          min="0"
          max={Math.max(0, n - 1)}
          value={Math.max(0, i)}
          disabled={!n}
          aria-label="Day of season"
          aria-valuetext={i < 0 ? "All days" : formatDate(date, "long")}
          title="Drag to pick a day · ← → step · Esc for all days"
          onChange={(e) => onDate(dates[Number(e.target.value)])}
          onClick={(e) => i < 0 && onDate(dates[Number(e.currentTarget.value)])}
        />
      </div>
      <div className="scrub-months" aria-hidden="true">
        {months.map(([d, k]) => (
          <span key={d} style={{ left: `${(k / Math.max(1, n)) * 100}%` }}>{formatDate(d, "monthShort")}</span>
        ))}
      </div>
    </div>
  );
}

export default function Dock({ analysis, year, onYear, dates, daily, date, playing, onDate, onPlay, peak }) {
  return (
    <footer className="dock">
      {analysis && year ? <SeasonStrip analysis={analysis} year={year} onYear={onYear} /> : <div className="dock-block" />}
      <DayScrubber dates={dates} daily={daily} date={date} playing={playing} onDate={onDate} onPlay={onPlay} peak={peak} />
    </footer>
  );
}
