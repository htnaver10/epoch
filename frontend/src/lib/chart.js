// Small pure helpers behind the hand-drawn SVG charts: scales, ticks, paths, number formats.

export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/** Linear scale from a domain to a pixel range. */
export const linear = (d0, d1, r0, r1) => {
  const span = d1 - d0 || 1;
  return (v) => r0 + ((v - d0) / span) * (r1 - r0);
};

/** Log scale; values at or below zero are pinned to the low end of the range. */
export const log = (d0, d1, r0, r1) => {
  const l0 = Math.log(d0), span = Math.log(d1) - l0 || 1;
  return (v) => (v > 0 ? r0 + ((Math.log(v) - l0) / span) * (r1 - r0) : r0);
};

/** Round, evenly spaced axis ticks from zero up to (at least) `max`: 1, 2 or 5 x 10^n apart. */
export function niceTicks(max, count = 4) {
  if (!(max > 0)) return [0, 1];
  const raw = max / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].find((m) => m * mag >= raw) * mag;
  const ticks = [];
  for (let i = 0; i * step < max + step - 1e-9 * step; i++) ticks.push(Number((i * step).toPrecision(12)));
  return ticks;
}

/** SVG path through [x, y] points; a null y breaks the line instead of drawing through a gap. */
export function linePath(points) {
  let d = "", pen = false;
  for (const [x, y] of points) {
    if (y === null || y === undefined || Number.isNaN(y)) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`;
    pen = true;
  }
  return d;
}

/** Closed area between a line and a flat baseline, one sub-path per unbroken run. */
export function areaPath(points, baseline) {
  let d = "", run = [];
  const flush = () => {
    if (run.length > 1)
      d += `M${run[0][0].toFixed(1)} ${baseline.toFixed(1)}${run.map(([x, y]) => `L${x.toFixed(1)} ${y.toFixed(1)}`).join("")}L${run[run.length - 1][0].toFixed(1)} ${baseline.toFixed(1)}Z`;
    run = [];
  };
  for (const p of points) {
    if (p[1] === null || p[1] === undefined || Number.isNaN(p[1])) flush();
    else run.push(p);
  }
  flush();
  return d;
}

/** Closed band between an upper and a lower line sharing the same x positions. */
export function bandPath(xs, lower, upper) {
  if (!xs.length) return "";
  const up = xs.map((x, i) => `${i ? "L" : "M"}${x.toFixed(1)} ${upper[i].toFixed(1)}`).join("");
  const down = xs.map((x, i) => `L${x.toFixed(1)} ${lower[i].toFixed(1)}`).reverse().join("");
  return `${up}${down}Z`;
}

/** 12,345 -> "12.3k", 1,234,567 -> "1.2M"; small values keep their digits. */
export function compact(v) {
  if (v === null || v === undefined) return "–";
  const a = Math.abs(v);
  if (a >= 1e6) return `${(v / 1e6).toFixed(a >= 1e7 ? 0 : 1)}M`;
  if (a >= 1e4) return `${Math.round(v / 1e3)}k`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(1).replace(/\.0$/, "")}k`;
  return `${Math.round(v)}`;
}

export const integer = (v) => (v === null || v === undefined ? "–" : Math.round(v).toLocaleString("en-CA"));

/** A multiple of the median: "4.0×", "13×", "0.18×". */
export function times(v) {
  if (v === null || v === undefined) return "–";
  return `${v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2)}×`;
}

/** 0.96 -> "96th". */
export function ordinal(n) {
  const v = Math.round(n);
  const s = v % 10 === 1 && v % 100 !== 11 ? "st" : v % 10 === 2 && v % 100 !== 12 ? "nd" : v % 10 === 3 && v % 100 !== 13 ? "rd" : "th";
  return `${v}${s}`;
}

/** Signed percentage: 0.54 -> "+54%". */
export const signedPercent = (v) => (v === null || v === undefined ? "–" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(Math.round(v * 100))}%`);

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const LONG_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

/** "2023-07-04" -> "Jul 4" (or "Jul 4, 2023", or "July 2023"). Pure string work: no timezone drift. */
export function formatDate(iso, style = "short") {
  const [y, m, d] = iso.split("-").map(Number);
  if (style === "month") return `${LONG_MONTHS[m - 1]} ${y}`;
  if (style === "monthShort") return MONTHS[m - 1];
  return style === "long" ? `${MONTHS[m - 1]} ${d}, ${y}` : `${MONTHS[m - 1]} ${d}`;
}
