// Multi-year analysis: like-for-like season ranking, a held-out-validated VIIRS -> MODIS
// bridge, a trend test and a bounded verdict. Every number is computed from the archive.

const sum = (a, n = a.length, from = 0) => {
  let s = 0;
  for (let i = Math.max(0, from); i < Math.min(n, a.length); i++) s += a[i];
  return s;
};
const sorted = (values) => [...values].sort((a, b) => a - b);

export function median(values) {
  if (!values.length) return null;
  const s = sorted(values), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function quantile(values, q) {
  if (!values.length) return null;
  const s = sorted(values), pos = (s.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

// Equal up to rounding: a bridged value must tie with the measured value it reproduces.
const same = (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(Math.abs(a), Math.abs(b));

/** Share of `others` below `value`, ties counted half (mid-rank percentile). */
export function percentileRank(value, others) {
  if (!others.length) return null;
  let below = 0;
  for (const o of others) below += same(o, value) ? 0.5 : o < value ? 1 : 0;
  return below / others.length;
}

/**
 * Robust z-score in log space: how many (MAD-scaled) deviations a season sits from the
 * median of the others. Season totals are heavy-tailed, so this is measured on ln(total).
 */
export function robustLogZ(value, others) {
  const logs = others.filter((o) => o > 0).map(Math.log);
  if (!(value > 0) || logs.length < 3) return null;
  const centre = median(logs);
  const spread = 1.4826 * median(logs.map((l) => Math.abs(l - centre)));
  return spread > 0 ? (Math.log(value) - centre) / spread : null;
}

// Daily activity for one sensor: area-weighted cell-days when the archive carries them
// (a 0.25 degree cell at 70N covers half the ground of one at 50N), plain cell-days otherwise.
const activity = (x) => x.w || x.cells;

/**
 * Totals per year for one region over the days [from, window) of the season, so a partial
 * season, or a chosen period, is compared against the same calendar span of every other year.
 */
export function yearTotals(seasons, region, window = seasons.season_days, from = 0) {
  return Object.entries(seasons.series[region] || {})
    .map(([year, s]) => {
      const row = { year: Number(year) };
      for (const k of ["modis", "viirs"]) {
        const x = s[k];
        const ok = x && x.days >= window;
        row[k] = ok ? sum(activity(x), window, from) : null;
        row[`${k}Cells`] = ok ? sum(x.cells, window, from) : null;
        row[`${k}Raw`] = ok ? sum(x.raw, window, from) : null;
        row[`${k}Days`] = x ? x.days : 0;
      }
      row.shared = s.shared && row.modis !== null && row.viirs !== null ? sum(s.shared, window, from) : null;
      return row;
    })
    .sort((a, b) => a.year - b.year);
}

const Z80 = 1.2816; // two-sided 80% interval under a normal error in log space

/**
 * VIIRS -> MODIS bridge fitted on the seasons both sensors flew, optionally leaving one out.
 *
 * VIIRS's advantage is not a constant factor: it is largest in quiet seasons (many small
 * fires MODIS misses) and shrinks in extreme ones. So the bridge is a power law,
 * ln(MODIS) = a + b ln(VIIRS), fitted by least squares. With too few seasons, or no spread
 * in activity, it falls back to the constant median ratio.
 */
export function fitBridge(totals, exclude = null) {
  const pts = totals.filter((t) => t.year !== exclude && t.modis > 0 && t.viirs > 0);
  if (pts.length < 3) return null;
  const ratios = pts.map((t) => t.viirs / t.modis);
  const ratio = median(ratios);
  const xs = pts.map((t) => Math.log(t.viirs)), ys = pts.map((t) => Math.log(t.modis));
  const n = pts.length, mx = sum(xs) / n, my = sum(ys) / n;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) ** 2;
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const power = n >= 5 && sxx > 1e-9;
  const b = power ? sxy / sxx : 1;
  const a = power ? my - b * mx : -Math.log(ratio);
  let sse = 0;
  for (let i = 0; i < n; i++) sse += (ys[i] - (a + b * xs[i])) ** 2;
  const s = Math.sqrt(sse / Math.max(1, n - (power ? 2 : 1)));
  const toModis = power ? (v) => Math.exp(a + b * Math.log(v)) : (v) => v / ratio;
  return {
    model: power ? "power" : "ratio",
    a, b, s, ratio,
    low: quantile(ratios, 0.25),
    high: quantile(ratios, 0.75),
    years: n,
    toModis,
    interval: (v) => [toModis(v) * Math.exp(-Z80 * s), toModis(v) * Math.exp(Z80 * s)],
  };
}

function ratioOnly(totals, exclude) {
  const ratios = totals
    .filter((t) => t.year !== exclude && t.modis > 0 && t.viirs > 0)
    .map((t) => t.viirs / t.modis);
  return ratios.length >= 3 ? median(ratios) : null;
}

/**
 * Leave-one-season-out test: predict each overlap season's MODIS total from its VIIRS total
 * using a bridge fitted on the other seasons. Also scores the simpler constant-ratio bridge
 * so the gain from the power law is visible rather than assumed.
 */
export function validateBridge(totals) {
  const results = [], naive = [];
  for (const t of totals) {
    if (!(t.modis > 0) || !(t.viirs > 0)) continue;
    const b = fitBridge(totals, t.year);
    if (!b) continue;
    const predicted = b.toModis(t.viirs);
    results.push({ year: t.year, actual: t.modis, predicted, error: (predicted - t.modis) / t.modis });
    const r = ratioOnly(totals, t.year);
    if (r) naive.push(Math.abs(t.viirs / r - t.modis) / t.modis);
  }
  const abs = results.map((r) => Math.abs(r.error));
  return {
    results,
    mdape: abs.length ? median(abs) : null,
    worst: abs.length ? Math.max(...abs) : null,
    ratioMdape: naive.length ? median(naive) : null,
    ratioWorst: naive.length ? Math.max(...naive) : null,
  };
}

/** MODIS-scale activity for every year: MODIS where flown, translated VIIRS alongside. */
export function harmonizedSeries(totals) {
  return totals.map((t) => {
    const b = t.viirs > 0 ? fitBridge(totals, t.year) : null;
    const fromViirs = b ? b.toModis(t.viirs) : null;
    const [lo, hi] = b ? b.interval(t.viirs) : [null, null];
    return {
      ...t,
      fromViirs,
      fromViirsLow: lo,
      fromViirsHigh: hi,
      equivalent: t.modis ?? fromViirs,
      basis: t.modis !== null ? "modis" : fromViirs !== null ? "viirs" : null,
    };
  });
}

export const VERDICTS = {
  high: "Unusually high",
  low: "Unusually quiet",
  typical: "Within the normal range",
  mixed: "Sensors disagree",
  insufficient: "Not enough history",
};

// A sensor that puts the season on the other side of its own median overrules a strong call.
const settle = (kind, crossPct) =>
  crossPct !== null && ((kind === "high" && crossPct < 0.5) || (kind === "low" && crossPct > 0.5)) ? "mixed" : kind;

function crossCheck(series, year) {
  const target = series.find((t) => t.year === year);
  if (!target || target.modis === null || target.viirs === null) return null;
  const others = series.filter((t) => t.year !== year && t.viirs !== null).map((t) => t.viirs);
  return others.length >= 5 ? percentileRank(target.viirs, others) : null;
}

/**
 * Bounded verdict for one year. Ranks the target against every *other* year on the same
 * sensor scale, cross-checks with VIIRS's own record, and reports the naive comparison a
 * reader would make by putting today's VIIRS detection count next to the pre-VIIRS record.
 */
export function verdict(totals, year, { minYears = 8, highPct = 0.9, lowPct = 0.1 } = {}) {
  const series = harmonizedSeries(totals);
  const target = series.find((t) => t.year === year);
  if (!target) return { kind: "insufficient", reason: "No observations for this season." };
  const baseline = series.filter((t) => t.year !== year && t.modis !== null).map((t) => t.modis);
  const value = target.equivalent;
  if (value === null || baseline.length < minYears)
    return {
      kind: "insufficient",
      reason: `Needs at least ${minYears} other MODIS seasons; has ${baseline.length}.`,
      target,
    };
  const pct = percentileRank(value, baseline);
  const base = median(baseline);
  // an area that has never burned in the record cannot be ranked against itself
  if (!(value > 0) && !baseline.some((b) => b > 0))
    return { kind: "insufficient", reason: "No fire activity recorded here in any season.", target };
  const likeForLike = base ? value / base : null;
  const crossPct = crossCheck(series, year);

  // Naive: this year's best-available raw count against the pre-VIIRS MODIS raw median.
  const preViirs = totals.filter((t) => t.viirsDays === 0 && t.modisRaw !== null).map((t) => t.modisRaw);
  // Only meaningful for a season VIIRS observed; earlier seasons have nothing to splice.
  const naive =
    preViirs.length && median(preViirs) && target.viirsRaw !== null ? target.viirsRaw / median(preViirs) : null;

  const strong = pct >= highPct ? "high" : pct <= lowPct ? "low" : "typical";
  const rank = 1 + baseline.filter((b) => b > value && !same(b, value)).length;
  const of = baseline.length + 1;
  return {
    kind: settle(strong, crossPct),
    strong,
    pct, rank, of, base, likeForLike, crossPct, naive,
    z: robustLogZ(value, baseline),
    record: rank === 1 ? "high" : rank === of ? "low" : null,
    third: pct >= 2 / 3 ? "upper" : pct <= 1 / 3 ? "lower" : "middle",
    // every season on this window and scale, and where the unusual tails begin
    field: series.filter((t) => t.equivalent !== null).map((t) => ({ year: t.year, value: t.equivalent })),
    lowCut: quantile(baseline, lowPct),
    highCut: quantile(baseline, highPct),
    basis: target.basis,
    target,
    detectionInflation: naive && likeForLike ? naive / likeForLike : null,
  };
}

const MIN_WINDOW = 14; // days of a season that must be archived before it is judged
const MIN_PERIOD = 7; // a chosen period may be as short as a week

/**
 * Verdict for a season as archived, over the whole season or over `period`, a span of its
 * days { from, to }. A season still in progress (or a sensor whose archive stops early) is
 * compared over the same calendar days of every other year, never against a longer span.
 */
export function assessSeason(seasons, region, year, period = null) {
  const full = seasons.season_days;
  const from = period ? Math.max(0, period.from) : 0, end = period ? Math.min(full, period.to) : full;
  const need = period ? MIN_PERIOD : MIN_WINDOW;
  const s = seasons.series[region]?.[year];
  if (!s) return { kind: "insufficient", reason: "No observations for this season.", window: end, from, partial: false };
  const mDays = s.modis?.days ?? 0, vDays = s.viirs?.days ?? 0;
  const window = Math.min(end, mDays || vDays || end);
  if (window - from < need)
    return { kind: "insufficient", reason: window <= from ? "This period is not archived yet." : `Only ${window - from} days archived so far.`, window, from, partial: true };
  const v = verdict(yearTotals(seasons, region, window, from), year);
  v.window = window;
  v.from = from;
  v.partial = window < end;
  v.crossWindow = v.crossPct === null ? null : window;
  v.naiveWindow = v.naive === null ? null : window;
  v.naiveLike = v.likeForLike;
  // VIIRS's archive can stop before MODIS's: judge the VIIRS-dependent figures on its window.
  if (v.kind !== "insufficient" && v.crossPct === null && vDays - from >= need && vDays < window) {
    const early = verdict(yearTotals(seasons, region, vDays, from), year);
    if (early.kind !== "insufficient") {
      if (early.crossPct !== null) {
        v.crossPct = early.crossPct;
        v.crossWindow = vDays;
        v.kind = settle(v.strong, early.crossPct);
      }
      if (early.naive !== null) {
        v.naive = early.naive;
        v.naiveLike = early.likeForLike;
        v.naiveWindow = vDays;
      }
    }
  }
  v.detectionInflation = v.naive && v.naiveLike ? v.naive / v.naiveLike : null;
  return v;
}

// ---- Trend -------------------------------------------------------------------------------

/** Standard normal upper-tail probability (Abramowitz & Stegun 26.2.17). */
function upperTail(z) {
  const t = 1 / (1 + 0.2316419 * z);
  const poly = t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return (Math.exp((-z * z) / 2) / Math.sqrt(2 * Math.PI)) * poly;
}

/** Mann-Kendall monotonic trend test (normal approximation, tie-corrected). */
export function mannKendall(values) {
  const n = values.length;
  if (n < 4) return null;
  let s = 0;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) s += Math.sign(values[j] - values[i]);
  const ties = new Map();
  for (const v of values) ties.set(v, (ties.get(v) || 0) + 1);
  let variance = n * (n - 1) * (2 * n + 5);
  for (const t of ties.values()) if (t > 1) variance -= t * (t - 1) * (2 * t + 5);
  variance /= 18;
  const z = variance > 0 ? (s - Math.sign(s)) / Math.sqrt(variance) : 0;
  return { s, z, p: z === 0 ? 1 : Math.min(1, 2 * upperTail(Math.abs(z))) };
}

/** Theil-Sen estimator: the median of all pairwise slopes, robust to a 2023-sized outlier. */
export function theilSen(xs, ys) {
  const slopes = [];
  for (let i = 0; i < xs.length; i++)
    for (let j = i + 1; j < xs.length; j++) if (xs[j] !== xs[i]) slopes.push((ys[j] - ys[i]) / (xs[j] - xs[i]));
  if (!slopes.length) return null;
  const slope = median(slopes);
  return { slope, intercept: median(ys.map((y, i) => y - slope * xs[i])) };
}

/**
 * Long-run change in a yearly quantity, measured on ln(value) so it reads as a percentage.
 * `pick` chooses the quantity; the default is the consistent MODIS record.
 */
export function trend(totals, pick = (t) => t.modis) {
  const pts = totals.map((t) => [t.year, pick(t)]).filter(([, v]) => v > 0);
  if (pts.length < 8) return null;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => Math.log(p[1]));
  const line = theilSen(xs, ys), test = mannKendall(ys);
  if (!line || !test) return null;
  return {
    perDecade: Math.exp(line.slope * 10) - 1,
    p: test.p,
    years: pts.length,
    first: xs[0],
    last: xs[xs.length - 1],
    at: (year) => Math.exp(line.intercept + line.slope * year),
  };
}

/** The trend a reader gets by splicing raw VIIRS counts onto the pre-VIIRS MODIS record. */
export const splicedTrend = (totals) => trend(totals, (t) => t.viirsRaw ?? t.modisRaw);

// ---- Within-season shape -----------------------------------------------------------------

/** ISO date of day `i` (0-based) of a season. */
export function seasonDate(seasons, year, i) {
  const [m, d] = seasons.season[0].split("-").map(Number);
  return new Date(Date.UTC(year, m - 1, d + i)).toISOString().slice(0, 10);
}

/**
 * Cumulative MODIS activity through the season for every year, plus the day-by-day
 * 10th / 50th / 90th percentile envelope of the complete seasons other than `year`.
 */
export function seasonCurves(seasons, region, year) {
  const days = seasons.season_days;
  const curves = Object.entries(seasons.series[region] || {})
    .filter(([, s]) => s.modis && s.modis.days > 0)
    .map(([y, s]) => {
      const src = activity(s.modis), n = Math.min(days, s.modis.days), values = new Array(n);
      let run = 0;
      for (let i = 0; i < n; i++) values[i] = run += src[i];
      return { year: Number(y), values, complete: n === days };
    })
    .sort((a, b) => a.year - b.year);
  const reference = curves.filter((c) => c.complete && c.year !== year);
  const band = Array.from({ length: days }, (_, i) => {
    const at = reference.map((c) => c.values[i]);
    return { lo: quantile(at, 0.1), mid: quantile(at, 0.5), hi: quantile(at, 0.9) };
  });
  return { days, curves, band, target: curves.find((c) => c.year === year) || null, reference: reference.length };
}

/**
 * The day (0-based) a season's running total first passed what a median season reaches by
 * its end, or null if it never did. Turns "4x the median" into a date.
 */
export function medianCrossing(curves) {
  const { target, band, days, reference } = curves;
  if (!target || reference < 3) return null;
  const goal = band[days - 1].mid;
  const i = target.values.findIndex((v) => v >= goal);
  return i < 0 ? null : i;
}

/**
 * The most cells either sensor lit on a single day in any season, for a region: one scale
 * for the day strip, so a quiet season looks quiet next to a busy one.
 */
export function recordDailyPeak(seasons, region) {
  let peak = 0;
  for (const s of Object.values(seasons.series[region] || {}))
    for (const k of ["modis", "viirs"]) if (s[k]) for (const c of s[k].cells) if (c > peak) peak = c;
  return peak;
}

/** Weekly MODIS-scale activity for the calendar heatmap (years x weeks). */
export function weeklyCalendar(seasons, region) {
  const totals = yearTotals(seasons, region);
  const weeks = Math.ceil(seasons.season_days / 7);
  return Object.entries(seasons.series[region] || {})
    .map(([year, s]) => {
      const useModis = s.modis && s.modis.days > 0;
      const src = useModis ? s.modis : s.viirs;
      let scale = useModis ? 1 : null;
      if (!useModis && src) {
        // The bridge is fitted on season totals, so a VIIRS-only year is rescaled by its
        // seasonal factor (or the plain ratio while the season is incomplete).
        const b = fitBridge(totals, Number(year));
        const t = totals.find((r) => r.year === Number(year));
        scale = !b ? null : t?.viirs > 0 ? b.toModis(t.viirs) / t.viirs : 1 / b.ratio;
      }
      const daily = src ? activity(src) : [];
      const cells = Array.from({ length: weeks }, (_, w) => {
        const from = w * 7, to = Math.min(from + 7, seasons.season_days);
        if (!src || scale === null || src.days < to) return null;
        let total = 0;
        for (let i = from; i < to; i++) total += daily[i];
        // per-day mean x 7 so a short final week is not drawn as falsely quiet
        return (total / (to - from)) * 7 * scale;
      });
      return { year: Number(year), basis: useModis ? "modis" : scale ? "viirs" : null, cells };
    })
    .sort((a, b) => a.year - b.year);
}
