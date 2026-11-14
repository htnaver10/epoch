import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import Icon from "./components/Icon";
import CommandBar from "./components/CommandBar";
import Dock from "./components/Dock";
import Drawer from "./components/Drawer";
import ZoneCard from "./components/ZoneCard";
import { Readout } from "./components/Readout";
import ActivityCalendar from "./components/ActivityCalendar";
import { ScopeBar, Legend } from "./components/ScopeBar";
import { verdictColor } from "./lib/theme";
import { summarize, groupByDate, dailySeries, mapCells, readView, viewSearch, writeTarget } from "./lib/analysis";
import { areaSeries, withArea, inArea, monthPeriods, writePeriod } from "./lib/area";
import { staticKeys, withoutStatic } from "./lib/harmonize";
import { cellDaysCsv, calendarCsv } from "./lib/export";
import { seasonDate, recordDailyPeak } from "./lib/seasons";
import { nearestCell, formatCoords, coverageOf } from "./lib/geo";
import { PLACES, nearestPlace } from "./lib/places";
import { PANELS } from "./lib/panels";
import { useSeasonAnalysis, useSeasonCurves } from "./lib/useSeasonAnalysis";
import { useRem, useViewport } from "./lib/hooks";
import { maxScaleFor } from "./lib/layout";
import { formatDate } from "./lib/chart";

// three.js and the coastline data are most of the bundle; load them apart from the shell.
const Globe = lazy(() => import("./components/Globe"));

// The analysis area is a latitude and longitude rectangle, not the national border, and the
// sectors are slices of it: every label says so.
const REGIONS = {
  canada: { short: "All", label: "Canada 50–70°N", name: "Canada, 50–70°N by 140–60°W", bounds: [[50, -140], [70, -60]] },
  west: { short: "West", label: "West 50–70°N", name: "West: 140–110°W (BC, AB, YT, NT)", bounds: [[50, -140], [70, -110]] },
  central: { short: "Central", label: "Central 50–70°N", name: "Central: 110–90°W (SK, MB, NU)", bounds: [[50, -110], [70, -90]] },
  east: { short: "East", label: "East 50–70°N", name: "East: 90–60°W (ON, QC, NL)", bounds: [[50, -90], [70, -60]] },
};
const EMPTY = [];
const IMAGE_OFF = { state: "off", date: null };
// Room the globe keeps clear, in rem: the readout on the left; on the right the season table
// or a target's report; and on a phone the report sheet along the bottom.
const READOUT_REM = 35.6, CALENDAR_REM = 31.8, CARD_REM = 31, SHEET_REM = 23, PHONE_READOUT_REM = 6;
const FOLDED_REM = 3.9; // the readout folded to one line, and the gap above it
// Layout steps, in rem of window width. They follow the presentation scale, because a
// larger scale leaves fewer rem to lay out in.
const PHONE_COLS = 69, CALENDAR_COLS = 98, ROOMY_COLS = 110, TABS_COLS = 83, SYSLINE_COLS = 104;
const AOI = "aoi"; // the key the area of interest takes among the record's regions
const RADII = [50, 100, 150, 250, 400]; // km: the sizes an area of interest steps through
const PLAY_MS = 300; // per day: the 92-day season plays in under half a minute
const FAR_KM_PER_PX = 3.5; // beyond this a 0.25° cell is smaller than the cursor
const targetKey = (t) => `${t.lat.toFixed(3)},${t.lon.toFixed(3)},${t.km}`;
const reduceMotion = () => typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const load = (path, signal) =>
  fetch(`${import.meta.env.BASE_URL}data/${path}`, { signal }).then((r) => {
    if (!r.ok) throw Error(`HTTP ${r.status}`);
    return r.json();
  });

function download(csv, name) {
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8;" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `epoch-${name}.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** A target restored from a shared link: a named place if the coordinates match one. */
function restoreTarget(at) {
  if (!at) return null;
  const place = PLACES.find((p) => Math.abs(p.lat - at.lat) < 0.002 && Math.abs(p.lon - at.lon) < 0.002);
  const target = { kind: place ? "place" : at.km ? "coords" : "cell", ...at, label: place ? place.name : "" };
  return { ...target, key: targetKey(target), n: 1 };
}

/** A key press that belongs to a text field or select, not to the console. */
function isTyping(el) {
  if (!el || !el.tagName) return false;
  if (el.isContentEditable || el.tagName === "TEXTAREA" || el.tagName === "SELECT") return true;
  return el.tagName === "INPUT" && !["range", "checkbox", "radio", "button"].includes(el.type);
}

export default function App({ onMode }) {
  const [initial] = useState(() => readView(window.location.search));
  const [seasons, setSeasons] = useState(null);
  const [cache, setCache] = useState({});
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [region, setRegion] = useState(initial.region);
  const [year, setYear] = useState(initial.year);
  const [pickedDate, setDate] = useState(initial.date);
  const [metric, setMetric] = useState(initial.metric);
  const [layer, setLayer] = useState(initial.layer);
  const [raw, setRaw] = useState(initial.view === "raw"); // the uncorrected view
  const [period, setPeriod] = useState(initial.period); // a span of the season's days, or null for all of it
  const [optical, setOptical] = useState(initial.optical);
  const [image, setImage] = useState(IMAGE_OFF);
  const [imageAttempt, setImageAttempt] = useState(0);
  const [panel, setPanel] = useState(initial.panel);
  const [scale, setScale] = useState(initial.scale);
  const [focus, setFocus] = useState(() => restoreTarget(initial.at));
  const [trail, setTrail] = useState(() => (focus ? [focus] : EMPTY)); // every target visited, numbered
  const [fit, setFit] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [notice, setNotice] = useState("");
  const [arriving, setArriving] = useState(() => !reduceMotion()); // the opening flight: the readout waits for the globe
  const requested = useRef(new Set());
  // static heat sources, from the record: dropped from every season's cell-days as they load
  const staticCells = useRef(new Set());
  const opened = useRef([]); // what Esc closes, oldest first: "focus", "panel", "day"
  const viewport = useViewport();
  const maxScale = maxScaleFor(viewport);
  const size = Math.min(scale, maxScale); // the presentation scale in effect: never more than the window can lay out
  const { rem } = useRem(size);
  const cols = viewport.w / rem;
  const wide = cols >= PHONE_COLS;

  // The multi-year record: small, and everything else hangs off it.
  useEffect(() => {
    const controller = new AbortController();
    load("seasons.json", controller.signal)
      .then((s) => {
        const years = Object.keys(s.series.canada).map(Number);
        // open on the record season: the one the product's claim is built on
        const total = (y) => (s.series.canada[y].modis?.w || EMPTY).reduce((a, b) => a + b, 0);
        const whole = years.filter((y) => s.series.canada[y].modis?.days === s.season_days);
        const record = (whole.length ? whole : years).reduce((best, y) => (total(y) > total(best) ? y : best));
        staticCells.current = staticKeys(s);
        setSeasons(s);
        setYear((y) => (y && years.includes(y) ? y : record));
      })
      .catch((e) => e.name !== "AbortError" && setError("The season record could not be loaded."));
    return () => controller.abort();
  }, [attempt]);

  // Cell-day records for the selected season, then its neighbours so stepping years is instant.
  useEffect(() => {
    if (!seasons || !year) return;
    const years = Object.keys(seasons.series.canada).map(Number);
    const controller = new AbortController();
    const fetchYear = (y, signal) =>
      load(`seasons/${y}.json`, signal).then((d) => setCache((c) => (c[y] ? c : { ...c, [y]: withoutStatic(d.records, staticCells.current) })));
    if (!cache[year]) {
      requested.current.add(year);
      fetchYear(year, controller.signal).catch((e) => {
        requested.current.delete(year);
        if (e.name !== "AbortError") setError(`The ${year} season could not be loaded.`);
      });
    }
    else
      for (const y of [year + 1, year - 1]) {
        if (!years.includes(y) || cache[y] || requested.current.has(y)) continue;
        requested.current.add(y);
        fetchYear(y).catch(() => requested.current.delete(y));
      }
    return () => controller.abort();
  }, [seasons, year, cache, attempt]);

  // An area's history needs every season, cell by cell. The rest of the record is fetched in
  // the background once the first season is on screen, so the first target answers at once.
  const allYears = useMemo(() => (seasons ? Object.keys(seasons.series.canada).map(Number) : EMPTY), [seasons]);
  const loadedYears = allYears.filter((y) => cache[y]).length;
  const historyReady = allYears.length > 0 && loadedYears === allYears.length;
  const hasRecord = Boolean(seasons && year && cache[year]);
  const wantsHistory = Boolean(focus);
  useEffect(() => {
    if (!hasRecord || historyReady) return undefined;
    const fetchRest = () => {
      for (const y of allYears) {
        if (requested.current.has(y)) continue;
        requested.current.add(y);
        load(`seasons/${y}.json`)
          .then((d) => setCache((c) => (c[y] ? c : { ...c, [y]: withoutStatic(d.records, staticCells.current) })))
          .catch(() => requested.current.delete(y));
      }
    };
    const t = setTimeout(fetchRest, wantsHistory ? 0 : 1800);
    return () => clearTimeout(t);
  }, [hasRecord, historyReady, wantsHistory, allYears]);

  // The presentation scale: one multiplier on the root size, so the whole console grows.
  useEffect(() => {
    document.documentElement.style.setProperty("--scale", String(size));
  }, [size]);

  useEffect(() => {
    const t = setTimeout(() => setArriving(false), 3400);
    return () => clearTimeout(t);
  }, []);

  // The area in scope: a built-in sector, or the target once its history is in. A target is
  // treated exactly like a sector: its daily series is built from the cell-day records.
  const focusKey = focus ? focus.key : null;
  // a target outside the analysis area has no record to put in scope: its report says so
  const inside = Boolean(focus) && coverageOf(focus.lat, focus.lon, focus.km) !== "outside";
  const areaRecord = useMemo(
    () => (seasons && inside && historyReady ? withArea(seasons, AOI, areaSeries(cache, seasons, focus)) : null),
    // the cache is complete and no longer changes once historyReady is true
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [seasons, historyReady, focusKey, inside],
  );
  const record = areaRecord || seasons;
  const scope = areaRecord ? AOI : region;
  const span = period && seasons && period.to <= seasons.season_days ? period : null;
  const analysis = useSeasonAnalysis(record, scope, span);
  const curves = useSeasonCurves(record, scope, year);
  const verdict = analysis && year ? analysis.verdicts.get(year) : null;
  const tone = verdictColor(verdict?.kind);
  const peak = useMemo(() => (record ? recordDailyPeak(record, scope) : 0), [record, scope]);

  const records = year ? cache[year] : undefined;
  const coverage = seasons && year ? seasons.series.canada[year] : null; // identical in every sector
  const archived = coverage ? Math.min(seasons.season_days, Math.max(coverage.modis?.days ?? 0, coverage.viirs?.days ?? 0)) : 0;
  const dates = useMemo(
    () => (seasons && year ? Array.from({ length: archived }, (_, i) => seasonDate(seasons, year, i)) : []),
    [seasons, year, archived],
  );
  // A date from a shared link only counts if it exists in the season it points at.
  const date = pickedDate !== "all" && dates.length && !dates.includes(pickedDate) ? "all" : pickedDate;
  // Last day VIIRS covers: undefined = all of it, null = VIIRS has no record of this season.
  const viirsThrough = !coverage?.viirs
    ? null
    : coverage.viirs.days >= archived
      ? undefined
      : seasonDate(seasons, year, coverage.viirs.days - 1);
  const noViirs = Boolean(coverage) && !coverage.viirs; // a season before VIIRS launched
  const viirsEnded = date !== "all" && typeof viirsThrough === "string" && date > viirsThrough;

  const bounds = REGIONS[region].bounds;
  const all = records || EMPTY;
  // The globe always draws the sector, for context. The figures, the day strip and the CSV
  // cover the area in scope: the sector, or the target's circle.
  const sectorRows = useMemo(() => all.filter((r) => r[2] >= bounds[0][1] && r[2] < bounds[1][1]), [all, bounds]);
  const scopeRows = useMemo(
    () => (areaRecord && focus ? all.filter((r) => inArea(focus, r[1], r[2])) : sectorRows),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [all, sectorRows, areaRecord, focusKey],
  );
  const byDate = useMemo(() => groupByDate(scopeRows), [scopeRows]);
  const daily = useMemo(() => dailySeries(byDate, dates, metric), [byDate, dates, metric]);
  // one day if a day is picked, otherwise the period, otherwise the whole season
  const firstDay = span ? dates[span.from] : null, lastOfSpan = span ? dates[Math.min(span.to, dates.length) - 1] : null;
  const within = useCallback(
    (list) => (date !== "all" ? list.filter((r) => r[0] === date) : firstDay ? list.filter((r) => r[0] >= firstDay && r[0] <= lastOfSpan) : list),
    [date, firstDay, lastOfSpan],
  );
  const selected = useMemo(() => within(scopeRows), [within, scopeRows]);
  const shown = useMemo(() => within(sectorRows), [within, sectorRows]);
  const summary = useMemo(() => summarize(selected, viirsThrough), [selected, viirsThrough]);
  const mapPoints = useMemo(() => mapCells(shown, layer, viirsThrough), [shown, layer, viirsThrough]);
  const peakDay = useMemo(
    () => daily.reduce((best, d) => (d.occupied > (best?.occupied ?? 0) ? d : best), null)?.date,
    [daily],
  );
  const imageryDate = date !== "all" ? date : peakDay || `${year || 2025}-07-15`;
  const loading = !error && (!seasons || !records);
  const lastDay = dates[dates.length - 1];
  // how the area and the period in scope are named, everywhere
  const months = useMemo(() => (seasons ? monthPeriods(seasons) : EMPTY), [seasons]);
  const month = span ? months.find((m) => m.from === span.from && m.to === span.to) : null;
  const periodLabel = !dates.length
    ? ""
    : !span
      ? `${formatDate(dates[0], "monthShort")}–${formatDate(seasonDate(seasons, year, seasons.season_days - 1), "monthShort")}`
      : month
        ? month.key
        : `${formatDate(dates[span.from])} – ${formatDate(seasonDate(seasons, year, span.to - 1))}`;
  const seasonLabel = year ? `${periodLabel} ${year}` : "";
  const targetName = focus ? `${focus.label || formatCoords(focus.lat, focus.lon)} · ${focus.km ? `${focus.km} km` : "cell"}` : "";
  const areaName = areaRecord ? targetName : REGIONS[region].label;

  useEffect(() => {
    if (!year) return;
    const query = viewSearch({ region, year, date, metric, layer, view: raw ? "raw" : "corrected", period: writePeriod(period), optical, panel, scale: size, at: writeTarget(focus) });
    window.history.replaceState(null, "", `${window.location.pathname}?${query}${window.location.hash}`);
  }, [region, year, date, metric, layer, raw, period, optical, panel, size, focus]);

  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(""), 4000);
    return () => clearTimeout(t);
  }, [notice]);

  // Playback runs the season once: on its last day it is no longer running.
  const running = playing && dates.length > 0 && date !== lastDay;
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setDate((d) => dates[Math.min(dates.length - 1, dates.indexOf(d) + 1)]), PLAY_MS);
    return () => clearInterval(t);
  }, [running, dates]);

  // What Esc closes next: whatever was opened last.
  const hasFocus = Boolean(focus), hasPanel = Boolean(panel), hasDay = date !== "all";
  useEffect(() => {
    const mark = (what, on) => {
      const list = opened.current.filter((x) => x !== what);
      opened.current = on ? [...list, what] : list;
    };
    mark("focus", hasFocus);
  }, [hasFocus]);
  useEffect(() => {
    opened.current = [...opened.current.filter((x) => x !== "panel"), ...(hasPanel ? ["panel"] : [])];
  }, [hasPanel]);
  useEffect(() => {
    opened.current = [...opened.current.filter((x) => x !== "day"), ...(hasDay ? ["day"] : [])];
  }, [hasDay]);

  // A control clicked with the mouse gives the keyboard back to the console, so Space still
  // plays and Esc still closes. A control reached with Tab keeps the focus the user gave it.
  useEffect(() => {
    const release = (e) => {
      if (e.detail === 0) return;
      const control = e.target.closest?.("button, summary, input[type=range]");
      if (control && control === document.activeElement) control.blur();
    };
    window.addEventListener("click", release);
    return () => window.removeEventListener("click", release);
  }, []);

  const chooseYear = useCallback((y) => {
    if (!y) return;
    setYear(y);
    setDate("all");
    setPlaying(false);
    setError("");
  }, []);
  const chooseDate = useCallback((d) => {
    setDate(d);
    setPlaying(false);
  }, []);
  const chooseRegion = useCallback((key) => {
    setRegion(key);
    setFocus(null);
    setPlaying(false);
    setFit((f) => f + 1);
  }, []);
  const clearFocus = useCallback(() => {
    setFocus(null);
    setFit((f) => f + 1); // pull back to the sector
  }, []);
  const goHome = useCallback(() => {
    setFocus(null);
    setTrail(EMPTY);
    setPanel(null);
    setPlaying(false);
    setRaw(false);
    setPeriod(null);
    setDate("all");
    setFit((f) => f + 1);
  }, []);
  // Grow or shrink the area of interest. It stays the same numbered target.
  const resizeTarget = useCallback(
    (step) => {
      if (!focus) return;
      const i = RADII.findIndex((r) => r >= focus.km);
      const km = focus.km === 0 ? RADII[0] : RADII[Math.max(0, Math.min(RADII.length - 1, (i < 0 ? RADII.length - 1 : i) + step))];
      if (km === focus.km) return;
      const next = { ...focus, km, kind: focus.kind === "cell" ? "zone" : focus.kind, keepZoom: false };
      next.key = targetKey(next);
      setTrail((list) => list.map((t) => (t.key === focus.key ? next : t)));
      setFocus(next);
    },
    [focus],
  );
  const togglePanel = useCallback((key) => setPanel((p) => (p === key ? null : key)), []);
  const togglePlay = useCallback(() => {
    if (!running && date === lastDay) setDate("all"); // from the end, play starts over
    setPlaying(!running);
  }, [running, date, lastDay]);
  const clearTrail = useCallback(() => {
    // the target in focus stays, renumbered: its card is still open
    const kept = focus ? [{ ...focus, n: 1, keepZoom: false }] : EMPTY;
    setTrail(kept);
    if (focus) setFocus({ ...focus, n: 1 });
  }, [focus]);
  const resize = useCallback(
    (step) => {
      const next = Math.max(0.8, Math.min(maxScale, Math.round((size + step) * 10) / 10));
      if (next === size) setNotice(step > 0 ? "Largest size this window can show" : "Smallest size");
      else setScale(next);
    },
    [size, maxScale],
  );

  // Aim at a target: it joins the numbered trail (or is found on it) and takes the focus.
  const aim = useCallback(
    (target) => {
      const key = targetKey(target);
      const known = trail.find((t) => t.key === key);
      const entry = known || { ...target, key, n: trail.length + 1 };
      if (!known) setTrail([...trail, entry]);
      setFocus(target.keepZoom ? { ...entry, keepZoom: true } : entry);
    },
    [trail],
  );

  // A click on the globe. Up close: the nearest hotspot cell, or the ground there. From far
  // out a single cell is smaller than the cursor, so the click takes the zone around it,
  // named after the nearest town, and the camera goes in.
  const onPick = useCallback(
    ({ lat, lon, kmPerPixel }) => {
      if (kmPerPixel > FAR_KM_PER_PX) {
        const near = nearestPlace(lat, lon);
        aim({ kind: "zone", lat, lon, km: 150, label: near ? `Near ${near.name}` : "" });
        return;
      }
      const cell = nearestCell(mapPoints, lat, lon, Math.max(16, kmPerPixel * 9));
      aim(cell ? { kind: "cell", lat: cell[0], lon: cell[1], km: 0, label: "", keepZoom: true } : { kind: "zone", lat, lon, km: 100, label: "", keepZoom: true });
    },
    [mapPoints, aim],
  );

  // The command line: every result type becomes one of the moves the interface already has.
  const run = useCallback(
    (action) => {
      if (action.type === "place" || action.type === "coords") {
        aim({ kind: action.type, lat: action.lat, lon: action.lon, km: action.km, label: action.type === "place" ? action.name : "" });
        // show the whole theatre's data so the target is never filtered out of view
        if (!(action.lon >= bounds[0][1] && action.lon < bounds[1][1])) setRegion("canada");
      } else if (action.type === "year") chooseYear(action.year);
      else if (action.type === "date") {
        if (action.year !== year) setYear(action.year);
        chooseDate(action.date);
      } else if (action.type === "peak") {
        if (peakDay) chooseDate(peakDay);
      } else if (action.type === "region") chooseRegion(action.key);
      else if (action.type === "panel") setPanel(action.key);
    },
    [aim, bounds, year, peakDay, chooseYear, chooseDate, chooseRegion],
  );

  // The summary: home, on the record season, with the whole assessment open. One key, so a
  // walkthrough has somewhere to end.
  const summarise = useCallback(() => {
    goHome();
    if (analysis) chooseYear(analysis.years.find((y) => analysis.verdicts.get(y)?.rank === 1) ?? year);
    setPanel("brief");
  }, [goHome, chooseYear, analysis, year]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || isTyping(e.target)) return;
      const tag = e.target.tagName;
      const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      // a control that holds the keyboard focus keeps its own activation key
      if (key === " " && (tag === "BUTTON" || tag === "SUMMARY" || tag === "A")) return;
      if (key === "ArrowRight" || key === "ArrowLeft") {
        if (!dates.length) return;
        e.preventDefault(); // also stops the day slider stepping on its own, so every route agrees
        setPlaying(false);
        setDate((d) => {
          const i = d === "all" ? (key === "ArrowRight" ? -1 : dates.length) : dates.indexOf(d);
          return dates[Math.max(0, Math.min(dates.length - 1, i + (key === "ArrowRight" ? 1 : -1)))] ?? d;
        });
      } else if (key === " ") {
        e.preventDefault();
        togglePlay();
      } else if (key === "Escape") {
        // close whatever was opened last: a target, a panel, a day
        const top = opened.current[opened.current.length - 1];
        if (top === "focus") clearFocus();
        else if (top === "panel") setPanel(null);
        else if (top === "day") {
          setPlaying(false);
          setDate("all");
        } else setRaw(false);
      } else if ((key === "[" || key === "]") && analysis && year) {
        const years = analysis.years;
        chooseYear(years[years.indexOf(year) + (key === "]" ? 1 : -1)]);
      } else if (/^[1-5]$/.test(key)) togglePanel(PANELS[Number(key) - 1][0]);
      else if (key === "0") summarise();
      else if (key === "d") setLayer((l) => (l === "uncertain" ? "all" : "uncertain"));
      else if (key === "s") setOptical((o) => !o);
      else if (key === "h") goHome();
      else if (key === "c") clearTrail();
      else if (key === "r") {
        if (!e.repeat) setRaw(true); // held: the uncorrected view shows while the key is down
      } else if (key === "+" || key === "=") resize(0.1);
      else if (key === "-" || key === "_") resize(-0.1);
    };
    const onKeyUp = (e) => {
      if (e.key.toLowerCase() === "r" && !isTyping(e.target)) setRaw(false);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [dates, analysis, year, chooseYear, clearFocus, togglePanel, togglePlay, goHome, clearTrail, summarise, resize]);

  async function share() {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setNotice("Link copied");
    } catch {
      setNotice("Copy the address bar to share");
    }
  }

  const context = useMemo(() => ({ year, years: analysis?.years || [], dates }), [year, analysis, dates]);
  const slug = (text) => text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const fileStem = `${slug(areaRecord ? targetName : region)}-${year}${date !== "all" ? `-${date}` : span ? `-days-${span.from + 1}-${span.to}` : ""}`;
  const exportCalendar = useCallback(() => {
    if (analysis) download(calendarCsv(record, analysis, { area: areaName, period: span }), `calendar-${slug(areaName)}`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysis, record, areaName, span]);
  const status = error
    ? { tone: "error", text: error }
    : loading
      ? { tone: "busy", text: `Loading ${seasonLabel || "the record"}` }
      : inside && !areaRecord
        ? { tone: "busy", text: `Loading this area's history: ${loadedYears} of ${allYears.length} seasons` }
      : raw
        ? { tone: "alert", text: "Uncorrected view: raw detection counts, not comparable across sensors" }
        : layer === "uncertain" && noViirs
          ? { tone: "note", text: `VIIRS was not observing in ${year}: no disagreement to show` }
          : viirsEnded
            ? { tone: "note", text: `The VIIRS archive ends ${formatDate(viirsThrough)}: this day is MODIS only` }
            : date !== "all" && !selected.length
              ? { tone: "note", text: `No hotspots on ${formatDate(date)}` }
              : null;

  // What shares the stage with the globe. A target and a panel together leave no room for
  // the full readout, so it folds to its title line and the index.
  const docked = Boolean(analysis && year) && wide && !hasPanel && cols >= CALENDAR_COLS; // the calendar, on the right
  const compact = wide && hasFocus && (hasPanel || cols < (docked ? ROOMY_COLS : 79));
  const table = docked;
  const prompt = Boolean(analysis) && wide && !status && !hasFocus && !hasPanel && !running && trail.length === 0;
  const steps = [!wide && "phone", !table && "no-table", cols < SYSLINE_COLS && "lt-sysline", cols < TABS_COLS && "lt-tabs", viewport.h / rem < 56 && "short", viewport.h / rem < 49 && "shorter"].filter(Boolean).join(" ");

  const callout = focus && records ? (
    <ZoneCard key={focus.key} focus={focus} index={focus.n} rows={all} dates={dates} year={year} date={date} noViirs={noViirs} onDate={chooseDate} onRadius={resizeTarget} onClose={clearFocus} />
  ) : null;

  return (
    <div className={`console ${steps}${panel ? " has-drawer" : ""}${focus ? " has-focus" : ""}${arriving ? " arriving" : ""}`}>
      <header className="bar">
        <h1 className="brand">
          <button onClick={goHome} title="Home: the whole analysis area, all days (H)">EPOCH</button>
        </h1>
        {onMode && (
          <div className="seg mode" role="group" aria-label="Mode">
            <button className="on" aria-pressed="true">Calendar</button>
            <button onClick={() => onMode("ops")} title="Response scenarios: a fire near a town, the crews and aircraft that could meet it, and the plans they could follow">Scenarios</button>
          </div>
        )}
        <p className="sysline">
          <b>Burning activity calendar<i className="sys-where"> · Canada 50–70°N</i></b>
          <span>{analysis ? `Harmonized MODIS ${analysis.first}–${analysis.last} + VIIRS ${analysis.viirsFirst ?? "–"}–${analysis.last}` : "loading the record"}</span>
        </p>
        <CommandBar context={context} onRun={run} />
        <nav className="tabs" aria-label="Panels">
          {PANELS.map(([key, label], i) => (
            <button key={key} className={panel === key ? "on" : undefined} aria-pressed={panel === key} onClick={() => togglePanel(key)} disabled={!analysis} title={`${label} (${i + 1})`}>
              <b>{i + 1}</b>
              {label}
            </button>
          ))}
        </nav>
        <div className="actions">
          <span className="notice" role="status">{notice}</span>
          <button onClick={share} title="Copy a link to exactly this view" aria-label="Copy link"><Icon name="link" size="1.15rem" /></button>
          <button
            disabled={!selected.length}
            title="Download the hotspot cell-days of the area and period in scope as CSV"
            aria-label="Export CSV"
            onClick={() => download(cellDaysCsv(selected), fileStem)}
          >
            <Icon name="download" size="1.15rem" />
          </button>
        </div>
      </header>

      <ScopeBar
        regions={REGIONS}
        region={region}
        onRegion={chooseRegion}
        target={areaRecord ? targetName.replace(/ · (\d+) km$/, " $1 km") : ""}
        year={year}
        months={months}
        period={span}
        periodLabel={periodLabel}
        onPeriod={setPeriod}
        date={date}
        layer={layer}
        onLayer={setLayer}
        raw={raw}
        onRaw={setRaw}
        optical={optical}
        onOptical={setOptical}
        image={optical ? image : IMAGE_OFF}
        onRetryImage={() => setImageAttempt((a) => a + 1)}
        targets={trail.length}
        onClearTargets={clearTrail}
        scale={size}
        cells={mapPoints.length}
        loading={loading}
      />

      <main className="stage">
        <Suspense fallback={<div className="globe-stage" />}>
          <Globe
            bounds={bounds}
            sectors={REGIONS}
            region={region}
            points={mapPoints}
            sensorColours={layer === "uncertain"}
            allDays={date === "all"}
            raw={raw}
            optical={optical}
            imageryDate={imageryDate}
            imageAttempt={imageAttempt}
            onImage={setImage}
            playing={running}
            focus={focus}
            trail={trail}
            fit={fit}
            rem={rem}
            inset={wide && !compact ? READOUT_REM * rem : 0}
            reserve={wide ? ((docked ? CALENDAR_REM : 0) + (hasFocus ? CARD_REM : 0)) * rem : 0}
            edgeRight={docked ? CALENDAR_REM * rem : 0}
            edgeTop={compact ? FOLDED_REM * rem : 0}
            reserveBelow={wide ? 0 : (hasFocus ? SHEET_REM : -PHONE_READOUT_REM) * rem}
            onPick={onPick}
            onSector={chooseRegion}
            onTrail={setFocus}
            places={PLACES}
            onPlace={(p) => run({ type: "place", ...p })}
            callout={callout}
          />
        </Suspense>

        {analysis && year ? (
          <Readout seasons={record} analysis={analysis} year={year} areaName={areaName} periodLabel={periodLabel} raw={raw} compact={compact || !wide} />
        ) : (
          <section className="readout" aria-busy="true"><div className="shimmer" style={{ height: "12rem" }} /></section>
        )}
        {docked && (
          <ActivityCalendar
            docked
            seasons={record}
            analysis={analysis}
            year={year}
            onYear={chooseYear}
            period={span}
            onPeriod={setPeriod}
            areaName={areaName}
            onExport={exportCalendar}
            busy={inside && !areaRecord ? `Loading this area's history: ${loadedYears} of ${allYears.length} seasons` : ""}
          />
        )}
        <Legend layer={layer} noViirs={noViirs} allDays={date === "all"} raw={raw} />

        {status ? (
          <div className={`status status-${status.tone}`} role={status.tone === "error" ? "alert" : "status"}>
            {status.tone === "busy" && <i className="spinner" aria-hidden="true" />}
            {status.text}
            {status.tone === "error" && <button onClick={() => { setError(""); setAttempt((a) => a + 1); }}>Retry</button>}
          </div>
        ) : (
          prompt && (
            <p className="status status-prompt">
              Pick an area of interest: click the map, or press <kbd>/</kbd> and type a place
            </p>
          )
        )}

        {panel && analysis && year && (
          <Drawer
            panel={panel}
            onClose={() => setPanel(null)}
            ctx={{
              seasons: record, analysis, curves, year, regionName: areaName, date, period: seasonLabel, span, tone, daily, metric, viirsThrough, noViirs, viirsEnded, summary, peak,
              hasRecords: Boolean(records),
              onYear: chooseYear, onDate: chooseDate, onMetric: setMetric, onPeriod: setPeriod, onExportCalendar: exportCalendar,
            }}
          />
        )}
      </main>

      <Dock analysis={analysis} year={year} onYear={chooseYear} dates={dates} daily={daily} date={date} playing={running} onDate={chooseDate} onPlay={togglePlay} peak={peak} />
    </div>
  );
}
