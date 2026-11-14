// Single source for colours used inside the globe and charts (CSS tokens mirror these).
// One hue per meaning: the fire ramp is always an amount, white and violet are always the two
// sensors, steel blue is always the interface, amber is a caution, and the verdict tones are
// always a verdict. The interface is cool so that fire is the only warm thing on screen.
export const C = {
  modis: "#e9eef4", // MODIS: the reference record, neutral white
  viirs: "#b794ff", // VIIRS: violet
  split: "#8fa0b3", // both sensors, but never on the same day
  accent: "#6fd3ff", // steel blue: selection and action only
  warn: "#ffb224", // amber: caution
  text: "#dbe4ee",
  soft: "#aebccb",
  muted: "#8a9cae",
  faint: "#647a90",
  grid: "rgba(140, 170, 200, 0.12)",
  surface: "#05080c",
  high: "#ff4d3a",
  low: "#5f8fff",
  typical: "#3ddc97",
  mixed: "#ffb224",
  // globe: the true-colour base carries the land, so lines are thin and grey, as on NASA FIRMS
  ocean: "#03070b",
  land: "#121f2c",
  coast: "#c9d2da",
  border: "#b4bec8",
  graticule: "#3a4a5a",
};

/**
 * Ink that stays readable on a ramp colour ("rgb(r, g, b)"): black on the hot end, white on
 * the cool end. Only pure black and white clear 4.5:1 on every colour of the ramp; the
 * switch is at the luminance where the two contrast equally.
 */
export function inkOn(rgb) {
  const [r, g, b] = rgb.match(/\d+/g).map((v) => {
    const c = Number(v) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.179 ? "#000000" : "#ffffff";
}

/** Colour that carries a verdict: hot for high seasons, cool for quiet ones. */
export const verdictColor = (kind) =>
  kind === "high" ? C.high : kind === "low" ? C.low : kind === "mixed" ? C.mixed : kind === "typical" ? C.typical : C.faint;

const STOPS = [
  [0, [16, 24, 34]],
  [0.25, [84, 34, 22]],
  [0.5, [176, 60, 24]],
  [0.75, [246, 122, 40]],
  [1, [255, 226, 150]],
];

/** Sequential fire ramp for dark surfaces: slate -> ember -> flame -> pale gold. */
export function fireRamp(t) {
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < STOPS.length; i++) {
    const [t1, c1] = STOPS[i];
    const [t0, c0] = STOPS[i - 1];
    if (x <= t1) {
      const f = (x - t0) / (t1 - t0);
      const c = c0.map((v, k) => Math.round(v + (c1[k] - v) * f));
      return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
    }
  }
  return "rgb(255, 226, 150)";
}

const MAP_STOPS = [
  [0, [150, 12, 14]],
  [0.35, [232, 24, 20]],
  [0.7, [255, 104, 30]],
  [1, [255, 226, 120]],
];

/**
 * The hotspot ramp on the map. The base is a satellite image, so the dots are FIRMS red:
 * one day of burning is red, a cell that burned all season runs through orange to yellow.
 */
export function mapRamp(t) {
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < MAP_STOPS.length; i++) {
    const [t1, c1] = MAP_STOPS[i];
    const [t0, c0] = MAP_STOPS[i - 1];
    if (x <= t1) {
      const f = (x - t0) / (t1 - t0);
      const c = c0.map((v, k) => Math.round(v + (c1[k] - v) * f));
      return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
    }
  }
  return "rgb(255, 226, 120)";
}

/** Which instrument provides the daily true-colour image for a date. */
export function dailySensor(date) {
  return date >= "2015-11-24"
    ? { layer: "VIIRS_SNPP_CorrectedReflectance_TrueColor", name: "VIIRS SUOMI NPP" }
    : { layer: "MODIS_Terra_CorrectedReflectance_TrueColor", name: "MODIS TERRA" };
}

// The optical overlay: one NASA GIBS image of the theatre for one day, draped on the globe.
// [[south, west], [north, east]], a little wider than the analysis rectangle for context.
export const OPTICAL_BOUNDS = [[46, -146], [74, -52]];
const WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi";
export function opticalUrl(date, width = 2048) {
  const [[s, w], [n, e]] = OPTICAL_BOUNDS;
  const height = Math.round((width * (n - s)) / (e - w));
  return `${WMS}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=${dailySensor(date).layer}&STYLES=&FORMAT=image/jpeg&SRS=EPSG:4326&BBOX=${w},${s},${e},${n}&WIDTH=${width}&HEIGHT=${height}&TIME=${date}`;
}

/** NASA Blue Marble (500 m) for a lat/lon rectangle from GIBS: sharper than the bundled base. */
/**
 * Sentinel-2 cloudless (10 m, EOX) for a lat/lon rectangle: the close-up around a hotspot or an
 * area of interest, sharp enough to see rivers, roads, burn scars and clearings.
 */
export const SENTINEL_CREDIT = "Sentinel-2 cloudless by EOX (modified Copernicus Sentinel data)";
export function sentinelUrl(width, [[s, w], [n, e]], height = Math.min(2048, Math.round((width * (n - s)) / ((e - w) * Math.cos((((s + n) / 2) * Math.PI) / 180))))) {
  return `https://tiles.maps.eox.at/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=s2cloudless-2020&STYLES=&FORMAT=image/jpeg&SRS=EPSG:4326&BBOX=${w},${s},${e},${n}&WIDTH=${width}&HEIGHT=${height}`;
}

/** NASA FIRMS's own map at a spot and day: every detection there, with brightness, FRP and confidence. */
export const firmsUrl = (lat, lon, date) => `https://firms.modaps.eosdis.nasa.gov/map/#d:${date}..${date};@${lon.toFixed(3)},${lat.toFixed(3)},11.0z`;

export function basemapUrl(width = 4096, layer = "BlueMarble_NextGeneration", bounds = OPTICAL_BOUNDS, height = 0) {
  const [[s, w], [n, e]] = bounds;
  height ||= Math.round((width * (n - s)) / (e - w));
  return `${WMS}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=${layer}&STYLES=&FORMAT=image/jpeg&SRS=EPSG:4326&BBOX=${w},${s},${e},${n}&WIDTH=${width}&HEIGHT=${height}`;
}
