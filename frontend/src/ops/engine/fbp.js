// Fire behaviour, after the Canadian Forest Fire Behavior Prediction (FBP) System
// (Forestry Canada 1992, ST-X-3; Wotton, Alexander & Taylor 2009 update) and the hourly Fine
// Fuel Moisture Code of Van Wagner (1977). Rates of spread are in m/min, intensities in kW/m,
// wind in km/h at 10 m. This is the published model with the less common fuel types left out;
// it is a planning aid inside a simulation, not an operational prediction.

// a, b, c: rate-of-spread curve. q, bui0: build-up effect. cbh (m), cfl (kg/m2): crown.
export const FUELS = [
  { key: "NF", name: "Non-fuel" },
  { key: "C1", name: "Spruce-lichen woodland", a: 90, b: 0.0649, c: 4.5, q: 0.9, bui0: 72, cbh: 2, cfl: 0.75 },
  { key: "C2", name: "Boreal spruce", a: 110, b: 0.0282, c: 1.5, q: 0.7, bui0: 64, cbh: 3, cfl: 0.8 },
  { key: "C3", name: "Mature jack / lodgepole pine", a: 110, b: 0.0444, c: 3.0, q: 0.75, bui0: 62, cbh: 8, cfl: 1.15 },
  { key: "C4", name: "Immature jack / lodgepole pine", a: 110, b: 0.0293, c: 1.5, q: 0.8, bui0: 66, cbh: 4, cfl: 1.2 },
  { key: "D1", name: "Leafless aspen", a: 30, b: 0.0232, c: 1.6, q: 0.9, bui0: 32, cbh: 0, cfl: 0 },
  { key: "M2", name: "Boreal mixedwood, green", mix: 0.5, cbh: 6, cfl: 0.8 },
  { key: "O1", name: "Grass", a: 250, b: 0.035, c: 1.7, q: 1, bui0: 1, cbh: 0, cfl: 0, grass: true },
  { key: "WET", name: "Treed wetland, bog and fen", wet: true, cbh: 3, cfl: 0.6 },
  { key: "SCR", name: "Scrub and regeneration", mix: 0.25, cbh: 3, cfl: 0.4 },
  { key: "URB", name: "Built-up", urban: true },
];
export const FUEL = Object.fromEntries(FUELS.map((f, i) => [f.key, i]));

// Head fire intensity classes used by Canadian agencies (kW/m, upper bounds).
export const HFI_CLASS_BOUNDS = [10, 500, 2000, 4000, 10000, Infinity];
export const hfiClass = (hfi) => HFI_CLASS_BOUNDS.findIndex((b) => hfi < b) + 1;

const GRASS_LOAD = 0.35; // kg/m2, standing grass
const GRASS_CURING = 0.6; // spread factor at about 80% cured: 0.176 + 0.02 * (80 - 58.8)
const FOLIAR_MOISTURE = 100; // %, summer
const URBAN_ROS = [0.4, 4]; // m/min, structure-to-structure spread at calm and at 40 km/h

/** Fine fuel moisture content (%) from the code, and back. */
export const moistureOf = (ffmc) => (147.2 * (101 - ffmc)) / (59.5 + ffmc);
export const ffmcOf = (m) => (59.5 * (250 - m)) / (147.2 + m);

/** One hour of drying or wetting of the fine fuels (Van Wagner 1977), no rain. */
export function hourlyFfmc(ffmc, temp, rh, ws) {
  const h = Math.max(0, Math.min(100, rh));
  const mo = moistureOf(ffmc);
  const tail = 0.18 * (21.1 - temp) * (1 - Math.exp(-0.115 * h));
  const ed = 0.942 * h ** 0.679 + 11 * Math.exp((h - 100) / 10) + tail;
  const ew = 0.618 * h ** 0.753 + 10 * Math.exp((h - 100) / 10) + tail;
  let m = mo;
  if (mo > ed) {
    const k = (0.424 * (1 - (h / 100) ** 1.7) + 0.0694 * Math.sqrt(ws) * (1 - (h / 100) ** 8)) * 0.0579 * Math.exp(0.0365 * temp);
    m = ed + (mo - ed) * 10 ** -k;
  } else if (mo < ew) {
    const k = (0.424 * (1 - ((100 - h) / 100) ** 1.7) + 0.0694 * Math.sqrt(ws) * (1 - ((100 - h) / 100) ** 8)) * 0.0579 * Math.exp(0.0365 * temp);
    m = ew - (ew - mo) * 10 ** -k;
  }
  return Math.max(0, Math.min(101, ffmcOf(m)));
}

const fineFuelFactor = (ffmc) => {
  const m = moistureOf(ffmc);
  return 91.9 * Math.exp(-0.1386 * m) * (1 + m ** 5.31 / 4.93e7);
};
/** Initial Spread Index. A negative wind gives the index for backing spread. */
export const isiOf = (ffmc, ws) => 0.208 * fineFuelFactor(ffmc) * (ws > 40 ? 12 * (1 - Math.exp(-0.0818 * (ws - 28))) : Math.exp(0.05039 * ws));

export const wetlandShare = (bui) => Math.max(0.12, Math.min(0.8, (bui - 40) / 130));
const curve = (f, isi) => f.a * (1 - Math.exp(-f.b * isi)) ** f.c;
const buildup = (f, bui) => (bui > 0 ? Math.exp(50 * Math.log(f.q) * (1 / bui - 1 / f.bui0)) : 1);
const byKey = (key) => FUELS[FUEL[key]];

/** Rate of spread (m/min) on level ground for a fuel at an Initial Spread Index. */
export function rosOf(fuel, isi, bui, ws = 0) {
  if (!fuel.a && !fuel.mix && !fuel.wet && !fuel.urban) return 0;
  if (fuel.urban) return URBAN_ROS[0] + (URBAN_ROS[1] - URBAN_ROS[0]) * Math.min(1, Math.max(0, ws) / 40);
  // muskeg: wet ground under black spruce. It carries little fire in a normal year and burns
  // like the spruce around it once a drought has dried the peat (taken here from the build-up index)
  if (fuel.wet) return rosOf(byKey("C2"), isi, bui, ws) * wetlandShare(bui);
  if (fuel.mix != null) {
    const c2 = byKey("C2"), d1 = byKey("D1");
    return fuel.mix * curve(c2, isi) * buildup(c2, bui) + 0.2 * (1 - fuel.mix) * curve(d1, isi) * buildup(d1, bui);
  }
  if (fuel.grass) return curve(fuel, isi) * GRASS_CURING;
  return curve(fuel, isi) * buildup(fuel, bui);
}

/** Surface fuel consumption (kg/m2). */
export function surfaceConsumption(fuel, ffmc, bui) {
  switch (fuel.key) {
    case "C1":
      return ffmc > 84 ? 0.75 + 0.75 * Math.sqrt(1 - Math.exp(-0.23 * (ffmc - 84))) : 0.75 - 0.75 * Math.sqrt(1 - Math.exp(-0.23 * (84 - ffmc)));
    case "C2":
      return 5 * (1 - Math.exp(-0.0115 * bui));
    case "C3":
    case "C4":
      return 5 * (1 - Math.exp(-0.0164 * bui)) ** 2.24;
    case "D1":
      return 1.5 * (1 - Math.exp(-0.0183 * bui));
    case "M2":
    case "SCR":
      return fuel.mix * surfaceConsumption(byKey("C2"), ffmc, bui) + (1 - fuel.mix) * surfaceConsumption(byKey("D1"), ffmc, bui);
    case "O1":
      return GRASS_LOAD;
    case "WET":
      return surfaceConsumption(byKey("C2"), ffmc, bui) * wetlandShare(bui);
    case "URB":
      return 3; // a nominal load: structures are counted separately, not modelled as fuel
    default:
      return 0;
  }
}

/** Length-to-breadth ratio of the fire ellipse. */
export const lengthToBreadth = (ws, grass = false) =>
  grass ? (ws >= 1 ? 1.1 * ws ** 0.464 : 1) : 1 + 8.729 * (1 - Math.exp(-0.03 * ws)) ** 2.155;

/**
 * Spread away from a burning point at angle theta from the downwind direction: the radius of
 * the fire ellipse with head rate h, back rate b and length-to-breadth lb, measured from the
 * ignition point (which sits at the rear focus side of the ellipse).
 */
export function ellipseRate(h, b, lb, theta) {
  if (h <= 0) return 0;
  const a = (h + b) / 2, w = a / lb, c = (h - b) / 2;
  const cos = Math.cos(theta), sin = Math.sin(theta);
  return (w * w * c * cos + a * w * Math.sqrt(w * w * cos * cos + (a * a - c * c) * sin * sin)) / (w * w * cos * cos + a * a * sin * sin);
}

/**
 * The same point of the ellipse, as the speed of the fire front along its own normal there:
 * the rate Byram's intensity is defined on. At the widest point this is the flank rate, though
 * the ray from the ignition point to it is much longer.
 */
export function ellipseNormalRate(h, b, lb, theta) {
  if (h <= 0) return 0;
  const r = ellipseRate(h, b, lb, theta);
  const a = (h + b) / 2, w = a / lb, c = (h - b) / 2;
  const x = r * Math.cos(theta), y = r * Math.sin(theta);
  const nx = (x - c) / (a * a), ny = y / (w * w);
  return Math.max(0, (x * nx + y * ny) / Math.hypot(nx, ny));
}

/** Upslope spread factor for a slope given as rise over run; gentle slowing downslope. */
export const slopeFactor = (slope) => (slope > 0 ? Math.exp(3.533 * Math.min(slope, 0.7) ** 1.2) : Math.max(0.6, 1 + 0.6 * Math.max(slope, -0.66)));

/**
 * Everything the spread model needs for one weather state: per fuel, the head and back
 * rates, the ellipse, and the terms that turn a rate of spread into an intensity.
 */
export function behaviour({ ws, ffmc, bui }) {
  const isi = isiOf(ffmc, ws), bisi = isiOf(ffmc, -ws);
  const fuels = FUELS.map((f) => {
    const head = rosOf(f, isi, bui, ws);
    if (!(head > 0)) return { head: 0, back: 0, lb: 1, sfc: 0, rso: Infinity, cfl: 0 };
    const back = f.urban ? head * 0.3 : rosOf(f, bisi, bui, 0);
    const sfc = surfaceConsumption(f, ffmc, bui);
    const crown = (f.cfl || 0) * (f.mix ?? 1);
    // critical surface intensity for crowning, and the rate of spread that reaches it
    const csi = f.cbh ? 0.001 * f.cbh ** 1.5 * (460 + 25.9 * FOLIAR_MOISTURE) ** 1.5 : Infinity;
    return { head, back, lb: lengthToBreadth(ws, Boolean(f.grass)), sfc, rso: sfc > 0 ? csi / (300 * sfc) : Infinity, cfl: crown };
  });
  return { isi, bisi, fuels };
}

/** Crown fraction burned at a rate of spread. */
export const crownFraction = (fb, ros) => (ros > fb.rso ? 1 - Math.exp(-0.23 * (ros - fb.rso)) : 0);
/** Fire line intensity (kW/m), Byram: 300 x fuel consumed (kg/m2) x rate of spread (m/min). */
export const intensity = (fb, ros) => 300 * (fb.sfc + fb.cfl * crownFraction(fb, ros)) * ros;
