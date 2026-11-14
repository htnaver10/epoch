// A scenario: one theatre on one day. Weather comes from the bundled hourly record, the
// fire from the satellite hotspots up to the opening time, and the units from a roster that
// is invented for the simulation (real types, real bases; the tasking is not a record of events).
import { FUEL, hourlyFfmc } from "./fbp.js";
import { PROFILES, buildTheatre } from "./theatre.js";
import { makeUnit } from "./units.js";
import { World } from "./world.js";

export const SCENARIOS = {
  yellowknife: {
    profile: PROFILES.shield, day: 0, start: 13, gust: 0.4, daylight: [5 * 60 + 25, 21 * 60 + 25],
    brief: "Fire ZF015 west of the city on Highway 3. Evacuation was ordered the next evening.",
    roster: [
      ["FIREBOSS", "T644", "air"], ["FIREBOSS", "T645", "air"], ["FIREBOSS", "T646", "air"], ["FIREBOSS", "T647", "air"],
      ["CL415", "T271", "air"], ["CL415", "T272", "air"], ["ELECTRA", "T416", "air"], ["ELECTRA", "T417", "air"], ["BIRDDOG", "BD14", "air"],
      ["B212", "H21", "air"], ["B212", "H22", "air"], ["B212", "H23", "air"], ["AS350", "H31", "air"],
      ["CREW", "CRW1", "yard"], ["CREW", "CRW2", "yard"], ["IA", "IA1", "air"], ["IA", "IA2", "air"], ["IA", "IA3", "air"],
      ["DOZER", "DZR1", "yard"], ["DOZER", "DZR2", "yard"], ["DOZER", "DZR3", "yard"],
      ["ENGINE", "ENG1", "fire"], ["ENGINE", "ENG2", "fire"], ["TENDER", "TND1", "fire"], ["SPU", "SPU1", "fire"], ["SPU", "SPU2", "yard"],
      ["MEDIC", "MED1", "fire"], ["MEDIC", "MED2", "hospital"], ["POLICE", "RCMP1", "police"], ["POLICE", "RCMP2", "police"],
    ],
  },
  hayriver: {
    profile: PROFILES.plain, day: 0, start: 15.5, gust: 0.5, daylight: [5 * 60 + 35, 21 * 60 + 25],
    brief: "Fire SS052 ran east from the Kakisa area on a west wind. Enterprise burned that evening and Hay River was evacuated.",
    roster: [
      ["ELECTRA", "T416", "air"], ["ELECTRA", "T417", "air"], ["FIREBOSS", "T640", "air"], ["FIREBOSS", "T641", "air"], ["FIREBOSS", "T642", "air"], ["FIREBOSS", "T643", "air"], ["BIRDDOG", "BD12", "air"],
      ["B212", "H24", "air"], ["B212", "H25", "air"], ["AS350", "H33", "air"],
      ["CREW", "CRW3", "yard"], ["CREW", "CRW4", "yard"], ["IA", "IA4", "air"], ["IA", "IA5", "air"], ["DOZER", "DZR4", "yard"], ["DOZER", "DZR5", "yard"],
      ["ENGINE", "ENG1", "fire"], ["ENGINE", "ENG2", "fire"], ["TENDER", "TND1", "fire"], ["SPU", "SPU3", "fire"],
      ["MEDIC", "MED1", "hospital"], ["POLICE", "RCMP1", "police"], ["POLICE", "RCMP2", "police"],
    ],
  },
  jasper: {
    profile: PROFILES.montane, day: 2, start: 7, gust: 0.5, valley: 150, daylight: [5 * 60 + 50, 21 * 60 + 45],
    adjusted: "Wind turned along the Athabasca valley (150°/330°): a reanalysis cell cannot resolve the channelling that drove this fire",
    brief: "Jasper wildfire complex in the Athabasca valley. The south fire reached the town on 24 July.",
    // heavy tankers work from the Hinton base, north-east of the mapped area
    bases: { tanker: { lat: 53.3188, lon: -117.7533, name: "Hinton tanker base" } },
    roster: [
      ["ELECTRA", "T481", "tanker"], ["ELECTRA", "T482", "tanker"], ["CL215T", "T201", "tanker"], ["CL215T", "T202", "tanker"], ["CL215T", "T203", "tanker"], ["CL215T", "T204", "tanker"], ["BIRDDOG", "BD131", "tanker"],
      ["B212", "H41", "air"], ["B212", "H42", "air"], ["B212", "H43", "air"], ["AS350", "H51", "air"], ["AS350", "H52", "air"],
      ["CREW", "CRW5", "yard"], ["CREW", "CRW6", "yard"], ["IA", "IA6", "air"], ["IA", "IA7", "air"], ["DOZER", "DZR6", "yard"], ["DOZER", "DZR7", "yard"],
      ["ENGINE", "ENG1", "fire"], ["ENGINE", "ENG2", "fire"], ["TENDER", "TND1", "fire"], ["SPU", "SPU4", "fire"], ["SPU", "SPU5", "fire"],
      ["MEDIC", "MED1", "hospital"], ["POLICE", "RCMP1", "police"], ["POLICE", "RCMP2", "police"],
    ],
  },
  synthetic: {
    profile: PROFILES.shield, start: 13, daylight: [5 * 60 + 30, 21 * 60 + 30], brief: "Synthetic theatre.",
    roster: [
      ["FIREBOSS", "T644", "air"], ["FIREBOSS", "T645", "air"], ["CL415", "T271", "air"], ["CL415", "T272", "air"], ["ELECTRA", "T416", "air"], ["ELECTRA", "T417", "air"], ["BIRDDOG", "BD14", "air"],
      ["B212", "H21", "air"], ["B212", "H22", "air"], ["AS350", "H31", "air"], ["CREW", "CRW1", "yard"], ["CREW", "CRW2", "yard"], ["IA", "IA1", "air"], ["IA", "IA2", "air"],
      ["DOZER", "DZR1", "yard"], ["DOZER", "DZR2", "yard"], ["ENGINE", "ENG1", "fire"], ["ENGINE", "ENG2", "fire"], ["SPU", "SPU1", "fire"], ["MEDIC", "MED1", "hospital"], ["POLICE", "RCMP1", "police"],
    ],
  },
};

const median = (values, fallback) => {
  const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  return v.length ? v[v.length >> 1] : fallback;
};

/**
 * Weather as a function of scenario minutes. Wind, temperature and humidity are the hourly
 * record; the fine fuel moisture code is carried forward hour by hour from the day's value.
 * `startHourLocal` counts from local midnight of the data's date and may run past 24.
 */
export function makeWeather(data, startHourLocal, gustShare = 0.4, valley = null) {
  const hours = data.weather?.length ? data.weather : [{ ws: 18, wd: 250, gust: 30, temp: 24, rh: 30 }];
  const day = (data.hotspots || []).filter((h) => (h.t || "").slice(0, 10) === data.date);
  const bui = median(day.map((h) => h.bui), median((data.hotspots || []).map((h) => h.bui), 90));
  const ffmc = [median(day.map((h) => h.ffmc), median((data.hotspots || []).map((h) => h.ffmc), 89))];
  for (let i = 1; i < hours.length; i++) ffmc.push(hourlyFfmc(ffmc[i - 1], hours[i].temp, hours[i].rh, hours[i].ws));
  const first = startHourLocal - (data.utcOffset ?? -6); // index of the opening hour in the UTC record
  const weather = (t) => {
    // past the end of the record, the last day repeats: the daily cycle matters more than the detail
    let i = Math.max(0, Math.floor(first + t / 60));
    if (i >= hours.length) i = hours.length - 24 + ((i - hours.length) % 24);
    const h = hours[i];
    const gust = h.gust ?? h.ws;
    // the reanalysis wind is an hourly mean over a wide cell and runs low; a fire answers to the gusts too
    let wd = h.wd, ws = h.ws + gustShare * Math.max(0, gust - h.ws);
    if (valley != null) {
      // in a mountain valley the wind runs along the valley, whichever way the flow aloft leans
      const along = Math.cos(((h.wd - valley) * Math.PI) / 180);
      wd = along >= 0 ? valley : valley + 180;
      ws *= 0.6 + 0.4 * Math.abs(along);
    }
    return { ws, mean: h.ws, wd, gust, temp: h.temp, rh: h.rh, ffmc: ffmc[i], bui };
  };
  weather.hours = hours.length - first;
  return weather;
}

/** The same weather with the wind turned and scaled: one member of a planning ensemble. */
export const perturbed = (weather, turn, scale) => (t) => {
  const w = weather(t);
  return { ...w, ws: w.ws * scale, wd: w.wd + turn };
};

function homes(theatre, config) {
  const { data, town } = theatre;
  const [lat0, lon0] = theatre.centre;
  const local = (b) => ({ x: (b.lon - lon0) * 111.32 * Math.cos((lat0 * Math.PI) / 180), y: (b.lat - lat0) * 110.574, name: b.name, off: true });
  const airport = (data.airports || []).filter((a) => (a.k || "airport") === "airport").sort((a, b) => (b.runways?.length || 0) - (a.runways?.length || 0) || Math.hypot(a.x - town.x, a.y - town.y) - Math.hypot(b.x - town.x, b.y - town.y))[0];
  const station = (kind) => (data.stations || []).filter((s) => s.k === kind).sort((a, b) => Math.hypot(a.x - town.x, a.y - town.y) - Math.hypot(b.x - town.x, b.y - town.y))[0];
  const fallback = { x: town.x, y: town.y, name: town.name };
  const air = airport ? { x: airport.x, y: airport.y, name: airport.icao || airport.n, runway: airport.runways?.[0] } : fallback;
  const fire = station("fire") || fallback;
  // the wildfire yard: on a road by the airport
  const node = theatre.roads.nearest(air.x, air.y - 0.4);
  const yard = node.node >= 0 ? { x: theatre.roads.nodes[node.node].x, y: theatre.roads.nodes[node.node].y, name: "Wildfire base" } : fire;
  return { air, fire: { ...fire, name: fire.n || fire.name }, yard, hospital: station("hospital") || fire, police: station("police") || fire, tanker: config.bases?.tanker ? local(config.bases.tanker) : air };
}

/** Units for a theatre, parked a little apart at their bases. */
export function makeRoster(theatre, config) {
  const home = homes(theatre, config), count = {};
  return config.roster.map(([type, call, base]) => {
    const b = home[base] || home.fire, k = (count[base] = (count[base] || 0) + 1) - 1;
    const air = base === "air" || base === "tanker";
    // a row along the apron, or along the street
    const x = b.x + (air ? (k % 7) * 0.09 - 0.27 : (k % 4) * 0.035 - 0.05), y = b.y + (air ? -0.12 - Math.floor(k / 7) * 0.1 : Math.floor(k / 4) * 0.035);
    return makeUnit(type, call, { x, y, name: (b.name || b.n || base).toUpperCase(), off: Boolean(b.off), runway: air && b.runway ? b.runway.slice(0, 4) : null }, { hdg: air ? 0 : 90 });
  });
}

/**
 * Burned ground and live edge from the hotspots seen before the opening time. A hotspot is
 * a pixel some hundreds of metres across, so the shape is approximate: detections are
 * stamped as discs, small gaps closed, and edge near a detection of the last day is live.
 */
export function seedFromHotspots(world, { opens, liveHours = 20 }) {
  const { theatre, fire } = world, { n } = theatre;
  const stamp = new Uint8Array(n * n), recent = new Uint8Array(n * n);
  let used = 0;
  for (const h of theatre.data.hotspots || []) {
    const at = Date.parse(h.t);
    if (!(at <= opens) || !theatre.inside(h.x, h.y)) continue;
    used++;
    const radius = /MODIS/i.test(h.sensor || "") ? 3 : 2, live = opens - at <= liveHours * 3600e3;
    const k = theatre.toCell(h.x, h.y), c0 = k % n, r0 = Math.floor(k / n);
    for (let r = r0 - radius - 2; r <= r0 + radius + 2; r++)
      for (let c = c0 - radius - 2; c <= c0 + radius + 2; c++) {
        if (r < 0 || c < 0 || r >= n || c >= n) continue;
        const d = Math.hypot(r - r0, c - c0);
        if (d <= radius) stamp[r * n + c] = 1;
        if (live && d <= radius + 2) recent[r * n + c] = 1;
      }
  }
  // close gaps of a cell or two between detections
  const grow = (src, by) => {
    let cur = src;
    for (let k = 0; k < Math.abs(by); k++) {
      const out = new Uint8Array(n * n);
      for (let r = 0; r < n; r++)
        for (let c = 0; c < n; c++) {
          const i = r * n + c;
          const a = cur[i], b = c > 0 && cur[i - 1], d = c < n - 1 && cur[i + 1], e = r > 0 && cur[i - n], f = r < n - 1 && cur[i + n];
          out[i] = by > 0 ? (a || b || d || e || f ? 1 : 0) : a && (c === 0 || b) && (c === n - 1 || d) && (r === 0 || e) && (r === n - 1 || f) ? 1 : 0;
        }
      cur = out;
    }
    return cur;
  };
  const mask = grow(grow(stamp, 2), -2);
  // one fire per connected patch of wildland: a detection over a built-up cell is not taken as the town burning
  const wild = (i) => theatre.grid.fuel[i] !== 0 && theatre.grid.fuel[i] !== FUEL.URB;
  const label = new Int32Array(n * n).fill(-1), patches = [];
  for (let s = 0; s < n * n; s++) {
    if (!mask[s] || label[s] >= 0 || !wild(s)) continue;
    const cells = [], stack = [s];
    label[s] = patches.length;
    while (stack.length) {
      const u = stack.pop(), c = u % n, r = (u - c) / n;
      cells.push(u);
      for (const v of [c > 0 ? u - 1 : -1, c < n - 1 ? u + 1 : -1, r > 0 ? u - n : -1, r < n - 1 ? u + n : -1, c > 0 && r > 0 ? u - n - 1 : -1, c < n - 1 && r < n - 1 ? u + n + 1 : -1, c > 0 && r < n - 1 ? u + n - 1 : -1, c < n - 1 && r > 0 ? u - n + 1 : -1])
        if (v >= 0 && mask[v] && label[v] < 0 && wild(v)) { label[v] = patches.length; stack.push(v); }
    }
    patches.push(cells);
  }
  patches.sort((a, b) => b.length - a.length);
  const fires = [];
  for (const cells of patches.slice(0, 12)) {
    if (cells.length < 4) continue;
    const dead = cells.filter((c) => !recent[c]), live = cells.filter((c) => recent[c]);
    const id = fire.seed(dead, { live: false, at: -600 });
    if (live.length) fire.seed(live, { fire: id, live: true, at: -240 });
    fires.push({ id, cells: cells.length, live: live.length });
  }
  fire.refresh(0);
  return { used, fires };
}

/** Build the world for a scenario from its theatre data. */
export function openScenario(data, { seed = 1 } = {}) {
  const config = SCENARIOS[data.id] || SCENARIOS.synthetic;
  const named = config.place ? (data.places || []).find((p) => p.n === config.place) : null;
  const theatre = buildTheatre(data, { profile: config.profile, place: named ? { name: named.n, x: named.x, y: named.y } : null });
  const hour = (config.day || 0) * 24 + config.start; // hours after local midnight of the data's date
  const weather = makeWeather(data, hour, config.gust, config.valley);
  const world = new World({ theatre, weather, units: makeRoster(theatre, config), seed, clock: config.start * 60, daylight: config.daylight });
  const opens = Date.parse(`${data.date}T00:00:00Z`) + (hour - (data.utcOffset ?? -6)) * 3600e3;
  world.opens = opens;
  world.brief = config.brief;
  world.adjusted = config.adjusted || "";
  world.seeded = seedFromHotspots(world, { opens });
  for (const u of world.units) u.nextIdle = world.random() * 45;
  return world;
}
