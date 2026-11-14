// A made-up theatre in the bundled data's shape, for tests and for running the console
// before real map data is loaded: a town on a lake shore, a highway, lakes, and a fire to the west.
import { noise2 } from "./theatre.js";

export function syntheticTheatre({ n = 400, cell = 0.16, seed = 5 } = {}) {
  const half = (n * cell) / 2;
  const cover = new Uint8Array(n * n);
  const town = { name: "Testville", x: 12, y: -2 };
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++) {
      const x = (c + 0.5) * cell - half, y = (r + 0.5) * cell - half;
      const lake = noise2(x / 5, y / 5, seed) * 0.7 + noise2(x / 1.6, y / 1.6, seed + 1) * 0.3;
      let code = 3;
      if (x - y * 0.4 > 20 || lake > 0.68) code = 1; // a big lake to the east, small lakes inland
      else if (lake > 0.62) code = 2;
      else if (noise2(x / 3, y / 3, seed + 2) > 0.8) code = 6;
      if (code !== 1 && Math.hypot(x - town.x, (y - town.y) * 1.3) < 2.4) code = 7;
      cover[r * n + c] = code;
    }
  const rle = [];
  for (let i = 0; i < n * n; ) {
    let j = i;
    while (j < n * n && cover[j] === cover[i]) j++;
    rle.push(cover[i], j - i);
    i = j;
  }
  const road = (c, name, points) => ({ c, n: name, p: points });
  const line = (x0, y0, x1, y1, steps) => Array.from({ length: steps + 1 }, (_, i) => [+(x0 + ((x1 - x0) * i) / steps).toFixed(3), +(y0 + ((y1 - y0) * i) / steps).toFixed(3)]);
  const roads = [road("hwy", "Hwy 3", line(-32, -2, 14, -2, 46)), road("main", "Lake Rd", line(12, -2, 12, 14, 16)), road("track", null, line(-10, -2, -10, -20, 18))];
  for (let k = -2; k <= 2; k++) roads.push(road("local", null, line(10, -2 + k * 0.6, 14, -2 + k * 0.6, 4)), road("local", null, line(12 + k * 0.6, -4, 12 + k * 0.6, 0, 4)));
  const buildings = [];
  for (let i = 0; i < 1800; i++) {
    const a = noise2(i * 0.37, 1.5, seed) * Math.PI * 2, d = noise2(i * 0.11, 9.5, seed) * 2.1;
    buildings.push([+(town.x + Math.cos(a) * d).toFixed(2), +(town.y + (Math.sin(a) * d) / 1.3).toFixed(2)]);
  }
  const weather = Array.from({ length: 73 }, (_, h) => {
    const local = (h + 18) % 24, day = Math.max(0, Math.sin(((local - 6) / 14) * Math.PI));
    return { t: `h${h}`, ws: 12 + 16 * day, wd: 262, gust: 20 + 26 * day, temp: 12 + 14 * day, rh: 70 - 45 * day };
  });
  return {
    id: "synthetic", name: "Testville", region: "XX", date: "2023-08-15", utcOffset: -6, centre: [62, -114], sizeKm: n * cell,
    town, places: [{ n: "Testville", k: "town", x: town.x, y: town.y, pop: 20000 }],
    water: [], islands: [], streams: [], roads, rail: [], built: [], buildings,
    airports: [{ n: "Testville Airport", icao: "CXXX", k: "airport", x: 9, y: -4.5, runways: [[7.8, -4.5, 10.2, -4.5, "asphalt"]] }],
    stations: [{ n: "Fire Hall 1", k: "fire", x: 12.2, y: -1.8 }, { n: "Hospital", k: "hospital", x: 12.8, y: -2.6 }, { n: "Detachment", k: "police", x: 11.6, y: -2.2 }],
    cover: { n, cell, rle },
    elev: null,
    hotspots: Array.from({ length: 40 }, (_, i) => ({ x: +(-2 + noise2(i * 0.9, 3, seed) * 4).toFixed(3), y: +(-5 + noise2(i * 0.9, 8, seed) * 5).toFixed(3), t: "2023-08-15T10:00Z", sensor: "VIIRS", ffmc: 91, bui: 110 })),
    weather, perimeter: null, sources: ["synthetic"],
  };
}
