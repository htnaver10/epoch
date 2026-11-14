// Where the fire is going if nothing more is done: a run of the spread model forward from
// now, without ember spotting, under the forecast weather. Everything the planner decides is
// read off this: when the town is reached, by which route, and which part of the perimeter
// the damage comes from.
import { UNBURNED } from "./fire.js";

/** Project a fire `hours` ahead. Returns the projected simulation and its arrival field. */
export function project(fire, hours, options = {}) {
  const sim = fire.clone({ spotting: false, ...options });
  sim.advance(fire.t + hours * 60);
  return { sim, arrival: sim.arrival, src: sim.src, from: fire.t, to: fire.t + hours * 60 };
}

/**
 * The same projection in slices, so a page can spread it over several frames:
 * call step(ms) until it returns the finished forecast.
 */
export function startProjection(fire, hours, options = {}) {
  const sim = fire.clone({ spotting: false, ...options }), from = fire.t, to = from + hours * 60;
  return {
    step(budgetMs = 6) {
      const start = performance.now();
      while (sim.t < to && performance.now() - start < budgetMs) sim.advance(Math.min(to, sim.t + 20));
      return sim.t >= to ? { sim, arrival: sim.arrival, src: sim.src, from, to } : null;
    },
  };
}

/** The earliest the projected fire reaches any of `cells`. */
export function impact(forecast, cells) {
  let at = UNBURNED, cell = -1;
  for (const c of cells) if (forecast.arrival[c] < at) { at = forecast.arrival[c]; cell = c; }
  return { at, cell, minutes: at === UNBURNED ? Infinity : Math.max(0, at - forecast.from) };
}

/** The projected route of the fire to a cell, from the present perimeter. */
export function routeTo(forecast, cell) {
  const path = [];
  for (let c = cell; c >= 0 && path.length < 4000; c = forecast.src[c]) {
    path.push(c);
    if (forecast.arrival[c] <= forecast.from) break;
  }
  return path.reverse();
}

/**
 * How much of the projected damage starts at each cell of today's perimeter: the number of
 * cells that burn downstream of it, with cells of the town counted `townWeight` times.
 * Suppression goes first where this is highest.
 */
export function threat(forecast, isTown, townWeight = 60) {
  const { arrival, src, from } = forecast, size = arrival.length;
  const order = [];
  for (let i = 0; i < size; i++) if (arrival[i] > from && arrival[i] !== UNBURNED) order.push(i);
  order.sort((a, b) => arrival[a] - arrival[b]);
  const root = new Int32Array(size).fill(-1), weight = new Float32Array(size);
  for (const v of order) {
    const s = src[v];
    if (s < 0) continue;
    const r = arrival[s] <= from ? s : root[s];
    if (r < 0) continue;
    root[v] = r;
    weight[r] += isTown[v] ? townWeight : 1;
  }
  return weight;
}

/**
 * For every cell the forecast burns, how much burns after it and because of it: the size of
 * its branch of the spread tree, with cells of the town counted `townWeight` times. A cell
 * with a large branch that has not burned yet is a place where the fire can be cut off:
 * treating it delays everything downstream. (The same idea as fuel-treatment placement on a
 * minimum-travel-time spread graph: treat the major travel routes, not the edge.)
 */
export function downstream(forecast, isTown, townWeight = 60) {
  const { arrival, src, from } = forecast, size = arrival.length;
  const order = [];
  for (let i = 0; i < size; i++) if (arrival[i] > from && arrival[i] !== UNBURNED) order.push(i);
  order.sort((a, b) => arrival[b] - arrival[a]); // last to burn first: leaves before their parents
  const weight = new Float32Array(size);
  for (const v of order) {
    weight[v] += isTown[v] ? townWeight : 1;
    const s = src[v];
    if (s >= 0 && arrival[s] > from) weight[s] += weight[v];
  }
  return { weight, order };
}
