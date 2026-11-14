// Isochrones: lines of equal arrival time on the forecast, by marching squares over the grid.

/**
 * @param field   Float32Array n x n, row 0 at the south edge
 * @param ok      (cell) => whether the cell takes part (water and rock do not)
 * @param levels  values to contour
 * @param cap     value to use where the field is infinite (beyond the forecast)
 * @returns for each level, a flat array of segment ends in cell units: [c0, r0, c1, r1, ...]
 */
export function contours(field, n, ok, levels, cap) {
  const out = levels.map(() => []);
  const value = (i) => (field[i] === Infinity ? cap : field[i]);
  for (let r = 0; r < n - 1; r++)
    for (let c = 0; c < n - 1; c++) {
      const i = r * n + c;
      if (!ok(i) || !ok(i + 1) || !ok(i + n) || !ok(i + n + 1)) continue;
      const a = value(i), b = value(i + 1), d = value(i + n + 1), e = value(i + n); // SW, SE, NE, NW
      const lo = Math.min(a, b, d, e), hi = Math.max(a, b, d, e);
      for (let k = 0; k < levels.length; k++) {
        const L = levels[k];
        if (L <= lo || L > hi) continue;
        const pts = [];
        const cross = (v0, v1, x0, y0, x1, y1) => {
          if (v0 < L === v1 < L) return;
          const t = (L - v0) / (v1 - v0);
          pts.push(c + x0 + (x1 - x0) * t, r + y0 + (y1 - y0) * t);
        };
        cross(a, b, 0, 0, 1, 0);
        cross(b, d, 1, 0, 1, 1);
        cross(e, d, 0, 1, 1, 1);
        cross(a, e, 0, 0, 0, 1);
        if (pts.length >= 4) out[k].push(pts[0], pts[1], pts[2], pts[3]);
        if (pts.length === 8) out[k].push(pts[4], pts[5], pts[6], pts[7]);
      }
    }
  return out;
}
