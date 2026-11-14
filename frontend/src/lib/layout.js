// Layout arithmetic that does not need the DOM.

const BASE_REM = 13, MAX_REM = 22; // the root size before scaling: mirrors the clamp() in index.css
// The smallest window, in rem, that the desktop layout holds together in.
const MIN_COLS = 86, MIN_LINES = 44;

/** The largest presentation scale a window of `w` by `h` px can lay out, never below 1. */
export function maxScaleFor({ w, h }) {
  const base = Math.max(BASE_REM, Math.min(MAX_REM, Math.min(0.009 * w, 0.015 * h)));
  const fit = Math.min(w / (MIN_COLS * base), h / (MIN_LINES * base));
  return Math.max(1, Math.min(1.8, Math.floor(fit * 10 + 1e-6) / 10));
}
