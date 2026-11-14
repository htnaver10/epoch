import { useEffect, useLayoutEffect, useRef, useState } from "react";

/** Measured width of an element, so SVG charts draw at true pixel size and text never scales. */
export function useWidth(initial = 640) {
  const ref = useRef(null);
  const [width, setWidth] = useState(initial);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const apply = () => setWidth((w) => {
      const next = Math.max(220, Math.round(el.clientWidth));
      return next === w ? w : next;
    });
    apply();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(apply);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

export const BASE_REM = 13; // root font size the layout is drawn for, before any scaling

const readRem = () => parseFloat(getComputedStyle(document.documentElement).fontSize) || BASE_REM;

/**
 * Pixel size of one rem, and the same as a multiplier of the design size. The root size
 * grows with the viewport (and the presentation scale), so anything laid out in script,
 * such as chart heights and the room the globe keeps clear, has to follow it.
 */
export function useRem(scale = 1) {
  const [rem, setRem] = useState(() => (typeof document === "undefined" ? BASE_REM : readRem()));
  useEffect(() => {
    const update = () => setRem(readRem());
    const frame = requestAnimationFrame(update); // after the new root size has been applied
    window.addEventListener("resize", update);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", update);
    };
  }, [scale]);
  return { rem, k: rem / BASE_REM };
}

/** The window's size, kept current. */
export function useViewport() {
  const [viewport, setViewport] = useState(() => ({ w: window.innerWidth, h: window.innerHeight }));
  useEffect(() => {
    const update = () => setViewport({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);
  return viewport;
}
