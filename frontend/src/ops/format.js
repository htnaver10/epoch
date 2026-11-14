// Text forms shared by the OPS panels.

/** Minutes as HH:MM. With `wrap`, a time of day; without, a duration that may pass 24 hours. */
export function clockText(minutes, wrap = false) {
  if (!Number.isFinite(minutes)) return "--:--";
  const m = Math.max(0, Math.floor(wrap ? ((minutes % 1440) + 1440) % 1440 : minutes));
  return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
}
export const integer = (v) => Math.round(v).toLocaleString("en-CA");
/** Hectares, to the precision that means something. */
export const hectares = (ha) => (ha >= 100 ? integer(ha) : ha >= 10 ? ha.toFixed(0) : ha.toFixed(1));
/** Litres as thousands. */
export const kilolitres = (l) => (l >= 1e6 ? `${(l / 1e6).toFixed(2)} ML` : `${Math.round(l / 1000)} kL`);
