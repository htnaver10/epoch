// Inline icons on a 24 px grid. Stroked unless listed in FILLED.
const PATHS = {
  play: "M8 5.5v13l11-6.5z",
  pause: "M7.5 5.5h3v13h-3zM13.5 5.5h3v13h-3z",
  left: "M14.5 6l-6 6 6 6",
  right: "M9.5 6l6 6-6 6",
  close: "M6.5 6.5l11 11M17.5 6.5l-11 11",
  download: "M12 4v11M7.5 10.5L12 15l4.5-4.5M5 19.5h14",
  info: "M12 11v6M12 7.6v.4M4.5 4.5h15v15h-15z",
  search: "M10.5 17a6.5 6.5 0 1 0 0-13 6.5 6.5 0 0 0 0 13zM15.3 15.3L20 20",
  link: "M10 13.5a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1.2 1.2M14 10.5a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1.2-1.2",
};
const FILLED = new Set(["play", "pause"]);

export default function Icon({ name, size = 16 }) {
  const filled = FILLED.has(name);
  return (
    <svg
      className="icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke={filled ? "none" : "currentColor"}
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
