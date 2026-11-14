// Unit symbols. Aircraft are plan-view silhouettes pointing up the page, as on a flight
// tracker; ground units are a framed designator, the frame being the friendly-force rectangle.
// Drawn on a 64 unit square, centred, for both the map sprites and the roster.

export const SILHOUETTES = {
  // four-engine airtanker
  tanker: "M32 4c2 0 3 3 3 8v12l23 10v5l-23-4v14l8 6v4l-11-3-11 3v-4l8-6V35L6 39v-5l23-10V12c0-5 1-8 3-8zM17 27v7M24 24v8M40 24v8M47 27v7",
  // high-wing twin flying boat with tip floats
  skimmer: "M32 5c2.5 0 3.5 3 3.5 8v8h22v6l-22 2v15l7 5v4l-10.500-2.500L21.500 53v-4l7-5V29l-22-2v-6h22v-8c0-5 1-8 3.500-8zM9 19v10M55 19v10M22 18v9M42 18v9",
  // single-engine floatplane
  fireboss: "M32 6c2 0 3 2.500 3 7v9h21v5l-21 2v14l6 4v4l-9-2-9 2v-4l6-4V29L8 27v-5h21v-9c0-4.500 1-7 3-7zM23 30v16M41 30v16",
  // light twin
  lead: "M32 8c1.800 0 2.800 2.500 2.800 7v9l19 6v4l-19-3v13l6 4v3l-8.800-2-8.800 2v-3l6-4V31l-19 3v-4l19-6v-9c0-4.500 1-7 2.800-7zM23 25v6M41 25v6",
  // helicopter: fuselage, boom, tail rotor, and the rotor disc
  helo: "M32 14c4.500 0 7 3.500 7 9v8c0 3-1.500 5-3.500 6l-1 16h4.500v3H25v-3h4.500l-1-16c-2-1-3.500-3-3.500-6v-8c0-5.500 2.500-9 7-9z",
};
export const ROTOR = "M32 4a22 22 0 1 0 0.010 0zM10 26h44M32 4v44";

export const SYMBOL = {
  ELECTRA: "tanker", Q400: "tanker", CL415: "skimmer", CL215T: "skimmer", FIREBOSS: "fireboss", BIRDDOG: "lead", B212: "helo", AS350: "helo",
};
export const DESIGNATOR = { ENGINE: "ENG", TENDER: "TND", SPU: "SPU", DOZER: "DZR", CREW: "CRW", IA: "IA", MEDIC: "MED", POLICE: "RCM" };

/** Draw a unit symbol, white on transparent, onto a canvas context scaled to `size` pixels. */
export function drawSymbol(ctx, type, size) {
  const s = size / 64;
  ctx.save();
  ctx.scale(s, s);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  const key = SYMBOL[type];
  if (key) {
    // a dark keyline first, so the symbol reads over fire and over water
    const path = new Path2D(SILHOUETTES[key]);
    ctx.strokeStyle = "rgba(2,4,6,0.95)";
    ctx.lineWidth = 7;
    ctx.stroke(path);
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2.6;
    ctx.fill(path);
    ctx.stroke(path);
    if (key === "helo") {
      const rotor = new Path2D(ROTOR);
      ctx.lineWidth = 5;
      ctx.strokeStyle = "rgba(2,4,6,0.9)";
      ctx.stroke(rotor);
      ctx.lineWidth = 1.8;
      ctx.strokeStyle = "#fff";
      ctx.stroke(rotor);
    }
  } else {
    ctx.fillStyle = "rgba(2,4,6,0.92)";
    ctx.fillRect(5, 15, 54, 34);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 3.5;
    ctx.strokeRect(6.500, 16.500, 51, 31);
    ctx.fillStyle = "#fff";
    ctx.font = "600 21px 'IBM Plex Mono', ui-monospace, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(DESIGNATOR[type] || "?", 32, 33.500);
  }
  ctx.restore();
}
