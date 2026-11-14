// Aircraft as small solid models, built from a few shapes each: enough to tell an airtanker
// from a skimmer from a helicopter at a glance, and to see one bank into a turn. A model is
// one unit long, nose towards -Z, +Y up, so that turning it about Y by minus its heading
// points it along its track on the map.
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";

/** A body of revolution along Z from a profile of [radius, z]; `squash` flattens it top to bottom. */
function body(profile, squash = 1, segments = 12) {
  const g = new THREE.LatheGeometry(profile.map(([r, z]) => new THREE.Vector2(r, z)), segments);
  g.rotateX(Math.PI / 2);
  g.scale(1, squash, 1);
  return g;
}
/** A flat panel from a planform of [x, z] points: a wing or a tailplane. Mirrored left to right. */
function panel(half, thickness, y) {
  const outline = [...half, ...half.slice().reverse().map(([x, z]) => [-x, z])];
  const shape = new THREE.Shape(outline.map(([x, z]) => new THREE.Vector2(x, z)));
  const g = new THREE.ExtrudeGeometry(shape, { depth: thickness, bevelEnabled: false });
  g.rotateX(Math.PI / 2);
  g.translate(0, y + thickness / 2, 0);
  return g;
}
/** An upright panel from a profile of [z, y] points: a fin. */
function fin(profile, thickness, x = 0) {
  const shape = new THREE.Shape(profile.map(([z, y]) => new THREE.Vector2(z, y)));
  const g = new THREE.ExtrudeGeometry(shape, { depth: thickness, bevelEnabled: false });
  g.rotateY(-Math.PI / 2);
  g.translate(x + thickness / 2, 0, 0);
  return g;
}
/** A pod along Z: an engine nacelle or a float. */
function pod(radius, length, x, y, z, squash = 1) {
  const g = body([[0, -length / 2], [radius * 0.8, -length * 0.42], [radius, -length * 0.2], [radius, length * 0.2], [radius * 0.5, length * 0.45], [0, length / 2]], squash, 8);
  g.translate(x, y, z);
  return g;
}
const box = (w, h, d, x, y, z) => new THREE.BoxGeometry(w, h, d).translate(x, y, z);
const join = (parts) => mergeGeometries(parts.map((p) => (p.index ? p.toNonIndexed() : p)), false);

const BUILD = {
  // four-engine turboprop airliner turned airtanker: long fuselage, low straight wing
  tanker: () =>
    join([
      body([[0, -0.5], [0.03, -0.47], [0.05, -0.4], [0.054, -0.25], [0.054, 0.15], [0.036, 0.38], [0.013, 0.5], [0, 0.5]]),
      panel([[0, -0.13], [0.475, -0.075], [0.475, -0.005], [0, 0.03]], 0.016, -0.02),
      pod(0.022, 0.15, 0.15, -0.012, -0.14), pod(0.022, 0.15, -0.15, -0.012, -0.14), pod(0.022, 0.15, 0.3, -0.012, -0.12), pod(0.022, 0.15, -0.3, -0.012, -0.12),
      panel([[0, 0.35], [0.18, 0.4], [0.18, 0.45], [0, 0.46]], 0.012, 0.015),
      fin([[0.3, 0.03], [0.43, 0.17], [0.49, 0.17], [0.48, 0.03]], 0.012),
    ]),
  // twin-engine flying boat: deep hull, high wing with the engines on it, tip floats, T-tail
  skimmer: () =>
    join([
      body([[0, -0.5], [0.04, -0.46], [0.066, -0.36], [0.07, -0.1], [0.06, 0.14], [0.034, 0.32], [0.02, 0.5], [0, 0.5]], 1.15),
      panel([[0, -0.12], [0.7, -0.105], [0.7, 0.015], [0, 0.03]], 0.018, 0.1),
      box(0.05, 0.05, 0.14, 0, 0.085, -0.05),
      pod(0.032, 0.18, 0.2, 0.128, -0.1), pod(0.032, 0.18, -0.2, 0.128, -0.1),
      pod(0.02, 0.12, 0.6, 0.03, -0.04), pod(0.02, 0.12, -0.6, 0.03, -0.04), box(0.008, 0.07, 0.03, 0.6, 0.07, -0.04), box(0.008, 0.07, 0.03, -0.6, 0.07, -0.04),
      fin([[0.28, 0.05], [0.4, 0.27], [0.5, 0.27], [0.5, 0.04]], 0.014),
      panel([[0, 0.37], [0.25, 0.4], [0.25, 0.46], [0, 0.48]], 0.012, 0.265),
    ]),
  // single-engine agricultural aircraft on amphibious floats
  fireboss: () =>
    join([
      body([[0, -0.5], [0.035, -0.47], [0.05, -0.36], [0.05, -0.05], [0.036, 0.2], [0.014, 0.5], [0, 0.5]]),
      body([[0, -0.2], [0.035, -0.17], [0.03, 0.02], [0, 0.06]], 1.25).translate(0, 0.05, 0),
      panel([[0, -0.2], [0.8, -0.19], [0.8, -0.04], [0, -0.02]], 0.018, -0.03),
      pod(0.04, 0.72, 0.19, -0.16, -0.1, 0.9), pod(0.04, 0.72, -0.19, -0.16, -0.1, 0.9),
      box(0.012, 0.11, 0.04, 0.19, -0.09, -0.22), box(0.012, 0.11, 0.04, -0.19, -0.09, -0.22), box(0.012, 0.11, 0.04, 0.19, -0.09, 0.05), box(0.012, 0.11, 0.04, -0.19, -0.09, 0.05),
      panel([[0, 0.36], [0.28, 0.38], [0.28, 0.46], [0, 0.47]], 0.012, 0.01),
      fin([[0.3, 0.03], [0.42, 0.19], [0.5, 0.19], [0.49, 0.03]], 0.012),
    ]),
  // light high-wing twin: the bird dog
  lead: () =>
    join([
      body([[0, -0.5], [0.03, -0.46], [0.046, -0.36], [0.048, -0.05], [0.034, 0.22], [0.012, 0.5], [0, 0.5]]),
      panel([[0, -0.14], [0.52, -0.08], [0.52, -0.02], [0, 0.02]], 0.014, 0.04),
      pod(0.03, 0.22, 0.19, 0.04, -0.12), pod(0.03, 0.22, -0.19, 0.04, -0.12),
      panel([[0, 0.36], [0.2, 0.41], [0.2, 0.46], [0, 0.46]], 0.01, 0.06),
      fin([[0.26, 0.03], [0.44, 0.2], [0.5, 0.2], [0.48, 0.03]], 0.01),
    ]),
  // utility helicopter: cabin, tail boom, fin, skids. The rotor is separate so that it can turn.
  helo: () =>
    join([
      body([[0, -0.34], [0.05, -0.31], [0.085, -0.2], [0.092, -0.02], [0.07, 0.1], [0.025, 0.2], [0, 0.22]], 1.05),
      pod(0.02, 0.5, 0, 0.035, 0.38),
      fin([[0.5, 0.02], [0.57, 0.17], [0.62, 0.17], [0.6, 0.02]], 0.012),
      panel([[0, 0.42], [0.09, 0.42], [0.09, 0.47], [0, 0.47]], 0.008, 0.04),
      box(0.012, 0.012, 0.42, 0.085, -0.125, -0.08), box(0.012, 0.012, 0.42, -0.085, -0.125, -0.08),
      box(0.008, 0.05, 0.008, 0.085, -0.1, -0.2), box(0.008, 0.05, 0.008, -0.085, -0.1, -0.2), box(0.008, 0.05, 0.008, 0.085, -0.1, 0.04), box(0.008, 0.05, 0.008, -0.085, -0.1, 0.04),
      box(0.03, 0.04, 0.03, 0, 0.11, -0.06),
    ]),
};
const KEY = { ELECTRA: "tanker", Q400: "tanker", CL415: "skimmer", CL215T: "skimmer", FIREBOSS: "fireboss", BIRDDOG: "lead", B212: "helo", AS350: "helo" };
/** Length on screen, in design pixels, of each kind of aircraft. */
export const LENGTH_PX = { tanker: 54, skimmer: 46, fireboss: 36, lead: 36, helo: 44 };

const cache = {};
function shapes(kind) {
  if (cache[kind]) return cache[kind];
  const solid = BUILD[kind]();
  solid.computeVertexNormals();
  const entry = { solid, edges: new THREE.EdgesGeometry(solid, 28) };
  if (kind === "helo") {
    entry.rotor = join([box(1.16, 0.006, 0.05, 0, 0, 0), box(0.05, 0.006, 1.16, 0, 0, 0)]);
    entry.disc = new THREE.CircleGeometry(0.58, 32).rotateX(-Math.PI / 2);
    entry.tail = box(0.006, 0.2, 0.03, 0.022, 0.1, 0.6);
  }
  return (cache[kind] = entry);
}

/** One aircraft: a group to place, turn and scale, with its own material so it can take a colour. */
export function makeAircraft(type) {
  const kind = KEY[type] || "lead", s = shapes(kind);
  const group = new THREE.Group();
  const material = new THREE.MeshLambertMaterial({ color: 0x8fa0b3, emissive: 0x1a2530 });
  const lines = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 });
  group.add(new THREE.Mesh(s.solid, material), new THREE.LineSegments(s.edges, lines));
  let rotor = null;
  if (kind === "helo") {
    rotor = new THREE.Group();
    rotor.position.set(0, 0.135, -0.06);
    rotor.add(new THREE.Mesh(s.rotor, material), new THREE.Mesh(s.disc, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.1, side: THREE.DoubleSide, depthWrite: false })));
    group.add(rotor, new THREE.Mesh(s.tail, material));
  }
  return { group, material, lines, rotor, kind, px: LENGTH_PX[kind], bank: 0, pitch: 0, dispose: () => (material.dispose(), lines.dispose(), rotor?.children[1].material.dispose()) };
}

/** Free the shared geometry when the map goes away. */
export function disposeAircraft() {
  for (const key of Object.keys(cache)) {
    for (const g of Object.values(cache[key])) g.dispose();
    delete cache[key];
  }
}
