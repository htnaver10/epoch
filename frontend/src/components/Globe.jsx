import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { feature, mesh } from "topojson-client";
import land from "world-atlas/land-50m.json";
import { C, mapRamp, OPTICAL_BOUNDS, opticalUrl, basemapUrl, sentinelUrl } from "../lib/theme";
// country borders (1:50m over North America, 1:110m elsewhere) and provincial and state lines,
// from Natural Earth (public domain)
import borders from "../lib/borders.json";
import { rawWeight } from "../lib/analysis";
import { GRID, formatCoords } from "../lib/geo";
import { BASE_REM } from "../lib/hooks";

const DEG = Math.PI / 180;
const FOV = 35;
const TAN = Math.tan((FOV / 2) * DEG);
const EARTH_KM = 6371;
const MIN_DISTANCE = 1.0025, MAX_DISTANCE = 7; // down to about 16 km up, where the 10 m imagery is still sharp
const reduceMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

/** Latitude / longitude on a sphere of radius r, matching three's equirectangular UVs. */
function toVec(lat, lon, r = 1, out = new THREE.Vector3()) {
  const phi = (90 - lat) * DEG, theta = (lon + 180) * DEG;
  return out.set(-r * Math.sin(phi) * Math.cos(theta), r * Math.cos(phi), r * Math.sin(phi) * Math.sin(theta));
}
function fromVec(v) {
  const r = v.length();
  let lon = Math.atan2(v.z, -v.x) / DEG - 180;
  if (lon < -180) lon += 360;
  return { lat: 90 - Math.acos(Math.max(-1, Math.min(1, v.y / r))) / DEG, lon, d: r };
}

/** Camera distance at which a zone of `km` radius fills about a third of the view. */
const distanceFor = (km, floor = 70) => 1 + (4.5 * Math.max(km, floor)) / EARTH_KM / TAN;

/** Camera position that frames a lat/lon rectangle, leaving room for the HUD. */
function frame(bounds, aspect, centered = false) {
  const [[s, w], [n, e]] = bounds;
  const lat = (s + n) / 2 - (centered ? 0 : 2), lon = (w + e) / 2;
  const c = toVec(lat, lon), east = new THREE.Vector3(0, 1, 0).cross(c).normalize(), north = c.clone().cross(east);
  const p = new THREE.Vector3(), perp = new THREE.Vector3();
  let d = MIN_DISTANCE;
  const fit = (la, lo) => {
    toVec(la, lo, 1, p);
    const cos = p.dot(c);
    perp.copy(p).addScaledVector(c, -cos);
    d = Math.max(d, cos + Math.max(Math.abs(perp.dot(east)) / (TAN * aspect * 0.76), Math.abs(perp.dot(north)) / (TAN * 0.62)));
  };
  for (let lo = w; lo <= e; lo += 5) {
    fit(s, lo);
    fit(n, lo);
  }
  for (let la = s; la <= n; la += 5) {
    fit(la, w);
    fit(la, e);
  }
  if (centered) for (const [la, lo] of [[s,w],[s,e],[n,w],[n,e]]) fit(la, lo);
  return { lat, lon, d: Math.min(MAX_DISTANCE, d) };
}

/** Line segments along a path of [lat, lon] points, slightly above the surface. */
function pathSegments(points, r, out) {
  const a = new THREE.Vector3(), b = new THREE.Vector3();
  for (let i = 1; i < points.length; i++) {
    toVec(points[i - 1][0], points[i - 1][1], r, a);
    toVec(points[i][0], points[i][1], r, b);
    out.push(a.x, a.y, a.z, b.x, b.y, b.z);
  }
}
const along = (lat0, lon0, lat1, lon1, step = 1) => {
  const n = Math.max(1, Math.ceil(Math.max(Math.abs(lat1 - lat0), Math.abs(lon1 - lon0)) / step));
  return Array.from({ length: n + 1 }, (_, i) => [lat0 + ((lat1 - lat0) * i) / n, lon0 + ((lon1 - lon0) * i) / n]);
};
const rectangle = ([[s, w], [n, e]]) => [...along(s, w, s, e), ...along(s, e, n, e), ...along(n, e, n, w), ...along(n, w, s, w)];
const lines = (positions, color, opacity = 1) =>
  new THREE.LineSegments(
    new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(positions, 3)),
    new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity, depthWrite: false }),
  );

/** Land and ocean painted on a canvas: the base map needs no network and no imagery. */
function landTexture(size) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size / 2;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = C.ocean;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = C.land;
  const x = (lon) => ((lon + 180) / 360) * canvas.width, y = (lat) => ((90 - lat) / 180) * canvas.height;
  const geometry = feature(land, land.objects.land).features?.[0]?.geometry || feature(land, land.objects.land).geometry;
  const polygons = geometry.type === "MultiPolygon" ? geometry.coordinates : [geometry.coordinates];
  for (const polygon of polygons) {
    ctx.beginPath();
    for (const ring of polygon) {
      ring.forEach(([lon, lat], i) => (i ? ctx.lineTo(x(lon), y(lat)) : ctx.moveTo(x(lon), y(lat))));
      ctx.closePath();
    }
    ctx.fill("evenodd");
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  return texture;
}

const EARTH_VERT = `
  varying vec2 vUv; varying vec3 vNormal; varying vec3 vView;
  void main() {
    vUv = uv;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vView = normalize(-mv.xyz);
    gl_Position = projectionMatrix * mv;
  }`;
// Satellite imagery, toned down a little so red hotspots stay the brightest thing on the map.
const GRADE = `
  vec3 grade(vec3 c) {
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    return mix(vec3(l), c, 0.9) * 0.86;
  }`;
const EARTH_FRAG = GRADE + `
  uniform sampler2D map; uniform float uImage; varying vec2 vUv; varying vec3 vNormal; varying vec3 vView;
  void main() {
    vec3 c = texture2D(map, vUv).rgb;
    c = mix(c, grade(c), uImage);
    float facing = max(dot(vNormal, vView), 0.0);
    c *= mix(0.3, 1.0, pow(facing, 0.6));          // darken toward the limb
    c += vec3(0.04, 0.09, 0.15) * pow(1.0 - facing, 3.5); // thin horizon haze
    gl_FragColor = vec4(c, 1.0);
    #include <colorspace_fragment>
  }`;
/** A lat/lon rectangle as a curved patch just above the surface, with 0-1 UVs across it. */
function patchGeometry([[s, w], [n, e]], r = 1.0009, nx = 96, ny = 32) {
  const pos = [], uvs = [], index = [];
  for (let j = 0; j <= ny; j++)
    for (let i = 0; i <= nx; i++) {
      const v = toVec(s + ((n - s) * j) / ny, w + ((e - w) * i) / nx, r);
      pos.push(v.x, v.y, v.z);
      uvs.push(i / nx, j / ny);
    }
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const a = j * (nx + 1) + i, b = a + 1, c = a + nx + 1, d = c + 1;
      index.push(a, b, c, b, d, c);
    }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  g.setIndex(index);
  return g;
}

/**
 * Load an image as a texture without stalling the frame loop: decoded off the main thread
 * (createImageBitmap) where the browser can, so the globe keeps turning while a large image
 * arrives. Calls back with null on failure. Returns a cancel function.
 */
function loadTexture(url, done) {
  let cancelled = false;
  const finish = (texture) => {
    if (cancelled) return texture?.dispose();
    if (texture) {
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.anisotropy = 8;
      texture.needsUpdate = true;
    }
    done(texture);
  };
  const fallback = () => new THREE.TextureLoader().setCrossOrigin("anonymous").load(url, finish, undefined, () => finish(null));
  if (typeof createImageBitmap === "function" && typeof THREE.ImageBitmapLoader === "function") {
    const loader = new THREE.ImageBitmapLoader().setCrossOrigin("anonymous").setOptions({ imageOrientation: "flipY" });
    loader.load(url, (bitmap) => {
      const texture = new THREE.Texture(bitmap);
      texture.flipY = false; // flipped while decoding
      finish(texture);
    }, undefined, fallback);
  } else fallback();
  return () => (cancelled = true);
}

// Imagery draped over the globe: the vertex half, shared by every tile.
const DRAPE_VERT = `
  varying vec2 vUv; varying vec3 vNormal; varying vec3 vView;
  void main() {
    vUv = uv;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normalize(position));
    vView = normalize(-mv.xyz);
    gl_Position = projectionMatrix * mv;
  }`;
// An imagery tile: graded and shaded like the globe beneath, with no feathering, so
// neighbouring tiles meet edge to edge.
const TILE_FRAG = GRADE + `
  uniform sampler2D map; uniform float uOpacity; varying vec2 vUv; varying vec3 vNormal; varying vec3 vView;
  void main() {
    vec3 c = grade(texture2D(map, vUv).rgb);
    float facing = max(dot(vNormal, vView), 0.0);
    c *= mix(0.3, 1.0, pow(facing, 0.6));
    gl_FragColor = vec4(c, uOpacity);
    #include <colorspace_fragment>
  }`;
const ATMOSPHERE_FRAG = `
  varying vec3 vNormal;
  void main() {
    float i = pow(max(0.0, 0.66 - dot(vNormal, vec3(0.0, 0.0, 1.0))), 4.0);
    gl_FragColor = vec4(0.3, 0.46, 0.66, 1.0) * i * 0.7;
  }`;
const ATMOSPHERE_VERT = `
  varying vec3 vNormal;
  void main() { vNormal = normalize(normalMatrix * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;

// Hotspots: one symbol per grid cell. On the harmonized layers every cell is the same plain
// square, so the map reads as where and how long it burned and nothing else. On the
// disagreement layer the shape says which sensor saw the cell:
//   diamond = both sensors on the same day    triangle = MODIS only
//   ring    = VIIRS only                      open diamond = both, but on different days
// Colour and size say how long it burned (the fire ramp), so the map reads as heat from far
// away and as evidence up close. On the disagreement layer colour is the sensor instead.
// The cell's true 0.25 degree footprint is drawn when it is selected.
// Sizes are in design pixels (the layout's 13 px rem at 1x): uUnit converts them to the
// device pixels gl_PointSize counts in, so a mark is as large on a dense display as on a plain one.
const SPOT_VERT = `
  attribute vec3 color; attribute float weight; attribute float kind;
  uniform float uScale; uniform float uCell; uniform float uUnit;
  varying vec3 vColor; varying float vKind;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float cell = uCell * uScale / -mv.z;
    // the cell it stands for up close, but never too small for its shape to read, busier cells a
    // little bigger; only the uncorrected view passes a weight above 1, and there a mark may outgrow its cell.
    // Where the cells are packed tighter than the marks (a phone, or far out) the floor gives way, so the
    // theatre does not fill in solid.
    float crowd = clamp(cell / (4.3 * uUnit), 0.6, 1.0);
    gl_PointSize = kind > 3.5
      // harmonized layers: small dots, as on NASA FIRMS, so the ground shows between them up close
      ? clamp(cell * 0.5 * max(1.0, weight), (4.5 + 4.0 * weight) * uUnit * crowd, (8.0 + 8.0 * weight) * uUnit)
      : clamp(cell * 0.95 * max(1.0, weight), (7.0 + 6.0 * weight) * uUnit * crowd, (14.0 + 12.0 * weight) * uUnit);
    vColor = color; vKind = kind;
    gl_Position = projectionMatrix * mv;
  }`;
const SPOT_UNIT = (rem, pixelRatio) => (rem / BASE_REM) * pixelRatio;
const SPOT_FRAG = `
  varying vec3 vColor; varying float vKind;
  uniform float uOpacity;
  float triangle(vec2 p, float r) {
    const float k = 1.7320508;
    p.x = abs(p.x) - r; p.y = p.y + r / k;
    if (p.x + k * p.y > 0.0) p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
    p.x -= clamp(p.x, -2.0 * r, 0.0);
    return -length(p) * sign(p.y);
  }
  void main() {
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    q.y = -q.y;
    float d; // signed distance to the symbol's edge
    if (vKind < 0.5) d = abs(q.x) + abs(q.y) - 0.66;
    else if (vKind < 1.5) d = triangle(q + vec2(0.0, 0.08), 0.6);
    else if (vKind < 2.5) d = abs(length(q) - 0.5) - 0.15;
    else if (vKind < 3.5) d = abs(abs(q.x) + abs(q.y) - 0.56) - 0.13;
    else d = length(q) - 0.74; // plain round dot, as on NASA FIRMS: a cell, whichever sensor saw it
    float fill = 1.0 - smoothstep(-0.05, 0.05, d);
    // dark outline: legible on cloud and on neighbours; a hairline on the small round dots
    float halo = vKind > 3.5 ? 1.0 - smoothstep(0.06, 0.18, d) : 1.0 - smoothstep(0.12, 0.26, d);
    float a = max(fill, halo * 0.85);
    if (a < 0.02) discard;
    gl_FragColor = vec4(mix(vec3(0.0), vColor, fill), a * uOpacity);
    #include <colorspace_fragment>
  }`;

/** A gazetteer label on the globe. Mouse only: the command line is the keyboard route to places. */
function PlaceLabel({ place, onPlace }) {
  const kind = place.kind === "town" || place.kind === "city" ? "point" : place.kind === "province" || place.kind === "territory" || place.kind === "region" ? "area" : "feature";
  return (
    <button tabIndex={-1} data-lat={place.lat} data-lon={place.lon} data-solid={kind} data-w={(place.name.length * (kind === "area" ? 0.9 : 0.59) + 1.5).toFixed(2)} className={`place place-${kind}`} onClick={() => onPlace?.(place)}>
      <span>{place.name}</span>
    </button>
  );
}

// Sensor agreement for a merged cell [lat, lon, modis, viirs, same-day days, active days]:
// 0 both on the same day, 1 MODIS only, 2 VIIRS only, 3 both but on different days.
const kindOf = (p) => (p[4] ? 0 : p[2] && p[3] ? 3 : p[2] ? 1 : 2);
const SENSOR_COLORS = [C.text, C.modis, C.viirs, C.split];
const PLAIN = 4; // the round dot drawn on the harmonized layers
const BURN_DAYS = 45; // days burning at which a cell reaches the hot end of the ramp
const IMAGE_TIMEOUT_MS = 15000;

/**
 * The stage: a tactical three.js globe. Vector coastlines and borders over a painted base
 * map, the analysis sector, hotspots, an optional optical (satellite image) overlay, and a
 * camera that spins and zooms to whatever is focused. It owns its own per-frame HUD
 * readouts and positions the focus callout, so React never re-renders at frame rate.
 */
export default function Globe({ bounds, sectors, region, places = [], points, researchPoints = null, researchCellDegrees = 0.0034, centerBounds = false, optical, imageryDate, sensorColours = false, allDays = true, playing, focus, trail = [], fit, inset = 0, reserve = 0, reserveBelow = 0, edgeRight = 0, edgeTop = 0, raw = false, rem = 13, imageAttempt = 0, onImage, onPick, onSector, onPlace, onTrail, callout }) {
  const mount = useRef(null);
  const readout = useRef(null);
  const creditBox = useRef(null);
  const calloutBox = useRef(null);
  const labelBox = useRef(null);
  const placeBox = useRef(null);
  const trailBox = useRef(null);
  const api = useRef(null);
  const handlers = useRef({ onPick, onSector, onImage });
  const remPx = useRef(rem);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    handlers.current = { onPick, onSector, onImage };
    remPx.current = rem;
    api.current?.setRem(rem);
  }, [onPick, onSector, onImage, rem]);

  useEffect(() => {
    const el = mount.current;
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    } catch {
      setFailed(true); // no WebGL: the panels still work without the globe
      return undefined;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    // Two cameras: `base` is the one the controls and flights move, always looking straight down at
    // the globe; `camera` is what is drawn, the base tilted toward the horizon when 3D is on.
    const base = new THREE.PerspectiveCamera(FOV, 1, 0.01, 200);
    const camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 200);
    const still = reduceMotion();
    const disposables = [];
    const keep = (o) => (disposables.push(o), o);
    const cancels = []; // image loads still in flight when the globe goes away
    // fades: imagery eases in rather than popping, and the old picture stays until the new one is ready
    const fades = new Set();
    const fadeTo = (material, goal) => {
      material.userData.goal = goal;
      fades.add(material);
      dirty = true;
    };

    // --- base globe
    const texture = keep(landTexture(Math.min(8192, renderer.capabilities.maxTextureSize)));
    const earth = new THREE.Mesh(
      keep(new THREE.SphereGeometry(1, 160, 80)),
      keep(new THREE.ShaderMaterial({ uniforms: { map: { value: texture }, uImage: { value: 0 } }, vertexShader: EARTH_VERT, fragmentShader: EARTH_FRAG })),
    );
    scene.add(earth);
    // a summer true-colour Earth, bundled (derived from NASA Blue Marble): the painted land stands in
    // until it arrives, or if it cannot load
    cancels.push(loadTexture(`${import.meta.env.BASE_URL}data/blue-marble.jpg`, (image) => {
      if (!image) return;
      keep(image);
      earth.material.uniforms.map.value = image;
      earth.material.uniforms.uImage.value = 1;
      dirty = true;
    }));
    scene.add(new THREE.Mesh(
      keep(new THREE.SphereGeometry(1.16, 64, 32)),
      keep(new THREE.ShaderMaterial({ vertexShader: ATMOSPHERE_VERT, fragmentShader: ATMOSPHERE_FRAG, side: THREE.BackSide, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false })),
    ));
    const starPositions = [];
    let seed = 7;
    const random = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
    for (let i = 0; i < 500; i++) {
      const v = new THREE.Vector3(random() * 2 - 1, random() * 2 - 1, random() * 2 - 1).normalize().multiplyScalar(60);
      starPositions.push(v.x, v.y, v.z);
    }
    const stars = new THREE.Points(
      keep(new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(starPositions, 3))),
      keep(new THREE.PointsMaterial({ color: 0x9fb4c8, size: 1.2, sizeAttenuation: false, transparent: true, opacity: 0.4, depthWrite: false })),
    );
    scene.add(stars);

    // --- vector layers
    const graticule = [];
    for (let lat = -80; lat <= 80; lat += 10) pathSegments(along(lat, -180, lat, 180, 3), 1.0002, graticule);
    for (let lon = -180; lon < 180; lon += 10) pathSegments(along(-80, lon, 80, lon, 3), 1.0002, graticule);
    const coast = [], border = [];
    for (const line of mesh(land, land.objects.land).coordinates) pathSegments(line.map(([lon, lat]) => [lat, lon]), 1.0003, coast);
    for (const line of borders) pathSegments(line.map(([lon, lat]) => [lat, lon]), 1.0003, border);
    // as on NASA FIRMS: thin grey coasts and borders over the imagery, and only a whisper of grid
    const vectors = [lines(graticule, C.graticule, 0.22), lines(border, C.border, 0.38), lines(coast, C.coast, 0.42)];
    for (const v of vectors) scene.add(keep(v.geometry) && keep(v.material) && v);

    // --- optical overlay: a satellite image of the theatre for one day
    const patch = keep(patchGeometry(OPTICAL_BOUNDS, 1.00018, 192, 64));
    const opticalMaterial = keep(new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }));
    const opticalMesh = new THREE.Mesh(patch, opticalMaterial);
    opticalMesh.visible = false;
    opticalMesh.renderOrder = 1;
    scene.add(opticalMesh);
    // --- imagery tiles: sharp at every zoom, the way a map app or Google Earth does it. The globe
    // is cut into a quadtree of geographic tiles (level z: 180/2^z degrees a side, 512 px each).
    // Each frame the tiles in view are refined until one image pixel is no bigger than a screen
    // pixel, so the picture stays crisp from orbit down to 16 km. Sentinel-2 cloudless (10 m)
    // where it answers, NASA Blue Marble (500 m) if not. A tile still loading is covered by its
    // nearest loaded ancestor, and every tile fades in over what was there: the map never blanks.
    const TILE_PX = 512, TILE_MIN = 3, TILE_CACHE = 120, TILE_R = 1.0001, MAX_FLIGHT = 6;
    const tiles = new Map();
    const queue = [];
    let source = "sentinel", sentinelOk = false, sentinelFailed = 0, inFlight = 0, lodStamp = 0, needLod = true, lodAt = 0;
    const tileBounds = (z, x, y) => {
      const span = 180 / 2 ** z, n = 90 - y * span, w = -180 + x * span;
      return [[n - span, w], [n, w + span]];
    };
    const tileAt = (z, x, y) => {
      const key = `${z}/${x}/${y}`;
      let t = tiles.get(key);
      if (!t) tiles.set(key, (t = { key, z, x, y, bounds: tileBounds(z, x, y), state: "idle", mesh: null, used: 0, wanted: 0, priority: 0 }));
      return t;
    };
    const credit = () => {
      if (creditBox.current) creditBox.current.textContent = source === "sentinel" ? "Sentinel-2 cloudless by EOX (modified Copernicus Sentinel data) · NASA Blue Marble" : "NASA Blue Marble";
    };
    const dropTile = (t) => {
      t.cancel?.();
      if (t.mesh) {
        t.mesh.removeFromParent();
        t.mesh.geometry.dispose();
        t.mesh.material.uniforms.map.value?.dispose();
        t.mesh.material.dispose();
        fades.delete(t.mesh.material);
      }
      tiles.delete(t.key);
    };
    const request = (t) => {
      t.state = "loading";
      inFlight++;
      const url = source === "sentinel" ? sentinelUrl(TILE_PX, t.bounds, TILE_PX) : basemapUrl(TILE_PX, "BlueMarble_NextGeneration", t.bounds, TILE_PX);
      t.cancel = loadTexture(url, (image) => {
        inFlight--;
        t.cancel = null;
        if (!image) {
          t.state = "failed";
          // Sentinel-2 not answering at all (offline, blocked): switch every tile to NASA's
          if (source === "sentinel" && !sentinelOk && ++sentinelFailed >= 3) {
            source = "nasa";
            credit();
            for (const other of tiles.values()) if (other.state === "failed") other.state = "idle";
            needLod = true;
          }
          pump();
          return;
        }
        if (source === "sentinel") sentinelOk = true;
        if (!tiles.has(t.key)) return image.dispose();
        image.generateMipmaps = true;
        const material = new THREE.ShaderMaterial({ uniforms: { map: { value: image }, uOpacity: { value: 0 } }, vertexShader: DRAPE_VERT, fragmentShader: TILE_FRAG, transparent: true, depthWrite: false });
        material.userData.goal = 0;
        // fine enough that the patch never sags below the globe's own surface
        const span = 180 / 2 ** t.z, seg = Math.max(4, Math.min(64, Math.ceil(span / 0.35)));
        t.mesh = new THREE.Mesh(patchGeometry(t.bounds, TILE_R, seg, seg), material);
        t.mesh.renderOrder = -9 + t.z * 0.1; // finer tiles draw over coarser ones
        t.mesh.visible = false;
        material.userData.mesh = t.mesh;
        scene.add(t.mesh);
        t.state = "ready";
        needLod = true;
        dirty = true;
        pump();
      });
    };
    const pump = () => {
      while (inFlight < MAX_FLIGHT && queue.length) {
        const t = queue.shift();
        if (t.state === "idle" && t.wanted === lodStamp && tiles.has(t.key)) request(t);
      }
    };
    const frustum = new THREE.Frustum(), viewProjection = new THREE.Matrix4(), bound = new THREE.Sphere(), centre = new THREE.Vector3(), eye = new THREE.Vector3();
    const updateTiles = () => {
      lodStamp++;
      needLod = false;
      camera.updateMatrixWorld();
      viewProjection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
      frustum.setFromProjectionMatrix(viewProjection);
      const d = camera.position.length(), horizon = Math.acos(Math.min(1, 1 / d));
      eye.copy(camera.position).normalize();
      const pixel = (2 * TAN) / (size.h * Math.min(1.5, renderer.getPixelRatio())); // radians per screen pixel at unit distance
      const maxZ = source === "sentinel" ? 13 : 8;
      const shown = new Set(), wanted = [];
      const visit = (z, x, y) => {
        const t = tileAt(z, x, y), [[s, w], [n, e]] = t.bounds, span = 180 / 2 ** z;
        if (s >= 88 || n <= -88) return;
        toVec((s + n) / 2, (w + e) / 2, 1, centre);
        const radius = span * 0.72 * DEG;
        if (Math.acos(Math.max(-1, Math.min(1, centre.dot(eye)))) - radius > horizon) return;
        bound.set(centre, 2 * Math.sin(radius / 2) + 0.002);
        if (!frustum.intersectsSphere(bound)) return;
        const distance = Math.max(d - 1, camera.position.distanceTo(centre) - bound.radius);
        if (z < maxZ && (span * DEG) / TILE_PX > distance * pixel * 1.15) {
          for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) visit(z + 1, x * 2 + i, y * 2 + j);
          return;
        }
        t.used = lodStamp;
        if (t.state === "ready") return shown.add(t);
        if (t.state === "idle") {
          t.wanted = lodStamp;
          t.priority = z * 4 + distance * 40;
          wanted.push(t);
        }
        // until it arrives, the nearest ancestor that has
        for (let k = 1; z - k >= TILE_MIN; k++) {
          const a = tiles.get(`${z - k}/${x >> k}/${y >> k}`);
          if (a?.state === "ready") {
            a.used = lodStamp;
            shown.add(a);
            break;
          }
        }
      };
      for (let y = 0; y < 2 ** TILE_MIN; y++) for (let x = 0; x < 2 ** (TILE_MIN + 1); x++) visit(TILE_MIN, x, y);
      for (const t of tiles.values()) {
        if (!t.mesh) continue;
        const on = shown.has(t), material = t.mesh.material;
        if (on && material.userData.goal !== 1) {
          t.mesh.visible = true;
          fadeTo(material, 1);
        } else if (!on && material.userData.goal !== 0) fadeTo(material, 0);
      }
      wanted.sort((a, b) => a.priority - b.priority);
      queue.length = 0;
      queue.push(...wanted);
      pump();
      // forget what has not been needed for a while
      if (tiles.size > TILE_CACHE) {
        const old = [...tiles.values()].filter((t) => t.used < lodStamp && t.state !== "loading").sort((a, b) => a.used - b.used);
        for (const t of old.slice(0, tiles.size - TILE_CACHE)) dropTile(t);
      }
    };
    cancels.push(() => {
      for (const t of [...tiles.values()]) dropTile(t);
    });

    // --- sector outline, hotspots, focus reticle (filled in by the prop effects)
    const sectorGroup = new THREE.Group(), reticle = new THREE.Group();
    sectorGroup.renderOrder = 2;
    scene.add(sectorGroup, reticle);
    const spotMaterial = keep(new THREE.ShaderMaterial({
      uniforms: { uScale: { value: 1 }, uCell: { value: GRID * DEG }, uUnit: { value: 1 }, uOpacity: { value: 0.95 } },
      vertexShader: SPOT_VERT, fragmentShader: SPOT_FRAG, transparent: true, depthWrite: false,
    }));
    const spots = new THREE.Points(new THREE.BufferGeometry(), spotMaterial);
    spots.frustumCulled = false;
    spots.renderOrder = 3;
    scene.add(spots);

    // --- camera and controls
    const controls = new OrbitControls(base, renderer.domElement);
    controls.enablePan = false;
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = MIN_DISTANCE;
    controls.maxDistance = MAX_DISTANCE;
    controls.zoomSpeed = 0.7;
    let flight = null, size = { w: 1, h: 1 };
    // `shift` slides the globe's centre into the free part of the stage; it eases to `shiftGoal`.
    // `dirty` asks for a frame: nothing is drawn while nothing changes.
    let shift = 0, shiftGoal = 0, lift = 0, liftGoal = 0, leftEdge = 0, rightEdge = 0, topEdge = 0, dirty = true;
    controls.addEventListener("change", () => (dirty = true));
    let framed = null, centeredFrame = false; // bounds framed until the user moves the camera
    controls.addEventListener("start", () => (framed = null));
    const place = ({ lat, lon, d }) => {
      toVec(Math.max(-85, Math.min(85, lat)), lon, d, base.position);
      base.lookAt(0, 0, 0);
    };
    const flyTo = (to, seconds) => {
      const from = fromVec(base.position);
      let dLon = to.lon - from.lon;
      if (dLon > 180) dLon -= 360;
      if (dLon < -180) dLon += 360;
      if (still) {
        place(to);
        controls.update();
        return;
      }
      const angle = Math.hypot(to.lat - from.lat, dLon * Math.cos(((to.lat + from.lat) / 2) * DEG));
      const duration = seconds ?? Math.min(3.4, 1.3 + angle / 40 + Math.abs(Math.log(to.d - 0.98) - Math.log(from.d - 0.98)) * 0.32);
      // rise away from the surface mid-flight, more for longer hops: spin, then dive
      const lift = Math.min(1.1, angle / 70) * Math.max(0.15, Math.min(from.d, to.d) - 0.9);
      flight = { from, to, dLon, lift, start: performance.now(), duration: duration * 1000 };
      controls.enabled = false;
    };
    const cancelFlight = () => {
      if (!flight) return;
      flight = null;
      controls.enabled = true;
    };

    const applyShift = () => {
      if (Math.abs(shift) > 0.5 || Math.abs(lift) > 0.5) base.setViewOffset(size.w, size.h, -shift / 2, lift / 2, size.w, size.h);
      else base.clearViewOffset();
      base.updateProjectionMatrix();
      dirty = true;
    };
    const resize = () => {
      const w = Math.max(1, el.clientWidth), h = Math.max(1, el.clientHeight);
      size = { w, h };
      renderer.setSize(w, h, false);
      base.aspect = w / h;
      applyShift();
      spotMaterial.uniforms.uScale.value = (h * renderer.getPixelRatio()) / (2 * TAN);
      spotMaterial.uniforms.uUnit.value = SPOT_UNIT(remPx.current, renderer.getPixelRatio());
      if (framed && !flight) flyTo(frame(framed, (w - leftEdge) / h, centeredFrame), 0.5);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    resize();
    place({ lat: 18, lon: -235, d: 6.2 }); // start far out over the Pacific, then spin in

    // --- picking and the coordinate readout
    const raycaster = new THREE.Raycaster(), ndc = new THREE.Vector2(), hit = new THREE.Vector3();
    const unit = new THREE.Sphere(new THREE.Vector3(), 1);
    const locate = (event) => {
      const box = renderer.domElement.getBoundingClientRect();
      ndc.set(((event.clientX - box.left) / box.width) * 2 - 1, -((event.clientY - box.top) / box.height) * 2 + 1);
      raycaster.setFromCamera(ndc, camera);
      return raycaster.ray.intersectSphere(unit, hit) ? fromVec(hit) : null;
    };
    let down = null, cursor = null;
    const onDown = (e) => {
      down = e.button === 0 && !e.shiftKey ? { x: e.clientX, y: e.clientY } : null;
      cancelFlight();
    };
    const onUp = (e) => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return;
      const at = locate(e);
      // how many km one screen pixel covers under the cursor: the click tolerance
      const kmPerPixel = ((base.position.length() - 1) * 2 * TAN * EARTH_KM) / size.h;
      if (at) handlers.current.onPick?.({ lat: at.lat, lon: at.lon, kmPerPixel });
    };
    const onMove = (e) => (cursor = locate(e));
    const onLeave = () => (cursor = null);

    // Zoom like a map: the wheel eases the height up or down by a share of itself, so steps feel
    // the same from orbit and from 20 km, and zooming in moves toward the spot under the cursor.
    let zoomGoal = null;
    const wrapLon = (v) => (v > 180 ? v - 360 : v < -180 ? v + 360 : v);
    const zoomTo = (k, at = null) => {
      cancelFlight();
      framed = null;
      const cur = zoomGoal || fromVec(base.position);
      const h = cur.d - 1, nh = Math.min(MAX_DISTANCE - 1, Math.max(MIN_DISTANCE - 1, h * k));
      let { lat, lon } = cur;
      if (at && nh < h) {
        const f = (1 - nh / h) * 0.95;
        lat += (at.lat - lat) * f;
        lon += wrapLon(at.lon - lon) * f;
      }
      zoomGoal = { lat, lon, d: 1 + nh };
      dirty = true;
    };
    const onWheel = (e) => {
      e.preventDefault();
      e.stopPropagation(); // the orbit controls' own zoom jumps too far near the ground
      const dy = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY;
      zoomTo(Math.exp(Math.max(-240, Math.min(240, dy)) * 0.0028), dy < 0 ? locate(e) : null);
    };
    el.addEventListener("wheel", onWheel, { capture: true, passive: false });
    controls.addEventListener("start", () => (zoomGoal = null));

    // Tilt, as in Google Earth: right-drag (or shift-drag) up and down, or the 3D button.
    let tilting = null, tilt = 0, tiltGoal = 0;
    const onTiltDown = (e) => {
      if (e.button !== 2 && !(e.button === 0 && e.shiftKey)) return;
      tilting = e.clientY;
      controls.enabled = false;
    };
    const onTiltMove = (e) => {
      if (tilting === null) return;
      tiltGoal = Math.max(0, Math.min(65, tiltGoal - (e.clientY - tilting) * 0.35));
      tilting = e.clientY;
      dirty = true;
    };
    const onTiltUp = () => {
      if (tilting === null) return;
      tilting = null;
      if (!flight) controls.enabled = true;
    };
    const noMenu = (e) => e.preventDefault();
    el.addEventListener("pointerdown", onTiltDown, { capture: true });
    window.addEventListener("pointermove", onTiltMove);
    window.addEventListener("pointerup", onTiltUp);
    el.addEventListener("contextmenu", noMenu);
    const dom = renderer.domElement;
    dom.addEventListener("pointerdown", onDown);
    dom.addEventListener("pointerup", onUp);
    dom.addEventListener("pointermove", onMove);
    dom.addEventListener("pointerleave", onLeave);
    dom.addEventListener("wheel", cancelFlight, { passive: true });

    // --- frame loop
    const anchor = { lat: 0, lon: 0, km: 0, on: false };
    const projected = new THREE.Vector3(), normal = new THREE.Vector3(), toCamera = new THREE.Vector3();
    const project = (lat, lon) => {
      toVec(lat, lon, 1, normal);
      toCamera.copy(camera.position).sub(normal).normalize();
      const facing = normal.dot(toCamera);
      const depth = camera.position.distanceTo(normal);
      projected.copy(normal).project(camera);
      return { x: (projected.x * 0.5 + 0.5) * size.w, y: (-projected.y * 0.5 + 0.5) * size.h, facing, depth };
    };
    // The drawn camera: the base camera, tilted about the point below it when 3D is on (less and less
    // as it climbs, none from orbit), with a near plane that follows it down to the ground.
    const up = new THREE.Vector3(), northward = new THREE.Vector3(), eastward = new THREE.Vector3();
    const syncView = () => {
      camera.copy(base);
      const h = base.position.length() - 1;
      const angle = tilt * Math.max(0, Math.min(1, (0.9 - h) / 0.6)) * DEG;
      if (angle > 0.002) {
        up.copy(base.position).normalize();
        eastward.set(0, 1, 0).cross(up).normalize();
        northward.copy(up).cross(eastward);
        camera.position.copy(up).addScaledVector(up, h * Math.cos(angle)).addScaledVector(northward, -h * Math.sin(angle));
        camera.up.copy(up);
        camera.lookAt(up);
      }
      camera.near = Math.max(0.00005, (camera.position.length() - 1) * 0.25);
      camera.far = 80;
      camera.updateProjectionMatrix();
      camera.updateMatrixWorld();
    };
    let raf = 0, last = "";
    const tick = (now) => {
      raf = requestAnimationFrame(tick);
      if (flight) {
        const t = Math.min(1, (now - flight.start) / flight.duration), k = ease(t);
        const { from, to, dLon, lift } = flight;
        place({
          lat: from.lat + (to.lat - from.lat) * k,
          lon: from.lon + dLon * k,
          d: Math.exp(Math.log(from.d - 0.98) + (Math.log(to.d - 0.98) - Math.log(from.d - 0.98)) * k) + 0.98 + lift * Math.sin(Math.PI * t),
        });
        dirty = true;
        if (t >= 1) {
          flight = null;
          controls.enabled = true;
          controls.update();
        }
      } else {
        const d = base.position.length();
        controls.rotateSpeed = Math.max(0.03, Math.min(0.7, (d - 1) * 0.45)); // finer control up close
        controls.update();
      }
      if (Math.abs(shiftGoal - shift) > 0.5 || Math.abs(liftGoal - lift) > 0.5) {
        shift += (shiftGoal - shift) * (still ? 1 : 0.14);
        lift += (liftGoal - lift) * (still ? 1 : 0.14);
        applyShift();
      }
      // the coordinate readout follows the pointer even when the picture is still
      const text = cursor ? formatCoords(cursor.lat, cursor.lon) : "";
      if (text !== last && readout.current) readout.current.textContent = last = text;
      if (zoomGoal && !flight) {
        const cur = fromVec(base.position), k = still ? 1 : 0.2;
        const dLon = wrapLon(zoomGoal.lon - cur.lon), h0 = cur.d - 1, h1 = zoomGoal.d - 1;
        const h = Math.exp(Math.log(h0) + (Math.log(h1) - Math.log(h0)) * k);
        place({ lat: cur.lat + (zoomGoal.lat - cur.lat) * k, lon: cur.lon + dLon * k, d: 1 + h });
        if (Math.abs(Math.log(h1 / h)) < 0.003 && Math.abs(zoomGoal.lat - cur.lat) + Math.abs(dLon) < 1e-4) zoomGoal = null;
        dirty = true;
      }
      if (tilt !== tiltGoal) {
        tilt = Math.abs(tiltGoal - tilt) < 0.05 || still ? tiltGoal : tilt + (tiltGoal - tilt) * 0.12;
        dirty = true;
      }
      if (needLod && now - lodAt > 100) dirty = true;
      for (const m of fades) {
        const u = m.uniforms.uOpacity, goal = m.userData.goal;
        // in quickly, out slowly: what is replaced stays under its replacement until that is opaque
        const step = goal > u.value ? 0.08 : 0.03;
        u.value = still ? goal : u.value + Math.sign(goal - u.value) * Math.min(Math.abs(goal - u.value), step);
        if (u.value === goal) {
          fades.delete(m);
          if (goal === 0 && m.userData.mesh) m.userData.mesh.visible = false;
        }
        dirty = true;
      }
      if (!dirty) return;
      dirty = false;
      syncView();
      if (now - lodAt > 100) {
        lodAt = now;
        updateTiles();
      } else needLod = true;
      stars.position.copy(camera.position);
      renderer.render(scene, camera);

      // HUD: callout position, sector labels, coordinate readout
      const box = calloutBox.current;
      let cardBox = null; // where the card sits this frame, so map labels under it can be hidden
      if (box) {
        if (anchor.on) {
          const p = project(anchor.lat, anchor.lon);
          const show = p.facing > 0.12;
          box.style.visibility = show ? "visible" : "hidden";
          // the ring and its distance label travel with the camera; the card pops out on arrival
          const card = box.lastElementChild;
          if (flight) box.dataset.flying = "1";
          else if (box.dataset.flying) {
            delete box.dataset.flying;
            // now that it can be seen, the card takes the keyboard focus it could not take while hidden
            if (document.activeElement === document.body) card?.focus?.({ preventScroll: true });
          }
          const cw = card ? card.offsetWidth : 304, ch = card ? card.offsetHeight : 320;
          // the ring's radius on screen: the card stands clear of it instead of landing on it
          const ring = Math.min(size.h * 0.3, ((Math.max(anchor.km, 22) / EARTH_KM) * size.h) / (2 * TAN * p.depth));
          const gap = ring + 26;
          box.style.setProperty("--ring", `${Math.round(ring)}px`);
          const limit = size.w - rightEdge - 10; // the stage's free right edge
          const fitsRight = p.x + gap + cw <= limit, fitsLeft = p.x - gap - cw >= leftEdge + 10;
          const side = fitsRight || !fitsLeft ? "right" : "left";
          box.dataset.side = side;
          // slide the card up or down its leader line, and sideways if neither side has room,
          // so it never leaves the stage
          // (and under whatever is folded along the top; a card taller than the room scrolls)
          box.style.setProperty("--room", `${Math.round(size.h - topEdge - 20)}px`);
          const lift = Math.max(10 + topEdge - p.y, Math.min(-34, size.h - 10 - ch - p.y));
          const x0 = side === "right" ? p.x + gap : p.x - gap - cw;
          const nudge = Math.max(Math.min(leftEdge + 10, limit - cw), Math.min(limit - cw, x0)) - x0;
          box.style.setProperty("--lift", `${Math.round(lift)}px`);
          box.style.setProperty("--nudge", `${Math.round(side === "right" ? nudge : -nudge)}px`);
          box.style.transform = `translate(${Math.round(p.x)}px, ${Math.round(p.y)}px)`;
          if (show && !flight && card) cardBox = { x0: x0 + nudge - 6, x1: x0 + nudge + cw + 6, y0: p.y + lift - 6, y1: p.y + lift + ch + 6 };
        } else box.style.visibility = "hidden";
      }
      for (const group of [labelBox.current, trailBox.current]) {
        if (!group) continue;
        for (const node of group.children) {
          const p = project(Number(node.dataset.lat), Number(node.dataset.lon));
          node.style.visibility = p.facing > 0.2 ? "visible" : "hidden";
          node.style.transform = `translate(${Math.round(p.x)}px, ${Math.round(p.y)}px) translate(-50%, -100%)`;
        }
      }
      const towns = placeBox.current;
      if (towns) {
        // place names only once the view is close enough for them to mean something
        const altitude = (base.position.length() - 1) * EARTH_KM;
        const zoom = altitude < 3400 ? "near" : altitude < 7000 ? "mid" : "far";
        if (towns.dataset.zoom !== zoom) towns.dataset.zoom = zoom;
        if (zoom !== "far") {
          const r = remPx.current, taken = []; // labels already placed this frame: later ones that would collide are dropped
          // up close the towns claim their room first; from the middle distance only the areas show
          for (const wanted of zoom === "near" ? ["point", "feature", "area"] : ["area"]) {
            for (const node of towns.children) {
              const kind = node.dataset.solid;
              if (kind !== wanted) continue;
              const p = project(Number(node.dataset.lat), Number(node.dataset.lon));
              let show = p.facing > 0.25 && p.x > leftEdge - 40 && p.x < size.w + 40 && p.y > -20 && p.y < size.h + 20;
              if (show) {
                const w = Number(node.dataset.w) * r, x0 = kind === "point" ? p.x - 6 : p.x - w / 2;
                // not where another label already is, and not under the target card
                if (taken.some((t) => x0 < t.x1 + r && x0 + w > t.x0 - r && Math.abs(p.y - t.y) < 2.1 * r)) show = false;
                else if (cardBox && x0 < cardBox.x1 && x0 + w > cardBox.x0 && p.y > cardBox.y0 && p.y < cardBox.y1 + r) show = false;
                else taken.push({ x0, x1: x0 + w, y: p.y });
              }
              node.style.visibility = show ? "" : "hidden";
              if (show) node.style.transform = `translate(${Math.round(p.x)}px, ${Math.round(p.y)}px)`;
            }
          }
        }
      }
    };
    raf = requestAnimationFrame(tick);

    api.current = {
      scene, camera, flyTo, frame: (b, centered = false) => frame(b, (size.w - leftEdge) / size.h, centered), anchor, sectorGroup, reticle, spots, opticalMesh, opticalMaterial, first: true,
      // room the HUD takes on the left, the report card on the right, and (phones) a sheet below
      setShift(left, right, below, docked = 0, top = 0) {
        leftEdge = left;
        rightEdge = docked;
        topEdge = top;
        dirty = true;
        shiftGoal = left - right;
        liftGoal = below;
        if (this.first) {
          shift = shiftGoal; // no slide on the first frame
          lift = liftGoal;
          applyShift();
        }
      },
      invalidate: () => (dirty = true),
      zoomBy: (k) => zoomTo(k),
      tilt: () => ((tiltGoal = tiltGoal > 0 ? 0 : 50), (dirty = true)),
      setRem: (rem) => ((spotMaterial.uniforms.uUnit.value = SPOT_UNIT(rem, renderer.getPixelRatio())), (dirty = true)),
      setFramed: (b, centered = false) => { framed = b; centeredFrame = centered; },
      current: () => fromVec(base.position),
    };
    return () => {
      for (const cancel of cancels) cancel();
      cancelAnimationFrame(raf);
      observer.disconnect();
      controls.dispose();
      dom.removeEventListener("pointerdown", onDown);
      dom.removeEventListener("pointerup", onUp);
      dom.removeEventListener("pointermove", onMove);
      dom.removeEventListener("pointerleave", onLeave);
      dom.removeEventListener("wheel", cancelFlight);
      el.removeEventListener("wheel", onWheel, { capture: true });
      el.removeEventListener("pointerdown", onTiltDown, { capture: true });
      window.removeEventListener("pointermove", onTiltMove);
      window.removeEventListener("pointerup", onTiltUp);
      el.removeEventListener("contextmenu", noMenu);
      for (const group of [sectorGroup, reticle]) group.traverse((o) => (o.geometry?.dispose(), o.material?.dispose()));
      spots.geometry.dispose();
      opticalMaterial.map?.dispose();
      for (const d of disposables) d.dispose();
      renderer.dispose();
      dom.remove();
      api.current = null;
    };
  }, []);

  // Keep the globe centred in the space the left-hand HUD leaves free.
  useEffect(() => {
    api.current?.setShift(inset, reserve, reserveBelow, edgeRight, edgeTop);
  }, [inset, reserve, reserveBelow, edgeRight, edgeTop]);

  // Sector outline, and the camera move that frames it.
  useEffect(() => {
    const g = api.current;
    if (!g) return;
    g.sectorGroup.traverse((o) => (o.geometry?.dispose(), o.material?.dispose()));
    g.sectorGroup.clear();
    const dim = [], bright = [];
    for (const s of Object.values(sectors)) pathSegments(rectangle(s.bounds), 1.0004, dim);
    pathSegments(rectangle(bounds), 1.00045, bright);
    g.sectorGroup.add(lines(dim, C.accent, 0.22), lines(bright, C.accent, 0.95));
    g.invalidate();
  }, [bounds, sectors]);

  useEffect(() => {
    const g = api.current;
    if (!g || focus) return;
    const first = g.first;
    g.first = false;
    g.flyTo(g.frame(bounds, centerBounds), first ? 4.6 : undefined);
    g.setFramed(bounds, centerBounds);
    // `focus` is read, not watched: clearing a focus is handled by `fit` changing
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bounds, fit, centerBounds]);

  // Focus: spin and zoom to the target, ring it, and pin the callout to it.
  useEffect(() => {
    const g = api.current;
    if (!g) return;
    g.reticle.traverse((o) => (o.geometry?.dispose(), o.material?.dispose()));
    g.reticle.clear();
    g.anchor.on = Boolean(focus);
    g.reticle.visible = Boolean(focus);
    g.invalidate();
    if (!focus) return;
    g.setFramed(null);
    g.anchor.lat = focus.lat;
    g.anchor.lon = focus.lon;
    g.anchor.km = focus.km;
    g.first = false;
    // a plain ring on the surface: the area radius, or a tight ring around a single cell
    const radius = (focus.km || 22) / EARTH_KM;
    const c = toVec(focus.lat, focus.lon), east = new THREE.Vector3(0, 1, 0).cross(c).normalize(), north = c.clone().cross(east);
    const ring = [];
    const at = (a, r) => c.clone().multiplyScalar(Math.cos(r)).addScaledVector(east, Math.cos(a) * Math.sin(r)).addScaledVector(north, Math.sin(a) * Math.sin(r)).multiplyScalar(1.0005);
    for (let i = 0; i < 96; i++) {
      const p = at((i / 96) * 2 * Math.PI, radius), q = at(((i + 1) / 96) * 2 * Math.PI, radius);
      ring.push(p.x, p.y, p.z, q.x, q.y, q.z);
    }
    const pulse = lines(ring, focus.km ? "#ffffff" : "#ffd43b", 0.95);
    g.reticle.add(pulse);
    if (!focus.km) {
      const h = GRID / 2;
      const cell = [];
      pathSegments(rectangle([[focus.lat - h, focus.lon - h], [focus.lat + h, focus.lon + h]]), 1.0005, cell);
      g.reticle.add(lines(cell, "#ffd43b", 0.8));
    }
    const here = g.current();
    // a single hotspot cell: in close, as on NASA FIRMS, with the ground around it in detail
    const wanted = focus.km ? distanceFor(focus.km) : distanceFor(5, 5);
    g.flyTo({ lat: focus.lat, lon: focus.lon, d: focus.keepZoom ? Math.min(here.d, Math.max(wanted, MIN_DISTANCE)) : wanted });
  }, [focus]);

  // Hotspots.
  useEffect(() => {
    const g = api.current;
    if (!g) return;
    const n = researchPoints ? researchPoints.length : points.length;
    const position = new Float32Array(n * 3), color = new Float32Array(n * 3), weight = new Float32Array(n), kind = new Float32Array(n);
    const v = new THREE.Vector3(), c = new THREE.Color();
    // days burning in view: the honest amount, the same for either sensor. The uncorrected
    // view draws raw detection counts instead, whichever sensor reported more.
    const weigh = (p) => (raw ? rawWeight(Math.max(p[2], p[3])) : allDays ? Math.sqrt(Math.min(1, p[5] / BURN_DAYS)) : 0.55);
    // marks overlap from far away: draw the busiest cells last, so the hot ones stay on top
    const ordered = researchPoints ? researchPoints.map((p) => [p, p.weight ?? 0.65]) : points.map((p) => [p, weigh(p)]).sort((a, b) => a[1] - b[1]);
    g.spots.material.uniforms.uCell.value = (researchPoints ? researchCellDegrees : GRID) * DEG;
    for (let i = 0; i < n; i++) {
      const [p, w] = ordered[i];
      toVec(researchPoints ? p.lat : p[0], researchPoints ? p.lon : p[1], 1.0004, v);
      position.set([v.x, v.y, v.z], i * 3);
      kind[i] = researchPoints ? PLAIN : sensorColours ? kindOf(p) : PLAIN;
      c.set(researchPoints ? p.color : sensorColours ? SENSOR_COLORS[kind[i]] : mapRamp(raw ? 0.3 + 0.7 * Math.min(1, w / 2) : 0.2 + 0.8 * w));
      color.set([c.r, c.g, c.b], i * 3);
      weight[i] = w;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(position, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(color, 3));
    geometry.setAttribute("weight", new THREE.BufferAttribute(weight, 1));
    geometry.setAttribute("kind", new THREE.BufferAttribute(kind, 1));
    g.spots.geometry.dispose();
    g.spots.geometry = geometry;
    g.invalidate();
  }, [points, researchPoints, researchCellDegrees, sensorColours, allDays, raw]);

  // Optical overlay: fetch the day's image only while it is switched on. Its state goes up
  // to the scope line, the one place that says what is on screen.
  const shownDate = useRef(null);
  useEffect(() => {
    const g = api.current;
    const report = (state, date = null) => handlers.current.onImage?.({ state, date });
    if (!g) return undefined;
    if (!optical) {
      g.opticalMesh.visible = false;
      g.invalidate();
      report("off");
      return undefined;
    }
    if (playing) {
      // hold the last frame: one image per playback step would hammer NASA
      if (g.opticalMesh.visible) report("held", shownDate.current);
      else report("paused");
      return undefined;
    }
    let cancelled = false, giveUp = 0;
    const timer = setTimeout(() => {
      report("loading", imageryDate);
      // a request NASA never answers must not leave the console saying "acquiring" for ever
      giveUp = setTimeout(() => {
        if (cancelled) return;
        cancelled = true;
        report("error", imageryDate);
      }, IMAGE_TIMEOUT_MS);
      new THREE.TextureLoader().setCrossOrigin("anonymous").load(
        opticalUrl(imageryDate),
        (texture) => {
          clearTimeout(giveUp);
          if (cancelled) return texture.dispose();
          texture.colorSpace = THREE.SRGBColorSpace;
          texture.anisotropy = 8;
          g.opticalMaterial.map?.dispose();
          g.opticalMaterial.map = texture;
          g.opticalMaterial.opacity = 0.9;
          g.opticalMaterial.needsUpdate = true;
          g.opticalMesh.visible = true;
          g.invalidate();
          shownDate.current = imageryDate;
          report("on", imageryDate);
        },
        undefined,
        () => {
          clearTimeout(giveUp);
          if (!cancelled) report("error", imageryDate);
        },
      );
    }, 180); // let a scrub or playback settle before asking NASA for a frame
    return () => {
      cancelled = true;
      clearTimeout(timer);
      clearTimeout(giveUp);
    };
  }, [optical, imageryDate, playing, imageAttempt]);

  return (
    <div className="globe-stage">
      <div ref={mount} className="globe-canvas" role="img" aria-label="Globe of fire hotspots over the analysis area. Use the search box to go to a place or coordinates." />
      {failed ? (
        <>
          <div className="globe-fallback" role="status">
            <strong>GLOBE OFFLINE</strong>
            <span>This browser could not start WebGL. The readout, the season table and every panel still work, and target reports open here without the map.</span>
          </div>
          <div className="callout-anchor docked">{callout}</div>
        </>
      ) : (
        <>
          <div ref={labelBox} className="sector-labels">
            {Object.entries(sectors).map(([key, s]) =>
              key === "canada" ? null : (
                <button key={key} data-lat={s.bounds[1][0] + 0.6} data-lon={(s.bounds[0][1] + s.bounds[1][1]) / 2} className={key === region ? "on" : undefined} onClick={() => onSector?.(key === region ? "canada" : key)}>
                  {s.short}
                </button>
              ),
            )}
          </div>
          <div ref={placeBox} className="place-labels" data-zoom="far">
            {places.map((p) => (
              <PlaceLabel key={p.name} place={p} onPlace={onPlace} />
            ))}
          </div>
          <div ref={trailBox} className="trail">
            {trail.map((t) => (
              <button key={t.key} data-lat={t.lat} data-lon={t.lon} className={focus && t.key === focus.key ? "on" : undefined} aria-label={`Return to area ${t.n}${t.label ? `, ${t.label}` : ""}`} title={t.label || "Return to this area"} onClick={() => onTrail?.(t)}>
                {t.n}
              </button>
            ))}
          </div>
          <div ref={calloutBox} className="callout-anchor" data-side="right">
            {focus && <span className="ring-label">{focus.km ? `${focus.km} km` : "0.25° cell"}</span>}
            {callout}
          </div>
          <div className="globe-readout">
            <span ref={readout} />
          </div>
          <div className="globe-nav" role="group" aria-label="Map view">
            <button onClick={() => api.current?.zoomBy(0.45)} aria-label="Zoom in" title="Zoom in (or scroll toward a spot)">+</button>
            <button onClick={() => api.current?.zoomBy(2.2)} aria-label="Zoom out" title="Zoom out">−</button>
            <button onClick={() => api.current?.tilt()} aria-label="Tilt the view" title="Tilt toward the horizon, or back to straight down. Right-drag to tilt by hand">3D</button>
          </div>
          <p className="globe-credit">
            Imagery: <span ref={creditBox}>Sentinel-2 cloudless by EOX (modified Copernicus Sentinel data) · NASA Blue Marble</span>
          </p>
        </>
      )}
    </div>
  );
}
