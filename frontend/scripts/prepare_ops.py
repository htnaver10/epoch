"""Build the bundled map data for OPS mode (the wildfire-response simulator).

    python3 -I frontend/scripts/prepare_ops.py                 # all theatres
    python3 -I frontend/scripts/prepare_ops.py --only jasper   # one theatre
    python3 -I frontend/scripts/prepare_ops.py --check         # validate the written files only

Needs: pip install osmium shapely numpy pyproj pyshp pillow

Each theatre is a 64 km square centred on `centre`, in local kilometres:

    x = (lon - lon0) * 111.320 * cos(lat0)      east positive
    y = (lat - lat0) * 110.574                  north positive

so the square is exactly a longitude/latitude rectangle and x, y lie in [-32, +32].

Everything downloaded is cached under --cache (default ~/.cache/epoch-ops) and only ever
read as data. Run with `python3 -I` so nothing next to a download can be imported.

Writes frontend/public/data/ops/<id>.json for each theatre and index.json listing them.
Polygon rings are closed (first point repeated). Rasters are row-major with row 0 at the
south edge and column 0 at the west edge.

Things worth knowing before trusting the output:
  * OSM is read as it stands today, after the fires. Features mappers tagged destroyed:* are
    put back (see before_the_fire), so the Jasper burn scar is forest and the buildings lost
    in Jasper and Enterprise are present. Anything rebuilt or newly mapped since is also there.
  * Land-cover code 0 means "no polygon in OSM", not forest. Around Yellowknife it is mostly
    bare Shield rock; in Jasper it is mostly alpine ground above the treeline plus the 6% of
    the square that lies in British Columbia (not in the Alberta extract); around Hay River
    it includes two rectangular map sheets that were never imported.
  * A cover cell takes the commonest code of its 16 sub-cells, so a river narrower than
    about 100 m is a broken line of water cells. Use `water` and `streams` for barriers.
  * Hotspot times are UTC. `perimeter` is the final NBAC burned area for the whole season,
    outer rings only, clipped to the square: far larger than the fire on the scenario date.

Sources (see SOURCES below for the attribution lines written into each file):
  OpenStreetMap extracts from Geofabrik      roads, water, land cover, buildings, places
  Terrarium elevation tiles (Mapzen / AWS)   elevation grid
  CWFIS Fire M3 hotspots (NRCan)             satellite hotspots with fire-behaviour columns
  NBAC burned-area composite (NRCan)         final fire perimeters of the scenario year
  Open-Meteo archive API (ERA5 reanalysis)   hourly weather at the town
  OurAirports                                identifiers and aerodromes missing from OSM
"""
import argparse, base64, csv, io, json, math, random, re, time, urllib.parse, urllib.request, zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import osmium
import pyproj
import shapefile
import shapely
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import linemerge, unary_union

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "frontend" / "public" / "data" / "ops"
USER_AGENT = "ops-data-prep/1.0"

SIZE_KM = 64
HALF = SIZE_KM / 2
COVER_N, COVER_CELL = 400, 0.16      # land-cover raster: 400 x 400 cells of 160 m
SUPER = 4                            # each cover cell is decided from 4 x 4 sub-cells of 40 m
ELEV_N, ELEV_CELL = 200, 0.32        # elevation raster: 200 x 200 cells of 320 m
ELEV_ZOOM = 10                       # Terrarium zoom: 72 m per pixel at 62N, 93 m at 53N
MAX_BUILDINGS = 12000
DENSE_BUILDINGS = 3                  # this many buildings in one 160 m cell makes it built-up

# Simplification tolerances and size floors, in kilometres / square kilometres.
TOL_WATER, MIN_WATER, MIN_ISLAND = 0.04, 0.02, 0.05
TOL_STREAM, TOL_ROAD, TOL_RAIL, TOL_BUILT, TOL_PERIMETER = 0.03, 0.007, 0.01, 0.02, 0.1
MAX_LAKE_LABELS = 60

GEOFABRIK = "https://download.geofabrik.de/north-america/canada/"
CWFIS = "https://cwfis.cfs.nrcan.gc.ca/downloads/"
NBAC_VERSION = "20260513"
TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
OURAIRPORTS = "https://davidmegginson.github.io/ourairports-data/"

THEATRES = [
    {"id": "yellowknife", "name": "Yellowknife", "region": "NT", "centre": (62.47, -114.62),
     "date": "2023-08-15", "utcOffset": -6, "town": ("Yellowknife", 62.454, -114.372),
     "extracts": ["northwest-territories"]},
    {"id": "hayriver", "name": "Hay River", "region": "NT", "centre": (60.70, -115.95),
     "date": "2023-08-13", "utcOffset": -6, "town": ("Hay River", 60.816, -115.783),
     "extracts": ["northwest-territories"]},
    # The south-west 6% of the Jasper square is in British Columbia. Adding "british-columbia"
    # to its extracts would map that corner, at the price of a 1.2 GB download.
    {"id": "jasper", "name": "Jasper", "region": "AB", "centre": (52.78, -117.95),
     "date": "2024-07-22", "utcOffset": -6, "town": ("Jasper", 52.873, -118.082),
     "extracts": ["alberta"]},
]

SOURCES = {
    "osm": "Map data (roads, water, land cover, buildings, places, aerodromes, stations) "
           "(c) OpenStreetMap contributors, ODbL 1.0, via Geofabrik extracts dated {osm_date}",
    "elev": "Elevation: Mapzen Terrarium terrain tiles (AWS Open Data), zoom 10. Canada terrain "
            "data contains information licensed under the Open Government Licence - Canada; "
            "ArcticDEM terrain data DEM(s) were created from DigitalGlobe, Inc., imagery and "
            "funded under National Science Foundation awards 1043681, 1559691, and 1542736; "
            "GMTED2010 and SRTM terrain data courtesy of the U.S. Geological Survey",
    "hotspots": "Hotspots: Canadian Forest Service. Canadian Wildland Fire Information System "
                "(CWFIS), Fire M3 hotspots, Natural Resources Canada, Canadian Forest Service, "
                "Northern Forestry Centre, Edmonton, Alberta. https://cwfis.cfs.nrcan.gc.ca. "
                "Open Government Licence - Canada",
    "perimeter": "Burned area: National Burned Area Composite (NBAC) {year}, release "
                 + NBAC_VERSION + ", Canadian Forest Service, Natural Resources Canada. "
                 "Open Government Licence - Canada",
    "weather": "Weather: Open-Meteo historical weather API (ERA5 / ERA5-Land reanalysis, "
               "Copernicus Climate Change Service), CC BY 4.0, https://open-meteo.com",
    "airports": "Aerodrome identifiers and aerodromes absent from OpenStreetMap: OurAirports "
                "(public domain), https://ourairports.com/data/",
}

# --- OpenStreetMap tag tables ------------------------------------------------------------

ROAD_CLASS = {
    "motorway": "hwy", "trunk": "hwy", "primary": "hwy",
    "motorway_link": "hwy", "trunk_link": "hwy", "primary_link": "hwy",
    "secondary": "main", "tertiary": "main", "secondary_link": "main", "tertiary_link": "main",
    "residential": "local", "unclassified": "local", "service": "local",
    "living_street": "local", "road": "local",
    "track": "track",
}
PRIVATE_SERVICE = {"parking_aisle", "driveway", "drive-through", "emergency_access"}
PLACE_KIND = {"city": "city", "town": "town", "village": "village", "hamlet": "hamlet",
              "locality": "locality", "suburb": "locality", "neighbourhood": "locality",
              "quarter": "locality", "isolated_dwelling": "locality"}
STATION_KIND = {"fire_station": "fire", "hospital": "hospital", "police": "police"}
RAIL_KINDS = {"rail", "narrow_gauge", "light_rail"}

# Land-cover codes: 0 unmapped, 1 water, 2 wetland, 3 wood, 4 scrub/heath, 5 grass/farmland,
# 6 bare rock/scree/sand/ice, 7 built-up, 8 other non-fuel.
COVER_NATURAL = {
    "water": 1, "wetland": 2, "wood": 3,
    "scrub": 4, "heath": 4, "fell": 4, "tundra": 4,
    "grassland": 5,
    "bare_rock": 6, "scree": 6, "sand": 6, "beach": 6, "shingle": 6, "glacier": 6,
    "rock": 6, "mud": 6, "shoal": 6, "blockfield": 6,
}
COVER_LANDUSE = {
    "reservoir": 1, "basin": 1, "forest": 3,
    "meadow": 5, "farmland": 5, "grass": 5, "orchard": 5, "recreation_ground": 5,
    "village_green": 5, "cemetery": 5, "farmyard": 5, "allotments": 5,
    "residential": 7, "commercial": 7, "retail": 7, "industrial": 7,
    "education": 7, "religious": 7, "institutional": 7,
    "quarry": 8, "landfill": 8, "brownfield": 8, "construction": 8, "railway": 8,
}
COVER_LEISURE = {"golf_course": 5, "pitch": 5}
BUILT_LANDUSE = {"residential", "commercial", "retail", "industrial"}
COVER_PRIORITY = [1, 7, 2, 3, 4, 5, 6, 8, 0]   # ties inside one cell go to the earliest

# Today's OSM records the fires: burnt forest and lost buildings carry a "destroyed:" prefix.
DESTROYED = ("destroyed:building", "destroyed:natural", "destroyed:landuse")
NODE_WAY_KEYS = ("highway", "railway", "waterway", "building", "aeroway", "amenity",
                 "emergency", "place", "natural", "landuse", "leisure") + DESTROYED
AREA_KEYS = ("natural", "landuse", "waterway", "leisure", "aeroway", "amenity", "emergency",
             "building", "place") + DESTROYED


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


# --- downloads ---------------------------------------------------------------------------

def fetch(url, path, pause=0.0):
    """Download `url` to `path` unless it is already cached. Returns the path."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log("download", url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = path.with_name(path.name + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    partial.replace(path)
    if pause:
        time.sleep(pause)
    return path


# --- geometry helpers --------------------------------------------------------------------

class Theatre:
    """One 64 km square and everything collected for it."""

    def __init__(self, spec):
        self.spec = spec
        self.id = spec["id"]
        self.lat0, self.lon0 = spec["centre"]
        self.kx = 111.320 * math.cos(math.radians(self.lat0))
        self.ky = 110.574
        self.west, self.east = self.lon0 - HALF / self.kx, self.lon0 + HALF / self.kx
        self.south, self.north = self.lat0 - HALF / self.ky, self.lat0 + HALF / self.ky
        self.box = box(-HALF, -HALF, HALF, HALF)
        self.seen = set()                  # (kind, osm id): a border object is in two extracts
        self.roads = []                    # (class, name, [(node id | None, x, y), ...])
        self.lines = defaultdict(list)     # "rail" | ("river" | "stream", name) -> LineStrings
        self.runways = []                  # (x1, y1, x2, y2, surface)
        self.buildings = []                # (x, y)
        self.areas = []                    # (cover code, polygon, tags)
        self.points = []                   # (tags, x, y)
        self.unmapped = Counter()          # natural= / landuse= values with no cover code
        self.restored = Counter()          # what before_the_fire() put back

    def xy(self, lon, lat):
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky

    def lonlat(self, x, y):
        return self.lon0 + x / self.kx, self.lat0 + y / self.ky

    def project(self, lonlat):
        """(N, 2) lon/lat array -> local kilometres; the callback shapely.transform wants."""
        return (lonlat - (self.lon0, self.lat0)) * (self.kx, self.ky)

    def overlaps(self, west, south, east, north):
        return west <= self.east and east >= self.west and south <= self.north and north >= self.south

    def first_time(self, kind, osm_id):
        key = (kind, osm_id)
        if key in self.seen:
            return False
        self.seen.add(key)
        return True


def r3(v):
    return round(float(v), 3)


def ring(coords):
    return [[r3(x), r3(y)] for x, y in coords]


def polygons(geom):
    """The Polygon parts of any geometry (drops lines and points left over from clipping)."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [p for g in geom.geoms for p in polygons(g)]
    return []


def linestrings(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type in ("MultiLineString", "GeometryCollection"):
        return [l for g in geom.geoms for l in linestrings(g)]
    return []


def clip_segment(x0, y0, x1, y1, m=HALF):
    """Liang-Barsky: the (t0, t1) part of a segment inside the square, or None."""
    t0, t1 = 0.0, 1.0
    dx, dy = x1 - x0, y1 - y0
    for p, q in ((-dx, x0 + m), (dx, m - x0), (-dy, y0 + m), (dy, m - y0)):
        if p == 0:
            if q < 0:
                return None
        else:
            t = q / p
            if p < 0:
                if t > t1:
                    return None
                t0 = max(t0, t)
            else:
                if t < t0:
                    return None
                t1 = min(t1, t)
    return (t0, t1) if t0 < t1 else None


def clip_nodes(nodes):
    """Clip a way [(id, x, y), ...] to the square, keeping node ids on the surviving vertices.

    Returns the pieces inside the square. Vertices created on the edge get id None.
    """
    inside = [abs(x) <= HALF and abs(y) <= HALF for _, x, y in nodes]
    if all(inside):
        return [nodes]
    pieces, current = [], []
    for i in range(len(nodes) - 1):
        a, b = nodes[i], nodes[i + 1]
        if inside[i] and inside[i + 1]:
            start, end = a, b
        else:
            span = clip_segment(a[1], a[2], b[1], b[2])
            if span is None:
                continue
            t0, t1 = span
            at = lambda t: (None, r3(a[1] + t * (b[1] - a[1])), r3(a[2] + t * (b[2] - a[2])))
            start = a if inside[i] else at(t0)
            end = b if inside[i + 1] else at(t1)
        if not current:
            current.append(start)
        current.append(end)
        if not inside[i + 1]:
            pieces.append(current)
            current = []
    if current:
        pieces.append(current)
    return [p for p in pieces if len(p) >= 2]


def simplify_pinned(points, pinned, tolerance):
    """Douglas-Peucker that never removes a pinned vertex. Returns the indices kept."""
    def distance(i, a, b):
        (px, py), (ax, ay), (bx, by) = points[i], points[a], points[b]
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
        return math.hypot(px - ax - t * dx, py - ay - t * dy)

    keep = {i for i, p in enumerate(pinned) if p} | {0, len(points) - 1}
    anchors = sorted(keep)
    stack = list(zip(anchors, anchors[1:]))
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        worst, index = max((distance(i, a, b), i) for i in range(a + 1, b))
        if worst > tolerance:
            keep.add(index)
            stack += [(a, index), (index, b)]
    return sorted(keep)


# --- OpenStreetMap -----------------------------------------------------------------------

def before_the_fire(tags):
    """Undo what mappers recorded after the fire, so the map shows the scenario date.

    A polygon or building with no current tag but a destroyed: one gets the old value back
    (the Jasper burn scar is destroyed:natural=wood). Where a current tag exists it stays:
    natural=scrub + destroyed:natural=wood is regrowth on an older burn.
    """
    for key in ("building", "natural", "landuse"):
        if key not in tags and "destroyed:" + key in tags:
            tags[key] = tags["destroyed:" + key]
            tags["_restored"] = key
    if tags.get("landuse") == "brownfield" and "wildfire" in tags.get("note", "").lower():
        tags["landuse"] = "residential"      # "Jasper wildfire (2024): destroyed built up areas"
        tags["_restored"] = "landuse"
    return tags


def cover_code(tags):
    """Land-cover code of a tagged polygon, or None when the tags say nothing about cover."""
    natural, landuse = tags.get("natural"), tags.get("landuse")
    if natural == "water" or tags.get("waterway") == "riverbank":
        return 1
    if natural in COVER_NATURAL:
        return COVER_NATURAL[natural]
    if landuse in COVER_LANDUSE:
        return COVER_LANDUSE[landuse]
    if tags.get("leisure") in COVER_LEISURE:
        return COVER_LEISURE[tags["leisure"]]
    if tags.get("aeroway") == "apron":
        return 8
    return None


def is_point_feature(tags):
    return (tags.get("place") in PLACE_KIND or tags.get("natural") == "peak"
            or tags.get("aeroway") in ("aerodrome", "heliport")
            or (tags.get("aeroway") == "helipad" and "name" in tags)
            or tags.get("amenity") in STATION_KIND
            or tags.get("emergency") == "ambulance_station")


def read_osm(pbf, theatres):
    """One streaming read of an extract, filing every relevant object under its theatre(s)."""
    west = min(t.west for t in theatres) - 0.02
    east = max(t.east for t in theatres) + 0.02
    south = min(t.south for t in theatres) - 0.02
    north = max(t.north for t in theatres) + 0.02
    wkb = osmium.geom.WKBFactory()
    stats = Counter()

    def on_node(o):
        tags = {t.k: t.v for t in o.tags}
        if not is_point_feature(tags):
            return
        lon, lat = o.location.lon, o.location.lat
        for th in theatres:
            if th.overlaps(lon, lat, lon, lat) and th.first_time("n", o.id):
                th.points.append((tags, *th.xy(lon, lat)))

    def on_way(o):
        tags = o.tags
        building = "building" in tags or "destroyed:building" in tags
        highway = ROAD_CLASS.get(tags.get("highway"))
        rail = tags.get("railway") in RAIL_KINDS
        waterway = tags.get("waterway")
        stream = waterway == "river" or (waterway == "stream" and "name" in tags)
        runway = tags.get("aeroway") == "runway"
        if not (building or highway or rail or stream or runway):
            return
        if building and not (highway or rail or stream or runway):
            # Buildings are small: one corner decides whether the rest is worth reading.
            first = o.nodes[0].location
            if not (first.valid() and west <= first.lon <= east and south <= first.lat <= north):
                return
        ids, lons, lats = [], [], []
        for n in o.nodes:
            loc = n.location
            if loc.valid():
                ids.append(n.ref); lons.append(loc.lon); lats.append(loc.lat)
        if len(ids) < 2 or max(lons) < west or min(lons) > east or max(lats) < south or min(lats) > north:
            return
        for th in theatres:
            if not th.overlaps(min(lons), min(lats), max(lons), max(lats)):
                continue
            if not th.first_time("w", o.id):
                continue
            pts = [th.xy(lon, lat) for lon, lat in zip(lons, lats)]
            if building:
                closed = ids[0] == ids[-1] and len(pts) > 1
                core = pts[:-1] if closed else pts
                cx = sum(p[0] for p in core) / len(core)
                cy = sum(p[1] for p in core) / len(core)
                if abs(cx) <= HALF and abs(cy) <= HALF:
                    th.buildings.append((cx, cy))
                    th.restored["buildings"] += "building" not in tags
            if highway:
                add_road(th, tags, highway, ids, pts)
            if rail and tags.get("service") not in ("yard", "spur"):
                th.lines["rail"].append(LineString(pts))
            if stream:
                th.lines[(waterway, tags.get("name"))].append(LineString(pts))
            if runway:
                (x1, y1), (x2, y2) = pts[0], pts[-1]
                if ids[0] != ids[-1]:
                    th.runways.append((x1, y1, x2, y2, tags.get("surface")))

    def add_road(th, tags, cls, ids, pts):
        if tags.get("area") == "yes":
            return
        if cls == "local":
            # Public roads only: no driveways, parking aisles or gated service roads.
            if tags.get("service") in PRIVATE_SERVICE or tags.get("access") in ("private", "no"):
                return
        name = tags.get("name") or tags.get("ref")
        nodes = [(i, r3(x), r3(y)) for i, (x, y) in zip(ids, pts)]
        for piece in clip_nodes(nodes):
            th.roads.append((cls, name, piece))

    def on_area(o):
        tags = before_the_fire({t.k: t.v for t in o.tags})
        code = cover_code(tags)
        point = is_point_feature(tags)
        building = "building" in tags and not o.from_way()   # closed ways were counted as ways
        if code is None and not point and not building:
            natural, landuse = tags.get("natural"), tags.get("landuse")
            if not (natural or landuse):
                return
        try:
            geom = shapely.from_wkb(wkb.create_multipolygon(o))
        except Exception:
            stats["areas osmium could not assemble"] += 1
            return
        if geom.is_empty:
            return
        for th in theatres:
            if not th.overlaps(*geom.bounds):
                continue
            if not th.first_time("a" if o.from_way() else "r", o.orig_id()):
                continue
            local = shapely.transform(geom, th.project)
            if not local.is_valid:
                local = shapely.make_valid(local)
                stats["invalid areas repaired"] += 1
            clipped = polygons(local.intersection(th.box))
            if not clipped:
                continue
            if code is None and not point and not building:
                th.unmapped[f"natural={tags.get('natural')}" if tags.get("natural")
                            else f"landuse={tags.get('landuse')}"] += 1
                continue
            if code is not None:
                for p in clipped:
                    th.areas.append((code, p, tags))
                if tags.get("_restored") in ("natural", "landuse"):
                    th.restored[f"km2 of cover code {code}"] += round(sum(p.area for p in clipped), 2)
            if point or building:
                c = unary_union(clipped).representative_point() if point else local.centroid
                if abs(c.x) <= HALF and abs(c.y) <= HALF:
                    if point:
                        th.points.append((tags, c.x, c.y))
                    else:
                        th.buildings.append((c.x, c.y))
                        th.restored["buildings"] += tags.get("_restored") == "building"

    processor = (osmium.FileProcessor(str(pbf))
                 .with_locations()
                 .with_areas(osmium.filter.KeyFilter(*AREA_KEYS))
                 .with_filter(osmium.filter.KeyFilter(*NODE_WAY_KEYS)))
    for o in processor:
        if o.is_node():
            on_node(o)
        elif o.is_way():
            on_way(o)
        elif o.is_area():
            on_area(o)
    for key, count in stats.items():
        log(f"  {pbf.name}: {key}: {count}")


def build_roads(th):
    """Simplify roads between junctions; a vertex shared by two ways is never removed."""
    uses = Counter()
    for _, _, nodes in th.roads:
        for node_id, _, _ in nodes:
            if node_id is not None:
                uses[node_id] += 1
    out = []
    for cls, name, nodes in th.roads:
        pinned = [node_id is None or uses[node_id] > 1 for node_id, _, _ in nodes]
        points = [(x, y) for _, x, y in nodes]
        kept = simplify_pinned(points, pinned, TOL_ROAD)
        path = [[points[i][0], points[i][1]] for i in kept]
        path = [p for i, p in enumerate(path) if i == 0 or p != path[i - 1]]
        if len(path) >= 2:
            out.append({"c": cls, "n": name, "p": path})
    order = {"hwy": 0, "main": 1, "local": 2, "track": 3}
    out.sort(key=lambda r: order[r["c"]])
    return out


def build_lines(th):
    streams, rail = [], []
    for key, parts in th.lines.items():
        merged = linemerge(parts) if len(parts) > 1 else parts[0]
        clipped = linestrings(merged.intersection(th.box))
        if key == "rail":
            for line in clipped:
                line = line.simplify(TOL_RAIL)
                if line.length > 0.02:
                    rail.append(ring(line.coords))
        else:
            kind, name = key
            for line in clipped:
                line = line.simplify(TOL_STREAM)
                if line.length > 0.05:
                    streams.append({"n": name, "k": kind, "p": ring(line.coords), "_len": line.length})
    streams.sort(key=lambda s: (s["k"] != "river", -s.pop("_len")))
    return streams, rail


def build_water(th):
    """Union the water polygons -> (outer rings, island rings, simplified geometry)."""
    water = unary_union([p for code, p, _ in th.areas if code == 1])
    water = shapely.simplify(water, TOL_WATER, preserve_topology=True)
    lakes = sorted((p for p in polygons(water) if p.area >= MIN_WATER), key=lambda p: -p.area)
    islands = sorted((Polygon(hole) for p in lakes for hole in p.interiors), key=lambda p: -p.area)
    islands = [p for p in islands if p.area >= MIN_ISLAND]
    return [ring(p.exterior.coords) for p in lakes], [ring(p.exterior.coords) for p in islands]


def build_built(th):
    built = unary_union([p for code, p, tags in th.areas
                         if code == 7 and tags.get("landuse") in BUILT_LANDUSE])
    built = shapely.simplify(built, TOL_BUILT, preserve_topology=True)
    parts = sorted((p for p in polygons(built) if p.area >= 0.002), key=lambda p: -p.area)
    return [ring(p.exterior.coords) for p in parts]


def build_cover(th):
    """Rasterise the tagged polygons on a 40 m grid, then take the commonest code per cell.

    Paint order decides overlaps: other cover first (largest polygon first, so a clearing
    inside a forest survives), then built-up, then water on top of everything.
    """
    size = COVER_N * SUPER
    px = COVER_CELL / SUPER
    grid = np.zeros((size, size), np.uint8)     # row 0 = south edge

    def paint(poly, code):
        minx, miny, maxx, maxy = poly.bounds
        c0, c1 = max(0, int((minx + HALF) / px)), min(size, int((maxx + HALF) / px) + 1)
        r0, r1 = max(0, int((miny + HALF) / px)), min(size, int((maxy + HALF) / px) + 1)
        if c1 <= c0 or r1 <= r0:
            return
        mask = Image.new("L", (c1 - c0, r1 - r0), 0)
        draw = ImageDraw.Draw(mask)
        # PIL puts pixel centres on integer coordinates, hence the half-pixel shift.
        to_px = lambda coords: [((x + HALF) / px - c0 - 0.5, (y + HALF) / px - r0 - 0.5) for x, y in coords]
        draw.polygon(to_px(poly.exterior.coords), fill=1)
        for hole in poly.interiors:
            draw.polygon(to_px(hole.coords), fill=0)
        grid[r0:r1, c0:c1][np.asarray(mask, bool)] = code

    rank = lambda code: 2 if code == 1 else 1 if code == 7 else 0
    for code, poly, _ in sorted(th.areas, key=lambda a: (rank(a[0]), -a[1].area)):
        paint(poly, code)

    counts = np.stack([(grid == code).reshape(COVER_N, SUPER, COVER_N, SUPER).sum(axis=(1, 3))
                       for code in COVER_PRIORITY])
    cover = np.array(COVER_PRIORITY, np.uint8)[counts.argmax(axis=0)]

    if th.buildings:
        b = np.array(th.buildings)
        cols = np.clip(((b[:, 0] + HALF) / COVER_CELL).astype(int), 0, COVER_N - 1)
        rows = np.clip(((b[:, 1] + HALF) / COVER_CELL).astype(int), 0, COVER_N - 1)
        density = np.zeros((COVER_N, COVER_N), int)
        np.add.at(density, (rows, cols), 1)
        cover[(density >= DENSE_BUILDINGS) & (cover != 1)] = 7

    flat = cover.ravel()
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    lengths = np.diff(np.r_[starts, flat.size])
    rle = [int(v) for pair in zip(flat[starts], lengths) for v in pair]
    return {"n": COVER_N, "cell": COVER_CELL, "rle": rle}, cover


def build_places(th):
    places, seen = [], set()
    for tags, x, y in th.points:
        name = tags.get("name")
        if not name or not (abs(x) <= HALF and abs(y) <= HALF):
            continue
        kind = PLACE_KIND.get(tags.get("place")) or ("peak" if tags.get("natural") == "peak" else None)
        if not kind or (kind, name) in seen:
            continue
        seen.add((kind, name))
        place = {"n": name, "k": kind, "x": r3(x), "y": r3(y), "pop": number(tags.get("population"))}
        if kind == "peak" and number(tags.get("ele")) is not None:
            place["ele"] = number(tags.get("ele"))
        places.append(place)
    # Lake labels: the largest named water bodies, one label per name.
    by_name = defaultdict(list)
    for code, poly, tags in th.areas:
        if code == 1 and tags.get("name") and tags.get("water") != "river" \
                and tags.get("waterway") != "riverbank":
            by_name[tags["name"]].append(poly)
    lakes = sorted(((sum(p.area for p in parts), name, max(parts, key=lambda p: p.area))
                    for name, parts in by_name.items()), key=lambda l: -l[0])
    for area, name, poly in lakes[:MAX_LAKE_LABELS]:
        c = poly.representative_point()
        places.append({"n": name, "k": "lake", "x": r3(c.x), "y": r3(c.y), "pop": None})
    order = ["city", "town", "village", "hamlet", "locality", "peak", "lake"]
    places.sort(key=lambda p: (order.index(p["k"]), -(p["pop"] or 0)))
    return places


def number(text):
    try:
        value = float(str(text).replace(",", "").split()[0])
    except (ValueError, IndexError):
        return None
    return int(value) if value == int(value) else value


def build_stations(th):
    out, seen = [], set()
    for tags, x, y in th.points:
        kind = STATION_KIND.get(tags.get("amenity")) or \
            ("ambulance" if tags.get("emergency") == "ambulance_station" else None)
        if not kind:
            continue
        name = tags.get("name") or {"fire": "Fire station", "hospital": "Hospital",
                                    "police": "Police", "ambulance": "Ambulance station"}[kind]
        key = (kind, round(x, 1), round(y, 1))
        if key in seen:
            continue
        seen.add(key)
        out.append({"n": name, "k": kind, "x": r3(x), "y": r3(y)})
    return sorted(out, key=lambda s: (s["k"], s["n"]))


def build_airports(th, cache):
    """Aerodromes from OSM, then identifiers and missing aerodromes from OurAirports."""
    airports = []
    for tags, x, y in th.points:
        aeroway = tags.get("aeroway")
        if aeroway not in ("aerodrome", "heliport", "helipad"):
            continue
        name = tags.get("name")
        text = " ".join(filter(None, [name, tags.get("aerodrome:type"), tags.get("aerodrome")])).lower()
        if "closed" in text:
            continue
        kind = "heliport" if aeroway != "aerodrome" else \
            "seaplane" if tags.get("seaplane") == "yes" or "water" in text or "seaplane" in text \
            or "float" in text else "airport"
        airports.append({"n": name, "icao": tags.get("icao"), "k": kind,
                         "x": r3(x), "y": r3(y), "runways": []})
    # A bare aeroway=aerodrome node beside a named aerodrome is the same airfield twice.
    named = [a for a in airports if a["n"] or a["icao"]]
    airports = [a for a in airports if a["n"] or a["icao"]
                or not any(math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < 3 for b in named)]

    rows = list(csv.DictReader(open(fetch(OURAIRPORTS + "airports.csv", cache / "ourairports" / "airports.csv"),
                                    encoding="utf-8", newline="")))
    runway_rows = defaultdict(list)
    ours = []
    for row in rows:
        try:
            lat, lon = float(row["latitude_deg"]), float(row["longitude_deg"])
        except ValueError:
            continue
        if row["type"] == "closed" or not th.overlaps(lon, lat, lon, lat):
            continue
        ours.append((row, *th.xy(lon, lat)))
    if ours:
        wanted = {row["id"] for row, _, _ in ours}
        with open(fetch(OURAIRPORTS + "runways.csv", cache / "ourairports" / "runways.csv"),
                  encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["airport_ref"] in wanted and row["closed"] != "1":
                    runway_rows[row["airport_ref"]].append(row)
    added = 0
    for row, x, y in ours:
        kind = {"heliport": "heliport", "seaplane_base": "seaplane"}.get(row["type"], "airport")
        # Canadian location indicators only (CYZF, CEN9); OurAirports' own "CA-0818" is not one.
        ident = next((c for c in (row["icao_code"], row["gps_code"], row["ident"])
                      if re.fullmatch(r"C[A-Z0-9]{3}", c or "")), None)
        near = [a for a in airports if math.hypot(a["x"] - x, a["y"] - y) < 1.5]
        match = min(near, key=lambda a: (a["k"] != kind, math.hypot(a["x"] - x, a["y"] - y)), default=None)
        if match and (match["k"] == kind or len(near) == 1) and not match.get("_matched"):
            match["_matched"] = True
            match["icao"] = match["icao"] or ident
            match["n"] = match["n"] or row["name"]
            if kind != "airport":
                match["k"] = kind
            match["_ours"] = row["id"]
            continue
        airport = {"n": row["name"], "icao": ident, "k": kind, "x": r3(x), "y": r3(y),
                   "runways": [], "_ours": row["id"], "_only_ours": True}
        airports.append(airport)
        added += 1

    # OSM runways go to the nearest non-heliport aerodrome within 3 km of their midpoint.
    for x1, y1, x2, y2, surface in th.runways:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        hosts = [a for a in airports if a["k"] != "heliport"] or airports
        host = min(hosts, key=lambda a: math.hypot(a["x"] - mx, a["y"] - my), default=None)
        if host and math.hypot(host["x"] - mx, host["y"] - my) < 3:
            host["runways"].append([r3(x1), r3(y1), r3(x2), r3(y2), surface])
    for airport in airports:
        ours_id = airport.pop("_ours", None)
        airport.pop("_matched", None)
        only_ours = airport.pop("_only_ours", False)
        csv_runways = runway_rows.get(ours_id, [])
        if not airport["runways"]:
            # No runway drawn in OSM: use OurAirports runway ends where it has them.
            for row in csv_runways:
                try:
                    a = th.xy(float(row["le_longitude_deg"]), float(row["le_latitude_deg"]))
                    b = th.xy(float(row["he_longitude_deg"]), float(row["he_latitude_deg"]))
                except ValueError:
                    continue
                airport["runways"].append([r3(a[0]), r3(a[1]), r3(b[0]), r3(b[1]),
                                           surface_name(row["surface"])])
        elif csv_runways:
            # Fill untagged OSM surfaces when OurAirports agrees on a single surface.
            surfaces = {surface_name(row["surface"]) for row in csv_runways} - {None}
            if len(surfaces) == 1:
                for runway in airport["runways"]:
                    runway[4] = runway[4] or next(iter(surfaces))
        for runway in airport["runways"]:
            runway[4] = runway[4] or "unknown"
        airport["src"] = "ourairports" if only_ours else "osm"
    order = {"airport": 0, "seaplane": 1, "heliport": 2}
    airports.sort(key=lambda a: (order[a["k"]], -len(a["runways"]), a["n"] or ""))
    return airports, added


def surface_name(code):
    code = (code or "").upper()
    for prefix, name in (("ASP", "asphalt"), ("CON", "concrete"), ("GRV", "gravel"), ("GRAVEL", "gravel"),
                         ("TURF", "grass"), ("GRASS", "grass"), ("WATER", "water"), ("DIRT", "dirt"),
                         ("SAND", "sand"), ("SNOW", "snow"), ("ICE", "ice")):
        if code.startswith(prefix):
            return name
    return code.lower() or None


# --- elevation ---------------------------------------------------------------------------

def build_elevation(th, cache):
    """Mean elevation of each 320 m cell from Terrarium tiles (bilinear, 4 x 4 samples a cell)."""
    z, n = ELEV_ZOOM, 2 ** ELEV_ZOOM

    def tile_xy(lon, lat):
        x = (lon + 180) / 360 * n
        y = (1 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2 * n
        return x, y

    x0, y0 = tile_xy(th.west, th.north)
    x1, y1 = tile_xy(th.east, th.south)
    tx0, tx1, ty0, ty1 = int(x0), int(x1), int(y0), int(y1)
    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), np.float32)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            path = fetch(TERRARIUM.format(z=z, x=tx, y=ty), cache / "terrarium" / str(z) / f"{tx}_{ty}.png", pause=0.1)
            rgb = np.asarray(Image.open(path).convert("RGB"), np.float32)
            metres = rgb[:, :, 0] * 256 + rgb[:, :, 1] + rgb[:, :, 2] / 256 - 32768
            mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = metres

    sub = 4
    offsets = (np.arange(ELEV_N * sub) + 0.5) * (ELEV_CELL / sub) - HALF
    lon = th.lon0 + offsets / th.kx
    lat = th.lat0 + offsets / th.ky
    fx, fy = tile_xy(lon, lat)
    px = (fx - tx0) * 256 - 0.5                      # pixel centres sit at +0.5
    py = (fy - ty0) * 256 - 0.5
    px = np.clip(px, 0, mosaic.shape[1] - 1.001)
    py = np.clip(py, 0, mosaic.shape[0] - 1.001)
    ix, iy = px.astype(int), py.astype(int)
    wx, wy = (px - ix)[None, :], (py - iy)[:, None]
    a = mosaic[np.ix_(iy, ix)]
    b = mosaic[np.ix_(iy, ix + 1)]
    c = mosaic[np.ix_(iy + 1, ix)]
    d = mosaic[np.ix_(iy + 1, ix + 1)]
    fine = a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + c * (1 - wx) * wy + d * wx * wy
    grid = fine.reshape(ELEV_N, sub, ELEV_N, sub).mean(axis=(1, 3))     # row 0 = south
    values = np.round(grid).astype("<i2")
    return {"n": ELEV_N, "cell": ELEV_CELL, "min": int(values.min()), "max": int(values.max()),
            "b64": base64.b64encode(values.tobytes()).decode("ascii")}


# --- hotspots, perimeters, weather -------------------------------------------------------

HOTSPOT_FIELDS = {  # schema name -> (CWFIS column, decimals); cfb is not in the archive files
    "frp": ("frp", 1), "ros": ("ros", 1), "hfi": ("hfi", 0), "ffmc": ("ffmc", 1), "bui": ("bui", 0),
    "isi": ("isi", 1), "ws": ("ws", 1), "wd": ("wd", 0), "tfc": ("tfc", 2), "cfb": ("cfb", 0),
}


def read_hotspots(cache, year, theatres):
    """Stream one year of CWFIS hotspots, keeping scenario date -3 days .. +2 days (UTC)."""
    archive = fetch(f"{CWFIS}hotspots/archive/{year}_hotspots.zip", cache / "hotspots" / f"{year}_hotspots.zip")
    windows = {}
    for th in theatres:
        day = date.fromisoformat(th.spec["date"])
        windows[th.id] = (str(day - timedelta(days=3)), str(day + timedelta(days=2)))
    first, last = min(w[0] for w in windows.values()), max(w[1] for w in windows.values())
    found = {th.id: [] for th in theatres}
    with zipfile.ZipFile(archive) as zf:
        member = next(name for name in zf.namelist() if name.lower().endswith(".csv"))
        with zf.open(member) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, "utf-8", errors="replace", newline=""))
            header = next(reader)
            col = {name: i for i, name in enumerate(header)}
            for row in reader:
                day = row[col["rep_date"]][:10]
                if day < first or day > last:
                    continue
                lat, lon = float(row[col["lat"]]), float(row[col["lon"]])
                for th in theatres:
                    lo, hi = windows[th.id]
                    if not (lo <= day <= hi and th.overlaps(lon, lat, lon, lat)):
                        continue
                    x, y = th.xy(lon, lat)
                    stamp = row[col["rep_date"]]
                    spot = {"x": r3(x), "y": r3(y), "t": f"{stamp[:10]}T{stamp[11:16]}Z",
                            "sensor": row[col["sensor"]]}
                    for key, (column, decimals) in HOTSPOT_FIELDS.items():
                        text = row[col[column]] if column in col else ""
                        if text in ("", "NA", "NULL") or (key == "frp" and float(text) == 0):
                            continue                 # a fire radiative power of 0 means "not given"
                        value = round(float(text), decimals)
                        spot[key] = int(value) if decimals == 0 else value
                    if row[col["fuel"]]:
                        spot["fuel"] = row[col["fuel"]]
                    found[th.id].append(spot)
    for spots in found.values():
        spots.sort(key=lambda s: (s["t"], s["x"], s["y"]))
    return found, header


def read_perimeters(cache, year, theatres):
    """NBAC burned-area polygons of `year` that reach each theatre, as outer rings."""
    stem = f"NBAC_{year}_{NBAC_VERSION}"
    archive = fetch(f"{CWFIS}nbac/{stem}.zip", cache / "nbac" / f"{stem}.zip")
    folder = cache / "nbac" / stem
    with zipfile.ZipFile(archive) as zf:
        for ext in ("shp", "shx", "dbf", "prj"):
            target = folder / f"{stem}.{ext}"
            if not target.exists():
                folder.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(f"{stem}.{ext}"))
    crs = pyproj.CRS.from_wkt((folder / f"{stem}.prj").read_text())
    to_lonlat = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_crs = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    found = {}
    reader = shapefile.Reader(str(folder / f"{stem}.shp"))
    fields = [f[0] for f in reader.fields[1:]]
    for th in theatres:
        edge = np.linspace(-HALF, HALF, 33)
        border = [(x, y) for x in edge for y in (-HALF, HALF)] + [(x, y) for y in edge for x in (-HALF, HALF)]
        bx, by = to_crs.transform(*zip(*(th.lonlat(x, y) for x, y in border)))
        bbox = (min(bx), min(by), max(bx), max(by))
        burned, fires = [], []
        for item in reader.iterShapeRecords(bbox=bbox):
            shp = item.shape
            if not shp.points:
                continue
            points = np.array(shp.points)
            lon, lat = to_lonlat.transform(points[:, 0], points[:, 1])
            local = th.project(np.column_stack([lon, lat]))
            parts = list(shp.parts) + [len(points)]
            touched = False
            for a, b in zip(parts, parts[1:]):
                if b - a < 4:
                    continue
                poly = Polygon(local[a:b])
                # Shapefile outer rings run clockwise; holes (unburned islands) are skipped.
                if poly.exterior.is_ccw or poly.bounds[0] > HALF or poly.bounds[2] < -HALF \
                        or poly.bounds[1] > HALF or poly.bounds[3] < -HALF:
                    continue
                if not poly.is_valid:
                    poly = shapely.make_valid(poly)
                piece = poly.intersection(th.box)
                if polygons(piece):
                    burned.extend(polygons(piece))
                    touched = True
            if touched:
                fires.append(dict(zip(fields, item.record)))
        merged = shapely.simplify(unary_union(burned), TOL_PERIMETER, preserve_topology=True) if burned else None
        rings = [ring(p.exterior.coords) for p in sorted(polygons(merged), key=lambda p: -p.area) if p.area >= 0.05]
        found[th.id] = (rings, fires, sum(p.area for p in polygons(merged)))
    return found


def read_weather(th, cache):
    """Hourly weather at the town from the scenario date 00Z through +72 h (73 values)."""
    _, lat, lon = th.spec["town"]
    start = date.fromisoformat(th.spec["date"])
    query = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": start + timedelta(days=3),
        "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m,temperature_2m,relative_humidity_2m"})
    data = json.loads(fetch(f"{OPEN_METEO}?{query}", cache / "weather" / f"{th.id}_{start}.json").read_text())
    assert data["hourly_units"]["wind_speed_10m"] == "km/h" and data["utc_offset_seconds"] == 0
    h = data["hourly"]
    rows = []
    for i, stamp in enumerate(h["time"][:73]):
        values = [h[k][i] for k in ("wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
                                    "temperature_2m", "relative_humidity_2m")]
        if any(v is None for v in values):
            continue
        rows.append({"t": stamp[:13] + ":00Z", "ws": values[0], "wd": values[1], "gust": values[2],
                     "temp": values[3], "rh": values[4]})
    return rows, (data["latitude"], data["longitude"], data.get("elevation"))


# --- assembly ----------------------------------------------------------------------------

def build(th, cache, hotspots, perimeter, osm_date):
    spec = th.spec
    town_name, town_lat, town_lon = spec["town"]
    tx, ty = th.xy(town_lon, town_lat)
    water, islands = build_water(th)
    streams, rail = build_lines(th)
    cover, _ = build_cover(th)
    random.seed(20230815)
    buildings = th.buildings if len(th.buildings) <= MAX_BUILDINGS else random.sample(th.buildings, MAX_BUILDINGS)
    airports, _ = build_airports(th, cache)
    weather, grid_point = read_weather(th, cache)
    rings, fires, _ = perimeter
    year = spec["date"][:4]
    sources = [SOURCES["osm"].format(osm_date=osm_date), SOURCES["elev"], SOURCES["hotspots"],
               SOURCES["perimeter"].format(year=year), SOURCES["weather"], SOURCES["airports"]]
    return {
        "id": th.id, "name": spec["name"], "region": spec["region"], "date": spec["date"],
        "utcOffset": spec["utcOffset"], "centre": [th.lat0, th.lon0], "sizeKm": SIZE_KM,
        "town": {"name": town_name, "x": r3(tx), "y": r3(ty)},
        "places": build_places(th),
        "water": water, "islands": islands, "streams": streams,
        "roads": build_roads(th), "rail": rail,
        "built": build_built(th),
        "buildings": sorted([round(x, 2), round(y, 2)] for x, y in buildings),
        "airports": airports,
        "stations": build_stations(th),
        "cover": cover,
        "elev": build_elevation(th, cache),
        "hotspots": hotspots,
        "weather": weather,
        "perimeter": rings or None,
        "sources": sources,
    }


def check(path):
    """Print layer counts and run the sanity checks on one written file. Returns problems."""
    d = json.loads(path.read_text())
    problems = []
    print(f"\n== {path.name}: {path.stat().st_size / 1e6:.2f} MB")
    for key in ("places", "water", "islands", "streams", "roads", "rail", "built", "buildings",
                "airports", "stations", "hotspots", "weather"):
        print(f"  {key:10} {len(d[key])}")
    print(f"  {'perimeter':10} {len(d['perimeter']) if d['perimeter'] else d['perimeter']}")
    print("  roads by class", dict(Counter(r["c"] for r in d["roads"])),
          "vertices", sum(len(r["p"]) for r in d["roads"]))
    print("  places by kind", dict(Counter(p["k"] for p in d["places"])))
    print("  airports", [(a["n"], a["icao"], a["k"], len(a["runways"])) for a in d["airports"]])
    print("  stations", [(s["k"], s["n"]) for s in d["stations"]])

    # Land cover.
    rle = d["cover"]["rle"]
    total = sum(rle[1::2])
    cover = np.repeat(np.array(rle[0::2], np.uint8), rle[1::2]).reshape(COVER_N, COVER_N) if total == COVER_N ** 2 else None
    if cover is None:
        problems.append(f"cover RLE sums to {total}")
    else:
        share = {int(c): round(float((cover == c).mean()) * 100, 1) for c in np.unique(cover)}
        print("  cover % by code", share, "RLE pairs", len(rle) // 2)
        half = COVER_N // 2
        quads = {"SW": cover[:half, :half], "SE": cover[:half, half:], "NW": cover[half:, :half], "NE": cover[half:, half:]}
        print("  water % by quadrant", {k: round(float((v == 1).mean()) * 100, 1) for k, v in quads.items()})

    # Elevation.
    elev = np.frombuffer(base64.b64decode(d["elev"]["b64"]), "<i2")
    if elev.size != ELEV_N ** 2:
        problems.append(f"elevation has {elev.size} values")
    elif not (-50 <= elev.min() and elev.max() <= 4500) or (elev.min(), elev.max()) != (d["elev"]["min"], d["elev"]["max"]):
        problems.append(f"elevation range {elev.min()}..{elev.max()}")
    print(f"  elevation {elev.size} values, {elev.min()}..{elev.max()} m, mean {elev.mean():.0f}")

    # Coordinate range.
    def walk(node):
        if isinstance(node, list):
            if len(node) >= 2 and all(isinstance(v, (int, float)) for v in node[:2]) and not isinstance(node[0], bool):
                yield node[0], node[1]
                if len(node) >= 4 and all(isinstance(v, (int, float)) for v in node[2:4]):
                    yield node[2], node[3]
            else:
                for item in node:
                    yield from walk(item)
        elif isinstance(node, dict):
            if "x" in node and "y" in node:
                yield node["x"], node["y"]
            for key, value in node.items():
                if key not in ("cover", "elev", "centre"):
                    yield from walk(value)
    worst = max((max(abs(x), abs(y)) for x, y in walk(d)), default=0)
    print(f"  largest |coordinate| {worst}")
    if worst > 33:
        problems.append(f"coordinate {worst} outside the theatre")

    # Town sits on built-up ground or next to buildings.
    tx, ty = d["town"]["x"], d["town"]["y"]
    near = sum(1 for x, y in d["buildings"] if math.hypot(x - tx, y - ty) <= 1.0)
    on_built = any(Polygon(r).contains(Point(tx, ty)) for r in d["built"] if len(r) >= 4)
    cell = None if cover is None else int(cover[min(COVER_N - 1, int((ty + HALF) / COVER_CELL)), min(COVER_N - 1, int((tx + HALF) / COVER_CELL))])
    print(f"  town ({tx}, {ty}): on built polygon {on_built}, buildings within 1 km {near}, cover cell {cell}")
    if not on_built and near == 0:
        problems.append("town reference is neither on built-up land nor near buildings")

    # Road graph: shared coordinates are the junctions.
    parent = {}
    def find(k):
        while parent.setdefault(k, k) != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for road in d["roads"]:
        keys = [tuple(p) for p in road["p"]]
        for a, b in zip(keys, keys[1:]):
            parent[find(a)] = find(b)
    components = Counter(find(k) for k in parent)
    local = [k for k in parent if math.hypot(k[0] - tx, k[1] - ty) <= 3.0]
    if local:
        main, size = Counter(find(k) for k in local).most_common(1)[0]
        print(f"  road graph: {len(parent)} vertices, {len(components)} components, largest {max(components.values())}; "
              f"within 3 km of town {len(local)} vertices, {size / len(local):.1%} in one component "
              f"(which holds {components[main]} vertices overall)")
        if size / len(local) < 0.9:
            problems.append("road graph near the town is fragmented")
    else:
        problems.append("no roads within 3 km of the town")

    if d["hotspots"]:
        times = [h["t"] for h in d["hotspots"]]
        print(f"  hotspots {min(times)} .. {max(times)}", dict(Counter(h["sensor"] for h in d["hotspots"])),
              "by day", dict(sorted(Counter(t[:10] for t in times).items())))
        print("  hotspot fields", dict(Counter(k for h in d["hotspots"] for k in h)))
    if d["weather"]:
        print(f"  weather {d['weather'][0]['t']} .. {d['weather'][-1]['t']}; max wind "
              f"{max(w['ws'] for w in d['weather'])} km/h, max gust {max(w['gust'] for w in d['weather'])}, "
              f"temp {min(w['temp'] for w in d['weather'])}..{max(w['temp'] for w in d['weather'])} C, "
              f"min RH {min(w['rh'] for w in d['weather'])}%")
    sizes = {k: len(json.dumps(v, separators=(",", ":"))) for k, v in d.items()}
    print("  bytes by layer", {k: v for k, v in sorted(sizes.items(), key=lambda kv: -kv[1]) if v > 20000})
    for problem in problems:
        print("  PROBLEM:", problem)
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "epoch-ops")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--only", action="append", help="theatre id (repeatable)")
    parser.add_argument("--check", action="store_true", help="only validate the files already written")
    args = parser.parse_args()
    specs = [s for s in THEATRES if not args.only or s["id"] in args.only]

    if not args.check:
        theatres = [Theatre(s) for s in specs]
        osm_dates = defaultdict(set)
        for name in sorted({e for s in specs for e in s["extracts"]}):
            pbf = fetch(f"{GEOFABRIK}{name}-latest.osm.pbf", args.cache / "osm" / f"{name}-latest.osm.pbf")
            group = [t for t in theatres if name in t.spec["extracts"]]
            stamp = osmium.io.Reader(str(pbf)).header().get("osmosis_replication_timestamp", "")[:10]
            for t in group:
                osm_dates[t.id].add(stamp or "unknown")
            log(f"reading {pbf.name} ({stamp}) for", ", ".join(t.id for t in group))
            read_osm(pbf, group)
        hotspots, perimeters = {}, {}
        for year in sorted({s["date"][:4] for s in specs}):
            group = [t for t in theatres if t.spec["date"].startswith(year)]
            log("hotspots", year)
            found, header = read_hotspots(args.cache, year, group)
            hotspots.update(found)
            log("  columns:", ",".join(header))
            log("perimeters", year)
            perimeters.update(read_perimeters(args.cache, year, group))
        args.out.mkdir(parents=True, exist_ok=True)
        for th in theatres:
            log("building", th.id, {"roads": len(th.roads), "areas": len(th.areas),
                                    "buildings": len(th.buildings), "points": len(th.points)})
            if th.unmapped:
                log("  polygons with no cover code:", dict(th.unmapped.most_common(12)))
            if th.restored:
                log("  restored to the pre-fire state:", dict(th.restored))
            if len(th.buildings) > MAX_BUILDINGS:
                log(f"  {len(th.buildings)} buildings sampled down to {MAX_BUILDINGS}")
            for fire in perimeters[th.id][1]:
                log("  NBAC fire:", {k: fire[k] for k in fire if k in ("NFIREID", "GID", "HS_SDATE", "HS_EDATE", "ADJ_HA", "POLY_HA", "ADMIN_AREA", "FIRECAUS")})
            data = build(th, args.cache, hotspots[th.id], perimeters[th.id], "/".join(sorted(osm_dates[th.id])))
            (args.out / f"{th.id}.json").write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))

    index = [{"id": s["id"], "name": s["name"], "region": s["region"], "date": s["date"],
              "centre": list(s["centre"]), "file": f"{s['id']}.json"}
             for s in THEATRES if (args.out / f"{s['id']}.json").exists()]
    (args.out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    problems = []
    for s in specs:
        problems += [f"{s['id']}: {p}" for p in check(args.out / f"{s['id']}.json")]
    raise SystemExit("\n".join(problems) if problems else 0)


if __name__ == "__main__":
    main()
