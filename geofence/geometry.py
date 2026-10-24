"""Small-area geometry on GeoJSON ([lon, lat]) without PostGIS or shapely.

Distances use a local equirectangular projection centred on the query point. For the ranges that
matter here (tens of metres to a few km) the error is well under GPS error.
"""
import math

R = 6371000.0


class GeometryError(ValueError):
    pass


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two bearings, 0..180."""
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def _xy(lat, lon, lat0, lon0):
    return (R * math.radians(lon - lon0) * math.cos(math.radians(lat0)), R * math.radians(lat - lat0))


def _lonlat(x, y, lat0, lon0):
    lat = lat0 + math.degrees(y / R)
    lon = lon0 + math.degrees(x / (R * math.cos(math.radians(lat0))))
    return lat, lon


def _seg(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy), cx, cy


def _ring_contains(px, py, ring):
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _pairs(coords):
    out = []
    for c in coords:
        if not (isinstance(c, (list, tuple)) and len(c) >= 2 and all(isinstance(v, (int, float)) for v in c[:2])):
            raise GeometryError("bad coordinate")
        lon, lat = float(c[0]), float(c[1])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise GeometryError("coordinate out of range")
        out.append((lat, lon))
    return out


def _parts(geom):
    """Yield (kind, data) with kind in point|line|polygon; data holds (lat, lon) tuples."""
    if not isinstance(geom, dict):
        raise GeometryError("geometry must be an object")
    t, c = geom.get("type"), geom.get("coordinates")
    if t == "Point":
        yield "point", _pairs([c])[0]
    elif t == "MultiPoint":
        for p in _pairs(c):
            yield "point", p
    elif t == "LineString":
        yield "line", _pairs(c)
    elif t == "MultiLineString":
        for ln in c:
            yield "line", _pairs(ln)
    elif t == "Polygon":
        yield "polygon", [_pairs(r) for r in c]
    elif t == "MultiPolygon":
        for poly in c:
            yield "polygon", [_pairs(r) for r in poly]
    else:
        raise GeometryError(f"unsupported geometry type {t!r}")


def geometry_kind(geom):
    kinds = {k for k, _ in _parts(geom)}
    return "polygon" if "polygon" in kinds else "line" if "line" in kinds else "point"


def bbox_of(geom):
    pts = []
    for kind, d in _parts(geom):
        if kind == "point":
            pts.append(d)
        elif kind == "line":
            pts.extend(d)
        else:
            for ring in d:
                pts.extend(ring)
    if not pts:
        raise GeometryError("empty geometry")
    lats, lons = [p[0] for p in pts], [p[1] for p in pts]
    return min(lats), min(lons), max(lats), max(lons)


def distance_to_geometry(lat, lon, geom):
    """-> (distance_m, nearest_lat, nearest_lon). Distance is 0 inside a polygon."""
    best = None

    def consider(d, x, y):
        nonlocal best
        if best is None or d < best[0]:
            best = (d, x, y)

    for kind, data in _parts(geom):
        if kind == "point":
            x, y = _xy(data[0], data[1], lat, lon)
            consider(math.hypot(x, y), x, y)
        elif kind == "line":
            xy = [_xy(p[0], p[1], lat, lon) for p in data]
            if len(xy) == 1:
                consider(math.hypot(*xy[0]), *xy[0])
            for a, b in zip(xy, xy[1:]):
                consider(*_seg(0.0, 0.0, a[0], a[1], b[0], b[1]))
        else:
            rings = [[_xy(p[0], p[1], lat, lon) for p in r] for r in data]
            if rings and _ring_contains(0.0, 0.0, rings[0]) and not any(_ring_contains(0.0, 0.0, h) for h in rings[1:]):
                return 0.0, lat, lon
            for ring in rings:
                for a, b in zip(ring, ring[1:] + ring[:1]):
                    consider(*_seg(0.0, 0.0, a[0], a[1], b[0], b[1]))
    if best is None:
        raise GeometryError("empty geometry")
    nlat, nlon = _lonlat(best[1], best[2], lat, lon)
    return best[0], nlat, nlon
