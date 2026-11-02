"""
Reverse geocoding for human-readable places ("NH 275 near Srirangapatna, Mandya") with a cache and a polite
throttle. Best effort: callers must work without it. Uses Geoapify if GEOAPIFY_API_KEY is set (better quota),
otherwise Nominatim (1 request/second, identifying User-Agent, as its usage policy requires).
"""
import os
import threading
import time

import httpx

_UA = {"User-Agent": "MargRakshak-Agent/1.0 (tourist safety console)"}
_cache = {}                       # cell -> (expires, result)
_lock = threading.Lock()
_last_call = [0.0]

# OSM class/type -> what kind of place this is. Drives the "what could be the issue" inference.
_KIND = [
    ({"highway": {"motorway", "trunk", "primary", "motorway_link", "trunk_link"}}, "highway"),
    ({"highway": {"secondary", "tertiary", "unclassified"}}, "main_road"),
    ({"highway": {"residential", "living_street", "service", "pedestrian", "footway"}}, "street"),
    ({"highway": {"track", "path", "bridleway"}}, "remote_track"),
    ({"natural": {"wood", "scrub", "cliff", "peak", "valley", "ridge", "cave_entrance"}}, "wilderness"),
    ({"natural": {"water", "bay", "beach", "coastline", "wetland"}, "waterway": {"river", "stream", "canal"}}, "waterside"),
    ({"landuse": {"forest"}, "boundary": {"protected_area", "national_park"}, "leisure": {"nature_reserve"}}, "wilderness"),
    ({"amenity": {"marketplace", "restaurant", "cafe", "bar", "pub", "fast_food", "bus_station", "cinema"},
      "shop": None, "tourism": {"attraction", "viewpoint", "museum", "gallery", "zoo", "theme_park"}}, "busy_public"),
    ({"railway": {"station", "halt", "tram_stop"}, "public_transport": {"station", "platform", "stop_position"}}, "transit"),
    ({"amenity": {"hospital", "clinic", "police", "fire_station"}}, "services"),
    ({"tourism": {"hotel", "hostel", "guest_house", "motel", "resort"}, "building": {"residential", "house", "apartments"}}, "lodging_or_home"),
    ({"landuse": {"residential", "commercial", "retail", "industrial"}}, "built_up"),
]


def _kind(cls, typ, addr_keys):
    for rules, kind in _KIND:
        for key, vals in rules.items():
            if (cls == key and (vals is None or typ in vals)):
                return kind
    return "built_up" if {"suburb", "neighbourhood", "city_district", "quarter"} & set(addr_keys) else "unknown"


def _cell(lat, lon):
    return f"{round(lat / 0.001)}:{round(lon / 0.001)}"           # ~110 m


def reverse(lat, lon, timeout=6.0, client=None):
    """Returns {name, road, locality, district, state, kind, raw_class, raw_type, source} or None."""
    key = _cell(lat, lon)
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] > time.time():
            return hit[1]
    res = None
    try:
        res = _geoapify(lat, lon, timeout, client) if os.environ.get("GEOAPIFY_API_KEY") else None
    except Exception:  # noqa: BLE001
        res = None
    if res is None:
        try:
            res = _nominatim(lat, lon, timeout, client)
        except Exception:  # noqa: BLE001
            res = None
    if res:
        with _lock:
            _cache[key] = (time.time() + 6 * 3600, res)
    return res


def _compose(road, locality, district, state, fallback):
    parts = [p for p in (road, locality, district) if p]
    seen, out = set(), []
    for p in parts:
        if p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    name = ", ".join(out[:3]) or fallback or None
    return name


def _nominatim(lat, lon, timeout, client):
    with _lock:                                                   # global throttle: 1 request / second
        wait = 1.1 - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
    c = client or httpx
    r = c.get("https://nominatim.openstreetmap.org/reverse", headers=_UA, timeout=timeout,
              params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 17, "addressdetails": 1, "accept-language": "en"})
    r.raise_for_status()
    d = r.json()
    a = d.get("address") or {}
    road = a.get("road") or a.get("pedestrian") or a.get("footway") or a.get("path")
    locality = a.get("suburb") or a.get("neighbourhood") or a.get("village") or a.get("town") or a.get("city") or a.get("hamlet")
    district = a.get("state_district") or a.get("county") or a.get("city_district")
    name = _compose(road, locality, district, a.get("state"), d.get("display_name", "")[:80])
    return {"name": name, "road": road, "locality": locality, "district": district, "state": a.get("state"),
            "kind": _kind(d.get("category") or d.get("class"), d.get("type"), a.keys()),
            "raw_class": d.get("category") or d.get("class"), "raw_type": d.get("type"), "source": "nominatim"}


def _geoapify(lat, lon, timeout, client):
    c = client or httpx
    r = c.get("https://api.geoapify.com/v1/geocode/reverse", timeout=timeout,
              params={"lat": lat, "lon": lon, "format": "json", "apiKey": os.environ["GEOAPIFY_API_KEY"]})
    r.raise_for_status()
    res = (r.json().get("results") or [None])[0]
    if not res:
        return None
    road = res.get("street")
    locality = res.get("suburb") or res.get("neighbourhood") or res.get("city") or res.get("town") or res.get("village")
    district = res.get("county") or res.get("district")
    cat = (res.get("category") or "")
    kind = ("highway" if cat.startswith("road") and "motorway" in str(res.get("result_type", "")) else
            "wilderness" if cat.startswith(("natural", "leisure.park", "national_park")) else
            "busy_public" if cat.startswith(("commercial", "catering", "tourism")) else
            "transit" if cat.startswith("public_transport") else
            "services" if cat.startswith(("healthcare", "service.police", "service.fire")) else
            "street" if road else "unknown")
    return {"name": _compose(road, locality, district, res.get("state"), res.get("formatted")), "road": road,
            "locality": locality, "district": district, "state": res.get("state"), "kind": kind,
            "raw_class": cat, "raw_type": res.get("result_type"), "source": "geoapify"}
