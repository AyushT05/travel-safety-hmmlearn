"""
Nearby emergency services with phone numbers where the sources have them, plus the national numbers that
always work. Best effort: every network call is optional and bounded.

Sources, in order: Geoapify Places (has phone/website for many Indian hospitals and police stations; needs
GEOAPIFY_API_KEY) and OpenStreetMap Overpass (free; phone tags are patchy, so a missing number is shown as
missing rather than invented).
"""
import os

import httpx

from .geo import haversine_m

_CATS = {"hospital": "amenity=hospital", "police": "amenity=police", "fire_station": "amenity=fire_station",
         "clinic": "amenity=clinic", "pharmacy": "amenity=pharmacy"}
_GEO_CATS = {"hospital": "healthcare.hospital", "police": "service.police", "fire_station": "service.fire_station",
             "clinic": "healthcare.clinic_or_praxis", "pharmacy": "healthcare.pharmacy"}
_ORDER = ("police", "hospital", "fire_station", "clinic", "pharmacy")
_OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]

# India. These are dialled from any phone, no signal-strength or SIM requirement for 112.
EMERGENCY_NUMBERS_IN = [
    {"label": "National emergency (police, fire, ambulance)", "number": "112"},
    {"label": "Police", "number": "100"},
    {"label": "Ambulance", "number": "108"},
    {"label": "Fire", "number": "101"},
    {"label": "Women helpline", "number": "1091"},
    {"label": "Tourist helpline", "number": "1363"},
    {"label": "Disaster management", "number": "1078"},
]


def emergency_numbers():
    return [dict(n) for n in EMERGENCY_NUMBERS_IN]


def _phone(tags):
    for k in ("phone", "contact:phone", "contact:mobile", "mobile", "phone:emergency"):
        v = tags.get(k)
        if v:
            return str(v).split(";")[0].strip()[:30]
    return None


def _from_geoapify(lat, lon, radius_m, per_category, timeout):
    key = os.environ.get("GEOAPIFY_API_KEY")
    if not key:
        return None
    r = httpx.get("https://api.geoapify.com/v2/places", timeout=timeout, params={
        "categories": ",".join(_GEO_CATS.values()), "filter": f"circle:{lon},{lat},{radius_m}",
        "bias": f"proximity:{lon},{lat}", "limit": 60, "apiKey": key})
    r.raise_for_status()
    by = {}
    for f in r.json().get("features", []):
        p, coords = f.get("properties") or {}, (f.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        cats = p.get("categories") or []
        cat = next((c for c, g in _GEO_CATS.items() if any(x.startswith(g.split("_or_")[0]) or x == g for x in cats)), None)
        if not cat:
            continue
        contact = p.get("contact") or {}
        raw = (p.get("datasource") or {}).get("raw") or {}
        by.setdefault(cat, []).append({
            "category": cat, "name": (p.get("name") or "")[:80] or None, "lat": coords[1], "lon": coords[0],
            "distance_km": round(haversine_m(lat, lon, coords[1], coords[0]) / 1000, 2),
            "phone": (contact.get("phone") or _phone(raw) or "")[:30] or None, "address": (p.get("formatted") or "")[:120] or None,
            "source": "geoapify"})
    return by


def _from_overpass(lat, lon, radius_m, per_category, timeout):
    q = "[out:json][timeout:10];(" + "".join(
        f"nwr[{tag}](around:{radius_m},{lat},{lon});" for tag in _CATS.values()) + ");out center tags qt 150;"
    elements = None
    for url in _OVERPASS:
        try:
            r = httpx.post(url, data={"data": q}, headers={"User-Agent": "MargRakshak-Agent/1.0"}, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            if "error" in str(j.get("remark", "")).lower():
                continue
            elements = j.get("elements", [])
            break
        except Exception:  # noqa: BLE001
            continue
    if elements is None:
        return None
    by = {}
    for el in elements:
        tags = el.get("tags", {})
        c = el if "lat" in el else el.get("center") or {}
        cat = next((k for k, t in _CATS.items() if tags.get("amenity") == t.split("=")[1]), None)
        if not cat or "lat" not in c:
            continue
        by.setdefault(cat, []).append({
            "category": cat, "name": (tags.get("name") or "")[:80] or None, "lat": c["lat"], "lon": c["lon"],
            "distance_km": round(haversine_m(lat, lon, c["lat"], c["lon"]) / 1000, 2),
            "phone": _phone(tags), "address": None, "source": "openstreetmap"})
    return by


def find_nearby_services(lat, lon, radius_m=8000, per_category=2, timeout=8.0):
    """Nearest services per category. Entries without a phone number are completed from the other source
    when the same place appears in both (matched by name or <150 m)."""
    merged = {}
    for fn in (_from_geoapify, _from_overpass):
        try:
            got = fn(lat, lon, radius_m, per_category, timeout)
        except Exception:  # noqa: BLE001
            got = None
        for cat, items in (got or {}).items():
            bucket = merged.setdefault(cat, [])
            for it in items:
                twin = next((b for b in bucket if (b["name"] and it["name"] and b["name"].lower() == it["name"].lower())
                             or haversine_m(b["lat"], b["lon"], it["lat"], it["lon"]) < 150), None)
                if twin:
                    twin["phone"] = twin["phone"] or it["phone"]
                    twin["address"] = twin.get("address") or it.get("address")
                else:
                    bucket.append(it)
    out = []
    for cat in _ORDER:
        out += sorted(merged.get(cat, []), key=lambda s: s["distance_km"])[:per_category]
    return out
