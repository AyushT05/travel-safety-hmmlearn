"""Nearby emergency services via Overpass (same categories as server.py's proxy). Best effort."""
import httpx

from .geo import haversine_m

_CATS = {"hospital": "amenity=hospital", "police": "amenity=police", "fire_station": "amenity=fire_station",
         "clinic": "amenity=clinic", "pharmacy": "amenity=pharmacy"}


def find_nearby_services(lat, lon, radius_m=5000, per_category=2, timeout=8.0):
    q = "[out:json][timeout:8];(" + "".join(
        f"node[{tag}](around:{radius_m},{lat},{lon});" for tag in _CATS.values()) + ");out body;"
    try:
        r = httpx.post("https://overpass-api.de/api/interpreter", data={"data": q},
                       headers={"User-Agent": "MargRakshak-Agent/1.0"}, timeout=timeout)
        r.raise_for_status()
        elements = r.json().get("elements", [])
    except Exception:  # noqa: BLE001
        return []
    by = {}
    for el in elements:
        tags = el.get("tags", {})
        cat = next((c for c, t in _CATS.items() if tags.get("amenity") == t.split("=")[1]), None)
        if not cat or "lat" not in el:
            continue
        by.setdefault(cat, []).append({
            "category": cat, "name": (tags.get("name") or "")[:80] or None,
            "lat": el["lat"], "lon": el["lon"],
            "distance_km": round(haversine_m(lat, lon, el["lat"], el["lon"]) / 1000, 2),
            "phone": tags.get("phone") or tags.get("contact:phone"),
        })
    out = []
    for cat in ("police", "hospital", "fire_station", "clinic", "pharmacy"):
        out += sorted(by.get(cat, []), key=lambda s: s["distance_km"])[:per_category]
    return out
