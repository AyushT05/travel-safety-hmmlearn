"""
"What could the problem be?" for an SOS, inferred from WHERE it came from and what was happening, never from
a generic list. Every candidate cites the evidence that raised it; with no evidence there is no candidate, and
the brief says so. These are possibilities for a human to check, not findings.
"""
import httpx

from .geo import haversine_m

# place kind (geocode.py) -> candidate issues with the reason they fit that kind of place
_PLACE = {
    "highway": [("vehicle_breakdown_or_collision", "SOS came from a highway, where breakdowns, collisions and being stranded are the common emergencies"),
                ("unsafe_to_stop", "stopping on a highway is itself hazardous, especially after dark")],
    "main_road": [("road_accident", "SOS came from a main road with traffic")],
    "remote_track": [("stranded_or_lost", "SOS came from a track or path away from regular roads, where getting stranded or lost is likely"),
                     ("no_help_nearby", "few people and services are usually near such places")],
    "wilderness": [("lost_or_injured_outdoors", "SOS came from forest/hill/protected terrain, where injury, getting lost and wildlife are the usual risks"),
                   ("hard_to_reach", "emergency vehicles may take long to reach this terrain")],
    "waterside": [("water_related_emergency", "SOS came from the waterside (river, lake or beach): drowning, slipping and sudden water-level rise are the usual risks")],
    "busy_public": [("theft_or_harassment", "SOS came from a busy public place (market, attraction or eatery), where theft, harassment and crowd crush happen"),
                    ("sudden_illness_in_crowd", "a medical event in a crowd can also trigger an SOS")],
    "transit": [("theft_or_missed_connection", "SOS came from a station or stop, where theft, confusion and crowding are common")],
    "services": [("medical_or_police_matter", "SOS came from a hospital, clinic or police location, so it may relate to a medical or police matter already underway")],
    "lodging_or_home": [("lodging_dispute_or_intrusion", "SOS came from lodging or a residential address")],
    "street": [("personal_safety_or_medical", "SOS came from a residential or local street")],
    "built_up": [("personal_safety_or_medical", "SOS came from a built-up area")],
}
_ZONE = {"robbery": "robbery", "assault": "assault", "harassment": "harassment", "poor_lighting": "poor lighting",
         "suspicious_activity": "suspicious activity", "other": "a safety concern"}


def weather_now(lat, lon, timeout=5.0):
    try:
        r = httpx.get("https://api.open-meteo.com/v1/forecast", timeout=timeout, params={
            "latitude": lat, "longitude": lon, "current": "weather_code,precipitation,wind_gusts_10m,apparent_temperature,visibility"})
        r.raise_for_status()
        return r.json().get("current") or {}
    except Exception:  # noqa: BLE001
        return {}


def likely_issues(place, night, zones, lat, lon, assessment=None, events=None, weather=None, alert_source=None):
    """Returns [{id, issue, why: [..], confidence: 'plausible'|'possible'}] (max 6), strongest evidence first."""
    out, seen = [], set()

    def add(i, why, conf="possible"):
        if i in seen:
            for o in out:
                if o["id"] == i and why not in o["why"]:
                    o["why"].append(why)
                    if conf == "plausible":
                        o["confidence"] = "plausible"
            return
        seen.add(i)
        out.append({"id": i, "issue": i.replace("_", " "), "why": [why], "confidence": conf})

    kind = (place or {}).get("kind")
    for i, why in _PLACE.get(kind, []):
        add(i, why)
    name = (place or {}).get("name")
    if name:
        for o in out:
            o["why"][0] = o["why"][0] + f" ({name})"

    for z in zones or []:
        d = haversine_m(lat, lon, z["lat"], z["lon"])
        if d <= (z.get("radius_m") or 500) + 300:
            cat = _ZONE.get(z.get("category"), "a safety concern")
            add(f"reported_{z.get('category', 'concern')}", f"a verified report of {cat} covers this spot (zone {int(d)} m away)", "plausible")

    for h in (assessment or {}).get("hazard_zones") or []:
        add(f"hazard_{h.get('type', 'zone')}", f"the zone safety score lists a {str(h.get('type', 'hazard')).replace('_', ' ')} here", "plausible")

    w = weather or {}
    code, gust, feels, vis, rain = w.get("weather_code"), w.get("wind_gusts_10m") or 0, w.get("apparent_temperature"), w.get("visibility"), w.get("precipitation") or 0
    if code in (95, 96, 99) or rain >= 12:
        add("severe_weather", "thunderstorm or very heavy rain was reported at this location when the SOS came in", "plausible")
    if gust >= 60:
        add("severe_weather", f"wind gusts around {round(gust)} km/h at this location")
    if feels is not None and feels >= 40:
        add("heat_exhaustion", f"it feels like {round(feels)}°C here")
    if feels is not None and feels <= 3:
        add("cold_exposure", f"it feels like {round(feels)}°C here")
    if vis is not None and vis < 300:
        add("poor_visibility_hazard", f"visibility is only about {int(vis)} m")

    for e in events or []:
        k = e.get("kind")
        if k == "abrupt_stop":
            add("vehicle_breakdown_or_collision", "the vehicle stopped abruptly shortly before the SOS" + (" with a collision-grade deceleration" if (e.get("detail") or {}).get("hard") else ""), "plausible")
        elif k == "route_deviation":
            add("lost_or_diverted", "the traveller was off the planned route before the SOS", "plausible")
        elif k in ("signal_lost", "repeated_signal_gaps"):
            add("no_help_nearby", "location updates were dropping out before the SOS (weak coverage can delay help)")
        elif k in ("erratic_movement", "circling"):
            add("lost_or_diverted", "movement was erratic or circular before the SOS")
        elif k == "in_hazard_zone":
            add("reported_hazard_area", "the traveller was inside a verified hazard zone", "plausible")

    if night and kind in (None, "unknown", "remote_track", "wilderness", "highway", "street"):
        add("personal_safety_after_dark", "the SOS came after dark in a place with few people around")
    if (assessment or {}).get("isolation_score") is not None and assessment["isolation_score"] >= 0.6:
        add("no_help_nearby", "the area looks sparsely populated, so nearby help may be limited")
    if alert_source == "sms":
        add("no_data_connection", "the SOS arrived by SMS, which means the phone had no data connection")

    rank = {"plausible": 0, "possible": 1}
    return sorted(out, key=lambda o: (rank[o["confidence"]], -len(o["why"])))[:6]
