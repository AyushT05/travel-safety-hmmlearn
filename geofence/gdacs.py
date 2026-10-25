"""GDACS event list -> risk_zones rows. Pure functions (no network); built against a real feed sample.

What the real feed taught us (checked on 86 events, 17 months of history):
  * The list only holds Orange/Red events and keeps old ones, so `iscurrent == "true"` (a STRING) is
    the only reliable "still active" flag. Only 2 of 86 were current. `todate` is not reliable for that:
    a current cyclone had a todate a day in the past.
  * Every feature is a Point (centroid). Real footprints come from a second call (properties.url.geometry),
    which is NOT used yet, so zones are approximate discs and are labelled and discounted as such.
  * Severity units differ per type (EQ magnitude, TC km/h, WF hectares) and floods carry no severity (0).
    So severity here comes from the GDACS alert level, not from the number.
  * Timestamps have no timezone; GDACS publishes UTC.
"""
from datetime import datetime, timedelta, timezone

SOURCE = "GDACS"
PREFIX = "gdacs:"
GDACS_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"

# (min_lat, min_lon, max_lat, max_lon): India plus a margin, because cyclones form at sea and
# affect the coast. Override with GEOFENCE_INTEREST_BBOX="2,62,42,102".
DEFAULT_INTEREST_BBOX = (2.0, 62.0, 42.0, 102.0)

HAZARD = {"TC": "cyclone", "FL": "flood", "EQ": "earthquake", "WF": "wildfire"}   # VO, DR deliberately skipped
SEVERITY = {"Red": "critical", "Orange": "high", "Green": "medium"}

ALIVE_TTL = timedelta(hours=6)       # a zone dies by itself if the hourly sync stops refreshing it
EQ_WINDOW = timedelta(hours=72)      # an earthquake is a hazard for aftershocks and damage, for a few days
APPROX_CONFIDENCE = 0.4              # a guessed disc around a centroid is weak evidence of the real footprint


def _dt(s):
    if not s:
        return None
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def approx_radius_m(etype, severity_value):
    """Placeholder footprints until the real polygons are used. Deliberately round numbers."""
    v = severity_value or 0.0
    if etype == "TC":
        return 250_000 if v >= 178 else 150_000 if v >= 118 else 100_000
    if etype == "EQ":
        return 100_000 if v >= 7 else 50_000 if v >= 6 else 25_000
    if etype == "FL":
        return 30_000
    return 20_000


def _in_bbox(lat, lon, bbox):
    return bbox[0] <= lat <= bbox[2] and bbox[1] <= lon <= bbox[3]


def feature_to_row(feat, now, bbox=DEFAULT_INTEREST_BBOX):
    """One GDACS feature -> a risk_zones row dict, or (None, reason) when it should be skipped."""
    p = feat.get("properties") or {}
    g = feat.get("geometry") or {}
    etype = p.get("eventtype")
    if etype not in HAZARD:
        return None, "type_not_used"
    if str(p.get("iscurrent")).lower() != "true":
        return None, "not_current"
    coords = g.get("coordinates")
    if g.get("type") != "Point" or not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return None, "no_point_geometry"
    lon, lat = float(coords[0]), float(coords[1])
    if not _in_bbox(lat, lon, bbox):
        return None, "outside_region"
    started = _dt(p.get("fromdate"))
    modified = _dt(p.get("datemodified")) or started or now
    if etype == "EQ":
        if started is None or now - started > EQ_WINDOW:
            return None, "earthquake_window_passed"
        expires = started + EQ_WINDOW
    else:
        expires = now + ALIVE_TTL
    sev = (p.get("severitydata") or {})
    sev_val = sev.get("severity") if isinstance(sev.get("severity"), (int, float)) else None
    level = p.get("alertlevel")
    row = {
        "name": str(p.get("name") or p.get("eventname") or f"GDACS {etype} {p.get('eventid')}")[:200],
        "hazard_type": HAZARD[etype],
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "radius_m": int(approx_radius_m(etype, sev_val)),
        "severity": SEVERITY.get(level, "medium"),
        "confidence": APPROX_CONFIDENCE,
        "source": SOURCE,
        "source_url": (p.get("url") or {}).get("report"),
        "source_ref": f"{PREFIX}{etype}:{p.get('eventid')}",     # one row per event, updated by each episode
        "observed_at": modified.isoformat(),
        "expires_at": expires.isoformat(),
        "status": "active",
        "attributes": {
            "footprint": "approximate_disc", "event_type": etype, "event_id": p.get("eventid"),
            "episode_id": p.get("episodeid"), "alert_level": level, "alert_score": p.get("alertscore"),
            "severity_value": sev_val, "severity_text": sev.get("severitytext"), "country": p.get("country"),
            "from": p.get("fromdate"), "to": p.get("todate"), "glide": p.get("glide") or None,
        },
    }
    return row, None


def parse_event_list(fc, now, bbox=DEFAULT_INTEREST_BBOX):
    """-> (rows, skipped_counts). Raises ValueError if the payload is not a usable feed.

    An empty or malformed feed is an ERROR, never "no hazards": the real list always carries many
    historical events, so zero features means the fetch or the API is broken.
    """
    feats = fc.get("features") if isinstance(fc, dict) else None
    if not isinstance(feats, list) or not feats:
        raise ValueError("GDACS feed has no features; refusing to treat that as 'no hazards'")
    rows, skipped, seen = [], {}, set()
    for f in feats:
        row, why = feature_to_row(f, now, bbox)
        if row is None:
            skipped[why] = skipped.get(why, 0) + 1
        elif row["source_ref"] not in seen:
            seen.add(row["source_ref"])
            rows.append(row)
    return rows, skipped


def refs_to_expire(existing_active_refs, rows):
    """GDACS zones we stored earlier that the feed no longer lists as current."""
    keep = {r["source_ref"] for r in rows}
    return sorted(r for r in existing_active_refs if r.startswith(PREFIX) and r not in keep)
