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

What the real cyclone geometry response taught us (event 1001338, 112 features, 470 KB):
  * It mixes hazard footprints with things that only LOOK like footprints. Used: the per-time-step
    wind fields (featuretype "WindRadii", classes Poly_Green/Orange/Red = winds of 60/90/120 km/h).
    NOT used: the cumulative "whole storm life" swaths (a 3.5 million km2 Green polygon covering the
    past track), the forecast uncertainty cone (track uncertainty, not wind), track lines, and the
    tiny per-fix point circles.
  * Later forecast steps can be degenerate (every vertex identical, area 0) once the storm weakens.
    Those carry no area and are skipped.
  * A wind field is a snapshot at a time, so each one becomes a zone with a validity window.
  * Only cyclones get footprints so far. Floods, earthquakes and wildfires stay approximate discs
    until their geometry responses have been seen.
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
            "geometry_url": (p.get("url") or {}).get("geometry"),
            "event_ref": f"{PREFIX}{etype}:{p.get('eventid')}",   # all zones of one event share this: one alert, not one per band
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


# ---------------------------------------------------------------------------------- cyclone footprints
BANDS = {"Poly_Red": ("critical", 120), "Poly_Orange": ("high", 90), "Poly_Green": ("medium", 60)}
MIN_AREA_KM2 = 1.0                       # below this a polygon is a degenerate point, not a wind field
HORIZON = timedelta(hours=96)            # forecast steps further ahead than this are not stored
EDGE_PADDING = timedelta(hours=12)       # validity added before the first and after the last step
ANALYSIS_CONFIDENCE = 0.7                # wind field for a time that has already happened
FORECAST_CONFIDENCE_FLOOR = 0.4          # forecast confidence falls by 0.1 per day ahead, down to this


def polygon_area_km2(rings):
    """Approximate area of a GeoJSON polygon (outer ring minus holes), in km2."""
    import math

    def one(ring):
        lat0 = sum(c[1] for c in ring) / len(ring)
        k = 111.195
        pts = [(c[0] * k * math.cos(math.radians(lat0)), c[1] * k) for c in ring]
        a = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))
        return abs(a) / 2.0
    if not rings or not rings[0]:
        return 0.0
    return max(0.0, one(rings[0]) - sum(one(h) for h in rings[1:] if h))


def _footprint_confidence(step_t, now):
    if step_t <= now:
        return ANALYSIS_CONFIDENCE
    days = (step_t - now).total_seconds() / 86400.0
    return round(max(FORECAST_CONFIDENCE_FLOOR, 0.65 - 0.1 * days), 2)


def tc_footprint_rows(parent, fc, now):
    """Wind-field polygons of one cyclone -> risk_zones rows, one per (time step, wind band).

    Each wind field is valid from the previous step to the next step. Overlapping windows mean that,
    at any moment, the fields on both sides of "now" apply, which covers the area swept while the
    storm moves between snapshots instead of leaving gaps.
    Returns [] when the response is valid but holds no wind-field area (a dissipated storm).
    Raises ValueError when the response is not a usable geometry feed.
    """
    from .geometry import GeometryError, bbox_of

    feats = fc.get("features") if isinstance(fc, dict) else None
    if not isinstance(feats, list):
        raise ValueError("GDACS geometry response has no features list")
    steps = {}
    for f in feats:
        p, g = f.get("properties") or {}, f.get("geometry") or {}
        if p.get("featuretype") != "WindRadii" or p.get("Class") not in BANDS:
            continue
        if str(p.get("visible")).lower() == "false" or g.get("type") != "Polygon":
            continue
        t = _dt(p.get("polygondate"))
        if t is not None:
            steps.setdefault(t, {})[p["Class"]] = g
    times = sorted(steps)
    rows = []
    for i, t in enumerate(times):
        start = times[i - 1] if i > 0 else t - EDGE_PADDING
        end = times[i + 1] if i < len(times) - 1 else t + EDGE_PADDING
        if end <= now or start > now + HORIZON:
            continue
        for cls, (sev, kmh) in BANDS.items():
            g = steps[t].get(cls)
            if not g or polygon_area_km2(g.get("coordinates")) < MIN_AREA_KM2:
                continue
            try:
                bbox_of(g)
            except GeometryError:
                continue
            attrs = {**parent["attributes"], "footprint": "wind_radii", "band_kmh": kmh,
                     "step_time": t.isoformat(), "valid_from": start.isoformat(), "valid_until": end.isoformat(),
                     "forecast": t > now}
            rows.append({
                "name": f"{parent['name']}: winds >= {kmh} km/h at {t:%d %b %H:%M} UTC"[:200],
                "hazard_type": "cyclone", "geometry": g, "radius_m": 0, "severity": sev,
                "confidence": _footprint_confidence(t, now), "source": SOURCE, "source_url": parent["source_url"],
                "source_ref": f"{parent['source_ref']}:{t:%Y%m%d%H%M}:{kmh}",
                "observed_at": now.isoformat(), "expires_at": end.isoformat(), "status": "active",
                "attributes": attrs,
            })
    return rows


def enrich_with_footprints(rows, fetch, now):
    """Replace each cyclone's approximate disc with real wind fields where the geometry is available.

    * geometry fetched, wind fields found  -> the disc is replaced by the polygons
    * geometry fetched, no wind-field area -> the event is dropped (a dissipated storm is not a hazard)
    * geometry could not be fetched/parsed -> the disc is KEPT, so an outage never hides a cyclone
    """
    out = []
    report = {"footprint_rows": 0, "fallback_discs": 0, "dropped_no_wind_field": 0, "errors": []}
    for r in rows:
        url = (r.get("attributes") or {}).get("geometry_url")
        if (r.get("attributes") or {}).get("event_type") != "TC" or not url:
            out.append(r)
            continue
        try:
            fp = tc_footprint_rows(r, fetch(url), now)
        except Exception as e:  # noqa: BLE001
            out.append(r)
            report["fallback_discs"] += 1
            report["errors"].append(f"{r['source_ref']}: {type(e).__name__}")
            continue
        if fp:
            out.extend(fp)
            report["footprint_rows"] += len(fp)
        else:
            report["dropped_no_wind_field"] += 1
    return out, report
