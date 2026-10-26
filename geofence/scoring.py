"""Time-aware, accuracy-aware hazard scoring for one position.

Design choices that differ from the plain weighted average in the guide, and why:
  * The guide's R = sum(w*s*c*f) / sum(w) divides by every factor's weight, so one verified flood
    closure gets diluted by any number of irrelevant factors. Here each applicable factor yields a
    contribution x_i = severity * confidence * freshness * time_weight * proximity, and the score is
    the largest contribution plus a small bonus for corroborating hazards of a DIFFERENT group.
  * Connectivity is capped below the "warning" level: weak signal raises the risk of a delayed SOS,
    it is not physical danger.
  * Missing data is never treated as safe or as unsafe. Callers pass the groups that were actually
    checked, and the rest come back as `unchecked`.
All weights and thresholds are starting hypotheses, not calibrated values.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .geometry import GeometryError, angle_diff, bearing_deg, distance_to_geometry, geometry_kind, signed_distance_to_geometry
from .sun import is_dark, sun_times, local_solar_date
from .zones import Zone, aware

SEVERITY = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
WARN_BASE_M = {"low": 100.0, "medium": 200.0, "high": 400.0, "critical": 800.0}
LEAD_TIME_S = 30.0            # extra warning distance = speed * this
DEFAULT_ACCURACY_M = 50.0     # used when the fix gives no accuracy: do not assume it is good
APPROACH_ANGLE_DEG = 60.0
MIN_MOVING_SPEED_MPS = 1.0
CLOSING_MIN_M = 10.0          # distance to a zone must shrink by at least this (more with a poor fix) to count as approaching
CLOSING_MAX_M = 30.0
DEFAULT_BUFFER_M = {"point": 100.0, "line": 30.0, "polygon": 0.0}
STALE_FLOOR = 0.05

PROX_FACTOR = {"inside": 1.0, "uncertain_boundary": 0.8, "approaching": 0.6, "near": 0.3, "outside": 0.0}
LEVELS = (("none", 0.0), ("caution", 0.15), ("warning", 0.40), ("danger", 0.65))


@dataclass(frozen=True)
class Profile:
    group: str                 # lighting | crime | connectivity | road | disaster | restricted
    label: str
    day_w: float
    night_w: float
    cap: float                 # maximum contribution this hazard type can make on its own
    half_life_days: Optional[float]   # freshness decay when the zone has no expires_at


PROFILES = {
    "poor_lighting":     Profile("lighting", "Poorly lit area", 0.05, 1.0, 0.70, 730),
    "crime":             Profile("crime", "Reported crime / security risk", 0.6, 1.0, 0.90, 180),
    "weak_connectivity": Profile("connectivity", "Weak mobile coverage", 0.5, 0.5, 0.35, 365),
    "road_condition":    Profile("road", "Poor road condition", 0.6, 0.8, 0.80, 180),
    "road_closure":      Profile("road", "Road closure", 1.0, 1.0, 1.00, 30),
    "flood":             Profile("disaster", "Flood", 1.0, 1.0, 1.00, 7),
    "landslide":         Profile("disaster", "Landslide", 1.0, 1.0, 1.00, 7),
    "cyclone":           Profile("disaster", "Cyclone", 1.0, 1.0, 1.00, 7),
    "earthquake":        Profile("disaster", "Earthquake", 1.0, 1.0, 1.00, 7),
    "wildfire":          Profile("disaster", "Wildfire", 1.0, 1.0, 1.00, 7),
    "other_disaster":    Profile("disaster", "Disaster alert", 1.0, 1.0, 1.00, 7),
    "restricted_area":   Profile("restricted", "Restricted area", 0.8, 0.8, 0.80, None),
}
ALL_GROUPS = ("lighting", "crime", "connectivity", "road", "disaster")


@dataclass(frozen=True)
class Sample:
    lat: float
    lon: float
    t: datetime                      # when the tourist is (or will be) here
    accuracy_m: Optional[float] = None   # 0 for a planned route point, None for "fix gave none"
    heading_deg: Optional[float] = None
    speed_mps: Optional[float] = None


@dataclass
class Proximity:
    state: str
    distance_m: float
    buffer_m: float
    warn_m: float
    bearing_to_zone: Optional[float] = None


@dataclass
class Factor:
    zone_key: str
    name: str
    hazard_type: str
    group: str
    state: str
    distance_m: float
    severity: str
    confidence: float
    freshness: float
    time_weight: float
    contribution: float
    source: str
    source_url: Optional[str]
    explanation: str


@dataclass
class Assessment:
    score: float
    level: str
    factors: list
    unchecked: list
    excluded: list
    is_dark: bool
    evaluated_at: datetime
    explanation: str


def level_for(score):
    lvl = "none"
    for name, floor in LEVELS:
        if score >= floor:
            lvl = name
    return lvl


def freshness(zone: Zone, t: datetime):
    """-> (value 0..1, None) or (None, reason) when the zone must be left out."""
    if zone.status != "active":
        return None, "inactive"
    t = aware(t)
    if zone.valid_from is not None and t < aware(zone.valid_from):
        return None, "not_yet_valid"
    if zone.expires_at is not None:
        return (None, "expired_before_arrival") if t > aware(zone.expires_at) else (1.0, None)
    prof = PROFILES[zone.hazard_type]
    if prof.half_life_days is None:
        return 1.0, None
    if zone.observed_at is None:
        return 0.5, None            # age unknown: trust it half as much
    age_days = max(0.0, (t - aware(zone.observed_at)).total_seconds() / 86400.0)
    f = 0.5 ** (age_days / prof.half_life_days)
    return (f, None) if f >= STALE_FLOOR else (None, "stale")


def zone_proximity(sample: Sample, zone: Zone, prev: Optional[Sample] = None) -> Proximity:
    signed, nlat, nlon = signed_distance_to_geometry(sample.lat, sample.lon, zone.geometry)
    dist = max(0.0, signed)                              # metres to the geometry, 0 inside a polygon
    buf = zone.radius_m if zone.radius_m is not None else DEFAULT_BUFFER_M[geometry_kind(zone.geometry)]
    acc = DEFAULT_ACCURACY_M if sample.accuracy_m is None else max(0.0, sample.accuracy_m)
    # Negative gap: inside the buffered zone. For polygons the SIGNED distance carries the depth, so a
    # position deep inside is "inside" and one within the accuracy circle of the edge is "uncertain".
    gap = signed - buf
    speed = sample.speed_mps or 0.0
    warn = WARN_BASE_M.get(zone.severity, 200.0) + speed * LEAD_TIME_S
    bearing = bearing_deg(sample.lat, sample.lon, nlat, nlon) if dist > 0 else None

    if gap <= -acc:
        state = "inside"
    elif gap <= acc:
        state = "uncertain_boundary"                    # the accuracy circle overlaps the boundary
    elif gap <= warn:
        state = "near"
        if sample.heading_deg is not None and speed >= MIN_MOVING_SPEED_MPS and bearing is not None:
            if angle_diff(sample.heading_deg, bearing) <= APPROACH_ANGLE_DEG:
                state = "approaching"
        elif prev is not None:
            # No heading: infer approach from the distance shrinking since an earlier fix. A phone lying
            # still still jitters by several metres, so (a) a phone that reports it is not moving never
            # "approaches", and (b) the shrinkage must clearly exceed the fix's own uncertainty.
            speed_known = sample.speed_mps is not None and sample.speed_mps >= 0
            moving = (not speed_known) or sample.speed_mps >= MIN_MOVING_SPEED_MPS
            pd, _, _ = distance_to_geometry(prev.lat, prev.lon, zone.geometry)
            if moving and pd - dist > max(CLOSING_MIN_M, min(2 * acc, CLOSING_MAX_M)):
                state = "approaching"
    else:
        state = "outside"
    return Proximity(state, dist, buf, warn, bearing)


_STATE_TEXT = {"inside": "inside this area", "uncertain_boundary": "at the edge of this area (position uncertain)",
               "approaching": "approaching this area", "near": "near this area"}


def _explain(zone, prof, prox, dark, f, sample, rise_set):
    where = _STATE_TEXT.get(prox.state, prox.state)
    if prox.state in ("approaching", "near") and prox.distance_m > prox.buffer_m:
        where += f", about {int(round(prox.distance_m - prox.buffer_m, -1))} m away"
    parts = [f"{prof.label}: {zone.name}. You are {where}."]
    if prof.night_w > prof.day_w and dark:
        parts.append("You will be here after dark.")
    elif prof.night_w > prof.day_w and not dark:
        parts.append("Matters mainly after dark.")
    if zone.expires_at is None and f < 0.9:
        parts.append("The report is not recent.")
    parts.append(f"Source: {zone.source}, confidence {int(round(zone.confidence * 100))}%.")
    return " ".join(parts)


def assess_point(sample: Sample, zones, coverage=None, prev: Optional[Sample] = None) -> Assessment:
    t = aware(sample.t)
    dark = is_dark(t, sample.lat, sample.lon)
    rise_set = sun_times(local_solar_date(t, sample.lon), sample.lat, sample.lon)
    factors, excluded = [], []
    for z in zones:
        prof = PROFILES.get(z.hazard_type)
        if prof is None:
            excluded.append((z.key, "unknown_hazard_type"))
            continue
        f, why = freshness(z, t)
        if f is None:
            excluded.append((z.key, why))
            continue
        try:
            prox = zone_proximity(sample, z, prev)
        except GeometryError:
            excluded.append((z.key, "bad_geometry"))
            continue
        pf = PROX_FACTOR[prox.state]
        if pf == 0.0:
            continue
        w = prof.night_w if dark else prof.day_w
        sev = SEVERITY.get(z.severity, 0.5)
        contrib = min(sev * max(0.0, min(1.0, z.confidence)) * f * w * pf, prof.cap)
        if contrib < 0.01:
            continue
        factors.append(Factor(z.key, z.name, z.hazard_type, prof.group, prox.state, round(prox.distance_m, 1),
                              z.severity, z.confidence, round(f, 3), w, round(contrib, 3), z.source,
                              z.source_url, _explain(z, prof, prox, dark, f, sample, rise_set)))
    factors.sort(key=lambda x: x.contribution, reverse=True)

    score = 0.0
    if factors:
        top = factors[0]
        other_groups = {x.group for x in factors[1:] if x.contribution >= 0.3 and x.group != top.group}
        score = min(1.0, top.contribution + 0.1 * len(other_groups))
    unchecked = [g for g in ALL_GROUPS if coverage is not None and g not in coverage]
    level = level_for(score)
    if factors:
        expl = " ".join(f.explanation for f in factors[:3])
    else:
        expl = "No known hazard matches this position and time. This is not a guarantee of safety."
    if unchecked:
        expl += " Not checked (no data): " + ", ".join(unchecked) + "."
    return Assessment(round(score, 3), level, factors, unchecked, excluded, dark, t, expl)
