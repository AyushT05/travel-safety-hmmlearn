"""Route planning, road path extraction, and safety pre-assessment.

Determines routes based on chosen start point, intermediate destinations, and final destination,
calculates road geometry via routing services (OSRM with graceful geodesic fallback), and assesses
safety risks along the route.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .geometry import haversine_m
from .scoring import Sample, assess_point, level_for

log = logging.getLogger("geofence.route")

DEFAULT_SPEED_MPS = 8.3       # ~30 km/h in city/hills
HIGHWAY_SPEED_MPS = 13.9      # ~50 km/h average intercity in India
DEFAULT_STEP_M = 200.0

OSRM_DEFAULT_URL = os.environ.get("OSRM_BASE_URL", "https://router.project-osrm.org")


def densify(points: List[Tuple[float, float]], step_m: float = DEFAULT_STEP_M) -> List[Tuple[float, float, float]]:
    """points: [(lat, lon)]. Returns [(lat, lon, cumulative_m)] with a sample about every step_m."""
    if not points:
        return []
    out: List[Tuple[float, float, float]] = [(points[0][0], points[0][1], 0.0)]
    cum = 0.0
    for (a_lat, a_lon), (b_lat, b_lon) in zip(points, points[1:]):
        seg = haversine_m(a_lat, a_lon, b_lat, b_lon)
        n = max(1, int(seg // step_m) + (1 if seg % step_m > 1 else 0))
        for i in range(1, n + 1):
            f = min(1.0, i * step_m / seg) if seg > 0 else 1.0
            out.append((a_lat + (b_lat - a_lat) * f, a_lon + (b_lon - a_lon) * f, cum + seg * f))
        cum += seg
    return out


@dataclass
class RouteAssessment:
    score: float
    level: str
    segments: list
    total_m: float
    sample_count: int
    route_source: str
    assumptions: dict
    notes: list = field(default_factory=list)


def assess_route(points, start, zones, coverage=None, speed_mps=DEFAULT_SPEED_MPS, step_m=DEFAULT_STEP_M,
                 route_source="polyline"):
    """route_source: "polyline" (a routed path) or "straight_line" (only stops are known).

    With straight_line the path between stops is a guess, so the result says so; it must not be
    presented as a verified route.
    """
    samples = densify(points, step_m)
    segments, current, worst = [], None, 0.0
    for lat, lon, cum in samples:
        eta = start + timedelta(seconds=cum / speed_mps)
        a = assess_point(Sample(lat, lon, eta, accuracy_m=0.0), zones, coverage)
        worst = max(worst, a.score)
        top = a.factors[0] if a.factors else None
        if top is None or a.level == "none":       # a negligible factor is not worth a warning
            current = None
            continue
        if current is not None and current["zone_key"] == top.zone_key:
            current["to_m"], current["eta_to"] = cum, eta
            if a.score > current["score"]:
                current.update(score=a.score, level=a.level, explanation=a.explanation, state=top.state)
        else:
            current = {"zone_key": top.zone_key, "name": top.name, "hazard_type": top.hazard_type,
                       "from_m": cum, "to_m": cum, "eta_from": eta, "eta_to": eta, "score": a.score,
                       "level": a.level, "state": top.state, "explanation": a.explanation,
                       "source": top.source, "source_url": top.source_url}
            segments.append(current)
    notes = []
    if route_source != "polyline":
        notes.append("Path between stops is approximated by straight lines; the real road may pass different places.")
    notes.append("Arrival times assume a constant speed; re-check if you are delayed.")
    total = samples[-1][2] if samples else 0.0
    return RouteAssessment(round(worst, 3), level_for(worst), segments, total, len(samples), route_source,
                           {"speed_mps": speed_mps, "step_m": step_m}, notes)


def fetch_road_route(
    waypoints: List[Tuple[float, float]],
    travel_mode: str = "driving",
    client: Optional[httpx.Client] = None,
    timeout_s: float = 8.0,
) -> Dict[str, Any]:
    """Fetch real turn-by-turn road polyline and duration from OSRM.

    waypoints: [(lat, lon), (lat, lon), ...]
    Returns dict with keys:
      - polyline: [[lat, lon], ...]
      - distance_m: float
      - duration_s: float
      - legs: list of leg dicts with distance, duration, steps
      - source: "osrm" or "fallback_geodesic"
    """
    if len(waypoints) < 2:
        return {
            "polyline": [[p[0], p[1]] for p in waypoints],
            "distance_m": 0.0,
            "duration_s": 0.0,
            "legs": [],
            "source": "trivial",
        }

    # Format OSRM coordinate string: lon,lat;lon,lat;...
    coords_str = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in waypoints)
    url = f"{OSRM_DEFAULT_URL.rstrip('/')}/route/v1/{travel_mode}/{coords_str}?overview=full&geometries=geojson&steps=true"

    should_close = False
    if client is None:
        client = httpx.Client(timeout=timeout_s)
        should_close = True

    try:
        resp = client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "Ok" and data.get("routes"):
                primary = data["routes"][0]
                # OSRM coordinates are [lon, lat] -> convert to [lat, lon]
                geom = primary.get("geometry", {}).get("coordinates", [])
                polyline = [[c[1], c[0]] for c in geom]
                legs_out = []
                for leg in primary.get("legs", []):
                    legs_out.append({
                        "distance_m": float(leg.get("distance", 0.0)),
                        "duration_s": float(leg.get("duration", 0.0)),
                        "summary": leg.get("summary", ""),
                    })
                return {
                    "polyline": polyline if len(polyline) >= 2 else [[p[0], p[1]] for p in waypoints],
                    "distance_m": float(primary.get("distance", 0.0)),
                    "duration_s": float(primary.get("duration", 0.0)),
                    "legs": legs_out,
                    "source": "osrm",
                }
    except Exception as exc:
        log.warning("OSRM route fetch failed (%s), falling back to densified geodesic: %s", type(exc).__name__, exc)
    finally:
        if should_close:
            client.close()

    # Fallback: Densified direct interpolation between stops with Indian highway speed estimate
    dense_pts = densify(waypoints, step_m=100.0)
    total_m = dense_pts[-1][2] if dense_pts else 0.0
    est_duration_s = total_m / HIGHWAY_SPEED_MPS
    polyline = [[p[0], p[1]] for p in dense_pts]

    # Approximate legs
    legs_out = []
    for (lat1, lon1), (lat2, lon2) in zip(waypoints, waypoints[1:]):
        leg_dist = haversine_m(lat1, lon1, lat2, lon2)
        legs_out.append({
            "distance_m": round(leg_dist, 1),
            "duration_s": round(leg_dist / HIGHWAY_SPEED_MPS, 1),
            "summary": "Direct segment",
        })

    return {
        "polyline": polyline,
        "distance_m": round(total_m, 1),
        "duration_s": round(est_duration_s, 1),
        "legs": legs_out,
        "source": "fallback_geodesic",
    }


def plan_and_assess_route(
    waypoint_dicts: List[Dict[str, Any]],
    start_time: Optional[datetime] = None,
    zones: Optional[List[Any]] = None,
    travel_mode: str = "driving",
    client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """Complete route determination and risk assessment pipeline.

    waypoint_dicts: [
        {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
        {"name": "Mathura", "lat": 27.4924, "lon": 77.6737}, # intermediate (optional)
        {"name": "Agra", "lat": 27.1767, "lon": 78.0081}      # final destination
    ]
    """
    valid_points = []
    names = []
    for wp in waypoint_dicts:
        if wp and wp.get("lat") is not None and wp.get("lon") is not None:
            valid_points.append((float(wp["lat"]), float(wp["lon"])))
            names.append(str(wp.get("name", "Waypoint")))

    if len(valid_points) < 2:
        return {
            "error": "At least 2 valid waypoints with latitude and longitude are required.",
            "success": False,
        }

    route_data = fetch_road_route(valid_points, travel_mode=travel_mode, client=client)
    polyline = route_data["polyline"]
    total_m = route_data["distance_m"]
    duration_s = route_data["duration_s"]

    # Calculate waypoint milestones
    t_start = start_time or datetime.now(timezone.utc)
    waypoint_milestones = []
    cum_dist = 0.0
    for idx, (lat, lon) in enumerate(valid_points):
        leg_dist = route_data["legs"][idx - 1]["distance_m"] if idx > 0 and idx - 1 < len(route_data["legs"]) else 0.0
        cum_dist += leg_dist
        cum_dur_s = (cum_dist / total_m * duration_s) if total_m > 0 else 0.0
        eta = t_start + timedelta(seconds=cum_dur_s)
        waypoint_milestones.append({
            "order": idx,
            "name": names[idx],
            "lat": lat,
            "lon": lon,
            "distance_from_start_km": round(cum_dist / 1000.0, 2),
            "estimated_arrival": eta.isoformat(),
        })

    # Assess hazard zones along the road
    assessment_dict = None
    if zones is not None:
        pts_tuple = [(p[0], p[1]) for p in polyline]
        speed_mps = (total_m / duration_s) if duration_s > 0 else DEFAULT_SPEED_MPS
        ass = assess_route(
            pts_tuple,
            start=t_start,
            zones=zones,
            speed_mps=max(5.0, min(speed_mps, 25.0)),
            route_source=route_data["source"],
        )
        assessment_dict = {
            "risk_score": ass.score,
            "risk_level": ass.level,
            "hazard_segments": ass.segments,
            "notes": ass.notes,
        }

    return {
        "success": True,
        "source": route_data["source"],
        "total_distance_km": round(total_m / 1000.0, 2),
        "total_duration_hours": round(duration_s / 3600.0, 2),
        "waypoints": waypoint_milestones,
        "legs": route_data["legs"],
        "polyline": polyline,
        "assessment": assessment_dict,
    }
