"""Route pre-assessment: sample the route, estimate arrival at each sample, assess each one."""
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

from .geometry import haversine_m
from .scoring import Sample, assess_point, level_for

DEFAULT_SPEED_MPS = 8.3      # about 30 km/h, an assumption the caller should override per travel mode
DEFAULT_STEP_M = 200.0


def densify(points, step_m=DEFAULT_STEP_M):
    """points: [(lat, lon)]. Returns [(lat, lon, cumulative_m)] with a sample about every step_m."""
    if not points:
        return []
    out, cum = [(points[0][0], points[0][1], 0.0)], 0.0
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
