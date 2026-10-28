"""Route deviation detection and spatial cross-track analysis.

Monitors tourist GPS fixes against the planned route polyline. Detects when a tourist drifts off
the designated route beyond safety tolerances, filters GPS jitter via temporal hysteresis,
and generates/auto-resolves 'deviation' alerts in the database.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from .geometry import haversine_m

# Deviation thresholds in meters
CORRIDOR_SAFE_M = 500.0         # Normal corridor along road / highway
CORRIDOR_CAUTION_M = 1200.0     # Slight detour (e.g. gas station, rest stop)
CORRIDOR_DEVIATION_M = 1800.0   # Significant deviation requiring alert
RESOLVE_TOLERANCE_M = 400.0     # Must return within this distance to clear alert

CONSECUTIVE_OFF_ROUTE_THRESHOLD = 3   # 3 consecutive fixes before alerting (~15s)
CONSECUTIVE_ON_ROUTE_THRESHOLD = 3    # 3 consecutive fixes to auto-clear


def point_to_segment_distance_m(
    lat: float, lon: float,
    a_lat: float, a_lon: float,
    b_lat: float, b_lon: float,
) -> Tuple[float, Tuple[float, float], float]:
    """Calculate distance in meters from point P to line segment AB.

    Returns: (distance_m, (closest_lat, closest_lon), segment_projection_ratio_t)
    """
    # Deg to rad
    r_lat = math.radians((a_lat + b_lat) / 2.0)
    cos_lat = math.cos(r_lat)

    # Local equirectangular planar projection (accurate over tens of kilometers)
    # x in meters east, y in meters north relative to A
    DEG_TO_M = 111319.5
    ax, ay = 0.0, 0.0
    bx = (b_lon - a_lon) * cos_lat * DEG_TO_M
    by = (b_lat - a_lat) * DEG_TO_M
    px = (lon - a_lon) * cos_lat * DEG_TO_M
    py = (lat - a_lat) * DEG_TO_M

    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq <= 1e-6:
        # Segment A and B are identical
        dist = haversine_m(lat, lon, a_lat, a_lon)
        return dist, (a_lat, a_lon), 0.0

    # Projection factor t along AB
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t_clamped = max(0.0, min(1.0, t))

    closest_x = ax + t_clamped * dx
    closest_y = ay + t_clamped * dy

    # Re-project to lat/lon
    closest_lat = a_lat + closest_y / DEG_TO_M
    closest_lon = a_lon + (closest_x / DEG_TO_M) / cos_lat if cos_lat != 0 else a_lon

    dist_m = haversine_m(lat, lon, closest_lat, closest_lon)
    return dist_m, (closest_lat, closest_lon), t_clamped


def calculate_cross_track(
    lat: float, lon: float,
    polyline: List[List[float]],
) -> Dict[str, Any]:
    """Finds closest point and distance from (lat, lon) to polyline [[lat, lon], ...].

    Returns:
      - distance_m: float
      - closest_point: [lat, lon]
      - segment_index: int
      - progress_ratio: float (0.0 to 1.0 along the polyline)
    """
    if not polyline:
        return {
            "distance_m": 0.0,
            "closest_point": [lat, lon],
            "segment_index": 0,
            "progress_ratio": 0.0,
        }

    if len(polyline) == 1:
        d = haversine_m(lat, lon, polyline[0][0], polyline[0][1])
        return {
            "distance_m": d,
            "closest_point": polyline[0],
            "segment_index": 0,
            "progress_ratio": 1.0,
        }

    best_dist = float("inf")
    best_point = polyline[0]
    best_idx = 0
    best_t = 0.0

    # Calculate segment lengths for along-track progress
    seg_lengths = []
    total_len = 0.0
    for i in range(len(polyline) - 1):
        slen = haversine_m(polyline[i][0], polyline[i][1], polyline[i + 1][0], polyline[i + 1][1])
        seg_lengths.append(slen)
        total_len += slen

    for i in range(len(polyline) - 1):
        a_lat, a_lon = polyline[i][0], polyline[i][1]
        b_lat, b_lon = polyline[i + 1][0], polyline[i + 1][1]
        dist, closest, t = point_to_segment_distance_m(lat, lon, a_lat, a_lon, b_lat, b_lon)
        if dist < best_dist:
            best_dist = dist
            best_point = [closest[0], closest[1]]
            best_idx = i
            best_t = t

    # Along-track distance up to the closest point
    dist_so_far = sum(seg_lengths[:best_idx]) + (seg_lengths[best_idx] * best_t if best_idx < len(seg_lengths) else 0.0)
    progress_ratio = (dist_so_far / total_len) if total_len > 0 else 0.0

    return {
        "distance_m": round(best_dist, 1),
        "closest_point": best_point,
        "segment_index": best_idx,
        "progress_ratio": round(min(1.0, max(0.0, progress_ratio)), 4),
    }


@dataclass
class TripDeviationState:
    trip_id: str
    user_id: str
    status: str = "on_route"          # "on_route" | "caution" | "deviated"
    consecutive_off_route: int = 0
    consecutive_on_route: int = 0
    max_deviation_m: float = 0.0
    first_deviated_at: Optional[float] = None
    last_alert_id: Optional[int] = None
    last_update_ts: float = field(default_factory=time.time)


class RouteDeviationTracker:
    """Stateful route deviation detector with hysteresis filtering to prevent flapping."""

    def __init__(self, db=None, clock=time.time):
        self.db = db
        self.clock = clock
        self._states: Dict[str, TripDeviationState] = {}

    def get_state(self, trip_id: str, user_id: str) -> TripDeviationState:
        if trip_id not in self._states:
            self._states[trip_id] = TripDeviationState(trip_id=trip_id, user_id=user_id)
        return self._states[trip_id]

    def evaluate_fix(
        self,
        trip_id: str,
        user_id: str,
        lat: float,
        lon: float,
        planned_polyline: List[List[float]],
        trip_destination_name: str = "destination",
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluates one GPS fix against the planned route.

        Returns analysis summary, alert actions, and updated deviation metrics.
        """
        now = timestamp or self.clock()
        state = self.get_state(trip_id, user_id)
        state.last_update_ts = now

        xt = calculate_cross_track(lat, lon, planned_polyline)
        dev_m = xt["distance_m"]

        alert_action = "none"
        new_alert_id = None
        message = None

        if dev_m >= CORRIDOR_DEVIATION_M:
            state.consecutive_off_route += 1
            state.consecutive_on_route = 0
            state.max_deviation_m = max(state.max_deviation_m, dev_m)

            if state.first_deviated_at is None:
                state.first_deviated_at = now

            if state.consecutive_off_route >= CONSECUTIVE_OFF_ROUTE_THRESHOLD:
                state.status = "deviated"
                # Raise alert if not already open
                if state.last_alert_id is None and self.db is not None:
                    km_off = round(dev_m / 1000.0, 1)
                    message = (
                        f"Route deviation detected: Traveller is {km_off} km off planned route "
                        f"towards {trip_destination_name}. (Progress: {round(xt['progress_ratio'] * 100)}%)."
                    )
                    alert_payload = {
                        "user_id": user_id,
                        "travel_card_id": trip_id,
                        "type": "deviation",
                        "severity": "warning",
                        "status": "open",
                        "lat": lat,
                        "lon": lon,
                        "message": message,
                        "source": "system",
                    }
                    try:
                        new_alert_id = self.db.insert_alert(alert_payload)
                        state.last_alert_id = new_alert_id
                        alert_action = "raised"
                    except Exception as e:
                        # Log and continue; next fix will retry
                        pass
        elif dev_m >= CORRIDOR_SAFE_M:
            # In caution zone (500m - 1800m)
            state.consecutive_on_route = 0
            if state.status != "deviated":
                state.status = "caution"
        else:
            # Back within normal corridor (< 500m)
            state.consecutive_on_route += 1
            state.consecutive_off_route = 0

            if state.consecutive_on_route >= CONSECUTIVE_ON_ROUTE_THRESHOLD:
                if state.status == "deviated" and state.last_alert_id is not None:
                    # Auto-resolve the open alert
                    if self.db is not None:
                        try:
                            self.db.resolve_alert(state.last_alert_id)
                            alert_action = "resolved"
                        except Exception:
                            pass
                    state.last_alert_id = None
                    state.first_deviated_at = None
                    state.max_deviation_m = 0.0

                state.status = "on_route"

        return {
            "trip_id": trip_id,
            "status": state.status,
            "deviation_distance_m": dev_m,
            "closest_point": xt["closest_point"],
            "progress_pct": round(xt["progress_ratio"] * 100.0, 1),
            "consecutive_off_route": state.consecutive_off_route,
            "max_deviation_m": state.max_deviation_m,
            "alert_action": alert_action,
            "alert_id": state.last_alert_id,
            "message": message,
        }
