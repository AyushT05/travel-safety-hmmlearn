"""Tests for route determination, route deviation detection, and AI behavior tracking."""
import time
import pytest
from datetime import datetime, timezone

from geofence.route import densify, fetch_road_route, plan_and_assess_route, assess_route
from geofence.deviation import (
    point_to_segment_distance_m,
    calculate_cross_track,
    RouteDeviationTracker,
    CORRIDOR_SAFE_M,
    CORRIDOR_DEVIATION_M,
)
from agent.signals import extract_signals, Signal
from agent.scoring import score_signals, INDEPENDENT


class MockDb:
    def __init__(self):
        self.alerts = []
        self.resolved = []

    def insert_alert(self, alert):
        alert_id = len(self.alerts) + 1
        self.alerts.append({**alert, "id": alert_id})
        return alert_id

    def resolve_alert(self, alert_id):
        self.resolved.append(alert_id)


# ---------------------------------------------------------------------------
# 1. Route Planning & Waypoints Tests
# ---------------------------------------------------------------------------

def test_densify_empty_and_normal():
    assert densify([]) == []
    pts = [(28.6139, 77.2090), (27.1767, 78.0081)] # Delhi to Agra
    dense = densify(pts, step_m=5000.0)
    assert len(dense) > 2
    assert dense[0][0] == pts[0][0]
    assert dense[-1][0] == pts[-1][0]
    assert dense[-1][2] > 150000.0  # > 150 km


def test_plan_and_assess_route_with_intermediate_stops():
    waypoints = [
        {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
        {"name": "Mathura", "lat": 27.4924, "lon": 77.6737},
        {"name": "Agra", "lat": 27.1767, "lon": 78.0081},
    ]
    res = plan_and_assess_route(waypoints)
    assert res["success"] is True
    assert res["total_distance_km"] > 160.0
    assert res["total_duration_hours"] > 2.0
    assert len(res["waypoints"]) == 3
    assert res["waypoints"][0]["name"] == "Delhi"
    assert res["waypoints"][1]["name"] == "Mathura"
    assert res["waypoints"][2]["name"] == "Agra"
    assert len(res["polyline"]) >= 3


# ---------------------------------------------------------------------------
# 2. Route Deviation & Cross-track Distance Tests
# ---------------------------------------------------------------------------

def test_point_to_segment_distance():
    # Straight north-south segment along longitude 77.0
    a_lat, a_lon = 28.0, 77.0
    b_lat, b_lon = 29.0, 77.0

    # Point directly on segment midpoint
    dist, closest, t = point_to_segment_distance_m(28.5, 77.0, a_lat, a_lon, b_lat, b_lon)
    assert dist < 1.0  # within 1 meter
    assert abs(t - 0.5) < 0.01

    # Point 0.01 deg east of midpoint (~980 m at 28.5 deg latitude)
    dist_east, _, _ = point_to_segment_distance_m(28.5, 77.01, a_lat, a_lon, b_lat, b_lon)
    assert 900.0 < dist_east < 1100.0


def test_cross_track_on_route_vs_deviated():
    route_polyline = [
        [28.0, 77.0],
        [28.5, 77.0],
        [29.0, 77.0],
    ]

    # Exactly on route
    xt_on = calculate_cross_track(28.25, 77.0, route_polyline)
    assert xt_on["distance_m"] < 5.0
    assert 0.2 < xt_on["progress_ratio"] < 0.3

    # Off route by ~2.5 km (0.025 degrees longitude at ~28.5N)
    xt_off = calculate_cross_track(28.5, 77.025, route_polyline)
    assert xt_off["distance_m"] > 2000.0
    assert abs(xt_off["progress_ratio"] - 0.5) < 0.05


def test_route_deviation_tracker_lifecycle():
    db = MockDb()
    t0 = 1700000000.0
    clock_state = {"t": t0}
    tracker = RouteDeviationTracker(db=db, clock=lambda: clock_state["t"])

    polyline = [
        [28.0, 77.0],
        [28.5, 77.0],
        [29.0, 77.0],
    ]

    # Fix 1: On route
    res1 = tracker.evaluate_fix("trip1", "user1", 28.1, 77.0, polyline)
    assert res1["status"] == "on_route"
    assert res1["alert_action"] == "none"

    # Fixes 2, 3, 4: Drifting significantly off route (>2.5 km)
    clock_state["t"] += 10.0
    res2 = tracker.evaluate_fix("trip1", "user1", 28.3, 77.03, polyline)
    assert res2["consecutive_off_route"] == 1
    assert res2["alert_action"] == "none"  # Not yet 3 consecutive

    clock_state["t"] += 10.0
    res3 = tracker.evaluate_fix("trip1", "user1", 28.3, 77.035, polyline)
    assert res3["consecutive_off_route"] == 2
    assert res3["alert_action"] == "none"

    clock_state["t"] += 10.0
    res4 = tracker.evaluate_fix("trip1", "user1", 28.3, 77.04, polyline)
    assert res4["consecutive_off_route"] == 3
    assert res4["status"] == "deviated"
    assert res4["alert_action"] == "raised"
    assert len(db.alerts) == 1
    assert db.alerts[0]["type"] == "deviation"
    assert "Route deviation detected" in db.alerts[0]["message"]

    # Fixes 5, 6, 7: Returning back within normal corridor
    clock_state["t"] += 10.0
    tracker.evaluate_fix("trip1", "user1", 28.4, 77.001, polyline) # ~100m off
    clock_state["t"] += 10.0
    tracker.evaluate_fix("trip1", "user1", 28.45, 77.0, polyline)
    clock_state["t"] += 10.0
    res_return = tracker.evaluate_fix("trip1", "user1", 28.5, 77.0, polyline)
    assert res_return["status"] == "on_route"
    assert res_return["alert_action"] == "resolved"
    assert len(db.resolved) == 1


# ---------------------------------------------------------------------------
# 3. AI Behavioral Anomaly Signals and Scoring Tests
# ---------------------------------------------------------------------------

def test_signal_extraction_route_deviation():
    now = 1700000000.0
    trip = {
        "id": "card_dev",
        "user_id": "u_dev",
        "places": ["Delhi", "Agra"],
        "planned_route": {"polyline": [[28.0, 77.0], [29.0, 77.0]]},
    }
    # Fix located 3 km off route
    locs = [
        {"id": 1, "lat": 28.5, "lon": 77.03, "accuracy": 15.0, "speed": 10.0, "recorded_at": "2026-10-28T12:00:00Z"}
    ]
    snap = {"trip": trip, "locations": locs, "activity": None, "alerts": []}
    sigs, ctx = extract_signals(snap, zones=[], now=now)

    dev_sig = next((s for s in sigs if s.kind == "route_deviation"), None)
    assert dev_sig is not None
    assert dev_sig.detail["deviation_m"] > 2000.0


def test_signal_extraction_erratic_movement():
    now = 1700000000.0
    trip = {"id": "c1", "user_id": "u1"}
    locs = [
        {"id": 1, "lat": 28.0, "lon": 77.0, "accuracy": 10.0, "speed": 5.0, "recorded_at": "2026-10-28T12:00:00Z"}
    ]
    # HSMM reported ERRATIC state
    act = {"current_state": "ERRATIC", "duration_minutes": 4.0, "tail_probability": 0.01, "updated_at": "2026-10-28T12:00:00Z"}
    snap = {"trip": trip, "locations": locs, "activity": act, "alerts": []}
    sigs, _ = extract_signals(snap, zones=[], now=now)

    err_sig = next((s for s in sigs if s.kind == "erratic_movement"), None)
    assert err_sig is not None
    assert "Erratic" in err_sig.text
    assert err_sig.detail["state"] == "ERRATIC"


def test_scoring_correlation_deviation_plus_silence_escalates_to_high():
    now = 1700000000.0
    sig_dev = Signal("route_deviation", ["loc:1"], now - 1200, "Deviated 2.5 km off route", {"deviation_m": 2500.0})
    sig_lost = Signal("signal_lost", ["loc:1"], now - 1200, "No location update for 20 min", {"age_s": 1200, "near_zone_id": None})

    ctx = {"night": False}
    result = score_signals([sig_dev, sig_lost], ctx)

    # Correlation between route deviation and signal loss elevates priority to "high"
    assert result.level == "high"
    assert result.score >= 60
    assert len(result.kinds) >= 2
    assert "route_deviation" in result.kinds
    assert "signal_lost" in result.kinds
    assert any("went silent after deviating from planned route" in r for r in result.reasons)


def test_single_deviation_capped_at_medium():
    now = 1700000000.0
    sig_dev = Signal("route_deviation", ["loc:1"], now, "Deviated 4.0 km off route", {"deviation_m": 4000.0})
    ctx = {"night": False}
    result = score_signals([sig_dev], ctx)

    # Invariant: Single signal can never exceed medium
    assert result.level in ("low", "medium")
    assert result.score < 60
