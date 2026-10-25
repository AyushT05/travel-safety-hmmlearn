from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from geofence.geometry import angle_diff, bbox_of, bearing_deg, distance_to_geometry, haversine_m, signed_distance_to_geometry
from geofence.route import assess_route, densify
from geofence.scoring import Sample, assess_point, freshness, zone_proximity
from geofence.state import transition
from geofence.sun import is_dark, sun_times
from geofence.zones import Zone, zone_from_risk_row, zone_from_unsafe_row

IST = timezone(timedelta(hours=5, minutes=30))
DAY = datetime(2026, 10, 24, 10, 0, tzinfo=IST)
NIGHT = datetime(2026, 10, 24, 20, 30, tzinfo=IST)
LAT, LON = 12.3000, 76.6500
M = 1 / 111195.0     # degrees of latitude per metre


def pt(lat=LAT, lon=LON):
    return {"type": "Point", "coordinates": [lon, lat]}


def zone(hazard="poor_lighting", sev="medium", conf=0.6, radius=100, geom=None, key="risk:z1", **kw):
    return Zone(key=key, name="Test zone", hazard_type=hazard, severity=sev, geometry=geom or pt(),
                confidence=conf, source="test", radius_m=radius, observed_at=DAY - timedelta(days=1), **kw)


def at(metres_north, t=NIGHT, **kw):
    return Sample(LAT + metres_north * M, LON, t, **kw)


# ---------------------------------------------------------------- geometry
def test_line_distance_and_polygon_containment():
    line = {"type": "LineString", "coordinates": [[LON - 0.01, LAT], [LON + 0.01, LAT]]}
    d, _, _ = distance_to_geometry(LAT + 50 * M, LON, line)
    assert d == pytest.approx(50, abs=1)
    sq = {"type": "Polygon", "coordinates": [[[LON - .01, LAT - .01], [LON + .01, LAT - .01], [LON + .01, LAT + .01],
                                              [LON - .01, LAT + .01], [LON - .01, LAT - .01]]]}
    assert distance_to_geometry(LAT, LON, sq)[0] == 0.0
    holed = {"type": "Polygon", "coordinates": sq["coordinates"] + [[[LON - .001, LAT - .001], [LON + .001, LAT - .001],
             [LON + .001, LAT + .001], [LON - .001, LAT + .001], [LON - .001, LAT - .001]]]}
    assert distance_to_geometry(LAT, LON, holed)[0] > 0      # inside the hole is outside the zone
    assert bbox_of(sq) == (LAT - .01, LON - .01, LAT + .01, LON + .01)


def test_bearing_and_angle_wrap():
    assert bearing_deg(LAT, LON, LAT + .01, LON) == pytest.approx(0, abs=0.5)
    assert bearing_deg(LAT, LON, LAT, LON + .01) == pytest.approx(90, abs=0.5)
    assert angle_diff(350, 10) == 20


# ---------------------------------------------------------------- sun
def test_sun_times_bengaluru():
    r, s = sun_times(date(2026, 10, 24), 12.9716, 77.5946)
    r, s = r.astimezone(IST), s.astimezone(IST)
    assert abs((r.hour * 60 + r.minute) - (6 * 60 + 8)) <= 10
    assert abs((s.hour * 60 + s.minute) - (17 * 60 + 52)) <= 10


def test_is_dark():
    assert not is_dark(DAY, LAT, LON)
    assert is_dark(NIGHT, LAT, LON)
    assert is_dark(datetime(2026, 10, 24, 3, 0, tzinfo=IST), LAT, LON)


# ---------------------------------------------------------------- scoring
def test_lighting_matters_after_dark_only():
    z = [zone("poor_lighting")]
    assert assess_point(at(0, DAY), z).level == "none"
    night = assess_point(at(0, NIGHT), z)
    assert night.is_dark and night.level == "caution" and night.factors[0].hazard_type == "poor_lighting"


def test_connectivity_never_reaches_warning():
    a = assess_point(at(0), [zone("weak_connectivity", sev="critical", conf=1.0)])
    assert a.factors and a.level == "caution" and a.score <= 0.35


def test_critical_flood_inside_is_danger_whatever_the_time():
    z = [zone("flood", sev="critical", conf=0.9, radius=500, expires_at=NIGHT + timedelta(days=1))]
    assert assess_point(at(0, DAY), z).level == "danger"
    assert assess_point(at(0, NIGHT), z).level == "danger"


def test_one_hazard_is_not_diluted_by_many_irrelevant_zones():
    far = [zone("crime", key=f"risk:far{i}", geom=pt(LAT + 0.5, LON)) for i in range(25)]
    z = [zone("flood", sev="critical", conf=0.9, radius=500, expires_at=NIGHT + timedelta(days=1))] + far
    assert assess_point(at(0), z).level == "danger"


def test_corroboration_from_a_different_group_adds_a_little():
    flood = zone("flood", sev="medium", conf=1.0, radius=500, key="risk:f", expires_at=NIGHT + timedelta(days=1))
    crime = zone("crime", sev="high", conf=0.9, radius=500, key="risk:c")
    solo = assess_point(at(0), [flood])
    both = assess_point(at(0), [flood, crime])
    assert both.score > solo.score


def test_expired_before_arrival_is_excluded_and_reported():
    z = [zone("flood", sev="critical", conf=1.0, radius=500, expires_at=datetime(2026, 10, 24, 18, 0, tzinfo=IST))]
    late = assess_point(at(0, NIGHT), z)
    assert late.level == "none" and ("risk:z1", "expired_before_arrival") in late.excluded
    assert assess_point(at(0, datetime(2026, 10, 24, 15, 0, tzinfo=IST)), z).level == "danger"


def test_freshness_decay_and_stale_cutoff():
    z = replace(zone("crime"), observed_at=NIGHT - timedelta(days=180))
    assert freshness(z, NIGHT)[0] == pytest.approx(0.5, abs=0.01)
    assert freshness(replace(z, observed_at=NIGHT - timedelta(days=3000)), NIGHT) == (None, "stale")
    assert freshness(replace(z, status="resolved"), NIGHT) == (None, "inactive")
    assert freshness(replace(z, observed_at=None), NIGHT)[0] == 0.5


def test_gps_accuracy_decides_inside_vs_boundary():
    z = zone("crime", radius=100)
    assert zone_proximity(at(20, accuracy_m=10), z).state == "inside"
    assert zone_proximity(at(110, accuracy_m=50), z).state == "uncertain_boundary"
    assert zone_proximity(at(110, accuracy_m=5), z).state == "near"
    assert zone_proximity(at(5000, accuracy_m=5), z).state == "outside"
    # a fix with no accuracy is not trusted to be precise
    assert zone_proximity(at(110), z).state == "uncertain_boundary"


def test_direction_of_travel():
    z = zone("crime", radius=100, sev="high")
    south = at(-300, accuracy_m=5)
    toward = Sample(south.lat, south.lon, NIGHT, 5, heading_deg=0, speed_mps=5)
    away = Sample(south.lat, south.lon, NIGHT, 5, heading_deg=180, speed_mps=5)
    assert zone_proximity(toward, z).state == "approaching"
    assert zone_proximity(away, z).state == "near"
    still = Sample(south.lat, south.lon, NIGHT, 5)
    prev = Sample(LAT - 330 * M, LON, NIGHT - timedelta(seconds=10), 5)
    assert zone_proximity(still, z, prev).state == "approaching"


def test_unknown_coverage_is_reported_not_assumed_safe():
    a = assess_point(at(5000), [], coverage={"disaster", "crime"})
    assert a.unchecked == ["lighting", "connectivity", "road"] and "not a guarantee" in a.explanation.lower()
    assert assess_point(at(5000), []).unchecked == []


def test_bad_geometry_is_skipped_not_fatal():
    bad = zone("crime", geom={"type": "Point", "coordinates": ["x", 1]})
    a = assess_point(at(0), [bad])
    assert a.level == "none" and a.excluded == [("risk:z1", "bad_geometry")]


# ---------------------------------------------------------------- adapters
def test_unsafe_zone_adapter():
    row = {"id": 7, "category": "robbery", "severity": "critical", "description": "Phones snatched", "lat": LAT,
           "lon": LON, "radius_m": 300, "status": "verified", "created_at": "2026-10-01T00:00:00Z", "expires_at": None}
    z = zone_from_unsafe_row(row)
    assert (z.key, z.hazard_type, z.severity, z.confidence) == ("unsafe:7", "crime", "critical", 0.8)
    assert zone_from_unsafe_row({**row, "status": "dismissed"}) is None
    assert zone_from_unsafe_row({**row, "status": "pending"}).confidence < 0.5
    assert zone_from_unsafe_row({**row, "category": "poor_lighting"}).hazard_type == "poor_lighting"


def test_risk_row_adapter():
    z = zone_from_risk_row({"id": "abc", "name": "Ghat road", "hazard_type": "landslide", "severity": "high",
                            "geometry": pt(), "confidence": 0.7, "source": "NDMA SACHET",
                            "expires_at": "2026-10-25T00:00:00+00:00"})
    assert z.key == "risk:abc" and z.expires_at is not None and z.confidence == 0.7


# ---------------------------------------------------------------- route
ROUTE = [(LAT, LON), (LAT + 0.02, LON)]          # about 2.2 km due north


def test_densify_spacing():
    pts = densify(ROUTE, 200)
    assert pts[0][2] == 0 and pts[-1][2] == pytest.approx(haversine_m(*ROUTE[0], *ROUTE[1]), abs=1)
    assert all(b[2] - a[2] <= 201 for a, b in zip(pts, pts[1:]))


def test_route_flags_dark_segment_with_eta_and_not_by_day():
    lit = zone("poor_lighting", sev="high", conf=0.8, radius=0,
               geom={"type": "LineString", "coordinates": [[LON, LAT + 0.008], [LON, LAT + 0.012]]})
    night = assess_route(ROUTE, datetime(2026, 10, 24, 19, 0, tzinfo=IST), [lit])
    assert night.segments and night.segments[0]["hazard_type"] == "poor_lighting"
    assert night.segments[0]["eta_from"] < night.segments[0]["eta_to"] or night.segments[0]["from_m"] == night.segments[0]["to_m"]
    assert night.level in ("caution", "warning")
    day = assess_route(ROUTE, datetime(2026, 10, 24, 10, 0, tzinfo=IST), [lit])
    assert day.segments == [] and day.level == "none"


def test_route_respects_alert_expiry_at_eta():
    flood = zone("flood", sev="critical", conf=1.0, radius=0,
                 geom={"type": "LineString", "coordinates": [[LON - .001, LAT], [LON - .001, LAT + 0.02]]},
                 expires_at=datetime(2026, 10, 24, 19, 5, tzinfo=IST))
    r = assess_route(ROUTE, datetime(2026, 10, 24, 19, 0, tzinfo=IST), [flood], speed_mps=2.0)
    assert r.segments
    assert max(s["eta_to"] for s in r.segments) <= datetime(2026, 10, 24, 19, 5, tzinfo=IST)


def test_straight_line_route_is_labelled():
    r = assess_route(ROUTE, NIGHT, [], route_source="straight_line")
    assert any("straight lines" in n for n in r.notes)


# ---------------------------------------------------------------- alert state machine
def run(seq, start=1000.0, step=10.0):
    row, out, t = None, [], start
    for s in seq:
        d = transition(row, s, t)
        row = d.row
        out.append(d.emit)
        t += step
    return out


def test_one_approach_one_entry_then_quiet():
    assert run(["approaching", "approaching", "inside", "inside", "inside"]) == [
        "approach", None, "entry", None, None]


def test_boundary_flicker_while_inside_does_not_exit_or_realert():
    assert run(["inside", "uncertain_boundary", "inside", "uncertain_boundary"]) == ["entry", None, None, None]


def test_exit_needs_consecutive_away_samples():
    assert run(["inside", "near", "near", "inside", "near", "near", "outside"]) == [
        "entry", None, None, None, None, None, "exit"]


def test_reapproach_is_rate_limited():
    seq = ["approaching", "outside", "outside", "outside", "approaching"]
    assert run(seq, step=10) == ["approach", None, None, None, None]          # inside the 15 min cooldown
    assert run(seq, step=600)[-1] == "approach"                                # after it, warn again


def test_entry_right_after_approach_is_not_suppressed_but_reentry_is():
    row, now = None, 1000.0
    d = transition(row, "approaching", now); assert d.emit == "approach"; row = d.row
    d = transition(row, "inside", now + 5); assert d.emit == "entry"; row = d.row     # escalation is immediate
    for i in range(3):
        d = transition(row, "outside", now + 10 + i); row = d.row
    assert d.emit == "exit"
    d = transition(row, "inside", now + 30)                                          # re-entry inside 5 min
    assert d.emit is None and d.reason == "entry suppressed by cooldown"
    assert transition(d.row, "outside", now + 40).emit is None


# ---------------------------------------------------------------- polygon depth (regression: found with real GDACS polygons)
SQUARE = {"type": "Polygon", "coordinates": [[[LON - .01, LAT - .01], [LON + .01, LAT - .01], [LON + .01, LAT + .01],
                                              [LON - .01, LAT + .01], [LON - .01, LAT - .01]]]}      # about 2.2 km wide


def test_signed_distance_is_negative_depth_inside_and_positive_outside():
    centre, _, _ = signed_distance_to_geometry(LAT, LON, SQUARE)
    assert centre == pytest.approx(-1086, abs=10)                     # half the width (0.01 deg of longitude here)
    near_edge, _, _ = signed_distance_to_geometry(LAT, LON + .01 - 20 / 108642, SQUARE)
    assert -30 < near_edge < -10
    outside, _, _ = signed_distance_to_geometry(LAT, LON + .01 + 100 / 108642, SQUARE)
    assert outside == pytest.approx(100, abs=5)
    line, _, _ = signed_distance_to_geometry(LAT, LON, {"type": "LineString", "coordinates": [[LON, LAT + .001], [LON + .01, LAT + .001]]})
    assert line > 0


def test_polygon_zones_are_inside_when_deep_and_uncertain_only_near_the_edge():
    z = zone("flood", radius=None, geom=SQUARE)
    assert zone_proximity(at(0, accuracy_m=10), z).state == "inside"                 # was "uncertain_boundary" before the fix
    edge = Sample(LAT, LON + .01 - 20 / 108642, NIGHT, accuracy_m=50)
    assert zone_proximity(edge, z).state == "uncertain_boundary"
    out = Sample(LAT, LON + .01 + 150 / 108642, NIGHT, accuracy_m=5)          # 150 m outside, inside the 200 m warning distance
    assert zone_proximity(out, z).state == "near"
    assert zone_proximity(Sample(LAT, LON + 1.0, NIGHT, accuracy_m=5), z).state == "outside"


def test_buffer_deepens_a_polygon_zone():
    z = zone("flood", radius=300, geom=SQUARE)
    outside = Sample(LAT, LON + .01 + 100 / 108642, NIGHT, accuracy_m=5)            # 100 m outside, inside the 300 m buffer
    assert zone_proximity(outside, z).state == "inside"
