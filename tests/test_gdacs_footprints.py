import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from geofence import gdacs
from geofence.geometry import distance_to_geometry
from geofence.scoring import Sample, assess_point
from geofence.sync import sync_gdacs
from geofence.zones import zone_from_risk_row

FX = Path(__file__).parent / "fixtures"
LIST = json.loads((FX / "gdacs_list_sample.json").read_text())
GEOM = json.loads((FX / "gdacs_tc_geometry_sample.json").read_text())     # trimmed from a real response
NOW = datetime(2026, 10, 24, 14, 0, tzinfo=timezone.utc)
WORLD = (-90.0, -180.0, 90.0, 180.0)


def utc(d, h):
    return datetime(2026, 10, d, h, 0, tzinfo=timezone.utc)


def parent():
    rows, _ = gdacs.parse_event_list(LIST, NOW, WORLD)
    return next(r for r in rows if r["attributes"]["event_type"] == "TC")


def test_parent_row_remembers_where_to_fetch_the_geometry():
    assert parent()["attributes"]["geometry_url"].endswith("eventtype=TC&eventid=1001338&episodeid=35")


def test_only_real_wind_fields_become_zones():
    rows = gdacs.tc_footprint_rows(parent(), GEOM, NOW)
    assert len(rows) == 6                       # 3 bands now, 2 for 25/10 00:00 (Red is empty), 1 for 25/10 12:00
    assert {r["attributes"]["footprint"] for r in rows} == {"wind_radii"}
    assert all(r["geometry"]["type"] == "Polygon" and r["hazard_type"] == "cyclone" and r["radius_m"] == 0 for r in rows)
    # the whole-storm swaths (3.5 million km2), the uncertainty cone and the tracks were all in the input
    assert any(f["properties"].get("Class") == "Poly_Cones" for f in GEOM["features"])
    assert max(gdacs.polygon_area_km2(r["geometry"]["coordinates"]) for r in rows) < 1_000_000


def test_degenerate_later_steps_are_skipped():
    rows = gdacs.tc_footprint_rows(parent(), GEOM, NOW)
    assert not any("2026-10-26" in r["attributes"]["step_time"] for r in rows)
    red_25 = [r for r in rows if r["attributes"]["step_time"].startswith("2026-10-25T00") and r["attributes"]["band_kmh"] == 120]
    assert red_25 == []


def test_bands_map_to_severity_and_wind_speed():
    rows = gdacs.tc_footprint_rows(parent(), GEOM, NOW)
    by = {(r["attributes"]["step_time"][:13], r["attributes"]["band_kmh"]): r["severity"] for r in rows}
    assert by[("2026-10-24T12", 120)] == "critical" and by[("2026-10-24T12", 90)] == "high" and by[("2026-10-24T12", 60)] == "medium"


def test_validity_windows_overlap_so_a_moving_storm_leaves_no_gap():
    rows = gdacs.tc_footprint_rows(parent(), GEOM, NOW)
    w = {r["attributes"]["step_time"][:13]: (r["attributes"]["valid_from"][:13], r["expires_at"][:13]) for r in rows}
    assert w["2026-10-24T12"] == ("2026-10-24T00", "2026-10-25T00")
    assert w["2026-10-25T00"] == ("2026-10-24T12", "2026-10-25T12")
    assert all(datetime.fromisoformat(r["expires_at"]) > NOW for r in rows)


def test_confidence_falls_with_forecast_distance():
    rows = gdacs.tc_footprint_rows(parent(), GEOM, NOW)
    conf = {r["attributes"]["step_time"][:13]: r["confidence"] for r in rows}
    assert conf["2026-10-24T12"] == 0.7 and conf["2026-10-25T00"] == 0.61 and conf["2026-10-25T12"] == 0.56
    assert [r["attributes"]["forecast"] for r in rows if r["attributes"]["step_time"][:13] == "2026-10-24T12"] == [False] * 3


def test_row_fields_satisfy_database_constraints():
    for r in gdacs.tc_footprint_rows(parent(), GEOM, NOW):
        assert r["severity"] in ("low", "medium", "high", "critical") and 0 <= r["confidence"] <= 1
        assert r["source_ref"].startswith("gdacs:TC:1001338:") and r["status"] == "active"
        assert datetime.fromisoformat(r["observed_at"]) == NOW


def test_far_future_steps_beyond_the_horizon_are_dropped():
    later = NOW - timedelta(days=6)               # now is long before this storm: its steps are 5+ days ahead
    assert gdacs.tc_footprint_rows(parent(), GEOM, later) == []


def test_dissipated_storm_gives_no_rows_and_malformed_response_is_an_error():
    only_empty = {"features": [f for f in GEOM["features"] if f["properties"].get("key") == "10260000"]}
    assert gdacs.tc_footprint_rows(parent(), only_empty, NOW) == []
    with pytest.raises(ValueError):
        gdacs.tc_footprint_rows(parent(), {"nope": 1}, NOW)


# ------------------------------------------------------------------ enrich: never hide a cyclone on an outage
def test_enrich_replaces_the_disc_with_wind_fields():
    rows, rep = gdacs.enrich_with_footprints([parent()], lambda url: GEOM, NOW)
    assert rep["footprint_rows"] == 6 and rep["fallback_discs"] == 0
    assert "gdacs:TC:1001338" not in [r["source_ref"] for r in rows]


def test_enrich_keeps_the_disc_when_geometry_cannot_be_fetched_or_parsed():
    def boom(url): raise ConnectionError("down")
    rows, rep = gdacs.enrich_with_footprints([parent()], boom, NOW)
    assert [r["source_ref"] for r in rows] == ["gdacs:TC:1001338"] and rep["fallback_discs"] == 1
    rows, rep = gdacs.enrich_with_footprints([parent()], lambda url: {"nope": 1}, NOW)
    assert rows[0]["attributes"]["footprint"] == "approximate_disc" and rep["errors"]


def test_enrich_drops_a_cyclone_whose_wind_field_is_gone():
    empty = {"features": [f for f in GEOM["features"] if f["properties"].get("key") == "10260000"]}
    rows, rep = gdacs.enrich_with_footprints([parent()], lambda url: empty, NOW)
    assert rows == [] and rep["dropped_no_wind_field"] == 1


def test_non_cyclones_are_untouched():
    eq, _ = gdacs.feature_to_row(copy.deepcopy(next(f for f in LIST["features"] if f["properties"]["eventid"] == 1573141)),
                                 datetime(2026, 10, 24, 8, 30, tzinfo=timezone.utc))
    calls = []
    rows, _ = gdacs.enrich_with_footprints([eq], lambda url: calls.append(url), NOW)
    assert rows == [eq] and calls == []


# ------------------------------------------------------------------ sync end to end
class FakeStore:
    def __init__(self, existing=()):
        self.existing, self.upserted, self.expired = list(existing), [], []

    def active_refs(self, prefix): return [x for x in self.existing if x.startswith(prefix)]
    def upsert(self, rows): self.upserted += rows
    def expire(self, refs): self.expired += refs


def test_sync_swaps_the_old_disc_for_wind_fields_and_retires_it():
    def fetch(url):
        return GEOM if "getgeometry" in url else LIST
    store = FakeStore(["gdacs:TC:1001338"])                  # the disc stored by the previous hourly run
    out = sync_gdacs(store, fetch=fetch, now=NOW, bbox=WORLD)
    assert out["footprints"]["footprint_rows"] == 6
    assert "gdacs:TC:1001338" in store.expired               # replaced, so retired
    assert sum(1 for r in store.upserted if r["attributes"]["footprint"] == "wind_radii") == 6


def test_sync_survives_a_geometry_outage_with_the_disc():
    def fetch(url):
        if "getgeometry" in url:
            raise ConnectionError("down")
        return LIST
    store = FakeStore()
    out = sync_gdacs(store, fetch=fetch, now=NOW, bbox=WORLD)
    assert out["footprints"]["fallback_discs"] == 1
    assert "gdacs:TC:1001338" in [r["source_ref"] for r in store.upserted]


# ------------------------------------------------------------------ evaluator end to end with real polygons
def zones():
    out = []
    for i, r in enumerate(gdacs.tc_footprint_rows(parent(), GEOM, NOW)):
        out.append(zone_from_risk_row({**r, "id": f"w{i}"}))
    return out


def inside_red():
    z = next(z for z in zones() if z.severity == "critical" and z.attributes["step_time"].startswith("2026-10-24T12"))
    ring = z.geometry["coordinates"][0]
    lon = sum(c[0] for c in ring) / len(ring)
    lat = sum(c[1] for c in ring) / len(ring)
    assert distance_to_geometry(lat, lon, z.geometry)[0] == 0.0
    return lat, lon


def test_a_point_inside_120_kmh_winds_is_danger_while_the_field_applies():
    lat, lon = inside_red()
    a = assess_point(Sample(lat, lon, utc(24, 18), accuracy_m=10), zones())
    assert a.level == "danger" and a.factors[0].hazard_type == "cyclone" and a.factors[0].state == "inside"


def test_forecast_field_is_not_applied_before_its_time():
    lat, lon = inside_red()
    a = assess_point(Sample(lat, lon, utc(23, 18), accuracy_m=10), zones())      # a day before the field starts
    assert a.level == "none" and ("not_yet_valid" in [why for _, why in a.excluded])


def test_field_is_gone_after_its_window_and_far_points_are_safe():
    lat, lon = inside_red()
    assert assess_point(Sample(lat, lon, utc(27, 6), accuracy_m=10), zones()).level == "none"
    assert assess_point(Sample(12.3, 76.65, utc(24, 18), accuracy_m=10), zones()).level == "none"     # Mysuru
