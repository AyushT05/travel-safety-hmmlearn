import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from geofence import gdacs
from geofence.scoring import Sample, assess_point
from geofence.store import RiskZoneStore
from geofence.sync import sync_gdacs
from geofence.zones import zone_from_risk_row

FIX = json.loads((Path(__file__).parent / "fixtures" / "gdacs_list_sample.json").read_text())
NOW = datetime(2026, 10, 24, 8, 30, tzinfo=timezone.utc)
WORLD = (-90.0, -180.0, 90.0, 180.0)


def feat(eventid):
    return copy.deepcopy(next(f for f in FIX["features"] if f["properties"]["eventid"] == eventid))


def test_real_sample_only_current_events_in_region_are_ingested():
    rows, skipped = gdacs.parse_event_list(FIX, NOW)
    assert [r["source_ref"] for r in rows] == ["gdacs:EQ:1573141"]       # M7 in the India+margin box
    assert skipped == {"outside_region": 1, "not_current": 4, "type_not_used": 2}   # Pacific cyclone, history, VO and DR


def test_world_box_also_takes_the_current_cyclone():
    rows, _ = gdacs.parse_event_list(FIX, NOW, WORLD)
    tc = next(r for r in rows if r["hazard_type"] == "cyclone")
    assert (tc["severity"], tc["radius_m"], tc["source_ref"]) == ("critical", 250_000, "gdacs:TC:1001338")
    assert tc["confidence"] == gdacs.APPROX_CONFIDENCE and tc["attributes"]["footprint"] == "approximate_disc"
    assert tc["geometry"] == {"type": "Point", "coordinates": [123.8, 22.7]}
    assert tc["source_url"].startswith("https://www.gdacs.org/report.aspx")


def test_iscurrent_is_a_string_and_false_means_inactive():
    f = feat(1001326)                         # a real India cyclone, long over
    assert gdacs.feature_to_row(f, NOW)[1] == "not_current"
    f["properties"]["iscurrent"] = "true"     # test-only edit: pretend it is live
    row, why = gdacs.feature_to_row(f, NOW)
    assert why is None and row["hazard_type"] == "cyclone" and row["severity"] == "high"
    assert row["radius_m"] == 100_000          # 83 km/h tropical storm


def test_ongoing_events_get_a_short_ttl_not_the_unreliable_todate():
    f = feat(1001326); f["properties"]["iscurrent"] = "true"
    row, _ = gdacs.feature_to_row(f, NOW)
    assert datetime.fromisoformat(row["expires_at"]) == NOW + gdacs.ALIVE_TTL


def test_earthquake_is_a_hazard_for_72_hours_only():
    row, _ = gdacs.feature_to_row(feat(1573141), NOW)
    assert datetime.fromisoformat(row["expires_at"]) == datetime(2026, 10, 25, 14, 33, 40, tzinfo=timezone.utc)
    assert row["radius_m"] == 100_000
    late = NOW + timedelta(days=4)
    assert gdacs.feature_to_row(feat(1573141), late)[1] == "earthquake_window_passed"


def test_flood_without_a_severity_value_still_gets_a_severity_from_alert_level():
    f = feat(1104121); f["properties"]["iscurrent"] = "true"
    row, _ = gdacs.feature_to_row(f, NOW, WORLD)
    assert row["severity"] == "high" and row["attributes"]["severity_value"] == 0.0 and row["radius_m"] == 30_000


def test_empty_or_malformed_feed_is_an_error_not_no_hazards():
    for bad in ({}, {"features": []}, {"features": None}, [], None):
        with pytest.raises(ValueError):
            gdacs.parse_event_list(bad, NOW)


def test_rows_satisfy_the_database_constraints():
    rows, _ = gdacs.parse_event_list(FIX, NOW, WORLD)
    for r in rows:
        assert r["hazard_type"] in ("cyclone", "earthquake", "flood", "wildfire")
        assert r["severity"] in ("low", "medium", "high", "critical") and 0 <= r["confidence"] <= 1
        assert 0 <= r["radius_m"] <= 500_000 and r["geometry"]["type"] == "Point" and r["status"] == "active"
        assert r["source_ref"].startswith("gdacs:")


def test_refs_to_expire_only_touches_gdacs_rows_missing_from_the_feed():
    rows, _ = gdacs.parse_event_list(FIX, NOW)
    existing = ["gdacs:EQ:1573141", "gdacs:TC:old", "manual:keep-me"]
    assert gdacs.refs_to_expire(existing, rows) == ["gdacs:TC:old"]


def test_ingested_zone_feeds_the_evaluator_end_to_end():
    row, _ = gdacs.feature_to_row(feat(1573141), NOW)
    row = {**row, "id": "eq1"}
    z = zone_from_risk_row(row)
    lon, lat = row["geometry"]["coordinates"]
    inside = assess_point(Sample(lat, lon, NOW, accuracy_m=10), [z])
    assert inside.factors[0].hazard_type == "earthquake" and inside.level == "warning"     # 0.4: approximate
    far = assess_point(Sample(lat + 3, lon, NOW, accuracy_m=10), [z])                      # about 330 km away
    assert far.level == "none"
    after = assess_point(Sample(lat, lon, NOW + timedelta(days=4), accuracy_m=10), [z])
    assert after.level == "none" and after.excluded == [("risk:eq1", "expired_before_arrival")]


# ------------------------------------------------------------------ sync orchestration
class FakeStore:
    def __init__(self, existing=()):
        self.existing, self.upserted, self.expired = list(existing), [], []

    def active_refs(self, prefix): return [x for x in self.existing if x.startswith(prefix)]
    def upsert(self, rows): self.upserted += rows
    def expire(self, refs): self.expired += refs


def test_sync_upserts_and_retires():
    store = FakeStore(["gdacs:TC:gone"])
    out = sync_gdacs(store, fetch=lambda url: FIX, now=NOW, bbox=gdacs.DEFAULT_INTEREST_BBOX)
    assert out["ingested"] == 1 and out["expired"] == 1 and out["features"] == len(FIX["features"])
    assert store.expired == ["gdacs:TC:gone"] and store.upserted[0]["source_ref"] == "gdacs:EQ:1573141"


def test_failed_fetch_changes_nothing():
    store = FakeStore(["gdacs:TC:live"])

    def boom(url): raise httpx.ConnectError("down")
    with pytest.raises(httpx.ConnectError):
        sync_gdacs(store, fetch=boom, now=NOW)
    with pytest.raises(ValueError):
        sync_gdacs(store, fetch=lambda url: {"features": []}, now=NOW)
    assert store.upserted == [] and store.expired == []        # live hazards are NOT retired by a bad fetch


# ------------------------------------------------------------------ PostgREST requests
def make_store(handler):
    return RiskZoneStore("https://x.supabase.co", "svc", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_store_upsert_uses_on_conflict_source_ref_and_chunks():
    calls = []

    def h(req):
        calls.append(req)
        return httpx.Response(201)
    make_store(h).upsert([{"source_ref": f"gdacs:EQ:{i}"} for i in range(250)])
    assert len(calls) == 3
    assert calls[0].url.params["on_conflict"] == "source_ref"
    assert "resolution=merge-duplicates" in calls[0].headers["prefer"]
    assert calls[0].headers["authorization"] == "Bearer svc"


def test_store_expire_and_bbox_queries():
    seen = []

    def h(req):
        seen.append(req)
        return httpx.Response(200, json=[{"source_ref": "gdacs:EQ:1"}])
    s = make_store(h)
    assert s.active_refs("gdacs:") == ["gdacs:EQ:1"]
    assert seen[-1].url.params["source_ref"] == "like.gdacs:*" and seen[-1].url.params["status"] == "eq.active"
    s.expire(['gdacs:EQ:1', 'gdacs:TC:2'])
    assert seen[-1].method == "PATCH" and seen[-1].url.params["source_ref"] == 'in.("gdacs:EQ:1","gdacs:TC:2")'
    s.zones_in_bbox(12.0, 76.0, 13.0, 77.0, "2026-10-24T00:00:00+00:00")
    p = seen[-1].url.params
    assert p["min_lat"] == "lte.13.0" and p["max_lat"] == "gte.12.0" and p["min_lon"] == "lte.77.0" and p["max_lon"] == "gte.76.0"
    assert "expires_at.is.null" in p["or"]
