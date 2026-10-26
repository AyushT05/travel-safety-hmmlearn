import threading
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from geofence.monitor import GeofenceMonitor
from geofence.scoring import Sample
from geofence.store import SupabaseGeofenceDb
from geofence.zones import Zone

LAT, LON = 12.3000, 76.6500
M = 1 / 111195.0
T0 = datetime(2026, 10, 24, 15, 30, tzinfo=timezone.utc).timestamp()        # 21:00 IST, after dark
UID, CARD = "user-1", "card-1"


def zone(key="risk:z1", hazard="crime", sev="high", conf=0.9, radius=100, event_ref=None, expires=None, lat=LAT, lon=LON):
    attrs = {"event_ref": event_ref} if event_ref else {}
    return Zone(key=key, name="Test zone", hazard_type=hazard, severity=sev, confidence=conf, source="test",
                geometry={"type": "Point", "coordinates": [lon, lat]}, radius_m=radius, attributes=attrs,
                observed_at=datetime.fromtimestamp(T0, tz=timezone.utc) - timedelta(days=1), expires_at=expires)


class FakeDb:
    def __init__(self, zones):
        self.zones, self.states, self.alerts, self.resolved, self.saved = zones, {}, [], [], []
        self.fail_insert, self.zone_calls, self._id = False, 0, 100

    def zones_near(self, lat, lon):
        self.zone_calls += 1
        return self.zones

    def load_states(self, uid): return {k: dict(v) for k, v in self.states.get(uid, {}).items()}

    def save_state(self, uid, key, row):
        self.saved.append((key, row["state"]))
        self.states.setdefault(uid, {})[key] = dict(row)

    def insert_alert(self, a):
        if self.fail_insert:
            raise RuntimeError("alerts insert failed")
        if any(x["event_id"] == a["event_id"] for x in self.alerts):
            return None
        self._id += 1
        self.alerts.append({**a, "id": self._id})
        return self._id

    def resolve_alert(self, aid): self.resolved.append(aid)


class Clock:
    def __init__(self): self.t = T0
    def __call__(self): return self.t


def make(zones):
    clock, db = Clock(), FakeDb(zones)
    return GeofenceMonitor(db, clock=clock), db, clock


def fix(clock, metres_north, lid, dt=5, acc=8.0, **kw):
    clock.t += dt
    t = datetime.fromtimestamp(clock.t, tz=timezone.utc).isoformat()
    return {"id": lid, "user_id": UID, "travel_card_id": CARD, "lat": LAT + metres_north * M, "lon": LON,
            "accuracy": acc, "speed": 4.0, "created_at": t, "observed_at": t, **kw}


# ------------------------------------------------------------------ the full lifecycle
def test_approach_entry_exit_lifecycle():
    mon, db, clock = make([zone()])
    types = []
    for i, north in enumerate([-700, -300, -200, 0, 0, -500, -500, -500]):
        out = mon.handle_location(fix(clock, north, i + 1))
        types += [a["type"] for a in out.get("alerts", [])]
    assert types == ["geofence_approach", "geofence_entry", "geofence_exit"]
    approach, entry, exit_ = db.alerts
    assert (approach["severity"], approach["status"], approach["source"]) == ("warning", "open", "system")
    assert entry["severity"] == "critical" and entry["status"] == "open" and "risk area" in entry["message"]
    assert exit_["status"] == "resolved" and exit_["severity"] == "info" and exit_["resolved_at"]
    assert db.resolved == [approach["id"], entry["id"]]            # approach superseded by entry, entry closed by exit
    assert approach["user_id"] == UID and approach["travel_card_id"] == CARD and approach["lat"] is not None
    assert [s for _, s in db.saved] == ["approaching", "inside", "clear"]      # only real transitions are written


def test_no_alert_far_away_and_no_database_writes():
    mon, db, clock = make([zone()])
    for i in range(5):
        mon.handle_location(fix(clock, -5000, i + 1))
    assert db.alerts == [] and db.saved == []


# ------------------------------------------------------------------ which fixes count
def test_stale_backlog_fixes_never_alert():
    mon, db, clock = make([zone()])
    old = fix(clock, 0, 1, dt=0)
    clock.t += 600                                               # the fix is now 10 minutes old (offline upload)
    assert mon.handle_location(old) == {"skipped": "stale"}
    assert db.alerts == [] and db.zone_calls == 0


def test_out_of_order_and_duplicate_webhooks_are_ignored():
    mon, db, clock = make([zone()])
    first = fix(clock, 0, 1)
    older = fix(clock, 0, 2, dt=-3)
    assert "alerts" in mon.handle_location(first)
    assert mon.handle_location(first) == {"skipped": "out_of_order"}
    assert mon.handle_location(older) == {"skipped": "out_of_order"}
    assert len(db.alerts) == 1


def test_a_phone_clock_in_the_future_is_clamped_not_trusted():
    mon, db, clock = make([zone()])
    out = mon.handle_location(fix(clock, 0, 1, dt=3600))         # claims to be an hour ahead
    assert out["alerts"][0]["type"] == "geofence_entry"
    out = mon.handle_location(fix(clock, 0, 2, dt=5))            # later real fix must still be accepted
    assert out.get("skipped") != "out_of_order"


def test_busy_user_is_skipped_not_queued():
    mon, db, clock = make([zone()])
    u = mon._user(UID)
    assert u.lock.acquire(blocking=False)
    try:
        assert mon.handle_location(fix(clock, 0, 1)) == {"skipped": "busy"}
    finally:
        u.lock.release()


# ------------------------------------------------------------------ relevance, grouping, escalation
def test_a_zone_that_does_not_matter_now_is_ignored():
    lit = zone(hazard="poor_lighting", sev="medium", conf=0.6)
    mon, db, clock = make([lit])
    clock.t = datetime(2026, 10, 24, 5, 0, tzinfo=timezone.utc).timestamp()          # 10:30 IST, daylight
    out = mon.handle_location(fix(clock, 0, 1))
    assert out["alerts"] == [] and db.saved == []


def test_zones_of_one_event_raise_one_alert_and_escalate_when_it_gets_worse():
    outer = zone("risk:a", "cyclone", "medium", 0.9, 1500, event_ref="gdacs:TC:1")
    inner = zone("risk:b", "cyclone", "critical", 0.9, 200, event_ref="gdacs:TC:1")
    mon, db, clock = make([outer, inner])
    mon.handle_location(fix(clock, 800, 1))                      # inside the outer band only
    assert [a["type"] for a in db.alerts] == ["geofence_entry"] and db.alerts[0]["severity"] == "warning"
    mon.handle_location(fix(clock, 800, 2))
    assert len(db.alerts) == 1                                   # nothing changed: no repeat
    mon.handle_location(fix(clock, 0, 3))                        # now inside the inner (critical) band as well
    assert len(db.alerts) == 2 and db.alerts[1]["severity"] == "critical"
    assert db.resolved == [db.alerts[0]["id"]]                   # the milder alert is replaced
    assert set(k for k, _ in db.saved) == {"gdacs:TC:1"}         # one lifecycle for the whole event


def test_leaving_the_nearby_list_entirely_counts_as_leaving():
    mon, db, clock = make([zone()])
    mon.handle_location(fix(clock, 0, 1))
    db.zones = []                                                # tourist travelled far: zone no longer nearby
    types = []
    for i in range(3):
        types += [a["type"] for a in mon.handle_location(fix(clock, 0, 10 + i)).get("alerts", [])]
    assert types == ["geofence_exit"] and db.resolved == [db.alerts[0]["id"]]


# ------------------------------------------------------------------ failure and restart behaviour
def test_failed_alert_write_is_retried_on_the_next_fix():
    mon, db, clock = make([zone()])
    db.fail_insert = True
    with pytest.raises(RuntimeError):
        mon._group(mon._user(UID), UID, fix(clock, 0, 1), "risk:z1", db.zones,
                   Sample(LAT, LON, datetime.fromtimestamp(clock.t, tz=timezone.utc), 8.0), None, clock.t)
    assert db.saved == []                                        # state not saved, so nothing is lost
    assert mon.handle_location(fix(clock, 0, 2))["alerts"] == []     # logged and skipped, scoring unaffected
    db.fail_insert = False
    out = mon.handle_location(fix(clock, 0, 3))
    assert [a["type"] for a in out["alerts"]] == ["geofence_entry"]


def test_state_survives_a_restart_without_a_duplicate_alert():
    mon, db, clock = make([zone()])
    mon.handle_location(fix(clock, 0, 1))
    assert len(db.alerts) == 1
    reborn = GeofenceMonitor(db, clock=clock)                    # new process, same database
    out = reborn.handle_location(fix(clock, 0, 2))
    assert out["alerts"] == [] and len(db.alerts) == 1


def test_webhook_retry_with_same_location_id_cannot_duplicate_an_alert():
    mon, db, clock = make([zone()])
    mon.handle_location(fix(clock, 0, 7))
    db.states.clear()
    other = GeofenceMonitor(db, clock=clock)                     # lost memory AND state, replays the same webhook
    rec = fix(clock, 0, 7, dt=0)
    other.handle_location(rec)
    assert len(db.alerts) == 1                                   # same deterministic event_id was refused


# ------------------------------------------------------------------ Supabase layer (request shapes)
def make_db(handler, clock=None):
    c = clock or Clock()
    return SupabaseGeofenceDb("https://x.supabase.co", "svc", client=httpx.Client(transport=httpx.MockTransport(handler)),
                              clock=c, tile_ttl_s=60), c


def test_zones_are_cached_per_tile_and_fetched_with_a_wide_box():
    calls = []

    def h(req):
        calls.append(req)
        return httpx.Response(200, json=[])
    db, clock = make_db(h)
    db.zones_near(12.3, 76.65)
    db.zones_near(12.9, 76.10)                                   # same 1 degree tile: cached
    assert len(calls) == 2                                       # one risk_zones request + one unsafe_zones request
    risk = next(c for c in calls if c.url.path.endswith("/risk_zones"))
    assert risk.url.params["min_lat"] == "lte.18.0" and risk.url.params["max_lat"] == "gte.7.0"
    assert "updated_at.gt." in risk.url.params["and"] and "source.neq.GDACS" in risk.url.params["and"]
    clock.t += 120
    db.zones_near(12.3, 76.65)
    assert len(calls) == 4                                       # TTL expired: refetched


def test_zone_fetch_failure_uses_a_recent_cached_tile_but_not_an_old_one():
    state = {"fail": False}

    def h(req):
        if state["fail"]:
            return httpx.Response(500)
        return httpx.Response(200, json=[{"id": 1, "category": "robbery", "severity": "critical", "description": "x", "lat": 12.3,
                                          "lon": 76.65, "radius_m": 300, "status": "verified", "created_at": "2026-10-01T00:00:00Z"}]
                              if req.url.path.endswith("/unsafe_zones") else [])
    db, clock = make_db(h)
    assert len(db.zones_near(12.3, 76.65)) == 1
    state["fail"] = True
    clock.t += 300
    assert len(db.zones_near(12.3, 76.65)) == 1                  # 5 minutes old: still usable
    clock.t += 3600
    with pytest.raises(httpx.HTTPStatusError):
        db.zones_near(12.3, 76.65)                               # an hour old: refuse rather than guess


def test_alert_insert_resolve_and_state_requests():
    seen = []

    def h(req):
        seen.append(req)
        if req.method == "POST" and req.url.path.endswith("/alerts"):
            return httpx.Response(201, json=[{"id": 55}] if len(seen) == 1 else [])
        if req.method == "GET":
            return httpx.Response(200, json=[{"zone_key": "gdacs:TC:1", "state": "inside", "clear_streak": 1,
                                              "first_seen_at": "2026-10-24T15:30:00+00:00", "last_approach_at": None,
                                              "last_entry_at": "2026-10-24T15:31:00+00:00", "last_alert_id": 55, "last_level": "danger"}])
        return httpx.Response(204)
    db, clock = make_db(h)
    assert db.insert_alert({"event_id": "e1", "type": "geofence_entry"}) == 55
    assert seen[0].url.params["on_conflict"] == "event_id" and "ignore-duplicates" in seen[0].headers["prefer"]
    assert db.insert_alert({"event_id": "e1", "type": "geofence_entry"}) is None            # duplicate event_id
    db.resolve_alert(55)
    assert seen[-1].method == "PATCH" and seen[-1].url.params["id"] == "eq.55" and seen[-1].url.params["status"] == "eq.open"
    st = db.load_states("u1")["gdacs:TC:1"]
    assert st["state"] == "inside" and st["last_alert_id"] == 55 and isinstance(st["last_entry_at"], float)
    db.save_state("u1", "gdacs:TC:1", {**st, "last_approach_at": None})
    body = __import__("json").loads(seen[-1].content)
    assert seen[-1].url.params["on_conflict"] == "user_id,zone_key" and body["last_entry_at"].startswith("2026-10-24T15:31")
    assert body["last_alert_id"] == 55 and body["last_level"] == "danger"
