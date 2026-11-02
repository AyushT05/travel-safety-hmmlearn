import json

from agent.context import likely_issues
from agent.engine import Engine
from agent.insights import InsightService, validate_llm, build_packet, InsightRejected
from agent.situation import analyse
from agent.util import iso
from tests.fakes import FakeStore, FakeLLM, T0, NIGHT, FAR

# planned route: straight east along lat 12.30 from lon 76.60 to 76.80, stops at both ends
POLY = [[12.30, 76.60 + i * 0.01] for i in range(21)]
TRIP = {"id": "c1", "user_id": "u1", "full_name": "Asha Rao", "mobile": "+919812345670", "places": ["Mysuru", "Hassan"],
        "place_details": [{"name": "Mysuru", "lat": 12.30, "lon": 76.60}, {"name": "Hassan", "lat": 12.30, "lon": 76.80}],
        "planned_route": {"polyline": POLY}, "emergency_contacts": [{"name": "Ravi", "phone": "+919876543210"}]}


def fx(now, pts, step=5, acc=8.0, id0=1):
    """pts: list of (lat, lon, speed) ending at `now`, spaced `step` seconds."""
    out, n = [], len(pts)
    for i, (la, lo, sp) in enumerate(pts):
        out.append({"id": id0 + i, "lat": la, "lon": lo, "accuracy": acc, "speed": sp, "recorded_at": iso(now - (n - 1 - i) * step)})
    return out


def snap(locs, **kw):
    return {"trip": TRIP, "locations": locs, "activity": kw.get("activity"), "alerts": [], "assessment": kw.get("assessment")}


def kinds(events):
    return {e.kind: e for e in events}


def drive_then_stop(now, lat=12.30, lon=76.70, stop_s=240):
    pts = [(lat, lon - 0.003 + i * 0.0005, 12.0) for i in range(6)]            # ~12 m/s
    pts += [(lat, lon, 0.0)] * (stop_s // 5)
    return fx(now, pts)


def test_on_route_moving_is_normal_and_knows_destination():
    now = T0
    locs = fx(now, [(12.30, 76.65 + i * 0.0004, 10.0) for i in range(30)])
    sit, ev, status = analyse(snap(locs), [], now, lambda la, lo: {"name": "NH 275, Channarayapatna", "kind": "highway"})
    assert status == "normal" and not [e for e in ev if e.severity != "info"]
    assert sit["route"]["on_route"] and sit["route"]["destination"] == "Hassan"
    assert "NH 275" in sit["summary"] and "Hassan" in sit["summary"] and sit["movement"]["heading"] == "east"


def test_deviation_is_detected_with_distance_and_trend():
    now = T0
    locs = fx(now, [(12.30 + i * 0.0006, 76.70, 9.0) for i in range(30)])        # heading north, away from the road
    sit, ev, status = analyse(snap(locs), [], now)
    d = kinds(ev)["route_deviation"]
    assert d.severity in ("watch", "alert") and d.detail["trend"] == "away" and d.notify_user
    assert status in ("watch", "alert")


def test_abrupt_stop_on_highway_at_night_is_alert():
    now = NIGHT
    sit, ev, status = analyse(snap(drive_then_stop(now)), [], now, lambda la, lo: {"name": "NH 275", "kind": "highway"})
    e = kinds(ev)["abrupt_stop"]
    assert e.severity == "alert" and status == "alert" and e.notify_user
    assert e.detail["risky_place"] and e.detail["from_kmh"] >= 40


def test_stop_at_planned_stop_is_not_escalated():
    now = T0
    sit, ev, status = analyse(snap(drive_then_stop(now, lon=76.60005, stop_s=300)), [], now)
    assert kinds(ev)["abrupt_stop"].severity == "info" and status == "normal"


def test_red_light_is_not_an_incident():
    now = T0
    sit, ev, status = analyse(snap(drive_then_stop(now, stop_s=70)), [], now)
    assert "abrupt_stop" not in kinds(ev) or kinds(ev)["abrupt_stop"].severity == "info"
    assert status == "normal"


def test_signal_lost_and_restored_and_gaps():
    now = T0
    locs = fx(now - 400, [(12.30, 76.70 + i * 0.0004, 10.0) for i in range(10)])
    sit, ev, status = analyse(snap(locs), [], now)
    assert kinds(ev)["signal_lost"].severity in ("watch", "alert")
    # restored: a 4-minute gap that ended 30 s ago
    old = fx(now - 300, [(12.30, 76.70, 10.0)] * 3)
    new = fx(now, [(12.30, 76.71, 10.0)] * 4, id0=10)
    sit, ev, _ = analyse(snap(old + new), [], now)
    assert "signal_restored" in kinds(ev) and kinds(ev)["signal_restored"].notify_user


def test_circling_detected():
    import math
    now = T0
    pts = [(12.30 + 0.0006 * math.sin(i / 3), 76.70 + 0.0006 * math.cos(i / 3), 6.0) for i in range(120)]
    sit, ev, _ = analyse(snap(fx(now, pts, step=5)), [], now)
    assert "circling" in kinds(ev)


def test_traveller_ok_reply_softens_alert_but_keeps_it_visible():
    now = NIGHT
    reply = [{"id": 5, "kind": "abrupt_stop", "response": "ok", "responded_at": iso(now - 60)}]
    sit, ev, status = analyse(snap(drive_then_stop(now)), [], now, lambda la, lo: {"name": "NH 275", "kind": "highway"}, reply)
    assert kinds(ev)["abrupt_stop"].severity == "watch" and "traveller_confirmed_ok" in kinds(ev)
    assert kinds(ev)["abrupt_stop"].detail["softened_by_traveller_reply"] and status == "watch"


def test_need_help_reply_is_an_alert():
    now = T0
    reply = [{"id": 6, "kind": "check_in", "response": "need_help", "responded_at": iso(now - 30)}]
    _, ev, status = analyse(snap(fx(now, [(12.30, 76.7, 8.0)] * 10)), [], now, None, reply)
    assert status == "alert" and "traveller_needs_help" in kinds(ev)


# ------------------------------------------------------------------ service: notifications, LLM validation
class InsStore(FakeStore):
    def __init__(self):
        super().__init__()
        self.insight_rows, self.notifs, self.responses = {}, [], []

    def insights(self): return list(self.insight_rows.values())
    def get_insight(self, t): return self.insight_rows.get(t)
    def upsert_insight(self, row): self.insight_rows[row["travel_card_id"]] = {**self.insight_rows.get(row["travel_card_id"], {}), **row}
    def patch_insight(self, t, patch, where=None):
        r = self.insight_rows.get(t)
        if r and all(r.get(k) == v for k, v in (where or {}).items()):
            r.update(patch)
    def notification_responses(self, t, since): return self.responses
    def recent_notifications(self, t, since): return self.notifs
    def insert_notification(self, row): self.notifs.append({**row, "id": len(self.notifs) + 1, "created_at": iso(self.now)})


def service(store, llm=None, now=NIGHT, kind="highway"):
    store.now = now
    return InsightService(store, llm or FakeLLM(enabled=False), lambda la, lo: {"name": "NH 275", "kind": kind}, lambda: now)


def test_alert_notifies_traveller_once_within_cooldown_and_writes_insight():
    s = InsStore()
    svc = service(s)
    svc.begin_tick()
    sn = snap(drive_then_stop(NIGHT))
    svc.process(TRIP, sn, [], NIGHT)
    assert len(s.notifs) == 1 and s.notifs[0]["kind"] == "abrupt_stop"
    assert {a["id"] for a in s.notifs[0]["actions"]} == {"confirm_ok", "need_help"}
    row = s.insight_rows["c1"]
    assert row["status"] == "alert" and row["source"] == "template" and "NH 275" in row["admin_summary"]
    svc.begin_tick()
    svc.process(TRIP, sn, [], NIGHT)
    assert len(s.notifs) == 1                                   # cooldown


def test_normal_trip_sends_nothing_and_never_calls_llm():
    s, llm = InsStore(), FakeLLM(response={}, enabled=True)
    svc = service(s, llm, T0)
    svc.begin_tick()
    svc.process(TRIP, snap(fx(T0, [(12.30, 76.65 + i * 0.0004, 10.0) for i in range(30)])), [], T0)
    assert s.notifs == [] and llm.calls == 0 and s.insight_rows["c1"]["status"] == "normal"


GOOD = {"headline": "Stopped on NH 275 after dark", "admin_summary": "Asha stopped abruptly on NH 275 and has not moved for 4 minutes.",
        "events": [{"id": "e_abrupt_stop", "explanation": "Consistent with a breakdown or a planned pull-over."}],
        "likely_explanations": ["A breakdown or a rest stop."], "admin_actions": [{"action": "CONTACT_TOURIST", "detail": "Call to check in."}],
        "user_message": "Your vehicle stopped on NH 275. Please tap I'm OK if you are fine."}


def test_llm_text_replaces_template_and_personalises_the_nudge_for_watch_events():
    s, llm = InsStore(), FakeLLM(response=GOOD)
    svc = service(s, llm, T0, kind="street")
    # a watch-level stop (daytime, 4 min, not planned): wording waits for the model
    sn = snap(drive_then_stop(T0, stop_s=240))
    svc.begin_tick()
    svc.process(TRIP, sn, [], T0)
    row = s.insight_rows["c1"]
    assert llm.calls == 1 and row["source"] == "llm" and row["headline"] == GOOD["headline"]
    assert s.notifs and s.notifs[0]["body"] == GOOD["user_message"]
    assert "Asha" not in llm.last_user and "+9198" not in llm.last_user       # no names or numbers reach the model


def test_llm_failure_falls_back_to_template_and_still_notifies():
    s, llm = InsStore(), FakeLLM(fail=True)
    svc = service(s, llm, T0, kind="street")
    svc.begin_tick()
    svc.process(TRIP, snap(drive_then_stop(T0, stop_s=240)), [], T0)
    assert s.insight_rows["c1"]["source"] == "template" and s.insight_rows["c1"]["llm_error"]
    assert s.notifs


def _packet():
    sit, ev, st = analyse(snap(drive_then_stop(NIGHT)), [], NIGHT)
    return build_packet(sit, ev, st, TRIP, NIGHT)


def test_validation_rejects_harm_unknown_ids_actions_and_contact_details():
    p = _packet()
    for bad in ({"admin_summary": "The tourist is in danger."}, {"events": [{"id": "e_nope", "explanation": "x"}]},
                {"admin_actions": [{"action": "SEND_POLICE", "detail": "x"}]},
                {"user_message": "Call 9876543210 now"}, {"user_message": "See www.example.com"}):
        try:
            validate_llm({**GOOD, **bad}, p, True)
            raise AssertionError(bad)
        except InsightRejected:
            pass
    ok, msg = validate_llm(GOOD, p, True)
    assert msg and validate_llm(GOOD, p, False)[1] is None


# ------------------------------------------------------------------ SOS location reasoning
def test_likely_issues_are_derived_from_place_and_evidence_not_generic():
    place = {"name": "NH 275, Mandya", "kind": "highway"}
    out = likely_issues(place, True, [], 12.3, 76.7, {"isolation_score": 0.8}, [{"kind": "abrupt_stop", "detail": {"hard": True}}],
                        {"weather_code": 95, "precipitation": 14, "wind_gusts_10m": 20}, "sms")
    ids = {o["id"] for o in out}
    assert {"vehicle_breakdown_or_collision", "severe_weather", "no_data_connection", "no_help_nearby"} <= ids
    assert all(o["why"] for o in out)
    market = likely_issues({"name": "Devaraja Market", "kind": "busy_public"}, False, [], 12.3, 76.7)
    assert "theft_or_harassment" in {o["id"] for o in market} and "vehicle_breakdown_or_collision" not in {o["id"] for o in market}
    assert likely_issues(None, False, [], 12.3, 76.7) == []             # no evidence, no invented reasons


def test_panic_brief_carries_location_issues_contacts_and_numbers():
    s = InsStore()
    s.trips = [TRIP]
    s.snaps["c1"] = lambda now: {**snap(drive_then_stop(now)), "alerts": [list(s.alerts.values())[0]]}
    s.alerts = {9: {"id": 9, "user_id": "u1", "type": "panic", "status": "open", "severity": "critical", "created_at": iso(NIGHT - 10),
                    "lat": 12.30, "lon": 76.70, "source": "sms"}}
    s.zones = lambda now: []
    e = Engine(s, FakeLLM(enabled=False), lambda la, lo: [{"category": "police", "name": "Mandya PS", "lat": 12.31, "lon": 76.7, "distance_km": 1.2, "phone": "0823-2222"}],
               lambda: NIGHT, geocode=lambda la, lo: {"name": "NH 275, Mandya", "kind": "highway"})
    import agent.engine as eng
    eng.weather_now = lambda la, lo: {"weather_code": 0}
    e.handle_panic(9)
    b = s.incidents[0]["brief"]
    assert b["last_known"]["location_name"] == "NH 275, Mandya"
    assert any(i["id"] == "vehicle_breakdown_or_collision" for i in b["likely_issues"])
    assert {c["phone"] for c in b["contacts_shared"]} == {"+919812345670", "+919876543210"}
    assert any(n["number"] == "112" for n in b["emergency_numbers"]) and b["nearby_services"][0]["phone"] == "0823-2222"
    assert b["sos_source"] == "sms"
