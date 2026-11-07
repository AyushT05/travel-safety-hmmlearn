"""A planned meal / tea / rest halt must not be reported as an unusual stop; everything else still is."""
from agent.breaks import stop_allowance_min, within_expected_break, MEALS, REST_BREAK_MIN, GRACE_FACTOR
from agent.situation import analyse
from agent.signals import extract_signals
from agent.util import parse_ts
from tests.test_analyst import TRIP, fx, kinds
from tests.fakes import T0, NIGHT   # T0 = 12:30 local, NIGHT = 00:00 local

CAR = {**TRIP, "transport_mode": "car"}


def local(h, m=0):                      # 2026-10-17 at h:m IST, as epoch
    return parse_ts(f"2026-10-17T{h:02d}:{m:02d}:00+05:30")


def stop_at(now, minutes, lat=12.30, lon=76.70):
    n = minutes * 60 // 5
    return fx(now, [(lat, lon - 0.003 + i * 0.0005, 12.0) for i in range(6)] + [(lat, lon, 0.0)] * n)


def snap_for(trip, locs):
    return {"trip": trip, "locations": locs, "activity": None, "alerts": [], "assessment": None}


# ---- the allowance itself
def test_no_transport_mode_means_no_allowance_so_old_cards_behave_as_before():
    assert stop_allowance_min(TRIP, local(13)) == (0, None)
    assert stop_allowance_min({"transport_mode": "hovercraft"}, local(13)) == (0, None)


def test_meal_windows_give_grace_scaled_allowance():
    for name, lo, hi, dur in MEALS:
        mid = (lo + hi) // 2
        allow, why = stop_allowance_min(CAR, local(mid // 60, mid % 60))
        assert allow == round(dur * GRACE_FACTOR) and name in why


def test_outside_meal_windows_only_a_short_rest_is_expected():
    assert stop_allowance_min(CAR, local(11, 0)) == (round(REST_BREAK_MIN * 2), "rest / fuel stop")


def test_stop_that_starts_before_lunch_but_runs_into_it_is_still_a_lunch_stop():
    assert stop_allowance_min(CAR, local(11, 50), 45 * 60)[1] == "lunch time"
    assert stop_allowance_min(CAR, local(11, 50), 5 * 60)[1] == "rest / fuel stop"


def test_train_and_flight_allow_long_waits():
    assert stop_allowance_min({"transport_mode": "train"}, local(3))[0] == 120
    assert stop_allowance_min({"transport_mode": "flight"}, local(3))[0] == 360


def test_within_expected_break_boundary():
    allow = round(45 * GRACE_FACTOR)
    assert within_expected_break(CAR, local(13), allow * 60)[0] is True
    assert within_expected_break(CAR, local(13), allow * 60 + 1)[0] is False


# ---- end to end through the real analysis
def events_for(trip, now, minutes):
    _sit, events, _status = analyse(snap_for(trip, stop_at(now, minutes)), [], now)
    return kinds(events)


def test_lunch_stop_on_a_car_trip_is_not_flagged_but_the_same_stop_without_a_plan_is():
    assert "unplanned_stop" in events_for(TRIP, T0, 40)          # legacy card: flagged, as before
    assert "unplanned_stop" not in events_for(CAR, T0, 40)       # 40 min at lunchtime, within 68 min


def test_a_lunch_stop_that_runs_far_too_long_is_still_flagged():
    assert "unplanned_stop" in events_for(CAR, T0, 100)


def test_a_40_minute_stop_outside_meal_time_is_still_flagged():
    assert "unplanned_stop" in events_for(CAR, local(11, 40), 40)   # started 11:00, ends 11:40: no meal window


def test_night_stop_far_beyond_any_dinner_is_still_flagged():
    assert "unplanned_stop" in events_for(CAR, NIGHT, 90)


def activity(now, minutes, tail=0.04):
    from agent.util import iso
    return {"current_state": "RESTING", "updated_at": iso(now), "duration_minutes": minutes, "tail_probability": tail}


def rest_kinds(trip, now, minutes):
    snap = {**snap_for(trip, stop_at(now, minutes)), "activity": activity(now, minutes)}
    out = extract_signals(snap, [], now)
    sigs = out[0] if isinstance(out, tuple) else out
    return {x.kind for x in sigs}


def test_prolonged_rest_model_finding_is_excused_only_inside_the_meal_allowance():
    assert "prolonged_rest" in rest_kinds(TRIP, T0, 40)              # no plan: unchanged
    assert "prolonged_rest" not in rest_kinds(CAR, T0, 40)           # lunch, within allowance
    assert "prolonged_rest" in rest_kinds(CAR, T0, 100)              # way past it


def test_meal_table_matches_the_mobile_app():
    # Same literal table is asserted in LocationShare/__tests__/tripEstimate.test.mjs.
    assert MEALS == (("breakfast", 450, 600, 35), ("lunch", 750, 870, 45),
                     ("snack", 960, 1050, 20), ("dinner", 1170, 1290, 45))
    assert (REST_BREAK_MIN, GRACE_FACTOR) == (15, 1.5)
