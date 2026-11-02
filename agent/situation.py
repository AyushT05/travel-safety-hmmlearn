"""
Deterministic trip situation + abnormal-event detection.

This is the part that KNOWS things: where the traveller is, where they are heading, whether they follow the
planned route, whether they just stopped abruptly, lost signal, are circling, entered a hazard zone... Same
snapshot in, same answer out; no network, no LLM, so every event is reproducible and unit-testable.
The LLM (insights.py) only explains these facts in words; it cannot add, remove or re-grade an event.

Snapshot / zones: see signals.py. `place_of(lat, lon)` is an optional reverse-geocoder returning
{"name", "kind", ...} or None.
"""
import math
from dataclasses import dataclass, field
from statistics import median

from . import config
from .geo import haversine_m
from .signals import _fixes, _is_good, last_known
from .util import hhmm, is_night, parse_ts

SEV_RANK = {"info": 0, "watch": 1, "alert": 2}

# Detection thresholds (hand-set heuristics, like config.py's; tune from admin dismissals).
MOVING_MPS = 1.0                 # above this the traveller counts as moving
STOPPED_MPS = 0.8
ABRUPT_FROM_MPS = 8.0            # ~29 km/h: a vehicle, not a pedestrian
ABRUPT_WINDOW_S = 20             # slowing from vehicle speed to standstill within this is "abrupt"
HARD_STOP_DROP_MPS = 14.0        # a drop this large within HARD_STOP_WINDOW_S looks like a collision-grade stop
HARD_STOP_WINDOW_S = 6
STOP_CONFIRM_S = 60              # must still be stationary this long (a traffic light is not an incident)
UNPLANNED_STOP_MIN = 20
SPEEDING_MPS = 33.0              # 120 km/h
GAP_S = 120                      # a silence this long between fixes is a signal gap
GAP_REPEAT = 2                   # gaps in the last hour before it is "repeated"
DEV_WATCH_M, DEV_ALERT_M = 800.0, 1800.0
AT_STOP_M = 300.0                # within this of a planned stop = a planned stop
JUMP_MPS = 70.0
LOOP_PATH_M, LOOP_NET_M = 1500.0, 200.0


@dataclass
class Event:
    id: str
    kind: str
    severity: str                       # info | watch | alert
    at: float
    text: str
    detail: dict = field(default_factory=dict)
    evidence_ids: list = field(default_factory=list)
    notify_user: bool = False           # a human-facing nudge is appropriate for this event


# ------------------------------------------------------------------ geometry / route helpers
def _waypoints(trip):
    out = []
    for p in (trip.get("place_details") or []):
        if isinstance(p, dict) and p.get("_type") != "planned_route" and p.get("lat") is not None and p.get("lon") is not None:
            try:
                out.append({"name": str(p.get("name") or "")[:60] or None, "lat": float(p["lat"]), "lon": float(p["lon"])})
            except (TypeError, ValueError):
                continue
    return out


def _polyline(trip, waypoints):
    route = trip.get("planned_route")
    poly = route.get("polyline") if isinstance(route, dict) else None
    if isinstance(poly, list) and len(poly) >= 2:
        return poly, (route.get("source") or "planned_route")
    if len(waypoints) >= 2:
        return [[w["lat"], w["lon"]] for w in waypoints], "waypoints_straight_line"
    return None, None


def _route_len_m(poly):
    return sum(haversine_m(poly[i][0], poly[i][1], poly[i + 1][0], poly[i + 1][1]) for i in range(len(poly) - 1))


def _speed(f, prev):
    """m/s: the device-reported speed when present, else derived from the two fixes."""
    if f.get("speed") is not None and f["speed"] >= 0:
        return float(f["speed"])
    if prev is not None and f["t"] > prev["t"]:
        return haversine_m(prev["lat"], prev["lon"], f["lat"], f["lon"]) / (f["t"] - prev["t"])
    return None


def _bearing(a, b):
    y = math.sin(math.radians(b["lon"] - a["lon"])) * math.cos(math.radians(b["lat"]))
    x = (math.cos(math.radians(a["lat"])) * math.sin(math.radians(b["lat"]))
         - math.sin(math.radians(a["lat"])) * math.cos(math.radians(b["lat"])) * math.cos(math.radians(b["lon"] - a["lon"])))
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def _compass(deg):
    return ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"][int((deg + 22.5) // 45) % 8]


def _fmt_dur(seconds):
    m = int(round(seconds / 60))
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"


# ------------------------------------------------------------------ main entry
def analyse(snap, zones, now, place_of=None, responses=None):
    """Returns (situation: dict, events: list[Event], status: 'normal'|'watch'|'alert')."""
    from geofence.deviation import calculate_cross_track      # lazy: same import style as signals.py

    trip = snap.get("trip") or {}
    fixes = _fixes(snap.get("locations"))
    night = is_night(now)
    sit = {"now": now, "night": night, "local_time": hhmm(now), "has_fixes": bool(fixes)}
    events = []

    def ev(kind, severity, at, text, detail=None, ids=None, notify=False):
        events.append(Event(id=f"e_{kind}", kind=kind, severity=severity, at=at, text=text, detail=detail or {},
                            evidence_ids=ids or [], notify_user=notify))

    assess = snap.get("assessment") or {}
    sit["area"] = {"safety_status": assess.get("safety_status"), "isolation_score": assess.get("isolation_score")}
    isolated = (assess.get("isolation_score") or 0) >= 0.6

    if not fixes:
        sit["state"] = "no_data"
        return sit, events, "normal"

    last = fixes[-1]
    lk = last_known(fixes)
    age = now - last["t"]
    good = [f for f in fixes if _is_good(f)] or fixes
    cur = good[-1]
    place = place_of(cur["lat"], cur["lon"]) if place_of else None
    place_name = (place or {}).get("name")
    sit["position"] = {"lat": cur["lat"], "lon": cur["lon"], "at": cur["t"], "age_s": round(now - cur["t"]),
                       "accuracy_m": cur.get("acc"), "place": place_name, "place_kind": (place or {}).get("kind")}
    where = f"near {place_name}" if place_name else f"at {cur['lat']:.4f}, {cur['lon']:.4f}"

    # ---- movement: speed now, average moving speed, how long stationary
    spd_now = _speed(cur, good[-2] if len(good) > 1 else None)
    recent = [f for f in good if f["t"] >= now - 600]
    speeds = []
    for i, f in enumerate(recent):
        s = _speed(f, recent[i - 1] if i else None)
        if s is not None:
            speeds.append(s)
    moving_speeds = [s for s in speeds if s >= MOVING_MPS]
    avg_moving = (sum(moving_speeds) / len(moving_speeds)) if moving_speeds else None
    stationary_since = None
    if spd_now is not None and spd_now < STOPPED_MPS:
        stationary_since = cur["t"]
        for f in reversed(good):
            if haversine_m(f["lat"], f["lon"], cur["lat"], cur["lon"]) <= max(25.0, (f.get("acc") or 0)):
                stationary_since = f["t"]
            else:
                break
    stopped_s = (cur["t"] - stationary_since) if stationary_since is not None else 0
    heading = None
    if len(good) >= 3:
        far = next((f for f in reversed(good[:-1]) if haversine_m(f["lat"], f["lon"], cur["lat"], cur["lon"]) >= 40), None)
        if far and (spd_now or 0) >= MOVING_MPS:
            heading = _compass(_bearing(far, cur))
    state = ("signal_lost" if age >= config.SIGNAL_LOST_SEC else
             "moving" if (spd_now or 0) >= MOVING_MPS else "stopped")
    sit["movement"] = {"state": state, "speed_kmh": None if spd_now is None else round(spd_now * 3.6, 1),
                       "avg_moving_kmh": None if avg_moving is None else round(avg_moving * 3.6, 1),
                       "stopped_for_s": round(stopped_s), "heading": heading}

    # ---- route adherence + destination
    wps = _waypoints(trip)
    poly, poly_source = _polyline(trip, wps)
    route = {"has_route": bool(poly), "source": poly_source}
    dev_m = None
    if poly:
        xt = calculate_cross_track(cur["lat"], cur["lon"], poly)
        dev_m = xt["distance_m"]
        progress = xt["progress_ratio"]
        total = _route_len_m(poly)
        remaining_m = max(0.0, (1 - progress) * total)
        dest = wps[-1] if wps else None
        if dev_m > DEV_WATCH_M and dest:
            remaining_m = haversine_m(cur["lat"], cur["lon"], dest["lat"], dest["lon"])
        nxt = None
        for w in wps:
            wx = calculate_cross_track(w["lat"], w["lon"], poly)
            if wx["progress_ratio"] >= progress - 0.01 and haversine_m(w["lat"], w["lon"], cur["lat"], cur["lon"]) > AT_STOP_M:
                nxt = w
                break
        eta_min = round(remaining_m / avg_moving / 60) if avg_moving and avg_moving * 3.6 >= 5 else None
        # Trend of the cross-track distance over the last ~6 good fixes: are they moving away or back?
        trend = None
        if len(recent) >= 6:
            d_old = calculate_cross_track(recent[-6]["lat"], recent[-6]["lon"], poly)["distance_m"]
            trend = "away" if dev_m - d_old > 150 else "returning" if d_old - dev_m > 150 else "steady"
        route.update({"deviation_m": round(dev_m), "progress_pct": round(progress * 100, 1),
                      "on_route": dev_m <= 500, "trend": trend, "remaining_km": round(remaining_m / 1000, 1),
                      "eta_min": eta_min, "next_stop": (nxt or {}).get("name"),
                      "destination": (dest or {}).get("name"), "stops": [w["name"] for w in wps]})
        if dev_m >= DEV_WATCH_M:
            sev = "alert" if dev_m >= DEV_ALERT_M else "watch"
            moving_away = trend == "away"
            txt = (f"{round(dev_m / 1000, 1)} km off the planned route {where}"
                   f"{', still moving away from it' if moving_away else ', coming back toward it' if trend == 'returning' else ''}"
                   f"; {round(progress * 100)}% of the route completed.")
            ev("route_deviation", sev, cur["t"], txt, {"deviation_m": round(dev_m), "trend": trend, "progress_pct": round(progress * 100)},
               [f"loc:{cur.get('id')}"], notify=True)
    else:
        dest_names = [str(p)[:40] for p in (trip.get("places") or [])][:6]
        route.update({"destination": dest_names[-1] if dest_names else None, "stops": dest_names,
                      "note": "Itinerary stops have no coordinates, so route adherence cannot be checked."})
    sit["route"] = route

    # ---- signal: current loss, historical gaps, restoration
    gaps = []
    for a, b in zip(fixes, fixes[1:]):
        if b["t"] - a["t"] >= GAP_S and b["t"] >= now - 3600:
            gaps.append((a, b, b["t"] - a["t"]))
    sit["connectivity"] = {"age_s": round(age), "gaps_last_hour": len(gaps),
                           "longest_gap_s": round(max((g[2] for g in gaps), default=0))}
    if age >= config.SIGNAL_LOST_SEC:
        speeds3 = [s for s in speeds[-3:]]
        moving_before = bool(speeds3) and median(speeds3) > 0.7
        sev = "alert" if (age >= 600 or (moving_before and (night or isolated))) else "watch"
        ev("signal_lost", sev, last["t"],
           f"No location update for {_fmt_dur(age)}; last seen {where}{' while moving' if moving_before else ''}.",
           {"age_s": round(age), "moving_before_loss": moving_before}, [f"loc:{last.get('id')}"])
    elif gaps and now - gaps[-1][1]["t"] <= 300 and gaps[-1][2] >= 180:
        g = gaps[-1]
        ev("signal_restored", "info", g[1]["t"], f"Signal returned after {_fmt_dur(g[2])} without updates.",
           {"gap_s": round(g[2])}, notify=True)
    if len(gaps) >= GAP_REPEAT and age < config.SIGNAL_LOST_SEC:
        ev("repeated_signal_gaps", "watch", gaps[-1][1]["t"],
           f"{len(gaps)} signal gaps in the last hour (longest {_fmt_dur(max(g[2] for g in gaps))}).",
           {"gaps": len(gaps)})

    # ---- stops: abrupt stop vs ordinary stop vs unplanned long stop
    at_planned_stop = any(haversine_m(w["lat"], w["lon"], cur["lat"], cur["lon"]) <= AT_STOP_M for w in wps)
    services_here = (place or {}).get("kind") in ("services", "transit", "lodging_or_home")
    abrupt = None
    window = [f for f in good if f["t"] >= now - 900]
    for i in range(len(window) - 1, 0, -1):
        s_i = _speed(window[i], window[i - 1])
        if s_i is None or s_i > STOPPED_MPS:
            continue
        # walk back to the last fix that was at vehicle speed, within ABRUPT_WINDOW_S
        for j in range(i - 1, max(-1, i - 8), -1):
            s_j = _speed(window[j], window[j - 1] if j else None)
            if window[i]["t"] - window[j]["t"] > ABRUPT_WINDOW_S:
                break
            if s_j is not None and s_j >= ABRUPT_FROM_MPS:
                abrupt = {"from_kmh": round(s_j * 3.6), "drop_s": round(window[i]["t"] - window[j]["t"]),
                          "hard": (s_j - s_i) >= HARD_STOP_DROP_MPS and (window[i]["t"] - window[j]["t"]) <= HARD_STOP_WINDOW_S,
                          "at": window[i]["t"], "id": window[i].get("id")}
                break
        if abrupt:
            break
    if abrupt and state in ("stopped", "signal_lost") and (stopped_s >= STOP_CONFIRM_S or state == "signal_lost"):
        planned = at_planned_stop or services_here
        risky = night or isolated or (place or {}).get("kind") in ("highway", "remote_track", "wilderness")
        if planned:
            sev = "info"
        elif abrupt["hard"] or (risky and stopped_s >= 120) or stopped_s >= 600 or state == "signal_lost" and risky:
            sev = "alert"
        elif stopped_s >= 180 or (risky and stopped_s >= STOP_CONFIRM_S):
            sev = "watch"
        else:
            sev = "info"                       # a red light or a toll booth: recorded, not escalated
        ev("abrupt_stop", sev, abrupt["at"],
           f"Slowed from {abrupt['from_kmh']} km/h to a standstill within {abrupt['drop_s']} s {where}; "
           f"stationary for {_fmt_dur(stopped_s)}"
           f"{' at a planned stop or service location' if planned else ''}"
           f"{'; the deceleration was collision-grade' if abrupt['hard'] else ''}"
           f"{'; this is a risky place to stop (night, isolated or on a highway)' if risky and not planned else ''}.",
           {**{k: abrupt[k] for k in ("from_kmh", "drop_s", "hard")}, "stationary_s": round(stopped_s), "planned": planned,
            "risky_place": bool(risky)},
           [f"loc:{abrupt['id']}"], notify=(sev != "info"))
    elif state == "stopped" and stopped_s >= UNPLANNED_STOP_MIN * 60 and not at_planned_stop and not services_here:
        sev = "alert" if ((night or isolated) and stopped_s >= 40 * 60) else "watch"
        ev("unplanned_stop", sev, stationary_since,
           f"Stationary {where} for {_fmt_dur(stopped_s)}, not at a planned stop.",
           {"stationary_s": round(stopped_s)}, [f"loc:{cur.get('id')}"], notify=True)

    # ---- speed, erratic movement, circling, GPS quality
    if len(recent) >= 4 and sum(1 for s in speeds[-4:] if s >= SPEEDING_MPS) >= 4:
        ev("high_speed", "watch", cur["t"], f"Sustained speed around {round(median(speeds[-4:]) * 3.6)} km/h {where}.",
           {"speed_kmh": round(median(speeds[-4:]) * 3.6)}, notify=True)
    act = snap.get("activity") or {}
    reversals = 0
    for a, b, c in zip(good[-8:], good[-7:], good[-6:]):
        if (_speed(b, a) or 0) > 1.2 and (_speed(c, b) or 0) > 1.2:
            d = abs(_bearing(b, c) - _bearing(a, b))
            d = 360 - d if d > 180 else d
            reversals += d >= 110
    if act.get("current_state") == "ERRATIC" or reversals >= 2:
        ev("erratic_movement", "watch", cur["t"], f"Erratic movement {where}: repeated sharp direction changes.",
           {"reversals": reversals, "hsmm_state": act.get("current_state")}, notify=True)
    ten = [f for f in good if f["t"] >= now - 600]
    if len(ten) >= 8:
        path = sum(haversine_m(a["lat"], a["lon"], b["lat"], b["lon"]) for a, b in zip(ten, ten[1:]))
        net = haversine_m(ten[0]["lat"], ten[0]["lon"], ten[-1]["lat"], ten[-1]["lon"])
        if path >= LOOP_PATH_M and net <= LOOP_NET_M and (spd_now or 0) >= 2:
            ev("circling", "watch", cur["t"], f"Moving in circles {where}: {round(path / 1000, 1)} km travelled, "
               f"only {round(net)} m from where this 10 minutes began.", {"path_m": round(path), "net_m": round(net)}, notify=True)
    jumps = [(a, b) for a, b in zip(fixes, fixes[1:]) if 0 < b["t"] - a["t"] <= 300
             and (haversine_m(a["lat"], a["lon"], b["lat"], b["lon"]) - (a.get("acc") or 0) - (b.get("acc") or 0)) / (b["t"] - a["t"]) > JUMP_MPS]
    if jumps and jumps[-1][1]["t"] >= now - 900:
        ev("gps_jump", "watch", jumps[-1][1]["t"], "Position jumped implausibly fast between two fixes (GPS error or spoofing).",
           {"jumps": len(jumps)})
    last5 = fixes[-5:]
    if len(last5) >= 3 and sum(1 for f in last5 if f["acc"] is not None and f["acc"] > config.POOR_FIX_ACCURACY_M) >= 3:
        ev("gps_degraded", "info", last["t"], "Recent GPS accuracy is poor, so the position is uncertain.",
           {"accuracy_m": last["acc"]})

    # ---- hazards and area
    for z in zones or []:
        d = haversine_m(cur["lat"], cur["lon"], z["lat"], z["lon"])
        r = z.get("radius_m") or 500
        if d <= r:
            sev = "alert" if z.get("severity") == "critical" or night else "watch"
            ev("in_hazard_zone", sev, cur["t"], f"Inside a verified {z.get('category', 'hazard')} zone {where}.",
               {"zone_id": z["id"], "category": z.get("category"), "severity": z.get("severity")},
               [f"zone:{z['id']}"], notify=True)
            break
        if (spd_now or 0) >= 2.5 and len(good) >= 2:
            p = good[-2]
            dt = max(1.0, cur["t"] - p["t"])
            pl = cur["lat"] + (cur["lat"] - p["lat"]) / dt * 900
            po = cur["lon"] + (cur["lon"] - p["lon"]) / dt * 900
            if haversine_m(pl, po, z["lat"], z["lon"]) <= r + 300 and d > haversine_m(pl, po, z["lat"], z["lon"]):
                ev("approaching_hazard", "watch", cur["t"],
                   f"Heading toward a verified {z.get('category', 'hazard')} zone, about "
                   f"{max(1, round((d - r) / max(spd_now, 1) / 60))} min away at this speed.",
                   {"zone_id": z["id"], "category": z.get("category")}, [f"zone:{z['id']}"], notify=True)
                break
    if assess.get("safety_status") == "unsafe":
        ev("unsafe_area", "watch", cur["t"], "The area around the traveller is currently rated unsafe by the zone safety score.",
           {"status": "unsafe"}, notify=False)
    if night and isolated and state in ("moving", "stopped") and not any(e.kind == "abrupt_stop" for e in events):
        ev("isolated_at_night", "info", cur["t"], f"Traveller is in a sparsely populated area {where} after dark.", {"isolation": assess.get("isolation_score")})

    # ---- the traveller's own answers (check-ins)
    ok = [r for r in (responses or []) if r.get("response") == "ok" and r.get("responded_at")
          and now - parse_ts(r["responded_at"]) <= 1800]
    if ok:
        t = parse_ts(ok[0]["responded_at"])
        ev("traveller_confirmed_ok", "info", t, f"Traveller confirmed they are OK at {hhmm(t)}.", {"notification_id": ok[0].get("id")})
    helpme = [r for r in (responses or []) if r.get("response") == "need_help" and r.get("responded_at")
              and now - parse_ts(r["responded_at"]) <= 1800]
    if helpme:
        t = parse_ts(helpme[0]["responded_at"])
        ev("traveller_needs_help", "alert", t, f"Traveller replied that they need help at {hhmm(t)}.", {"notification_id": helpme[0].get("id")})

    # A confirmed-OK reply softens (never erases) events the traveller was asked about.
    if ok:
        for e in events:
            if e.kind in ("abrupt_stop", "unplanned_stop", "signal_lost", "erratic_movement") and e.severity == "alert":
                e.severity = "watch"
                e.detail["softened_by_traveller_reply"] = True

    real = [e for e in events if e.kind != "traveller_confirmed_ok"]
    status = "alert" if any(e.severity == "alert" for e in real) else "watch" if any(e.severity == "watch" for e in real) else "normal"
    sit["summary"] = describe(sit)
    return sit, sorted(events, key=lambda e: (-SEV_RANK[e.severity], -e.at)), status


def describe(sit):
    """One deterministic sentence an admin can read without any AI."""
    pos, mv, rt = sit.get("position"), sit.get("movement"), sit.get("route") or {}
    if not pos:
        return "No location data yet."
    where = f"near {pos['place']}" if pos.get("place") else f"at {pos['lat']:.4f}, {pos['lon']:.4f}"
    if mv["state"] == "signal_lost":
        s = f"Last seen {where} {_fmt_dur(pos['age_s'])} ago; no signal since."
    elif mv["state"] == "stopped":
        s = f"Stopped {where}{' for ' + _fmt_dur(mv['stopped_for_s']) if mv['stopped_for_s'] >= 60 else ''}."
    else:
        s = (f"Moving {mv['heading'] + ' ' if mv.get('heading') else ''}at {mv['speed_kmh']} km/h {where}."
             if mv.get("speed_kmh") is not None else f"Moving {where}.")
    if rt.get("has_route") and rt.get("deviation_m") is not None:
        s += (f" On the planned route ({rt['deviation_m']} m from it)" if rt["on_route"]
              else f" {round(rt['deviation_m'] / 1000, 1)} km off the planned route")
        if rt.get("destination"):
            s += f", {rt['remaining_km']} km from {rt['destination']}"
            if rt.get("eta_min"):
                s += f" (ETA about {_fmt_dur(rt['eta_min'] * 60)})"
        s += "."
    elif rt.get("destination"):
        s += f" Heading to {rt['destination']} (route adherence cannot be checked: stops have no coordinates)."
    return s
