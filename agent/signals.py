"""
Deterministic signal extraction. No network, no LLM, no randomness: the same
snapshot always yields the same signals, so every incident can be reproduced
and unit-tested.

A snapshot is a plain dict:
  trip        travel_cards row (id, user_id, places, emergency_contacts, ...)
  locations   rows {id, lat, lon, accuracy, speed, recorded_at}, any order
  activity    trip_activity_state row or None
  alerts      alerts rows for this user (recent)
  assessment  safety_assessments row or None (CLIENT-TRIGGERED: lower trust)
zones is a list of VERIFIED, unexpired unsafe_zones rows.
"""
from dataclasses import dataclass, field
import math
from statistics import median

from . import config
from .geo import haversine_m
from .util import parse_ts, is_night, hhmm


@dataclass
class Signal:
    kind: str                 # zone_inside | signal_lost | prolonged_rest | poor_gps | isolated_area | panic
    evidence_ids: list
    at: float                 # epoch seconds the fact became true / was observed
    text: str                 # deterministic English sentence
    detail: dict = field(default_factory=dict)


def _fixes(locations):
    out = []
    for r in locations or []:
        if r.get("lat") is None or r.get("lon") is None or not r.get("recorded_at"):
            continue
        out.append({
            "id": r.get("id"), "t": parse_ts(r["recorded_at"]),
            "lat": r["lat"], "lon": r["lon"],
            "acc": r.get("accuracy"), "speed": r.get("speed"),
        })
    out.sort(key=lambda f: f["t"])
    return out


def _is_good(f):
    return f["acc"] is None or f["acc"] <= config.GOOD_FIX_ACCURACY_M


def last_known(fixes):
    """Newest fix with usable accuracy, else newest fix at all."""
    good = [f for f in fixes if _is_good(f)]
    return (good or fixes or [None])[-1]


def extract_signals(snap, zones, now):
    """Returns (signals, ctx)."""
    fixes = _fixes(snap.get("locations"))
    sigs = []
    ctx = {"night": is_night(now), "has_fixes": bool(fixes), "last_fix": None, "age_s": None}

    # Panic and Deviation come from alerts, independent of whether any fix exists.
    for a in snap.get("alerts") or []:
        if a.get("type") == "panic" and a.get("status") in ("open", "acknowledged"):
            at = parse_ts(a["created_at"])
            sigs.append(Signal("panic", [f"alert:{a['id']}"], at,
                               f"Panic button pressed at {hhmm(at)}.",
                               {"alert_id": a["id"], "lat": a.get("lat"), "lon": a.get("lon")}))
        elif a.get("type") == "deviation" and a.get("status") in ("open", "acknowledged"):
            at = parse_ts(a["created_at"])
            msg = a.get("message") or f"Route deviation detected at {hhmm(at)}."
            sigs.append(Signal("route_deviation", [f"alert:{a['id']}"], at, msg,
                               {"alert_id": a["id"], "lat": a.get("lat"), "lon": a.get("lon"), "deviation_m": 2000.0}))

    if not fixes:
        return sigs, ctx

    last = fixes[-1]
    lk = last_known(fixes)
    age = now - last["t"]
    ctx.update({"last_fix": last, "last_known": lk, "age_s": age})
    good = [f for f in fixes if _is_good(f)]

    # --- zone_inside: use the newest GOOD fix; a bad fix cannot prove presence ---
    if good:
        g = good[-1]
        for z in zones or []:
            d = haversine_m(g["lat"], g["lon"], z["lat"], z["lon"])
            r = z.get("radius_m") or 500
            if d > r:
                continue
            # walk back through consecutive good fixes still in/at the edge of the zone
            entered = g["t"]
            for f in reversed(good):
                if haversine_m(f["lat"], f["lon"], z["lat"], z["lon"]) <= r + config.ZONE_EXIT_TOLERANCE_M:
                    entered = f["t"]
                else:
                    break
            dwell_s = g["t"] - entered
            margin = r - d
            uncertain = g["acc"] is not None and margin < g["acc"]
            sigs.append(Signal(
                "zone_inside", [f"zone:{z['id']}", f"loc:{g['id']}"], entered,
                f"Inside verified {z.get('category', 'hazard')} zone "
                f"(severity {z.get('severity', 'warning')}) since {hhmm(entered)}"
                f"{', position uncertain near the zone edge' if uncertain else ''}.",
                {"zone_id": z["id"], "category": z.get("category"), "severity": z.get("severity", "warning"),
                 "dwell_min": round(dwell_s / 60, 1), "margin_m": round(margin, 1), "uncertain": uncertain,
                 "last_seen": g["t"]}))

    # --- signal_lost ---
    if age >= config.SIGNAL_LOST_SEC:
        recent_speeds = [f["speed"] for f in fixes[-3:] if f["speed"] is not None and f["speed"] >= 0]
        moving = bool(recent_speeds) and median(recent_speeds) > 0.7
        near_zone = None
        for z in zones or []:
            d = haversine_m(lk["lat"], lk["lon"], z["lat"], z["lon"])
            if d <= (z.get("radius_m") or 500) + 300:
                near_zone = z["id"]
                break
        ev = [f"loc:{last['id']}"]
        for a in snap.get("alerts") or []:
            if a.get("type") == "dropout" and a.get("status") in ("open", "acknowledged"):
                ev.append(f"alert:{a['id']}")
        sigs.append(Signal(
            "signal_lost", ev, last["t"],
            f"No location update for {int(age // 60)} min; last fix at {hhmm(last['t'])}"
            f"{' while moving' if moving else ''}.",
            {"age_s": round(age), "moving_before_loss": moving, "near_zone_id": near_zone,
             "last_accuracy_m": last["acc"]}))

    # --- prolonged_rest: HSMM duration tail, only if the state was computed from the fixes we have ---
    act = snap.get("activity")
    if act and act.get("current_state") == "RESTING" and act.get("updated_at"):
        upd = parse_ts(act["updated_at"])
        dur = act.get("duration_minutes")
        tail = act.get("tail_probability")
        if (dur is not None and tail is not None and upd >= last["t"] - 120
                and dur >= config.REST_MIN_DURATION_MIN and tail <= config.REST_TAIL_P):
            sigs.append(Signal(
                "prolonged_rest", [f"act:{snap['trip']['id']}"], upd - dur * 60,
                f"Resting for about {round(dur)} min, which the duration model rates as unusual "
                f"(tail probability {float(tail):.2f}).",
                {"duration_min": float(dur), "tail_probability": float(tail)}))

    # --- poor_gps: confidence modifier, not an independent signal ---
    recent = fixes[-5:]
    if len(recent) >= 3 and sum(1 for f in recent if f["acc"] is not None and f["acc"] > config.POOR_FIX_ACCURACY_M) >= 3:
        sigs.append(Signal("poor_gps", [f"loc:{last['id']}"], last["t"],
                           "Recent GPS accuracy is poor, so the position is uncertain.",
                           {"last_accuracy_m": last["acc"]}))

    # --- isolated_area: context only; the assessment is triggered by the client app ---
    a = snap.get("assessment")
    if a and (a.get("isolation_score") or 0) >= config.ISOLATED_SCORE:
        sigs.append(Signal("isolated_area", [f"assess:{snap['trip']['user_id']}"], last["t"],
                           "The area around the last position looks isolated (few roads, buildings or settlements).",
                           {"isolation_score": a["isolation_score"], "source": "client_triggered_assessment"}))

    # --- erratic_movement: HSMM classification or sharp erratic velocity oscillations ---
    act = snap.get("activity")
    if act and act.get("current_state") == "ERRATIC":
        sigs.append(Signal(
            "erratic_movement", [f"act:{snap['trip'].get('id', 'trip')}"], last["t"],
            "Erratic movement behavior detected by HSMM (unusual velocity fluctuations / direction swings).",
            {"state": "ERRATIC", "duration_min": act.get("duration_minutes", 1)}))
    elif len(fixes) >= 5:
        # Check for rapid direction reversals (>120 deg) while moving
        reversals = 0
        for i in range(len(fixes) - 2):
            f1, f2, f3 = fixes[i], fixes[i + 1], fixes[i + 2]
            if (f1.get("speed") or 0) > 1.2 and (f2.get("speed") or 0) > 1.2:
                # Approximate bearing change
                b1 = math.atan2(f2["lon"] - f1["lon"], f2["lat"] - f1["lat"])
                b2 = math.atan2(f3["lon"] - f2["lon"], f3["lat"] - f2["lat"])
                diff = abs(math.degrees(b2 - b1))
                if diff > 180:
                    diff = 360 - diff
                if diff >= 110:
                    reversals += 1
        if reversals >= 2:
            sigs.append(Signal(
                "erratic_movement", [f"loc:{last['id']}"], last["t"],
                "Frequent sharp direction reversals and erratic movement detected.",
                {"reversals": reversals}))

    # --- distress_stop: sudden deceleration from vehicular speed to 0 in isolated/night setting ---
    if len(fixes) >= 4 and lk and (lk.get("speed") or 0) < 0.5:
        earlier_speeds = [f.get("speed") for f in fixes[-6:-2] if f.get("speed") is not None]
        if earlier_speeds and max(earlier_speeds) >= 10.0:   # Was moving > 36 km/h
            is_isolated = bool(a and (a.get("isolation_score") or 0) >= 0.4)
            if ctx.get("night") or is_isolated:
                sigs.append(Signal(
                    "distress_stop", [f"loc:{last['id']}"], last["t"],
                    f"Sudden stop from vehicular speed ({round(max(earlier_speeds) * 3.6)} km/h to standstill)"
                    f" in an {'isolated sector' if is_isolated else 'unfamiliar area at night'}.",
                    {"peak_speed_kmh": round(max(earlier_speeds) * 3.6, 1), "isolated": is_isolated}))

    # --- predictive_hazard: heading projection into active verified hazard zones ---
    if len(fixes) >= 2 and lk and (lk.get("speed") or 0) >= 2.5:
        prev_f = fixes[-2]
        dt = max(1.0, lk["t"] - prev_f["t"])
        v_lat = (lk["lat"] - prev_f["lat"]) / dt
        v_lon = (lk["lon"] - prev_f["lon"]) / dt
        # Project 20 minutes (1200 seconds) ahead
        proj_lat = lk["lat"] + v_lat * 1200.0
        proj_lon = lk["lon"] + v_lon * 1200.0
        for z in zones or []:
            proj_dist = haversine_m(proj_lat, proj_lon, z["lat"], z["lon"])
            r = (z.get("radius_m") or 500)
            if proj_dist <= r + 400:
                current_dist = haversine_m(lk["lat"], lk["lon"], z["lat"], z["lon"])
                if current_dist > proj_dist:  # Moving towards it
                    eta_min = round(max(1.0, (current_dist - r) / (lk.get("speed") * 60.0)))
                    sigs.append(Signal(
                        "predictive_hazard", [f"loc:{lk['id']}", f"zone:{z['id']}"], lk["t"],
                        f"Predictive alert: Trajectory heading toward {z.get('category', 'hazard')} zone "
                        f"({z.get('severity', 'warning')}) with estimated entry in ~{eta_min} min.",
                        {"zone_id": z["id"], "category": z.get("category"), "eta_min": eta_min}))
                    break

    # --- route_deviation: if trip has planned route or place_details, verify adherence ---
    has_dev_alert = any(s.kind == "route_deviation" for s in sigs)
    if not has_dev_alert and lk:
        trip = snap.get("trip") or {}
        route = trip.get("planned_route")
        polyline = route.get("polyline") if isinstance(route, dict) else None
        if not polyline:
            # Try to build from place_details if available
            p_details = trip.get("place_details") or []
            pts = [[float(p["lat"]), float(p["lon"])] for p in p_details if isinstance(p, dict) and p.get("lat") and p.get("lon")]
            if len(pts) >= 2:
                polyline = pts
        if polyline and len(polyline) >= 2:
            from geofence.deviation import calculate_cross_track
            xt = calculate_cross_track(lk["lat"], lk["lon"], polyline)
            if xt["distance_m"] >= 1800.0:
                km_off = round(xt["distance_m"] / 1000.0, 1)
                sigs.append(Signal(
                    "route_deviation", [f"loc:{lk['id']}"], lk["t"],
                    f"Tourist has deviated {km_off} km from planned route (progress {round(xt['progress_ratio'] * 100)}%).",
                    {"deviation_m": xt["distance_m"], "deviation_km": km_off, "progress_ratio": xt["progress_ratio"]}))

    return sigs, ctx
