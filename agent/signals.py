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

    # Panic comes from alerts, independent of whether any fix exists.
    for a in snap.get("alerts") or []:
        if a.get("type") == "panic" and a.get("status") in ("open", "acknowledged"):
            at = parse_ts(a["created_at"])
            sigs.append(Signal("panic", [f"alert:{a['id']}"], at,
                               f"Panic button pressed at {hhmm(at)}.",
                               {"alert_id": a["id"], "lat": a.get("lat"), "lon": a.get("lon")}))

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

    return sigs, ctx
