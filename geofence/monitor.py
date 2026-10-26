"""Live geofence: one location fix in, zero or more alerts out.

Runs beside the HSMM scoring and must never affect it: the caller invokes handle_location from a
background thread and swallows every exception. Rules enforced here:
  * Only LIVE fixes raise alerts. A phone that was offline uploads its backlog at once; judging an
    old position would say "you entered a zone" for a place the tourist left an hour ago.
  * Zones that belong to one event (a cyclone has several wind bands and time steps) form ONE group
    with one alert lifecycle; the group is judged by its strongest factor.
  * A zone that contributes nothing at this place and time (poor lighting by day) is treated as not
    there, so it neither alerts nor burns a cooldown.
  * If writing an alert fails, the state change is NOT saved, so the next fix retries it.
"""
import logging
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

from .scoring import CLOSING_MAX_M, CLOSING_MIN_M, Sample, assess_point
from .state import transition
from .zones import parse_dt

log = logging.getLogger("geofence")

LIVE_MAX_AGE_S = 120.0
ANCHOR_MIN_AGE_S = 10.0        # "approaching" compares with a fix at least this old, never the previous 5 s one
ANCHOR_MAX_AGE_S = 30.0
UNKNOWN_SPEED_ANCHOR_S = 20.0
MAX_FUTURE_SKEW_S = 300.0
ALERT_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")
ALERT_TYPE = {"approach": "geofence_approach", "entry": "geofence_entry", "exit": "geofence_exit"}
SEVERITY_FOR_LEVEL = {"caution": "info", "warning": "warning", "danger": "critical"}
RANK = {"none": 0, "caution": 1, "warning": 2, "danger": 3}


def group_key(zone):
    return zone.attributes.get("event_ref") or zone.key


class _User:
    def __init__(self):
        self.lock = threading.Lock()
        self.states = {}
        self.loaded = False
        self.hist = []              # [(epoch, Sample)] of the last ANCHOR_MAX_AGE_S seconds
        self.last_t = None


class GeofenceMonitor:
    def __init__(self, db, clock=time.time, max_users=5000):
        self.db, self.clock, self.max_users = db, clock, max_users
        self._users = OrderedDict()
        self._guard = threading.Lock()

    def _user(self, uid):
        with self._guard:
            u = self._users.pop(uid, None) or _User()
            self._users[uid] = u
            while len(self._users) > self.max_users:
                self._users.popitem(last=False)
            return u

    def handle_location(self, rec):
        """rec: the webhook's location record. Returns a small summary dict (used by tests and logs)."""
        uid = rec["user_id"]
        now = self.clock()
        t = parse_dt(rec.get("observed_at") or rec.get("recorded_at") or rec.get("created_at"))
        if t is None:
            return {"skipped": "no_timestamp"}
        age = now - t.timestamp()
        if age > LIVE_MAX_AGE_S:
            return {"skipped": "stale"}
        if age < -MAX_FUTURE_SKEW_S:
            t = datetime.fromtimestamp(now, tz=timezone.utc)      # a wrong phone clock must not freeze us
        u = self._user(uid)
        if not u.lock.acquire(blocking=False):
            return {"skipped": "busy"}                           # a newer fix will follow in seconds
        try:
            ts = t.timestamp()
            if u.last_t is not None and ts <= u.last_t:
                return {"skipped": "out_of_order"}
            u.last_t = ts
            if not u.loaded:
                u.states = self.db.load_states(uid)
                u.loaded = True
            speed = rec.get("speed")
            speed = speed if isinstance(speed, (int, float)) and speed >= 0 else None     # -1 / null = unknown
            sample = Sample(float(rec["lat"]), float(rec["lon"]), t, rec.get("accuracy"), None, speed)
            # Baseline for "am I getting closer?": the NEWEST fix that is old enough for real walking to clearly
            # exceed GPS noise at this speed and accuracy (the previous fix, 5 s ago, mistakes jitter for
            # movement), yet recent enough that a path that turned around is not judged against where it started.
            acc = sample.accuracy_m if sample.accuracy_m is not None else 50.0
            need_m = max(CLOSING_MIN_M, min(2 * acc, CLOSING_MAX_M))
            want_age = (UNKNOWN_SPEED_ANCHOR_S if not speed else
                        min(ANCHOR_MAX_AGE_S, max(ANCHOR_MIN_AGE_S, 1.5 * need_m / speed)))
            prev = next((smp for a, smp in reversed(u.hist) if ts - a >= want_age), None)
            u.hist = [(a, smp) for a, smp in u.hist if ts - a <= ANCHOR_MAX_AGE_S] + [(ts, sample)]
            groups = {}
            for z in self.db.zones_near(sample.lat, sample.lon):
                groups.setdefault(group_key(z), []).append(z)
            out = {"alerts": [], "groups": len(groups)}
            for gk, zs in groups.items():
                try:
                    emitted = self._group(u, uid, rec, gk, zs, sample, prev, now)
                except Exception:  # noqa: BLE001  one bad group must not stop the others
                    log.exception("geofence group failed: %s", gk)
                    continue
                if emitted:
                    out["alerts"].append(emitted)
            # A group that dropped out of the nearby list entirely (user travelled far) counts as "away".
            for gk in [k for k, r in u.states.items() if r["state"] != "clear" and k not in groups]:
                try:
                    emitted = self._group(u, uid, rec, gk, [], sample, prev, now)
                except Exception:  # noqa: BLE001
                    log.exception("geofence clear failed: %s", gk)
                    continue
                if emitted:
                    out["alerts"].append(emitted)
            return out
        finally:
            u.lock.release()

    # ------------------------------------------------------------------
    def _group(self, u, uid, rec, gk, zones, sample, prev, now):
        best = None
        for z in zones:
            a = assess_point(sample, [z], prev=prev)
            if a.factors and a.level != "none" and (best is None or a.score > best.score):
                best = a
        state = best.factors[0].state if best else "outside"
        level = best.level if best else "none"
        row_prev = u.states.get(gk)
        if row_prev is None and state in ("outside", "near"):
            return None                                           # never interacted with: nothing to track

        d = transition(row_prev, state, now)
        row, emit = dict(d.row), d.emit
        prev_alert = (row_prev or {}).get("last_alert_id")
        prev_level = (row_prev or {}).get("last_level")
        if emit is None and row["state"] in ("approaching", "inside") and RANK[level] > RANK.get(prev_level or "none", 0):
            emit = "entry" if row["state"] == "inside" else "approach"     # conditions got worse while there

        row["last_level"] = level if emit else prev_level
        row["last_alert_id"] = prev_alert
        new_alert_id = None
        if emit:
            alert = self._alert(uid, rec, gk, zones, best, emit, sample, now)
            new_alert_id = self.db.insert_alert(alert)             # raises on failure: state stays unsaved
            if emit == "exit":
                row["last_alert_id"] = None
                row["last_level"] = None
            elif new_alert_id is not None:
                row["last_alert_id"] = new_alert_id
        if row["state"] == "clear":
            row["last_alert_id"], row["last_level"] = None, None
        # supersede / close the previous open alert once the new situation is recorded
        if prev_alert and (emit in ("entry", "exit") or (emit == "approach" and new_alert_id) or row["state"] == "clear"):
            if prev_alert != new_alert_id:
                self.db.resolve_alert(prev_alert)

        changed = (row_prev is None or row["state"] != row_prev["state"] or emit is not None)
        if changed:
            self.db.save_state(uid, gk, row)
        u.states[gk] = row
        return {"group": gk, "type": ALERT_TYPE[emit], "level": level, "alert_id": new_alert_id} if emit else None

    def _alert(self, uid, rec, gk, zones, best, emit, sample, now):
        t_iso = sample.t.isoformat()
        base = {
            "user_id": uid, "travel_card_id": rec.get("travel_card_id"), "type": ALERT_TYPE[emit],
            "lat": sample.lat, "lon": sample.lon, "location_accuracy": sample.accuracy_m,
            "location_observed_at": t_iso, "observed_at": t_iso, "source": "system",
            "event_id": str(uuid.uuid5(ALERT_NS, f"{uid}|{gk}|{emit}|{rec.get('id') or int(sample.t.timestamp())}")),
        }
        if emit == "exit":
            name = zones[0].name if zones else "the risk area"
            return {**base, "severity": "info", "status": "resolved",
                    "resolved_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
                    "message": f"You have left: {name}."}
        sev = SEVERITY_FOR_LEVEL[best.level]
        lead = "Heads up: " if emit == "approach" else "You are in a risk area. "
        return {**base, "severity": sev, "status": "open", "message": (lead + best.explanation)[:600]}
