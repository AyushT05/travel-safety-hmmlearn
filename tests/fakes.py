import copy
import itertools

from agent.llm import LLMUnavailable
from agent.util import parse_ts

T0 = parse_ts("2026-10-17T07:00:00Z")          # 12:30 local (day)
NIGHT = parse_ts("2026-10-17T18:30:00Z")        # 00:00 local next day
ZONE = {"id": 7, "category": "robbery", "severity": "warning", "lat": 12.3000, "lon": 76.6400, "radius_m": 500}
FAR = (12.2800, 76.6000)                        # well outside the zone
IN = (12.3001, 76.6401)


def fixes(now, from_s, to_s, pos, step=5, acc=8.0, speed=0.0, id0=1):
    """Fixes from (now - from_s) to (now - to_s)."""
    out, t, i = [], now - from_s, id0
    while t <= now - to_s:
        out.append({"id": i, "lat": pos[0], "lon": pos[1], "accuracy": acc, "speed": speed,
                    "recorded_at": _iso(t)})
        t += step
        i += 1
    return out


def _iso(t):
    from agent.util import iso
    return iso(t)


class FakeStore:
    def __init__(self):
        self.trips, self.snaps, self.zones_, self.alerts, self.incidents = [], {}, [], {}, []
        self._ids = itertools.count(1)

    def active_trips(self, now):
        return list(self.trips)

    def active_trip_for_user(self, uid, now):
        return next((t for t in self.trips if t["user_id"] == uid), None)

    def snapshot(self, trip, now):
        key = trip["id"] or ("user", trip["user_id"])
        s = self.snaps[key]
        return copy.deepcopy(s(now) if callable(s) else s)

    def zones(self, now):
        return self.zones_

    def open_panic_alerts(self, now):
        return [a for a in self.alerts.values() if a["type"] == "panic" and a["status"] in ("open", "acknowledged")]

    def get_alert(self, i):
        return self.alerts.get(i)

    def find_incident(self, key):
        rows = [i for i in self.incidents if i["dedupe_key"] == key]
        return rows[-1] if rows else None

    def open_incident(self, key):
        return next((i for i in self.incidents if i["dedupe_key"] == key and i["status"] in ("open", "acknowledged")), None)

    def open_incidents(self, kind):
        return [i for i in self.incidents if i["kind"] == kind and i["status"] in ("open", "acknowledged")]

    def last_dismissed(self, key, since):
        rows = [i for i in self.incidents if i["dedupe_key"] == key and i["status"] == "dismissed"
                and parse_ts(i["resolved_at"]) >= since]
        return rows[-1] if rows else None

    def insert_incident(self, row):
        if self.open_incident(row["dedupe_key"]):
            return self.open_incident(row["dedupe_key"])
        row = {"id": f"inc{next(self._ids)}", "status": "open", "brief": None, "brief_source": None,
               "brief_generated_at": None, "brief_error": None, "clear_streak": 0, **row}
        self.incidents.append(row)
        return row

    def update_incident(self, iid, patch, where=None):
        row = next(i for i in self.incidents if i["id"] == iid)
        if where and any(row.get(k) != v for k, v in where.items()):
            return None
        row.update(patch)
        return row


class FakeLLM:
    def __init__(self, response=None, fail=False, enabled=True):
        self.response, self.fail, self.enabled, self.calls, self.last_user = response, fail, enabled, 0, None

    def generate_json(self, system, user):
        self.calls += 1
        self.last_user = user
        if self.fail:
            raise LLMUnavailable("down")
        return self.response(user) if callable(self.response) else self.response
