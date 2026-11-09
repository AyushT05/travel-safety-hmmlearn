"""Supabase REST (PostgREST) access for the agent, using the service-role key."""
import os
from datetime import datetime, timezone

import httpx

from . import config
from .util import iso, parse_ts


class AgentStore:
    def __init__(self, supabase_url=None, service_key=None):
        self.base = (supabase_url or os.environ["SUPABASE_URL"]).rstrip("/") + "/rest/v1"
        key = service_key or os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        self.h = {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.http = httpx.Client(timeout=15)

    def _get(self, table, params):
        r = self.http.get(f"{self.base}/{table}", headers=self.h, params=params)
        r.raise_for_status()
        return r.json()

    # ---- reads
    def active_trips(self, now):
        # Trip dates are India dates (the database compares in Asia/Kolkata); UTC made a trip look "not started" until 05:30.
        today = datetime.fromtimestamp(now + 19800, tz=timezone.utc).date().isoformat()
        params = {"status": "eq.active", "start_date": f"lte.{today}", "end_date": f"gte.{today}"}
        base = "id,user_id,full_name,mobile,places,place_details,planned_route,emergency_contacts,companions,start_date,end_date"
        try:
            return self._get("travel_cards", {**params, "select": base + ",transport_mode,departure_time,eta_plan,return_date"})
        except httpx.HTTPStatusError:
            # Migration 20261106 not applied yet: keep working without the transport columns.
            return self._get("travel_cards", {**params, "select": base})

    def active_trip_for_user(self, user_id, now):
        rows = [t for t in self.active_trips(now) if t["user_id"] == user_id]
        return rows[0] if rows else None

    def snapshot(self, trip, now):
        if trip.get("id"):
            locs = self._get("locations", {
                "travel_card_id": f"eq.{trip['id']}", "order": "recorded_at.desc",
                "limit": config.SNAPSHOT_LOCATION_ROWS, "select": "id,lat,lon,accuracy,speed,recorded_at"})
            act = self._get("trip_activity_state", {
                "trip_id": f"eq.{trip['id']}",
                "select": "current_state,duration_minutes,tail_probability,updated_at"})
        else:
            locs = self._get("locations", {
                "user_id": f"eq.{trip['user_id']}", "order": "recorded_at.desc",
                "limit": config.SNAPSHOT_LOCATION_ROWS, "select": "id,lat,lon,accuracy,speed,recorded_at"})
            act = []
        alerts = self._get("alerts", {
            "user_id": f"eq.{trip['user_id']}", "created_at": f"gte.{iso(now - 6 * 3600)}",
            "select": "id,type,severity,status,created_at,lat,lon,message,source"})
        assess = self._get("safety_assessments", {"user_id": f"eq.{trip['user_id']}", "select": "isolation_score,safety_status,assessed_at,hazard_zones,risk_factors"})
        return {"trip": trip, "locations": locs, "activity": act[0] if act else None,
                "alerts": alerts, "assessment": assess[0] if assess else None}

    def zones(self, now):
        rows = self._get("unsafe_zones", {"status": "eq.verified", "select": "id,category,severity,lat,lon,radius_m,expires_at"})
        return [z for z in rows if not z.get("expires_at") or parse_ts(z["expires_at"]) > now]

    def open_panic_alerts(self, now):
        return self._get("alerts", {"type": "eq.panic", "status": "in.(open,acknowledged)",
                                    "created_at": f"gte.{iso(now - 24 * 3600)}",
                                    "select": "id,user_id,type,severity,status,created_at,lat,lon,message"})

    def get_alert(self, alert_id):
        cols = "id,user_id,type,severity,status,created_at,lat,lon,message,source,travel_card_id,location_accuracy"
        try:
            rows = self._get("alerts", {"id": f"eq.{alert_id}", "select": cols})
        except httpx.HTTPStatusError:      # an older schema without some columns: retry with the originals
            rows = self._get("alerts", {"id": f"eq.{alert_id}", "select": "id,user_id,type,severity,status,created_at,lat,lon,message"})
        return rows[0] if rows else None

    # ---- incidents
    def find_incident(self, dedupe_key):
        rows = self._get("incidents", {"dedupe_key": f"eq.{dedupe_key}", "order": "opened_at.desc", "limit": 1, "select": "*"})
        return rows[0] if rows else None

    def open_incident(self, dedupe_key):
        rows = self._get("incidents", {"dedupe_key": f"eq.{dedupe_key}", "status": "in.(open,acknowledged)", "limit": 1, "select": "*"})
        return rows[0] if rows else None

    def open_incidents(self, kind):
        return self._get("incidents", {"kind": f"eq.{kind}", "status": "in.(open,acknowledged)", "select": "*"})

    def last_dismissed(self, dedupe_key, since_epoch):
        rows = self._get("incidents", {"dedupe_key": f"eq.{dedupe_key}", "status": "eq.dismissed",
                                       "resolved_at": f"gte.{iso(since_epoch)}", "order": "resolved_at.desc",
                                       "limit": 1, "select": "id,level"})
        return rows[0] if rows else None

    def insert_incident(self, row):
        r = self.http.post(f"{self.base}/incidents", headers={**self.h, "Prefer": "return=representation"}, json=row)
        if r.status_code == 409:       # lost a race on the partial unique index: someone else opened it
            return self.open_incident(row["dedupe_key"])
        r.raise_for_status()
        return r.json()[0]

    def update_incident(self, incident_id, patch, where=None):
        params = {"id": f"eq.{incident_id}", **{k: f"eq.{v}" for k, v in (where or {}).items()}}
        r = self.http.patch(f"{self.base}/incidents", headers={**self.h, "Prefer": "return=representation"},
                            params=params, json=patch)
        r.raise_for_status()
        rows = r.json()
        return rows[0] if rows else None

    # ---- AI trip insights + traveller notifications
    def insights(self):
        return self._get("trip_insights", {"select": "*"})

    def get_insight(self, trip_id):
        rows = self._get("trip_insights", {"travel_card_id": f"eq.{trip_id}", "select": "*"})
        return rows[0] if rows else None

    def upsert_insight(self, row):
        r = self.http.post(f"{self.base}/trip_insights", params={"on_conflict": "travel_card_id"},
                           headers={**self.h, "Prefer": "resolution=merge-duplicates,return=minimal"}, json=row)
        r.raise_for_status()

    def patch_insight(self, trip_id, patch, where=None):
        params = {"travel_card_id": f"eq.{trip_id}", **{k: f"eq.{v}" for k, v in (where or {}).items()}}
        r = self.http.patch(f"{self.base}/trip_insights", headers={**self.h, "Prefer": "return=minimal"},
                            params=params, json=patch)
        r.raise_for_status()

    def notification_responses(self, trip_id, since_epoch):
        return self._get("user_notifications", {
            "travel_card_id": f"eq.{trip_id}", "response": "not.is.null", "responded_at": f"gte.{iso(since_epoch)}",
            "order": "responded_at.desc", "select": "id,kind,response,responded_at"})

    def recent_notifications(self, trip_id, since_epoch):
        return self._get("user_notifications", {
            "travel_card_id": f"eq.{trip_id}", "created_at": f"gte.{iso(since_epoch)}",
            "order": "created_at.desc", "select": "id,kind,created_at,response,responded_at"})

    def insert_notification(self, row):
        r = self.http.post(f"{self.base}/user_notifications", headers={**self.h, "Prefer": "return=minimal"}, json=row)
        if r.status_code == 409:
            return
        r.raise_for_status()
