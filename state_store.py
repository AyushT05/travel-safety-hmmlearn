"""
state_store.py

Two implementations of the same tiny interface: load_state(trip_id) and
save_state(trip_id, state). pipeline.process_ping() doesn't know or care
which one is behind it.

InMemoryStateStore  - a plain dict, only useful for local testing. State
                      disappears the moment the process restarts, which is
                      exactly what happens on Render's free tier after
                      ~15 minutes of no requests. Never use this in
                      production for that reason.

SupabaseStateStore  - reads and writes a row per trip in a
                      `trip_activity_state` table via Supabase's REST API
                      (PostgREST), using the service-role key. This is what
                      makes the service safe to run on a tier that sleeps:
                      the process itself holds no per-trip memory, only the
                      trained model, which is read-only.
"""

import json
import os

import httpx


class InMemoryStateStore:
    def __init__(self):
        self._store = {}

    def load_state(self, trip_id):
        return self._store.get(trip_id)

    def save_state(self, trip_id, state):
        self._store[trip_id] = state


class SupabaseStateStore:
    """
    Expects a table created with:

        create table trip_activity_state (
          trip_id text primary key,
          state jsonb not null,
          current_state text,
          duration_minutes numeric,
          tail_probability numeric,
          llr numeric,
          updated_at timestamptz default now()
        );

    The whole pipeline state dict is stored as one jsonb blob (`state`);
    the four extra columns duplicate the LATEST score fields out of that
    blob purely so the dashboard can query/subscribe to them directly
    without having to unpack jsonb on the frontend.
    """

    def __init__(self, supabase_url=None, service_key=None):
        self.base_url = (supabase_url or os.environ["SUPABASE_URL"]).rstrip("/")
        self.service_key = service_key or os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        self.headers = {
            "apikey": self.service_key,
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json",
        }

    def load_state(self, trip_id):
        url = f"{self.base_url}/rest/v1/trip_activity_state"
        params = {"trip_id": f"eq.{trip_id}", "select": "state"}
        resp = httpx.get(url, headers=self.headers, params=params, timeout=10)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return None
        return rows[0]["state"]

    def save_state(self, trip_id, state, score=None):
        url = f"{self.base_url}/rest/v1/trip_activity_state"
        payload = {
            "trip_id": trip_id,
            "state": state,
        }
        if score:
            payload["current_state"] = score["current_state"]
            payload["duration_minutes"] = score["duration_minutes"]
            payload["tail_probability"] = score["tail_probability"]
            payload["llr"] = score["llr"]

        headers = dict(self.headers)
        # upsert: insert if trip_id is new, update if it already exists
        headers["Prefer"] = "resolution=merge-duplicates"
        resp = httpx.post(url, headers=headers, json=[payload], timeout=10)
        resp.raise_for_status()
