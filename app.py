"""
app.py

This is the file Render actually runs. Everything else in this directory
(simulator, features, hsmm_model, duration_fit, inference, fusion,
pipeline, train_and_save, state_store) is either training-time code or a
pure function this file calls. app.py's only jobs are:

  1. Load the trained model ONCE when the process starts (not per request).
  2. Expose one HTTP endpoint that a Supabase Database Webhook calls every
     time a new row is inserted into `locations`.
  3. Load this trip's state, run one step of the pipeline, save the state
     back, return the score.

IMPORTANT: this endpoint receives Supabase's Database Webhook payload
shape directly, NOT a hand-picked {trip_id, t, lat, lon} body. Supabase's
webhook UI does not let you remap field names or restructure the JSON, it
always sends:

    {
      "type": "INSERT",
      "table": "locations",
      "schema": "public",
      "record": { "id": ..., "user_id": ..., "lat": ..., "lon": ...,
                   "accuracy": ..., "speed": ..., "created_at": "..." },
      "old_record": null
    }

So the adaptation happens here, not in Supabase's config:
  - trip_id in this file's state store is actually `user_id`. The
    `locations` table has no travel_card/trip id column at all, only
    `user_id`, and the app only supports one active trip per user at a
    time (see ActiveTracking.js), so user_id is the correct, honest key
    to track state under, not a workaround.
  - `created_at` arrives as an ISO 8601 string and is converted to epoch
    seconds before being handed to pipeline.process_ping(), which only
    ever deals in plain numeric seconds.

A shared secret header is checked on every request, since this URL is
public on the internet the moment it's deployed to Render, and Supabase's
webhook UI lets you attach a custom header for exactly this purpose.

Run locally with:
    uvicorn app:app --reload --port 8000

On Render, the start command is:
    uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import os
import pickle
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

from pipeline import process_ping, initial_state
from state_store import InMemoryStateStore, SupabaseStateStore

app = FastAPI()

# Loaded once, at import time, when the Render process boots. Every request
# afterward reuses this same object; nothing here is retrained per request.
with open("model.pkl", "rb") as f:
    MODEL = pickle.load(f)
with open("durations.pkl", "rb") as f:
    FITTED_DURATIONS = pickle.load(f)

# Falls back to in-memory storage if Supabase env vars aren't set, so this
# is runnable locally without any credentials. Production MUST set
# SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY, or state silently resets on
# every restart (see state_store.py's docstring for exactly why).
if os.environ.get("SUPABASE_URL"):
    STORE = SupabaseStateStore()
else:
    STORE = InMemoryStateStore()

# Shared secret Supabase's webhook config sends back on every call. If this
# env var isn't set, the check is skipped entirely (fine for local testing,
# NOT fine once this is deployed and public).
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")


def _to_epoch_seconds(iso_timestamp: str) -> float:
    # Supabase sends e.g. "2026-09-17T05:12:03.482+00:00"
    dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


class LocationRecord(BaseModel):
    user_id: str
    lat: float
    lon: float
    accuracy: Optional[float] = None
    speed: Optional[float] = None
    created_at: str


class SupabaseWebhookPayload(BaseModel):
    type: str
    table: str
    record: LocationRecord
    schema_name: str = Field(alias="schema")
    old_record: Optional[dict] = None

    class Config:
        populate_by_name = True


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(payload: SupabaseWebhookPayload, x_webhook_secret: Optional[str] = Header(None)):
    if WEBHOOK_SECRET and x_webhook_secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="invalid webhook secret")

    if payload.type != "INSERT" or payload.table != "locations":
        # defensive: the webhook is configured to only fire on INSERT into
        # locations, but don't trust configuration alone to guarantee that
        return {"skipped": True}

    trip_id = payload.record.user_id
    t = _to_epoch_seconds(payload.record.created_at)

    state = STORE.load_state(trip_id)
    if state is None:
        state = initial_state()

    new_state, result = process_ping(
        state,
        {"t": t, "lat": payload.record.lat, "lon": payload.record.lon},
        MODEL,
        FITTED_DURATIONS,
    )

    if isinstance(STORE, SupabaseStateStore):
        STORE.save_state(trip_id, new_state, score=result)
    else:
        STORE.save_state(trip_id, new_state)

    return {
        "trip_id": trip_id,
        "buffering": result is None,
        "score": result,
    }
