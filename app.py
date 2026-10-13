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
      "record": { "id": ..., "user_id": ..., "travel_card_id": ...,
                   "lat": ..., "lon": ..., "accuracy": ..., "speed": ...,
                   "created_at": "..." },
      "old_record": null
    }

So the adaptation happens here, not in Supabase's config:
  - trip_id in this file's state store is `travel_card_id` when present,
    falling back to `user_id` only for older rows that predate the
    add_travel_card_id_to_locations.sql migration. travel_card_id is the
    correct key: it changes every time the person starts a genuinely new
    trip, so a new trip automatically gets a fresh state_store row instead
    of inheriting run-duration/state from whatever trip they were on last.
    Keying on user_id alone (the original design, before that migration
    existed) meant two completely unrelated trips from the same account
    would silently share one continuous activity-duration counter, which
    is exactly the bug this fallback chain now avoids for anyone updated
    to send travel_card_id, while not breaking anyone still on old data.
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
import base64
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import pymupdf
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

# Local-only fallback lock. Production uses PostgreSQL-backed locks so requests
# for the same trip serialize even across multiple Render workers/instances.
_LOCAL_LOCKS = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


@contextmanager
def _trip_lock(trip_id):
    if isinstance(STORE, SupabaseStateStore):
        token = STORE.acquire_trip_lock(trip_id)
        try:
            yield
        finally:
            STORE.release_trip_lock(trip_id, token)
    else:
        with _LOCAL_LOCKS_GUARD:
            lock = _LOCAL_LOCKS.setdefault(trip_id, threading.Lock())
        acquired = lock.acquire(timeout=25)
        if not acquired:
            raise TimeoutError(f"Timed out waiting for local scoring lock for trip {trip_id}")
        try:
            yield
        finally:
            lock.release()


def _check_webhook_secret(provided):
    # Fail closed in production: a public scoring endpoint must never silently
    # disable authentication because WEBHOOK_SECRET was omitted.
    if not WEBHOOK_SECRET:
        if os.environ.get("RENDER") or os.environ.get("SUPABASE_URL"):
            raise HTTPException(status_code=503, detail="WEBHOOK_SECRET is not configured")
        return
    if provided != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="invalid webhook secret")


def _to_epoch_seconds(iso_timestamp: str) -> float:
    # Supabase sends e.g. "2026-09-17T05:12:03.482+00:00"
    dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


class LocationRecord(BaseModel):
    id: Optional[int] = None
    user_id: str
    travel_card_id: Optional[str] = None
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


# -----------------------------------------------------------------------
# PDF-to-image rendering for document verification. Unrelated to the HSMM
# scoring; co-located only to avoid a second Render service. The
# verify-document edge function calls this for PDF uploads because Groq's
# vision endpoint accepts images only. Page 1 only (an ID is one page).
# Same shared-secret header check as /score.
# -----------------------------------------------------------------------

class PdfRenderRequest(BaseModel):
    pdfBase64: str
    max_dimension: int = 1600
    quality: int = 80


@app.post("/pdf-to-image")
def pdf_to_image(payload: PdfRenderRequest, x_webhook_secret: Optional[str] = Header(None)):
    _check_webhook_secret(x_webhook_secret)
    try:
        doc = pymupdf.open(stream=base64.b64decode(payload.pdfBase64), filetype="pdf")
        if doc.page_count == 0:
            raise HTTPException(status_code=400, detail="PDF has no pages")
        page = doc[0]
        scale = min(payload.max_dimension / max(page.rect.width, page.rect.height), 1.0)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale))
        jpeg = pix.tobytes("jpeg", jpg_quality=payload.quality)
        doc.close()
        return {"imageBase64": base64.b64encode(jpeg).decode("ascii"), "mimeType": "image/jpeg"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not render PDF: {e}")


@app.post("/score")
def score(payload: SupabaseWebhookPayload, x_webhook_secret: Optional[str] = Header(None)):
    _check_webhook_secret(x_webhook_secret)

    if payload.type != "INSERT" or payload.table != "locations":
        return {"skipped": True, "reason": "unsupported webhook event"}

    trip_id = payload.record.travel_card_id or payload.record.user_id
    t = _to_epoch_seconds(payload.record.created_at)
    location_id = payload.record.id

    # The lock covers load -> infer -> save as one critical section. This is a
    # distributed PostgreSQL lease in production, not merely a process mutex.
    try:
        with _trip_lock(trip_id):
            state = STORE.load_state(trip_id)
            if state is None:
                state = initial_state()

            # Webhook delivery can be concurrent or out of order. Location IDs
            # are monotonically increasing, so ignore duplicate/older events.
            last_id = state.get("last_location_id")
            last_t = state.get("last_ping_t")
            if location_id is not None and last_id is not None and location_id <= last_id:
                return {"trip_id": trip_id, "skipped": True, "reason": "duplicate_or_out_of_order", "location_id": location_id}
            if location_id is None and last_t is not None and t <= last_t:
                return {"trip_id": trip_id, "skipped": True, "reason": "duplicate_or_out_of_order"}

            new_state, result = process_ping(
                state,
                {"t": t, "lat": payload.record.lat, "lon": payload.record.lon},
                MODEL,
                FITTED_DURATIONS,
            )
            new_state["last_ping_t"] = t
            if location_id is not None:
                new_state["last_location_id"] = location_id

            if isinstance(STORE, SupabaseStateStore):
                STORE.save_state(trip_id, new_state, score=result)
            else:
                STORE.save_state(trip_id, new_state)

            return {
                "trip_id": trip_id,
                "location_id": location_id,
                "buffering": result is None,
                "score": result,
            }
    except TimeoutError as exc:
        # A non-2xx response lets the caller/logging identify contention rather
        # than silently processing against stale state.
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/check-dropouts")
def check_dropouts(x_webhook_secret: Optional[str] = Header(None)):
    """Run one dropout check; Supabase pg_cron invokes this periodically.

    The database migration owns the actual stale-trip query and idempotency.
    This endpoint is intentionally thin so the SQL function is also callable
    directly for manual testing.
    """
    _check_webhook_secret(x_webhook_secret)
    if not isinstance(STORE, SupabaseStateStore):
        raise HTTPException(status_code=503, detail="Supabase state store is not configured")
    try:
        result = STORE._rpc("check_location_dropouts", {})
        return {"ok": True, "result": result}
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Dropout check failed") from exc
