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
from typing import Optional, Union

import httpx
import pymupdf
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import process_ping, initial_state
from state_store import InMemoryStateStore, SupabaseStateStore

app = FastAPI(title="Margarakshak Safety & Routing Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once, at import time, when the Render process boots. Every request
# afterward reuses this same object; nothing here is retrained per request.
def _load_or_train():
    """Load model.pkl/durations.pkl; retrain in-process if they are missing,
    unreadable (numpy/hmmlearn version drift on the host), or were trained
    under a different feature definition than the code now computes."""
    from features import FEATURE_VERSION
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("durations.pkl", "rb") as f:
            fitted = pickle.load(f)
        if getattr(model, "feature_version", None) == FEATURE_VERSION:
            return model, fitted
        print("model.pkl feature_version mismatch; retraining in-process")
    except Exception as exc:  # noqa: BLE001
        print(f"could not load pickles ({exc!r}); retraining in-process")
    from train_and_save import train_models
    return train_models()


MODEL, FITTED_DURATIONS = _load_or_train()

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
    id: Optional[Union[int, str]] = None
    user_id: str
    travel_card_id: Optional[str] = None
    lat: float
    lon: float
    accuracy: Optional[float] = None
    speed: Optional[float] = None
    created_at: str
    # The row's real observation time. created_at in the webhook is the server's now(), which is the
    # wrong clock for deciding whether a fix is live (an offline backlog arrives all at once).
    observed_at: Optional[str] = None
    recorded_at: Optional[str] = None


class SupabaseWebhookPayload(BaseModel):
    type: str
    table: str
    record: LocationRecord
    schema_name: str = Field(alias="schema")
    old_record: Optional[dict] = None

    class Config:
        populate_by_name = True


# -----------------------------------------------------------------------
# Incident agent (see agent/). Deterministic signal fusion decides; the LLM only
# words briefs. Both endpoints use the same shared-secret check as /score.
#   POST /agent/tick   called every minute by pg_cron
#   POST /agent/panic  called by an AFTER INSERT trigger on alerts (type = panic).
#                      The existing alert/SMS flow is untouched and never waits on this.
# -----------------------------------------------------------------------
import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor

from agent.engine import Engine
from agent.llm import LLMClient
from agent.services import find_nearby_services
from agent.store import AgentStore

_AGENT_POOL = ThreadPoolExecutor(max_workers=2)
_AGENT = None
_AGENT_GUARD = threading.Lock()
_TICK_LOCK = threading.Lock()


def _agent_submit(fn, *args, **kwargs):
    def run():
        try:
            fn(*args, **kwargs)
        except Exception:  # noqa: BLE001
            logging.getLogger("agent").exception("background agent task failed")
    _AGENT_POOL.submit(run)


def _agent():
    global _AGENT
    with _AGENT_GUARD:
        if _AGENT is None:
            if not os.environ.get("SUPABASE_URL"):
                raise HTTPException(status_code=503, detail="agent requires SUPABASE_URL")
            _AGENT = Engine(AgentStore(), LLMClient(), find_nearby_services, _time.time, submit=_agent_submit)
        return _AGENT


@app.post("/agent/tick")
def agent_tick(x_webhook_secret: Optional[str] = Header(None)):
    _check_webhook_secret(x_webhook_secret)
    if not _TICK_LOCK.acquire(blocking=False):
        return {"skipped": "tick_in_progress"}
    try:
        return _agent().tick()
    finally:
        _TICK_LOCK.release()


@app.post("/agent/panic")
def agent_panic(payload: dict, x_webhook_secret: Optional[str] = Header(None)):
    _check_webhook_secret(x_webhook_secret)
    alert_id = payload.get("alert_id") or (payload.get("record") or {}).get("id")
    if not alert_id:
        raise HTTPException(status_code=422, detail="alert_id required")
    return _agent().handle_panic(str(alert_id))


# -----------------------------------------------------------------------
# Dynamic geo-fencing (see geofence/). Hazard zones live in Supabase risk_zones.
#   POST /geofence/sync/gdacs   called hourly by pg_cron; pulls current GDACS disaster events.
# A failed fetch raises before anything is written, so live hazards are never retired by an outage.
# -----------------------------------------------------------------------
from geofence.store import RiskZoneStore
from geofence.sync import sync_gdacs

_GDACS_LOCK = threading.Lock()

# Live geofence alerts: one location in, approach / entry / exit alerts out. Off unless GEOFENCE_ALERTS=on,
# so deploying the code changes nothing until you opt in. Runs on a small thread pool; every failure is
# logged and swallowed because HSMM scoring and SOS must never depend on it.
from geofence.monitor import GeofenceMonitor
from geofence.store import SupabaseGeofenceDb

_GEOFENCE_POOL = ThreadPoolExecutor(max_workers=3)
_GEOFENCE = None
_GEOFENCE_GUARD = threading.Lock()


def _geofence_enabled():
    return os.environ.get("GEOFENCE_ALERTS", "").lower() in ("1", "true", "on", "yes") and bool(os.environ.get("SUPABASE_URL"))


def _geofence_monitor():
    global _GEOFENCE
    with _GEOFENCE_GUARD:
        if _GEOFENCE is None:
            _GEOFENCE = GeofenceMonitor(SupabaseGeofenceDb())
        return _GEOFENCE


def _geofence_run(rec):
    try:
        _geofence_monitor().handle_location(rec)
    except Exception:  # noqa: BLE001
        logging.getLogger("geofence").exception("geofence evaluation failed")


def _geofence_submit(rec):
    try:
        if _geofence_enabled():
            _GEOFENCE_POOL.submit(_geofence_run, rec)
    except Exception:  # noqa: BLE001
        logging.getLogger("geofence").exception("geofence submit failed")


@app.post("/geofence/sync/gdacs")
def geofence_sync_gdacs(x_webhook_secret: Optional[str] = Header(None)):
    _check_webhook_secret(x_webhook_secret)
    if not os.environ.get("SUPABASE_URL"):
        raise HTTPException(status_code=503, detail="geofence sync requires SUPABASE_URL")
    if not _GDACS_LOCK.acquire(blocking=False):
        return {"skipped": "sync_in_progress"}
    try:
        return sync_gdacs(RiskZoneStore())
    except Exception as e:  # noqa: BLE001
        logging.getLogger("geofence").exception("gdacs sync failed")
        raise HTTPException(status_code=502, detail=f"GDACS sync failed: {type(e).__name__}")
    finally:
        _GDACS_LOCK.release()


@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------------------------------------------------
# Route determination, deviation analysis, and tourist safety scoring
# -----------------------------------------------------------------------
from geofence.route import plan_and_assess_route
from geofence.deviation import calculate_cross_track

class WaypointItem(BaseModel):
    name: Optional[str] = "Waypoint"
    lat: float
    lon: float

class RoutePlanRequest(BaseModel):
    waypoints: list[WaypointItem]
    travel_mode: Optional[str] = "driving"

class DeviationCheckRequest(BaseModel):
    lat: float
    lon: float
    polyline: list[list[float]]
    trip_id: Optional[str] = None
    user_id: Optional[str] = None

class SafetyScoreRequest(BaseModel):
    lat: float
    lon: float
    activity_state: Optional[str] = None
    is_night: Optional[bool] = False
    deviation_m: Optional[float] = 0.0
    isolation_score: Optional[float] = 0.0
    nearest_police_km: Optional[float] = None
    nearest_hospital_km: Optional[float] = None


@app.post("/route/plan")
def route_plan(payload: RoutePlanRequest):
    """Calculates road driving route between start, intermediate stops, and destination,
    and returns waypoint milestones, turn-by-turn road polyline, and risk assessment."""
    wps = [wp.model_dump() for wp in payload.waypoints]
    if len(wps) < 2:
        raise HTTPException(status_code=400, detail="At least 2 waypoints are required")
    # Fetch nearby zones if store is configured
    zones = None
    if isinstance(STORE, SupabaseStateStore) and hasattr(_geofence_monitor(), "db"):
        try:
            zones = _geofence_monitor().db.zones_near(wps[0]["lat"], wps[0]["lon"])
        except Exception:
            zones = None
    result = plan_and_assess_route(wps, zones=zones, travel_mode=payload.travel_mode or "driving")
    if not result.get("success"):
        raise HTTPException(status_code=422, detail=result.get("error", "Route planning failed"))
    return result


@app.post("/route/deviation")
def route_deviation(payload: DeviationCheckRequest):
    """Calculates cross-track distance from ping to polyline and evaluates deviation status."""
    if not payload.polyline:
        raise HTTPException(status_code=400, detail="Polyline required")
    xt = calculate_cross_track(payload.lat, payload.lon, payload.polyline)
    dev_m = xt["distance_m"]
    status = "on_route" if dev_m <= 500.0 else "caution" if dev_m <= 1500.0 else "deviated"
    return {
        "status": status,
        "deviation_distance_m": dev_m,
        "closest_point": xt["closest_point"],
        "progress_pct": round(xt["progress_ratio"] * 100.0, 1),
    }


@app.post("/safety/tourist-score")
def tourist_safety_score(payload: SafetyScoreRequest):
    """Calculates the transparent 0-100 Tourist Safety Score."""
    score = 100.0
    penalties = []

    dev_m = payload.deviation_m or 0.0
    if dev_m > 1800.0:
        p = min(35.0, 15.0 + (dev_m - 1800.0) / 200.0)
        score -= p
        penalties.append(f"-{round(p)}: Route deviation ({round(dev_m / 1000.0, 1)} km off planned route)")
    elif dev_m > 500.0:
        score -= 10.0
        penalties.append(f"-10: Detour from corridor ({round(dev_m)} m)")

    if payload.activity_state == "ERRATIC":
        score -= 25.0
        penalties.append("-25: Erratic movement pattern detected")
    elif payload.activity_state == "RESTING" and payload.is_night:
        score -= 10.0
        penalties.append("-10: Stationary during night hours")

    iso = payload.isolation_score or 0.0
    if iso > 0.6:
        p = round(iso * 20.0)
        score -= p
        penalties.append(f"-{p}: Remote / isolated area")

    if payload.nearest_police_km is not None and payload.nearest_police_km > 15.0:
        score -= 10.0
        penalties.append(f"-10: Distant emergency services ({round(payload.nearest_police_km, 1)} km to police)")

    if payload.is_night and (iso > 0.4 or dev_m > 1000.0):
        score -= 10.0
        penalties.append("-10: Nocturnal travel risk in unfamiliar sector")

    final_score = max(5.0, min(100.0, round(score, 1)))
    status = "safe" if final_score >= 80.0 else "caution" if final_score >= 50.0 else "unsafe"

    return {
        "safety_score": final_score,
        "status": status,
        "risk_score": round(100.0 - final_score, 1),
        "penalties": penalties,
    }


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

    _geofence_submit(payload.record.model_dump())      # best effort, never blocks or breaks scoring

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

            # Webhook delivery can be concurrent or out of order. Skip a ping only
            # if it is provably older: its id is not newer AND its timestamp is
            # not newer. Requiring both matters: if the locations table is ever
            # truncated/reset (ids restart at 1) the id check alone would
            # silently freeze this trip's activity state forever.
            last_id = state.get("last_location_id")
            last_t = state.get("last_ping_t")
            id_not_newer = (isinstance(location_id, int) and isinstance(last_id, int)
                            and location_id <= last_id)
            t_not_newer = last_t is not None and t <= last_t
            if (id_not_newer and t_not_newer) or (location_id is None and t_not_newer):
                return {"trip_id": trip_id, "skipped": True, "reason": "duplicate_or_out_of_order", "location_id": location_id}

            new_state, result = process_ping(
                state,
                {"t": t, "lat": payload.record.lat, "lon": payload.record.lon,
                 "speed": payload.record.speed, "accuracy": payload.record.accuracy},
                MODEL,
                FITTED_DURATIONS,
            )
            new_state["last_ping_t"] = t
            if isinstance(location_id, int):
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
