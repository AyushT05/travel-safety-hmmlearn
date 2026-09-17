"""
app.py

This is the file Render actually runs. Everything else in this directory
(simulator, features, hsmm_model, duration_fit, inference, fusion,
pipeline, train_and_save, state_store) is either training-time code or a
pure function this file calls. app.py's only jobs are:

  1. Load the trained model ONCE when the process starts (not per request).
  2. Expose one HTTP endpoint that Supabase can call every time a new GPS
     ping is inserted.
  3. Load this trip's state, run one step of the pipeline, save the state
     back, return the score.

Run locally with:
    uvicorn app:app --reload --port 8000

On Render, the start command is:
    uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import os
import pickle

from fastapi import FastAPI
from pydantic import BaseModel

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


class Ping(BaseModel):
    trip_id: str
    t: float       # epoch seconds
    lat: float
    lon: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(ping: Ping):
    state = STORE.load_state(ping.trip_id)
    if state is None:
        state = initial_state()

    new_state, result = process_ping(
        state, {"t": ping.t, "lat": ping.lat, "lon": ping.lon}, MODEL, FITTED_DURATIONS
    )

    if isinstance(STORE, SupabaseStateStore):
        STORE.save_state(ping.trip_id, new_state, score=result)
    else:
        STORE.save_state(ping.trip_id, new_state)

    return {
        "trip_id": ping.trip_id,
        "buffering": result is None,
        "score": result,
    }
