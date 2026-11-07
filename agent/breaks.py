"""Expected stops: a meal, tea or rest halt is not an anomaly.

The mobile app stores the traveller's transport mode and a time plan on the travel card
(`transport_mode`, `eta_plan`). This module turns that into one number: how long a stationary
stretch that STARTED at a given time of day is normally expected to last.

It only ever *removes* the two inactivity findings (`unplanned_stop`, `prolonged_rest`) for stops that
are still within that allowance. It never touches abrupt/collision-grade stops, hazard zones, route
deviation, signal loss or panic, and a stop that runs past the allowance is flagged exactly as before.
Cards without a transport mode (created before this feature) get no allowance, so nothing changes
for them.

Numbers mirror LocationShare/lib/tripEstimate.js (MEALS, REST_BREAK_MIN, STOP_GRACE_FACTOR). The two
live in separate repos, so each side has a test pinning the SAME literal table; change both together.
"""
from .util import local_dt

# (id, window start, window end) in minutes after local midnight, and planned duration in minutes.
MEALS = (
    ("breakfast", 7 * 60 + 30, 10 * 60, 35),
    ("lunch", 12 * 60 + 30, 14 * 60 + 30, 45),
    ("snack", 16 * 60, 17 * 60 + 30, 20),
    ("dinner", 19 * 60 + 30, 21 * 60 + 30, 45),
)
REST_BREAK_MIN = 15
GRACE_FACTOR = 1.5          # a halt may run 50% over plan before it counts as unusual
WINDOW_SLACK_MIN = 30       # people arrive a bit before the usual meal hour
ROAD_MODES = ("car", "bike", "bus", "cycle", "walk")
# Train / flight: long waits at stations and airports are normal.
TRANSIT_ALLOWANCE_MIN = {"train": 120, "flight": 360}


def stop_allowance_min(trip, stop_started_at, stopped_s=0):
    """Return (allowance_minutes, reason). (0, None) means: no special allowance.

    A stop counts as a meal stop if it OVERLAPS a meal window (so one that starts at 11:50 and runs
    into lunch is still a lunch stop). Stops that cross midnight are treated as plain rests."""
    mode = (trip or {}).get("transport_mode")
    if not mode:
        return 0, None
    if mode in TRANSIT_ALLOWANCE_MIN:
        return TRANSIT_ALLOWANCE_MIN[mode], f"{mode} journey: long waits at stations/airports are normal"
    if mode not in ROAD_MODES:
        return 0, None

    d = local_dt(stop_started_at)
    start = d.hour * 60 + d.minute
    end = start + (stopped_s or 0) / 60.0
    for name, lo, hi, dur in MEALS:
        if start <= hi + WINDOW_SLACK_MIN and end >= lo - WINDOW_SLACK_MIN:
            return round(dur * GRACE_FACTOR), f"{name} time"
    return round(REST_BREAK_MIN * 2), "rest / fuel stop"


def within_expected_break(trip, stop_started_at, stopped_s):
    """True when a stationary stretch of `stopped_s` seconds that began at `stop_started_at`
    is still inside what the plan expects. Returns (bool, allowance_min, reason)."""
    allow, why = stop_allowance_min(trip, stop_started_at, stopped_s)
    return bool(allow) and stopped_s <= allow * 60, allow, why
