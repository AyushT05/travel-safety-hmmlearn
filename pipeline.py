"""
pipeline.py

This is the piece that makes the live service actually work. Everything
built so far (features.py, hsmm_model.py, duration_fit.py, inference.py)
assumes you already have a clean 60-second window's worth of points. In
production, pings arrive one at a time, roughly every 5 seconds, and the
service needs to remember, PER TRIP, both the pings not yet aggregated into
a finished window and the trip's ongoing activity/duration state.

Two important design decisions, stated plainly:

1. State is a plain dict, not a Python object kept alive in memory.
   Render's free tier spins the service down after ~15 minutes of no
   traffic, which would silently wipe any in-memory state, exactly the same
   class of bug as the buffer-capped-duration bug from inference.py. So
   state for every trip is loaded from and saved back to the database on
   EVERY call. The service itself holds nothing between requests except the
   trained model, which is read-only and fine to keep in memory.

2. process_ping() is a pure function: (old_state, new_ping, model, fitted)
   -> (new_state, score_or_None). It doesn't know or care whether old_state
   came from Supabase, a local dict, or a test fixture. That's what makes it
   testable without a database and swappable onto any storage backend.
"""

import numpy as np

from features import feature_vector_from_raw
from duration_fit import tail_probability
from fusion import hsmm_evidence_llr

WINDOW_SEC = 60
DECODE_BUFFER_WINDOWS = 8


def initial_state():
    """The state dict for a brand-new trip, before any pings arrive."""
    return {
        "window_buffer": [],       # raw pings not yet folded into a window
        "window_start_t": None,
        "decode_buffer": [],       # last few window feature vectors (as lists)
        "current_state": None,
        "run_windows": 0,
        "last_score": None,        # most recent score dict, returned when
                                    # a ping arrives but doesn't close a window
    }


def process_ping(state, ping, model, fitted_durations):
    """
    state: dict as produced by initial_state() / a previous call's return
    ping: {"t": epoch_seconds, "lat": float, "lon": float}
    model: a trained hsmm_model.ExpandedHSMM
    fitted_durations: output of duration_fit.fit_gamma_distributions()

    Returns (new_state, score) where score is a dict (a window just closed
    and was scored) or None (still buffering, not enough time elapsed yet).
    """
    state = dict(state)  # don't mutate the caller's dict in place
    state["window_buffer"] = list(state["window_buffer"]) + [ping]

    if state["window_start_t"] is None:
        state["window_start_t"] = ping["t"]

    elapsed = ping["t"] - state["window_start_t"]
    if elapsed < WINDOW_SEC:
        # window not finished yet: nothing new to report
        return state, state["last_score"]

    # window finished: reduce the buffered pings to one feature vector
    feature_vector = feature_vector_from_raw(state["window_buffer"])

    decode_buffer = list(state["decode_buffer"]) + [feature_vector.tolist()]
    decode_buffer = decode_buffer[-DECODE_BUFFER_WINDOWS:]
    X = np.array(decode_buffer)

    macro_seq = model.decode_macro_states(X)
    latest_state = macro_seq[-1]

    if latest_state == state["current_state"]:
        run_windows = state["run_windows"] + 1
    else:
        run_windows = 1

    duration_sec = run_windows * WINDOW_SEC
    tp = tail_probability(duration_sec, latest_state, fitted_durations)
    llr = hsmm_evidence_llr(tp)

    score = {
        "current_state": latest_state,
        "duration_minutes": round(duration_sec / 60, 1),
        "tail_probability": round(tp, 4),
        "llr": round(llr, 3),
    }

    # reset for the next window, carry the decode buffer and run info forward
    state["window_buffer"] = []
    state["window_start_t"] = ping["t"]
    state["decode_buffer"] = decode_buffer
    state["current_state"] = latest_state
    state["run_windows"] = run_windows
    state["last_score"] = score

    return state, score


if __name__ == "__main__":
    # Local, no-database sanity check: feed one synthetic trip through
    # process_ping() one raw ping at a time, exactly as the live service
    # will receive pings, and confirm it behaves like inference.py did.
    from simulator import generate_trip, generate_dataset
    from hsmm_model import ExpandedHSMM
    from duration_fit import extract_segment_durations, fit_gamma_distributions
    from features import windows_from_points

    train_trips = generate_dataset(n_trips=150, seed=3)
    all_feats, all_labels = [], []
    for trip in train_trips:
        feats, labels, _ = windows_from_points(trip["points"])
        all_feats.append(feats)
        all_labels.extend(labels)
    model = ExpandedHSMM().fit_supervised(np.vstack(all_feats), all_labels)
    durations = extract_segment_durations(train_trips)
    fitted = fit_gamma_distributions(durations)

    rng = np.random.default_rng(99)
    live_trip = generate_trip(rng=rng, force_one_anomaly=True)

    state = initial_state()
    print(f"{'t(min)':>7} {'true_label':<20} {'state':<12} {'dur(min)':>9} {'tail_p':>8}")
    for p in live_trip:
        state, score = process_ping(state, p, model, fitted)
        if score and (p["true_state"] == "STATIONARY_ANOMALOUS" or score["tail_probability"] < 0.1):
            t_min = round(p["t"] / 60, 1)
            print(f"{t_min:>7} {p['true_state']:<20} {score['current_state']:<12} "
                  f"{score['duration_minutes']:>9} {score['tail_probability']:>8}")
