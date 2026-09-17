"""
duration_fit.py

Fits a duration distribution per macro activity state directly from
ground-truth segment lengths in the simulator output. This is the piece
that actually answers "is this 47-minute stop unusual", by giving us
P(duration > T | state), rather than a hand-picked cutoff.

Deliberately excludes STATIONARY_ANOMALOUS segments when fitting RESTING's
distribution: we're fitting what NORMAL resting looks like, and the
injected anomalies are, by construction, not normal. Including them would
quietly launder the anomaly into what counts as "expected", which defeats
the entire point.
"""

import numpy as np
from scipy import stats


def extract_segment_durations(trips):
    """
    trips: output of simulator.generate_dataset()
    Returns dict: macro_state -> list of segment durations (seconds),
    where macro_state is one of WALKING / RESTING / IN_VEHICLE
    (STATIONARY_ANOMALOUS durations are returned separately, not merged in).
    """
    durations = {"WALKING": [], "RESTING": [], "IN_VEHICLE": [], "STATIONARY_ANOMALOUS": []}

    for trip in trips:
        points = trip["points"]
        if not points:
            continue
        current_state = points[0]["true_state"]
        seg_start_t = points[0]["t"]
        for p in points[1:]:
            if p["true_state"] != current_state:
                durations[current_state].append(p["t"] - seg_start_t)
                current_state = p["true_state"]
                seg_start_t = p["t"]
        durations[current_state].append(points[-1]["t"] - seg_start_t)

    return durations


def fit_gamma_distributions(durations_by_state):
    """
    Fits a Gamma distribution to each NORMAL state's duration list
    (WALKING, RESTING, IN_VEHICLE only). Returns dict:
        state -> {"shape": a, "loc": loc, "scale": scale}
    Gamma's loc is fixed at 0 (durations can't be negative) for a stabler fit.
    """
    fitted = {}
    for state in ["WALKING", "RESTING", "IN_VEHICLE"]:
        d = np.array(durations_by_state[state], dtype=float)
        d = d[d > 0]
        a, loc, scale = stats.gamma.fit(d, floc=0)
        fitted[state] = {"shape": a, "loc": loc, "scale": scale}
    return fitted


def tail_probability(duration_sec, state, fitted_params):
    """
    P(duration > duration_sec | state), i.e. how unusual this duration is
    given what "normal" looks like for that activity. Small = unusual.
    """
    p = fitted_params[state]
    return float(stats.gamma.sf(duration_sec, a=p["shape"], loc=p["loc"], scale=p["scale"]))


if __name__ == "__main__":
    from simulator import generate_dataset

    trips = generate_dataset(n_trips=300, seed=11)
    durations = extract_segment_durations(trips)

    print("segment counts:", {k: len(v) for k, v in durations.items()})
    print("RESTING duration range (min):",
          round(min(durations["RESTING"]) / 60, 1), "-",
          round(max(durations["RESTING"]) / 60, 1))
    print("STATIONARY_ANOMALOUS duration range (min):",
          round(min(durations["STATIONARY_ANOMALOUS"]) / 60, 1), "-",
          round(max(durations["STATIONARY_ANOMALOUS"]) / 60, 1))

    fitted = fit_gamma_distributions(durations)
    print("\nfitted Gamma params:")
    for state, p in fitted.items():
        print(f"  {state}: shape={p['shape']:.2f} scale={p['scale']:.2f} "
              f"(implied mean = {p['shape']*p['scale']/60:.1f} min)")

    print("\ntail probabilities for various RESTING durations:")
    for mins in [10, 20, 30, 35, 45, 60, 90]:
        tp = tail_probability(mins * 60, "RESTING", fitted)
        print(f"  {mins:>3} min stationary -> P(normal rest this long or longer) = {tp:.4f}")
