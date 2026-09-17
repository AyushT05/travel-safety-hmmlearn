"""
inference.py

Live-trip inference, meant to be called every time a new location window is
available for an active trip (i.e. once every WINDOW_SEC=60s in production).

Two things are tracked separately, deliberately:

1. A SMALL decode buffer (a handful of windows), re-Viterbi-decoded each
   call, purely to get a stable current-macro-state label smoothed over a
   few windows of context, cheap to decode every call, no incremental
   forward-pass bookkeeping needed.
2. A PERSISTENT run counter, external to that buffer, tracking how many
   consecutive windows have carried the same macro-state label. This must
   NOT be bounded by the decode buffer's length: an early version of this
   file kept the run-duration counter inside the same fixed-size buffer,
   which silently capped every detected duration at the buffer length (30
   min) and made anything past that point invisible. That's exactly the
   kind of bug that quietly launders the anomaly you're trying to catch, so
   it's worth calling out rather than just fixing silently.
"""

import numpy as np
from collections import deque

from duration_fit import tail_probability


class LiveTripState:
    """One instance per active trip; call update() as each new window arrives."""

    def __init__(self, model, fitted_durations, decode_buffer_windows=8, window_sec=60):
        self.model = model
        self.fitted_durations = fitted_durations
        self.window_sec = window_sec
        self.decode_buffer = deque(maxlen=decode_buffer_windows)

        # persistent, unbounded-by-buffer run tracking
        self.current_state = None
        self.run_windows = 0

    def update(self, feature_vector):
        """
        feature_vector: single (4,) array from features.py for the newest window.
        Returns a dict with the current macro state, how long we've been in
        it, and the tail probability (how unusual that duration is).
        """
        self.decode_buffer.append(feature_vector)
        X = np.vstack(self.decode_buffer)

        macro_seq = self.model.decode_macro_states(X)
        latest_state = macro_seq[-1]

        if latest_state == self.current_state:
            self.run_windows += 1
        else:
            self.current_state = latest_state
            self.run_windows = 1

        duration_sec = self.run_windows * self.window_sec
        tp = tail_probability(duration_sec, self.current_state, self.fitted_durations)

        return {
            "current_state": self.current_state,
            "duration_minutes": round(duration_sec / 60, 1),
            "tail_probability": round(tp, 4),
        }


if __name__ == "__main__":
    from simulator import generate_trip
    from features import windows_from_points
    from hsmm_model import ExpandedHSMM
    from duration_fit import extract_segment_durations, fit_gamma_distributions
    from simulator import generate_dataset
    import numpy as np

    # train on a fresh batch
    train_trips = generate_dataset(n_trips=200, seed=3)
    all_feats, all_labels = [], []
    for trip in train_trips:
        feats, labels, _ = windows_from_points(trip["points"])
        all_feats.append(feats)
        all_labels.extend(labels)
    X_train = np.vstack(all_feats)
    model = ExpandedHSMM().fit_supervised(X_train, all_labels)

    durations = extract_segment_durations(train_trips)
    fitted = fit_gamma_distributions(durations)

    # simulate one live trip with an injected anomaly, window by window
    rng = np.random.default_rng(99)
    live_trip = generate_trip(rng=rng, force_one_anomaly=True)
    feats, true_labels, _ = windows_from_points(live_trip)

    live_state = LiveTripState(model, fitted, decode_buffer_windows=8)

    print(f"{'t(min)':>7} {'true_label':<20} {'state':<12} {'dur(min)':>9} {'tail_p':>8}")
    for i, (fv, true_lbl) in enumerate(zip(feats, true_labels)):
        result = live_state.update(fv)
        if true_lbl == "STATIONARY_ANOMALOUS" or result["tail_probability"] < 0.1:
            t_min = round(i * 60 / 60, 1)
            print(f"{t_min:>7} {true_lbl:<20} {result['current_state']:<12} "
                  f"{result['duration_minutes']:>9} {result['tail_probability']:>8}")
