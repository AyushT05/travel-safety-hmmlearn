"""
evaluate.py

Trains on one batch of synthetic trips, evaluates on a SEPARATE held-out
batch (different seed), comparing two detectors on the same held-out data:

  NAIVE      : flag once continuously-stationary duration exceeds a fixed
               cutoff (the "if inactive > 30 min, alert" approach almost
               every competing project in this space uses)
  HSMM-TAIL  : flag once the HSMM-modeled duration's tail probability drops
               below a chosen cutoff (this project's approach)

Evaluated at the level of "stationary runs" (a run = one continuous stretch
of RESTING or STATIONARY_ANOMALOUS ground truth). A run counts as a true
positive for a detector if the detector fires at some point during that
run AND the run's true label is STATIONARY_ANOMALOUS. A run counts as a
false positive if the detector fires during a run whose true label is
plain RESTING. This directly measures the thing that matters: does the
detector reliably catch injected anomalies without crying wolf on ordinary
long rests.

Also reports median time-to-detection (minutes into the anomalous run
before each detector first fires), since catching it 10 minutes sooner
matters in a real incident.
"""

import numpy as np

from simulator import generate_dataset
from features import windows_from_points
from hsmm_model import ExpandedHSMM
from duration_fit import extract_segment_durations, fit_gamma_distributions, tail_probability

NAIVE_THRESHOLD_MIN = 35     # chosen near the upper end of normal RESTING durations
HSMM_TAIL_CUTOFF = 0.05      # matches fusion.py's DISTRESS_LIKELIHOOD_FLOOR crossover


def train(seed):
    trips = generate_dataset(n_trips=250, seed=seed)
    all_feats, all_labels = [], []
    for trip in trips:
        feats, labels, _ = windows_from_points(trip["points"])
        all_feats.append(feats)
        all_labels.extend(labels)
    X = np.vstack(all_feats)
    model = ExpandedHSMM().fit_supervised(X, all_labels)

    durations = extract_segment_durations(trips)
    fitted = fit_gamma_distributions(durations)
    return model, fitted


def extract_stationary_runs(labels, window_sec=60):
    """
    Ground-truth runs of RESTING or STATIONARY_ANOMALOUS windows.
    Returns list of (start_idx, end_idx_inclusive, true_run_label) where
    true_run_label is STATIONARY_ANOMALOUS if any window in the run is,
    else RESTING.
    """
    runs = []
    i = 0
    n = len(labels)
    while i < n:
        if labels[i] in ("RESTING", "STATIONARY_ANOMALOUS"):
            j = i
            has_anomaly = False
            while j < n and labels[j] in ("RESTING", "STATIONARY_ANOMALOUS"):
                if labels[j] == "STATIONARY_ANOMALOUS":
                    has_anomaly = True
                j += 1
            runs.append((i, j - 1, "STATIONARY_ANOMALOUS" if has_anomaly else "RESTING"))
            i = j
        else:
            i += 1
    return runs


def evaluate_detectors(test_trips, model, fitted):
    results = {"NAIVE": {"tp": 0, "fp": 0, "fn": 0, "detect_delays": []},
               "HSMM": {"tp": 0, "fp": 0, "fn": 0, "detect_delays": []}}

    for trip in test_trips:
        feats, true_labels, _ = windows_from_points(trip["points"])
        runs = extract_stationary_runs(true_labels)

        # decode the whole trip once; reuse per-run
        decoded_macro = model.decode_macro_states(feats)

        for start, end, true_run_label in runs:
            naive_fired_at, hsmm_fired_at = None, None

            for k, idx in enumerate(range(start, end + 1)):
                duration_min = (k + 1)  # 1 window = 1 minute
                if naive_fired_at is None and duration_min >= NAIVE_THRESHOLD_MIN:
                    naive_fired_at = duration_min

                macro_state = decoded_macro[idx]  # will be RESTING (see hsmm_model docstring)
                tp = tail_probability(duration_min * 60, macro_state, fitted)
                if hsmm_fired_at is None and tp < HSMM_TAIL_CUTOFF:
                    hsmm_fired_at = duration_min

            is_anomaly = true_run_label == "STATIONARY_ANOMALOUS"
            for name, fired_at in [("NAIVE", naive_fired_at), ("HSMM", hsmm_fired_at)]:
                if is_anomaly:
                    if fired_at is not None:
                        results[name]["tp"] += 1
                        results[name]["detect_delays"].append(fired_at)
                    else:
                        results[name]["fn"] += 1
                else:
                    if fired_at is not None:
                        results[name]["fp"] += 1

    return results


def print_report(results):
    print(f"{'detector':<8} {'precision':>10} {'recall':>8} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'median_delay(min)':>18}")
    for name, r in results.items():
        tp, fp, fn = r["tp"], r["fp"], r["fn"]
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        median_delay = np.median(r["detect_delays"]) if r["detect_delays"] else float("nan")
        print(f"{name:<8} {precision:>10.3f} {recall:>8.3f} {tp:>4} {fp:>4} {fn:>4} "
              f"{median_delay:>18.1f}")


if __name__ == "__main__":
    print("Training on seed=100 ...")
    model, fitted = train(seed=100)

    print("Evaluating on held-out seed=200 ...")
    test_trips = generate_dataset(n_trips=250, seed=200)
    results = evaluate_detectors(test_trips, model, fitted)

    print()
    print_report(results)
