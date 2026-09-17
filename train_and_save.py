"""
train_and_save.py

Run this ONCE (locally, or as a one-off build step), not every time the
service starts. Training on 250 synthetic trips takes a few seconds, but
there's no reason to pay that cost on every Render restart, and it also
means the live service's behavior is pinned to a specific, reproducible
trained artifact rather than silently retraining (and potentially drifting)
on every deploy.

Produces two files in this directory:
    model.pkl      - the trained hsmm_model.ExpandedHSMM
    durations.pkl  - the fitted Gamma parameters from duration_fit.py

Re-run this whenever you retrain on new (eventually real) data, and
redeploy the service with the updated .pkl files.
"""

import pickle

import numpy as np

from simulator import generate_dataset
from features import windows_from_points
from hsmm_model import ExpandedHSMM
from duration_fit import extract_segment_durations, fit_gamma_distributions

TRAIN_SEED = 100
N_TRAIN_TRIPS = 250


def main():
    print(f"Generating {N_TRAIN_TRIPS} synthetic training trips (seed={TRAIN_SEED})...")
    trips = generate_dataset(n_trips=N_TRAIN_TRIPS, seed=TRAIN_SEED)

    all_feats, all_labels = [], []
    for trip in trips:
        feats, labels, _ = windows_from_points(trip["points"])
        all_feats.append(feats)
        all_labels.extend(labels)
    X = np.vstack(all_feats)

    print("Fitting HSMM emissions...")
    model = ExpandedHSMM().fit_supervised(X, all_labels)

    print("Fitting duration distributions...")
    durations = extract_segment_durations(trips)
    fitted = fit_gamma_distributions(durations)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("durations.pkl", "wb") as f:
        pickle.dump(fitted, f)

    print("Saved model.pkl and durations.pkl")
    for state, p in fitted.items():
        print(f"  {state}: implied mean duration = {p['shape'] * p['scale'] / 60:.1f} min")


if __name__ == "__main__":
    main()
