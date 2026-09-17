"""
hsmm_model.py

Implements the "state expansion" trick: approximate a Hidden Semi-Markov
Model using an ordinary HMM (hmmlearn.GaussianHMM) by expanding each macro
activity state into a short chain of sub-states. Forcing the chain through
K sub-states before it's allowed to leave enforces a minimum dwell time,
giving a non-memoryless (non-geometric) duration shape instead of the
single-step-transition duration distribution an ordinary HMM implies.

Design choice: STATIONARY_ANOMALOUS is deliberately NOT modeled as its own
HMM state. Kinematically, an anomalous stationary period looks identical to
a normal RESTING period (near-zero speed) — that's exactly why a fixed
threshold on speed alone can't distinguish them. The HMM's job is only to
say "which macro activity is this" (Walking / Resting / In-vehicle) and
track how long we've continuously been in it. Whether that duration is
anomalous is judged separately in duration_fit.py / fusion.py, against a
distribution fit from real segment lengths — that's where the actual
detection power comes from, not from the HMM itself.

Emissions are fit per MACRO state (not per sub-state) from labeled data:
sub-states within one macro state share the same Gaussian emission, since
their only purpose is to shape the duration distribution, not to represent
different observations.
"""

import numpy as np
from hmmlearn import hmm

MACRO_STATES = ["WALKING", "RESTING", "IN_VEHICLE"]

# Number of sub-states per macro state = enforced minimum dwell, in windows
# (WINDOW_SEC=60s each) before the chain is allowed to transition out.
# RESTING gets more sub-states since we care most about its duration shape.
N_SUBSTATES = {"WALKING": 2, "RESTING": 3, "IN_VEHICLE": 2}

# Probability of staying in the LAST sub-state of a macro chain another step
# (i.e. continuing the same activity) vs. transitioning to another macro
# state. Higher = activities tend to run longer before switching.
SELF_STAY_PROB = 0.85


class ExpandedHSMM:
    def __init__(self):
        self.expanded_labels = []   # index -> macro state name, for each sub-state
        self.macro_to_indices = {}  # macro state -> list of sub-state indices
        self.model = None

    # ---- structure construction -------------------------------------

    def _build_index_map(self):
        idx = 0
        for macro in MACRO_STATES:
            k = N_SUBSTATES[macro]
            self.macro_to_indices[macro] = list(range(idx, idx + k))
            self.expanded_labels.extend([macro] * k)
            idx += k
        self.n_states = idx

    def _build_transmat(self):
        """
        Block structure:
          - within a macro chain: sub-state i -> i+1 deterministically
          - last sub-state of a macro chain: self-loop with SELF_STAY_PROB,
            remaining probability mass split across the FIRST sub-state of
            every OTHER macro chain, weighted by empirical macro-to-macro
            transition frequency (set uniformly here; refine with real
            transition counts if you have them).
        """
        T = np.zeros((self.n_states, self.n_states))
        macros = MACRO_STATES
        for macro in macros:
            indices = self.macro_to_indices[macro]
            k = len(indices)
            for i in range(k - 1):
                T[indices[i], indices[i + 1]] = 1.0
            last = indices[-1]
            T[last, last] = SELF_STAY_PROB
            others = [m for m in macros if m != macro]
            remaining = 1.0 - SELF_STAY_PROB
            for m in others:
                entry = self.macro_to_indices[m][0]
                T[last, entry] = remaining / len(others)
        # normalize defensively (should already sum to 1 per row)
        T = T / T.sum(axis=1, keepdims=True)
        return T

    def _build_startprob(self):
        pi = np.zeros(self.n_states)
        for macro in MACRO_STATES:
            pi[self.macro_to_indices[macro][0]] = 1.0 / len(MACRO_STATES)
        return pi

    # ---- supervised emission fitting ---------------------------------

    def fit_supervised(self, feature_matrix, labels):
        """
        feature_matrix : (n_windows, 4) array from features.py
        labels         : list of true macro-state strings, same length
                         (only WALKING / RESTING / IN_VEHICLE rows are used;
                          STATIONARY_ANOMALOUS rows are folded into RESTING's
                          emission fit, since they ARE kinematically resting
                          — that's intentional, see module docstring)
        """
        self._build_index_map()

        labels = np.array(labels)
        labels_for_emission = np.where(labels == "STATIONARY_ANOMALOUS", "RESTING", labels)

        n_features = feature_matrix.shape[1]
        means = np.zeros((self.n_states, n_features))
        covars = np.ones((self.n_states, n_features))  # diag covariance

        for macro in MACRO_STATES:
            mask = labels_for_emission == macro
            if mask.sum() < 2:
                raise ValueError(f"Not enough labeled windows for state {macro}")
            mu = feature_matrix[mask].mean(axis=0)
            var = feature_matrix[mask].var(axis=0)
            var = np.maximum(var, 1e-3)  # floor to avoid degenerate covariance
            for idx in self.macro_to_indices[macro]:
                means[idx] = mu
                covars[idx] = var

        self.model = hmm.GaussianHMM(
            n_components=self.n_states, covariance_type="diag", init_params=""
        )
        self.model.startprob_ = self._build_startprob()
        self.model.transmat_ = self._build_transmat()
        self.model.means_ = means
        self.model.covars_ = covars
        self.model.n_features = n_features
        return self

    # ---- inference -----------------------------------------------------

    def decode_macro_states(self, feature_matrix):
        """
        Viterbi-decode a sequence of feature windows, returns the most
        likely MACRO state (collapsed from sub-states) per window.
        """
        _, state_seq = self.model.decode(feature_matrix, algorithm="viterbi")
        return [self.expanded_labels[s] for s in state_seq]

    def current_run_duration(self, macro_seq, window_sec=60):
        """
        Given a decoded macro-state sequence (oldest first), returns
        (current_macro_state, duration_seconds_in_that_state_so_far)
        counting consecutive identical labels ending at the last window.
        """
        if not macro_seq:
            return None, 0
        current = macro_seq[-1]
        run_len = 1
        for s in reversed(macro_seq[:-1]):
            if s == current:
                run_len += 1
            else:
                break
        return current, run_len * window_sec


if __name__ == "__main__":
    from simulator import generate_dataset
    from features import windows_from_points

    ds = generate_dataset(n_trips=60, seed=7)
    all_feats, all_labels = [], []
    for trip in ds:
        feats, labels, _ = windows_from_points(trip["points"])
        all_feats.append(feats)
        all_labels.extend(labels)
    X = np.vstack(all_feats)

    model = ExpandedHSMM().fit_supervised(X, all_labels)
    print("expanded state count:", model.n_states)
    print("expanded labels:", model.expanded_labels)

    # quick decode sanity check on one trip
    test_trip = ds[0]
    feats, true_labels, _ = windows_from_points(test_trip["points"])
    decoded = model.decode_macro_states(feats)
    agree = np.mean([
        d == t or (t == "STATIONARY_ANOMALOUS" and d == "RESTING")
        for d, t in zip(decoded, true_labels)
    ])
    print(f"decode agreement with ground truth on trip 0: {agree:.2%}")
