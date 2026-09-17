"""
fusion.py

Converts the HSMM's duration tail probability into a log-likelihood-ratio
(LLR) evidence term, meant to be summed with the OTHER evidence terms in the
project's risk fusion engine (CUSUM route-deviation alarm, geofence status,
time-of-day, connectivity state, etc. — those live in their own modules,
not here). This file only owns the HSMM's contribution to that sum.

Why LLR and not the raw tail probability directly: log-likelihood ratios
from conditionally-independent evidence sources ADD, so the overall fusion
engine can just sum every term without needing per-source rescaling. Using
the raw tail probability directly here instead would break that additivity
and force ad hoc reweighting between sources, which is exactly the
unprincipled "hand-tuned weighted sum" approach this whole design is meant
to avoid.

Derivation used here, stated plainly rather than dressed up:
    - Treat the fitted Gamma's tail probability p = P(duration > T | normal)
      as an approximation of the likelihood of observing "duration at least
      this long" under the NORMAL hypothesis.
    - Approximate the DISTRESS hypothesis as a flat/uninformative prior over
      duration (every duration equally "expected" if something's wrong),
      giving it a constant likelihood floor.
    - LLR = log(P(evidence | distress) / P(evidence | normal))
          ≈ log(floor / p)
    This is an approximation, not a rigorously derived likelihood ratio
    (the "distress" distribution is assumed, not fit from real distress
    data, because none exists). Say exactly that if asked: it's a
    principled STRUCTURE (log-odds evidence fusion) with a deliberately
    simple, honestly-stated approximation for the distress side, not a
    claim that the distress distribution itself was learned.
"""

import numpy as np

# Floor representing "background" likelihood under the distress hypothesis.
# Kept as an explicit, named constant (not buried in a formula) since it's
# the one genuinely assumed number in this module and should be easy to
# find and justify in a viva.
DISTRESS_LIKELIHOOD_FLOOR = 0.05

# Clip tail probability away from exact 0 to avoid -inf / overflow on very
# long durations (see the 175+ min rows in inference.py's output).
MIN_TAIL_PROB = 1e-6


def hsmm_evidence_llr(tail_probability):
    """
    tail_probability: P(duration > T | current macro state), from
    duration_fit.tail_probability(). Small p (unusual duration) should push
    the LLR positive (evidence toward distress); p near 1 (ordinary
    duration) should push it toward/below zero (evidence toward normal).
    """
    p = max(tail_probability, MIN_TAIL_PROB)
    llr = np.log(DISTRESS_LIKELIHOOD_FLOOR / p)
    return float(llr)


if __name__ == "__main__":
    print(f"{'tail_prob':>10} {'llr':>8}   interpretation")
    for tp in [0.90, 0.60, 0.30, 0.10, 0.05, 0.01, 0.001]:
        llr = hsmm_evidence_llr(tp)
        verdict = "toward NORMAL" if llr < 0 else "toward DISTRESS"
        print(f"{tp:>10.3f} {llr:>8.2f}   {verdict}")
