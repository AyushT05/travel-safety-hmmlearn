# HSMM Activity Segmentation

Duration-aware activity classification for the Tourist Safety Companion's
risk engine. Distinguishes "40 minutes stationary at a viewpoint" from "40
minutes stationary somewhere that isn't normal for this activity," which a
fixed-threshold inactivity check structurally cannot do.

## Files, in the order they build on each other

- `simulator.py` — generates synthetic labeled trip trajectories (Walking /
  Resting / In-vehicle / Stationary-anomalous), since no real distress
  dataset exists to train or evaluate against.
- `features.py` — converts raw GPS pings into 60-second windowed feature
  vectors (speed, speed variance, displacement radius, heading entropy).
  Same function is meant to run on real production pings, not just
  simulated ones.
- `hsmm_model.py` — the state-expansion HSMM: each macro activity state
  (Walking/Resting/In-vehicle) is expanded into a short chain of sub-states
  inside an ordinary `hmmlearn.GaussianHMM`, enforcing a minimum dwell time.
  Emissions are fit per macro state from labeled data; `Stationary-anomalous`
  is deliberately NOT its own HMM state (see the module docstring for why).
- `duration_fit.py` — fits a Gamma distribution to real segment durations
  per macro state, giving `P(duration > T | state)`. This is what actually
  judges whether a given duration is unusual, not the HMM itself.
- `inference.py` — online, per-trip inference: a small rolling decode buffer
  for a stable current-state label, plus a persistent (buffer-independent)
  run-duration counter.
- `fusion.py` — converts the duration tail probability into a
  log-likelihood-ratio evidence term, meant to be summed with the project's
  other evidence sources (CUSUM route deviation, geofence status, etc.) in
  the broader risk fusion engine. Only owns the HSMM's contribution.
- `evaluate.py` — head-to-head comparison against a naive fixed-threshold
  detector on held-out synthetic trips: precision, recall, and detection
  delay for both.

## Running it

```bash
pip install -r requirements.txt
python3 simulator.py     # sanity check: generates a few trips
python3 features.py      # sanity check: windows one trip
python3 hsmm_model.py    # trains and decodes one trip
python3 duration_fit.py  # fits Gamma distributions, prints tail probs
python3 inference.py     # simulates one live trip window-by-window
python3 evaluate.py      # the actual precision/recall comparison
```

## Current evaluated result (seed=100 train / seed=200 test, 250 trips each)

| detector | precision | recall | median detection delay |
|---|---|---|---|
| naive fixed-threshold (35 min) | 0.146 | 1.000 | 35 min |
| HSMM duration-tail (cutoff 0.05) | 0.429 | 0.971 | 64 min |

The HSMM approach roughly triples precision by not flagging ordinary long
rests, at the cost of firing later. This is a real tradeoff, not a free
win — say so plainly if asked. The tail-probability cutoff is a tunable
knob (see the sweep below); it also gives a natural way to drive graduated
escalation stages directly from one number instead of separate hand-picked
thresholds per stage:

| cutoff | precision | recall | median delay |
|---|---|---|---|
| 0.25 (→ WATCH stage) | 0.164 | 1.000 | 39 min |
| 0.10 (→ CONCERN stage) | 0.284 | 0.971 | 54 min |
| 0.05 (→ CRITICAL stage) | 0.429 | 0.971 | 64 min |

## Honest framing for a viva

This is an HSMM **approximated via state expansion** within a standard HMM
framework (the `hmmlearn` state-expansion trick), with duration
distributions fit **directly** from segmented trip data via
`scipy.stats.gamma`, not a full generalized Baum-Welch HSMM implementation.
That's a normal, defensible engineering tradeoff — describe it exactly this
way if asked, rather than as "a full Hidden Semi-Markov Model from first
principles."

The distress-side likelihood in `fusion.py` is an assumed flat floor
(`DISTRESS_LIKELIHOOD_FLOOR = 0.05`), not fit from real distress data,
because none exists. The structure (log-odds evidence fusion) is
principled; that one constant is a stated, deliberate simplification.

## Next steps to integrate into the real pipeline

1. Replace `simulator.py`'s synthetic feature generation with real GPS
   pings from the Supabase `locations` table, run through the same
   `features.py` windowing function.
2. Wrap `LiveTripState` in the Edge Function (or a small Python
   microservice it calls) that already receives each location batch.
3. Feed `fusion.hsmm_evidence_llr(...)`'s output into the broader log-odds
   fusion engine alongside CUSUM and geofence evidence terms.
4. Re-fit `duration_fit.py`'s Gamma parameters periodically on real trip
   data once enough is collected, instead of only simulator output.
