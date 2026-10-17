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

Synthetic data only. The simulator now models time-correlated GPS error, a
per-trip error scale (open sky to indoors) and Doppler-style reported speed, so
these numbers are NOT directly comparable with the earlier table (0.429
precision), which was measured on the older i.i.d.-noise simulator.

| detector | precision | recall | median detection delay |
|---|---|---|---|
| naive fixed-threshold (35 min) | 0.228 | 1.000 | 35 min |
| HSMM duration-tail (cutoff 0.05) | 0.550 | 0.957 | 64 min |

The HSMM approach is more than twice as precise by not flagging ordinary long
rests, at the cost of firing later. This is a real tradeoff, not a free win,
say so plainly if asked. The cutoff is a tunable knob and can drive graduated
escalation stages from one number:

| cutoff | precision | recall | median delay |
|---|---|---|---|
| 0.25 (WATCH stage) | 0.261 | 1.000 | 39 min |
| 0.10 (CONCERN stage) | 0.433 | 0.978 | 54 min |
| 0.05 (CRITICAL stage) | 0.550 | 0.957 | 64 min |

## Activity classification robustness

`features.py` now prefers the device's reported (Doppler) speed over speeds
differenced from positions, and drops fixes with accuracy worse than 50 m.
On an independently written GPS generator (30 simulated 15 min trips per
case, first 3 minutes excluded), per-window accuracy of the macro state:

| case | before | after |
|---|---|---|
| resting, very poor GPS (sigma 12 m, tau 5 s) | 2.5% | 100% |
| every other case tried (walk / rest / drive, clean and poor GPS) | 97% to 100% | 100% |

Caveat: the generator's reported speed is idealized (rest ~0.08 m/s noise).
Real Android fused-provider speed can be noisier; if resting shows up as
walking on a real phone, widen DOPPLER_REST_NOISE in simulator.py and retrain.
`model.pkl` is tagged with `FEATURE_VERSION`; `app.py` retrains in-process if the
pickle is missing, unreadable, or from a different feature version.

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
