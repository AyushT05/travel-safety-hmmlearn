"""
simulator.py

Generates synthetic tourist trip trajectories with ground-truth activity
labels, so we have (feature, true_state, true_duration) data to train and
evaluate the HSMM against. There is no real distress dataset for this
problem, so this synthetic generator is the foundation everything else is
validated against.

States:
    WALKING              - normal ambulatory movement
    RESTING              - stationary, expected (viewpoints, meals, photos)
    IN_VEHICLE           - fast, low heading variance, road-following
    STATIONARY_ANOMALOUS - stationary but NOT like a normal rest
                            (injected ground-truth anomaly)

Each segment is generated as a sequence of raw (timestamp, lat, lon) points
at ~1 sample / 5 seconds, which is then fed through features.py exactly the
way real GPS pings would be, so the training and production pipelines never
diverge.
"""

import numpy as np

STATES = ["WALKING", "RESTING", "IN_VEHICLE", "STATIONARY_ANOMALOUS"]

# Roughly-real-world segment duration ranges (seconds), used to generate
# ground truth. STATIONARY_ANOMALOUS durations are deliberately drawn from a
# range that overlaps the tail of RESTING, so a naive fixed threshold can be
# meaningfully compared against the duration-distribution approach later.
DURATION_RANGES_SEC = {
    "WALKING": (300, 2400),          # 5–40 min
    "RESTING": (300, 2100),          # 5–35 min  (normal rest)
    "IN_VEHICLE": (180, 3600),       # 3–60 min
    "STATIONARY_ANOMALOUS": (2400, 7200),  # 40–120 min (overlaps RESTING's tail)
}

SAMPLE_INTERVAL_SEC = 5

# meters/sec speed ranges per state (mean, std), used to draw per-sample speed
SPEED_PARAMS = {
    "WALKING": (1.3, 0.3),
    "RESTING": (0.0, 0.05),
    "IN_VEHICLE": (12.0, 3.0),
    "STATIONARY_ANOMALOUS": (0.0, 0.05),  # looks like RESTING kinematically
}

GPS_NOISE_STD_M = 4.0  # typical consumer GPS jitter


def _meters_to_latlon_delta(dx_m, dy_m, lat0):
    # crude local flat-earth approximation, fine at this scale
    dlat = dy_m / 111320.0
    dlon = dx_m / (111320.0 * np.cos(np.radians(lat0)))
    return dlat, dlon


def _generate_segment(state, lat0, lon0, t0, rng, force_duration_sec=None):
    lo, hi = DURATION_RANGES_SEC[state]
    duration_sec = force_duration_sec or rng.integers(lo, hi)
    n_samples = max(2, duration_sec // SAMPLE_INTERVAL_SEC)

    speed_mean, speed_std = SPEED_PARAMS[state]
    heading = rng.uniform(0, 2 * np.pi)

    points = []
    lat, lon = lat0, lon0
    t = t0
    for i in range(n_samples):
        speed = max(0.0, rng.normal(speed_mean, speed_std))
        if state == "WALKING":
            heading += rng.normal(0, 0.15)  # meanders
        elif state == "IN_VEHICLE":
            heading += rng.normal(0, 0.03)  # follows roads, low heading variance
        # RESTING / STATIONARY_ANOMALOUS: heading is meaningless, speed ~ 0

        dx = speed * SAMPLE_INTERVAL_SEC * np.cos(heading)
        dy = speed * SAMPLE_INTERVAL_SEC * np.sin(heading)
        dlat, dlon = _meters_to_latlon_delta(dx, dy, lat)
        lat += dlat
        lon += dlon

        # GPS noise
        noise_lat, noise_lon = _meters_to_latlon_delta(
            rng.normal(0, GPS_NOISE_STD_M), rng.normal(0, GPS_NOISE_STD_M), lat
        )
        obs_lat, obs_lon = lat + noise_lat, lon + noise_lon

        points.append(
            {
                "t": t,
                "lat": obs_lat,
                "lon": obs_lon,
                "true_state": state,
            }
        )
        t += SAMPLE_INTERVAL_SEC

    return points, lat, lon, t


def generate_trip(rng=None, n_segments=12, anomaly_prob=0.15, force_one_anomaly=False):
    """
    Generates one synthetic trip as a concatenation of segments.

    Returns a list of point dicts: {t, lat, lon, true_state}
    """
    if rng is None:
        rng = np.random.default_rng()

    lat, lon, t = 25.578, 91.893, 0  # arbitrary Northeast-India-ish start (Shillong area)
    trip = []

    placed_anomaly = False
    for i in range(n_segments):
        if force_one_anomaly and i == n_segments - 2 and not placed_anomaly:
            state = "STATIONARY_ANOMALOUS"
            placed_anomaly = True
        else:
            r = rng.random()
            if r < anomaly_prob and not placed_anomaly:
                state = "STATIONARY_ANOMALOUS"
                placed_anomaly = True
            else:
                state = rng.choice(["WALKING", "RESTING", "IN_VEHICLE"], p=[0.5, 0.35, 0.15])

        pts, lat, lon, t = _generate_segment(state, lat, lon, t, rng)
        trip.extend(pts)

    return trip


def generate_dataset(n_trips=200, anomaly_prob=0.15, seed=42):
    """
    Generates a labeled dataset of trips. Roughly anomaly_prob fraction of
    trips contain exactly one injected STATIONARY_ANOMALOUS segment; the rest
    are "clean" trips with only WALKING/RESTING/IN_VEHICLE.
    """
    rng = np.random.default_rng(seed)
    trips = []
    for i in range(n_trips):
        has_anomaly = rng.random() < anomaly_prob
        trip = generate_trip(rng=rng, force_one_anomaly=has_anomaly, anomaly_prob=0.0)
        trips.append({"trip_id": i, "points": trip, "has_anomaly": has_anomaly})
    return trips


if __name__ == "__main__":
    ds = generate_dataset(n_trips=5, seed=1)
    for trip in ds:
        states = [p["true_state"] for p in trip["points"]]
        print(f"trip {trip['trip_id']}: {len(trip['points'])} pts, "
              f"has_anomaly={trip['has_anomaly']}, "
              f"states_present={sorted(set(states))}")
