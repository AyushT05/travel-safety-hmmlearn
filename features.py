"""
features.py

Converts a raw stream of GPS pings into fixed-length windows, each reduced
to a small feature vector. This is the ONLY place raw lat/lon touches the
model — everything downstream (training, inference) consumes these feature
vectors, never raw coordinates directly. Using the exact same function here
for both training data and live production pings is what keeps the
simulator-trained model valid in production.

Feature vector per window (4 dims):
    mean_speed        - meters/sec, mean over window
    speed_var         - variance of speed within window
    displacement_radius - max distance (m) from window's centroid point
    heading_entropy   - Shannon entropy of the heading-change histogram
                         (high = erratic/wandering, low = smooth/consistent)
"""

import numpy as np

WINDOW_SEC = 60


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _heading(lat1, lon1, lat2, lon2):
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(np.radians(lat2))
    x = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - \
        np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(dlon)
    return np.degrees(np.arctan2(y, x)) % 360


def _window_feature_vector(points):
    """points: list of {t, lat, lon, true_state} covering one ~60s window."""
    return feature_vector_from_raw(points)


def feature_vector_from_raw(points):
    """
    Public entry point: same computation as _window_feature_vector, exposed
    under a clean name for external callers (e.g. the live scoring service)
    that pass in raw {t, lat, lon} dicts without a true_state field.
    """
    lats = np.array([p["lat"] for p in points])
    lons = np.array([p["lon"] for p in points])
    ts = np.array([p["t"] for p in points])

    if len(points) < 2:
        return np.array([0.0, 0.0, 0.0, 0.0])

    dists = _haversine_m(lats[:-1], lons[:-1], lats[1:], lons[1:])
    dts = np.diff(ts)
    dts[dts == 0] = 1
    speeds = dists / dts

    mean_speed = float(np.mean(speeds))
    speed_var = float(np.var(speeds))

    centroid_lat, centroid_lon = float(np.mean(lats)), float(np.mean(lons))
    radii = _haversine_m(lats, lons, centroid_lat, centroid_lon)
    displacement_radius = float(np.max(radii))

    if len(points) >= 3:
        headings = _heading(lats[:-1], lons[:-1], lats[1:], lons[1:])
        # bin headings into 8 compass sectors, compute Shannon entropy
        bins = np.histogram(headings, bins=8, range=(0, 360))[0]
        probs = bins / max(bins.sum(), 1)
        probs = probs[probs > 0]
        heading_entropy = float(-np.sum(probs * np.log2(probs))) if len(probs) else 0.0
    else:
        heading_entropy = 0.0

    return np.array([mean_speed, speed_var, displacement_radius, heading_entropy])


def windows_from_points(points, window_sec=WINDOW_SEC):
    """
    Splits a point stream into non-overlapping time windows and returns:
        feature_matrix : (n_windows, 4) array
        majority_labels: list of the most common true_state per window
                         (only meaningful for simulated/labeled data)
        window_start_ts: list of each window's start timestamp
    """
    if not points:
        return np.zeros((0, 4)), [], []

    t0 = points[0]["t"]
    buckets = {}
    for p in points:
        idx = int((p["t"] - t0) // window_sec)
        buckets.setdefault(idx, []).append(p)

    feats, labels, starts = [], [], []
    for idx in sorted(buckets.keys()):
        wpts = buckets[idx]
        feats.append(_window_feature_vector(wpts))
        # majority label within this window (ties broken by first-seen)
        states = [p["true_state"] for p in wpts]
        labels.append(max(set(states), key=states.count))
        starts.append(wpts[0]["t"])

    return np.array(feats), labels, starts


if __name__ == "__main__":
    from simulator import generate_trip
    import numpy as _np

    rng = _np.random.default_rng(0)
    trip = generate_trip(rng=rng, force_one_anomaly=True)
    feats, labels, starts = windows_from_points(trip)
    print(f"{len(trip)} raw points -> {feats.shape[0]} windows")
    print("first 5 windows:")
    for i in range(5):
        print(f"  t={starts[i]:>6}  label={labels[i]:<20} feat={feats[i].round(3)}")
    print("anomalous windows present:", "STATIONARY_ANOMALOUS" in labels)
