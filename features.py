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

# Bump whenever the meaning of the feature vector changes. app.py refuses to
# use a model.pkl trained under a different version (and retrains instead).
FEATURE_VERSION = 2


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


MAX_USABLE_ACCURACY_M = 50.0   # fixes reported worse than this are dropped
MIN_DOPPLER_FRACTION = 0.6     # need this share of valid reported speeds to trust them


def _smooth3(a):
    """3-point moving mean (edges kept) to damp fix-to-fix jitter."""
    if len(a) < 3:
        return a
    out = a.copy()
    out[1:-1] = (a[:-2] + a[1:-1] + a[2:]) / 3.0
    return out


def feature_vector_from_raw(points):
    """
    Public entry point used by both training and the live scoring service.
    points: raw dicts {t, lat, lon[, speed, accuracy]}.

    Speed source, in order of preference:
      1. The device's reported (Doppler) speed, median over the window. This
         is far less noisy than differencing positions, and it is what lets a
         stationary phone with poor GPS (indoors) still read as ~0 m/s.
      2. Fallback when the client sends no speed: differenced speed over
         3-point-smoothed positions (noticeably weaker; see README).
    Fixes with very poor reported accuracy are dropped first.
    """
    usable = [p for p in points
              if p.get("accuracy") is None or p["accuracy"] <= MAX_USABLE_ACCURACY_M]
    if len(usable) >= 3:
        points = usable

    if len(points) < 2:
        return np.array([0.0, 0.0, 0.0, 0.0])

    lats = np.array([p["lat"] for p in points])
    lons = np.array([p["lon"] for p in points])
    ts = np.array([p["t"] for p in points], dtype=float)

    reported = np.array([p["speed"] for p in points
                         if p.get("speed") is not None and p["speed"] >= 0], dtype=float)
    if len(reported) >= MIN_DOPPLER_FRACTION * len(points):
        mean_speed = float(np.median(reported))
        speed_var = float(np.var(reported))
    else:
        sl, so = _smooth3(lats), _smooth3(lons)
        dists = _haversine_m(sl[:-1], so[:-1], sl[1:], so[1:])
        dts = np.diff(ts)
        dts[dts <= 0] = 1
        speeds = dists / dts
        mean_speed = float(np.mean(speeds))
        speed_var = float(np.var(speeds))

    centroid_lat, centroid_lon = float(np.mean(lats)), float(np.mean(lons))
    radii = _haversine_m(lats, lons, centroid_lat, centroid_lon)
    displacement_radius = float(np.max(radii))

    if len(points) >= 3:
        headings = _heading(lats[:-1], lons[:-1], lats[1:], lons[1:])
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
