"""Regenerates the dashboard's parity fixture from the REAL Python evaluator.

The dashboard re-implements the zone scoring in JavaScript (src/lib/risk) so the map can colour zones for any
time of day. This script is how the two are kept honest: it evaluates a fixed set of cases here and writes the
expected numbers to a JSON file that the dashboard's vitest suite (test/riskModel.test.js) compares against.
Run it whenever geofence/scoring.py, zones.py or sun.py changes, then run the dashboard tests.

    python tools/gen_risk_parity.py ../Travel-Safety-main/WebDashboard/margarakshak/test/fixtures/risk_parity.json
"""
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from geofence.scoring import Sample, assess_point  # noqa: E402
from geofence.sun import sun_times  # noqa: E402
from geofence.zones import Zone  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
LAT, LON = 12.3, 76.65
DAY = datetime(2026, 10, 24, 10, 0, tzinfo=IST)
NIGHT = datetime(2026, 10, 24, 20, 30, tzinfo=IST)
OBS = DAY - timedelta(days=1)


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat()


def pt(lat=LAT, lon=LON):
    return {"type": "Point", "coordinates": [lon, lat]}


SQ = {"type": "Polygon", "coordinates": [[[LON - .01, LAT - .01], [LON + .01, LAT - .01], [LON + .01, LAT + .01],
                                          [LON - .01, LAT + .01], [LON - .01, LAT - .01]]]}
LINE = {"type": "LineString", "coordinates": [[LON - .01, LAT], [LON + .01, LAT]]}


def Z(key, hazard, sev, conf, geom, radius=None, observed=OBS, expires=None, status="active", valid_from=None):
    return Zone(key=key, name=key, hazard_type=hazard, severity=sev, geometry=geom, confidence=conf, source="t",
                radius_m=radius, observed_at=observed, expires_at=expires, status=status, valid_from=valid_from)


def spec(z):
    return {"key": z.key, "hazard": z.hazard_type, "severity": z.severity, "geometry": z.geometry,
            "confidence": z.confidence, "radiusM": z.radius_m,
            "observedAt": iso(z.observed_at) if z.observed_at else None,
            "expiresAt": iso(z.expires_at) if z.expires_at else None, "status": z.status,
            "validFrom": iso(z.valid_from) if z.valid_from else None}


def build():
    cases = []

    def add(name, zones, lat, lon, t):
        a = assess_point(Sample(lat, lon, t, accuracy_m=0.0), zones)
        cases.append({"name": name, "zones": [spec(z) for z in zones], "lat": lat, "lon": lon, "t": iso(t),
                      "score": a.score, "level": a.level, "dark": a.is_dark,
                      "top": a.factors[0].hazard_type if a.factors else None})

    tomorrow = NIGHT + timedelta(days=1)
    add("lighting by day", [Z("a", "poor_lighting", "medium", 0.6, pt(), 100)], LAT, LON, DAY)
    add("lighting at night", [Z("a", "poor_lighting", "medium", 0.6, pt(), 100)], LAT, LON, NIGHT)
    add("crime by day", [Z("a", "crime", "high", 0.9, pt(), 100)], LAT, LON, DAY)
    add("crime at night", [Z("a", "crime", "high", 0.9, pt(), 100)], LAT, LON, NIGHT)
    add("connectivity capped", [Z("a", "weak_connectivity", "critical", 1.0, pt(), 100)], LAT, LON, NIGHT)
    add("flood critical", [Z("a", "flood", "critical", 0.9, pt(), 500, expires=tomorrow)], LAT, LON, DAY)
    add("expired before arrival", [Z("a", "flood", "critical", 1.0, pt(), 500, expires=datetime(2026, 10, 24, 18, 0, tzinfo=IST))], LAT, LON, NIGHT)
    add("not yet valid", [Z("a", "cyclone", "critical", 0.7, pt(), 500, expires=tomorrow, valid_from=NIGHT + timedelta(hours=3))], LAT, LON, NIGHT)
    add("crime 180 days old", [Z("a", "crime", "high", 0.9, pt(), 100, observed=NIGHT - timedelta(days=180))], LAT, LON, NIGHT)
    add("crime stale", [Z("a", "crime", "high", 0.9, pt(), 100, observed=NIGHT - timedelta(days=3000))], LAT, LON, NIGHT)
    add("crime unknown age", [Z("a", "crime", "high", 0.9, pt(), 100, observed=None)], LAT, LON, NIGHT)
    add("resolved zone ignored", [Z("a", "flood", "critical", 1.0, pt(), 500, status="resolved")], LAT, LON, DAY)
    add("two groups corroborate", [Z("f", "flood", "medium", 1.0, pt(), 500, expires=tomorrow), Z("c", "crime", "high", 0.9, pt(), 500)], LAT, LON, NIGHT)
    add("polygon deep inside", [Z("p", "flood", "high", 0.8, SQ, expires=tomorrow)], LAT, LON, NIGHT)
    add("polygon far outside", [Z("p", "flood", "high", 0.8, SQ, expires=tomorrow)], LAT + 1.0, LON, NIGHT)
    add("line inside default buffer", [Z("l", "road_condition", "high", 0.7, LINE)], LAT + 0.00009, LON, NIGHT)
    add("point buffer deepens polygon", [Z("p", "flood", "high", 0.8, SQ, radius=300, expires=tomorrow)], LAT, LON + 0.01 + 100 / 108642, NIGHT)
    add("restricted area no decay", [Z("r", "restricted_area", "medium", 0.9, pt(), 200, observed=NIGHT - timedelta(days=4000))], LAT, LON, DAY)

    sun = []
    for name, lat, lon, d in [("Mysuru", 12.2958, 76.6394, date(2026, 10, 24)), ("Delhi", 28.6139, 77.2090, date(2027, 1, 15)),
                              ("Kolkata", 22.5726, 88.3639, date(2026, 6, 21)), ("Chennai", 13.0827, 80.2707, date(2026, 12, 21)),
                              ("Leh", 34.1526, 77.5771, date(2026, 12, 21))]:
        r, s = sun_times(d, lat, lon)
        sun.append({"name": name, "lat": lat, "lon": lon, "y": d.year, "m": d.month, "d": d.day, "rise": r.isoformat(), "set": s.isoformat()})
    return {"generated_by": "tools/gen_risk_parity.py (hmmlearn repo)", "cases": cases, "sun": sun}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "risk_parity.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    data = build()
    with open(out, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {len(data['cases'])} cases and {len(data['sun'])} sun cases to {out}")
