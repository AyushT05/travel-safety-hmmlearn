"""
Priority scoring: transparent additive rules, one reason string per rule that
fired. Two invariants the tests enforce:
  1. Automatic detection can never reach "critical"; only a panic alert can.
  2. A single independent signal can never exceed "medium". "High" requires
     at least two independent kinds of evidence (the correlation requirement).
"""
from dataclasses import dataclass, field

from . import config

INDEPENDENT = {
    "zone_inside",
    "signal_lost",
    "prolonged_rest",
    "route_deviation",
    "erratic_movement",
    "distress_stop",
}
ZONE_BASE = {"info": 6, "warning": 18, "critical": 34}


@dataclass
class Result:
    score: int
    level: str | None          # None when below the "low" threshold
    reasons: list = field(default_factory=list)
    kinds: list = field(default_factory=list)   # independent kinds present


def level_for(score):
    if score >= config.HIGH_MIN:
        return "high"
    if score >= config.MEDIUM_MIN:
        return "medium"
    if score >= config.LOW_MIN:
        return "low"
    return None


def score_signals(signals, ctx):
    by = {}
    for s in signals:
        by.setdefault(s.kind, []).append(s)

    if "panic" in by:
        return Result(100, "critical", ["Panic button pressed (confirmed SOS)."],
                      sorted({s.kind for s in signals if s.kind in INDEPENDENT}))

    pts, reasons = 0.0, []

    def add(p, why):
        nonlocal pts
        pts += p
        reasons.append(f"+{round(p)}: {why}")

    zone = None
    if "zone_inside" in by:
        zone = max(by["zone_inside"], key=lambda s: ZONE_BASE.get(s.detail["severity"], 18))
        d = zone.detail
        p = ZONE_BASE.get(d["severity"], 18)
        if d["dwell_min"] >= 10:
            p += 8
        if d["uncertain"]:
            p *= 0.6
        add(p, f"inside verified {d['category']} zone ({d['severity']}), dwell {d['dwell_min']} min"
               f"{', position uncertain' if d['uncertain'] else ''}")

    lost = by["signal_lost"][0] if "signal_lost" in by else None
    if lost:
        age = lost.detail["age_s"]
        p = 12 if age < 600 else 22 if age < 1800 else 30
        add(p, f"no location update for {age // 60} min")

    rest = by["prolonged_rest"][0] if "prolonged_rest" in by else None
    if rest:
        add(20 if rest.detail["tail_probability"] <= 0.02 else 15,
            f"unusually long rest ({round(rest.detail['duration_min'])} min)")

    dev = by["route_deviation"][0] if "route_deviation" in by else None
    if dev:
        dev_m = dev.detail.get("deviation_m", 1800.0)
        p = 15 if dev_m < 3000 else 22 if dev_m < 6000 else 28
        add(p, f"deviated from planned route ({round(dev_m / 1000.0, 1)} km off)")

    erratic = by["erratic_movement"][0] if "erratic_movement" in by else None
    if erratic:
        add(18, "erratic movement patterns detected")

    stop = by["distress_stop"][0] if "distress_stop" in by else None
    if stop:
        add(20, "abrupt vehicle standstill in isolated sector")

    pred = by["predictive_hazard"][0] if "predictive_hazard" in by else None
    if pred:
        add(12, f"trajectory projecting towards {pred.detail.get('category', 'hazard')} zone")

    # Correlation bonuses: this is where several weak signals become one strong one.
    if lost and (zone or lost.detail["near_zone_id"] is not None):
        add(25, "contact lost while inside or next to a verified hazard zone")
    if lost and rest:
        add(15, "went silent after an unusually long stop")
    if rest and zone:
        add(10, "long stop inside a verified hazard zone")
    if dev and lost:
        add(25, "went silent after deviating from planned route")
    if dev and zone:
        add(25, "route deviation led into a verified hazard zone")
    if dev and erratic:
        add(18, "erratic trajectory while deviating from planned route")
    if dev and rest:
        add(15, "prolonged stop following route deviation")
    if stop and ctx.get("night"):
        add(10, "abrupt vehicle stop at night")
    if erratic and zone:
        add(15, "erratic movement near a verified hazard zone")

    kinds = sorted(k for k in by if k in INDEPENDENT)
    if len(kinds) >= 1:   # context only amplifies something that already exists
        if "isolated_area" in by and rest:
            add(5, "area looks isolated (client-triggered assessment, lower trust)")
        if ctx.get("night") and len(kinds) >= 2:
            add(5, "night-time")

    score = int(round(pts))
    if len(kinds) < 2:
        capped = min(score, config.HIGH_MIN - 1)
        if capped < score:
            reasons.append(f"capped at {capped}: only one independent kind of evidence")
        score = capped
    if "poor_gps" in by:
        reasons.append("GPS accuracy is poor; position claims carry extra uncertainty")

    return Result(score, level_for(score), reasons, kinds)
