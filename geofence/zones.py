"""Zone model plus adapters from the two Supabase tables that hold hazards."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def aware(dt):
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def parse_dt(v):
    if v in (None, ""):
        return None
    if isinstance(v, datetime):
        return aware(v)
    try:
        return aware(datetime.fromisoformat(str(v).replace("Z", "+00:00")))
    except ValueError:
        return None


@dataclass(frozen=True)
class Zone:
    key: str                      # "risk:<uuid>" or "unsafe:<id>", also the alert dedupe key
    name: str
    hazard_type: str
    severity: str                 # low | medium | high | critical
    geometry: dict                # GeoJSON geometry
    confidence: float             # 0..1, how far the source can be trusted
    source: str
    radius_m: Optional[float] = None     # buffer around the geometry; None = default per geometry kind
    observed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    status: str = "active"
    source_url: Optional[str] = None
    attributes: dict = field(default_factory=dict)
    valid_from: Optional[datetime] = None   # hazard not in effect before this (e.g. a forecast wind field)


def zone_from_risk_row(r):
    return Zone(
        key=f"risk:{r['id']}", name=r.get("name") or "Risk zone", hazard_type=r["hazard_type"],
        severity=r.get("severity") or "medium", geometry=r["geometry"],
        confidence=float(r["confidence"]) if r.get("confidence") is not None else 0.5,
        source=r.get("source") or "unknown", radius_m=r.get("radius_m"),
        observed_at=parse_dt(r.get("observed_at")), expires_at=parse_dt(r.get("expires_at")),
        status=r.get("status") or "active", source_url=r.get("source_url"),
        attributes=r.get("attributes") or {},
        valid_from=parse_dt((r.get("attributes") or {}).get("valid_from")),
    )


_UNSAFE_TYPE = {"poor_lighting": "poor_lighting", "robbery": "crime", "assault": "crime",
                "harassment": "crime", "suspicious_activity": "crime", "other": "crime"}
_UNSAFE_SEV = {"info": "low", "warning": "medium", "critical": "critical"}
# Community reports: a verified report is trusted much more than a pending one.
_UNSAFE_CONF = {"verified": 0.8, "pending": 0.35}


def zone_from_unsafe_row(r):
    """unsafe_zones (user/admin reports). Dismissed and expired reports return None."""
    conf = _UNSAFE_CONF.get(r.get("status"))
    if conf is None:
        return None
    return Zone(
        key=f"unsafe:{r['id']}", name=(r.get("description") or r.get("category") or "Reported zone")[:80],
        hazard_type=_UNSAFE_TYPE.get(r.get("category"), "crime"),
        severity=_UNSAFE_SEV.get(r.get("severity"), "medium"),
        geometry={"type": "Point", "coordinates": [r["lon"], r["lat"]]},
        confidence=conf, source=f"community report ({r.get('status')})",
        radius_m=r.get("radius_m") or 500,
        observed_at=parse_dt(r.get("created_at")), expires_at=parse_dt(r.get("expires_at")),
    )
