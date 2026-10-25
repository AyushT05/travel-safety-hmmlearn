"""Orchestrates one GDACS sync run. Call from an authenticated endpoint on a schedule (hourly)."""
import os
from datetime import datetime, timezone

import httpx

from . import gdacs


def _bbox():
    raw = os.environ.get("GEOFENCE_INTEREST_BBOX")
    if not raw:
        return gdacs.DEFAULT_INTEREST_BBOX
    vals = tuple(float(x) for x in raw.split(","))
    if len(vals) != 4:
        raise ValueError("GEOFENCE_INTEREST_BBOX needs min_lat,min_lon,max_lat,max_lon")
    return vals


def _fetch(url):
    r = httpx.get(url, timeout=30, headers={"Accept": "application/json", "User-Agent": "Margarakshak/1.0"})
    r.raise_for_status()
    return r.json()


def sync_gdacs(store, fetch=_fetch, now=None, bbox=None):
    """Any failure raises BEFORE the database is touched, so a bad fetch can never retire live hazards
    or write garbage. (Zones also expire on their own if syncs stop: see gdacs.ALIVE_TTL.)"""
    now = now or datetime.now(timezone.utc)
    fc = fetch(gdacs.GDACS_URL)
    rows, skipped = gdacs.parse_event_list(fc, now, bbox or _bbox())
    rows, footprints = gdacs.enrich_with_footprints(rows, fetch, now)
    existing = store.active_refs(gdacs.PREFIX)
    expire = gdacs.refs_to_expire(existing, rows)
    store.upsert(rows)
    store.expire(expire)
    return {"features": len(fc["features"]), "ingested": len(rows), "expired": len(expire), "skipped": skipped,
            "footprints": footprints}
