"""PostgREST access to risk_zones with the service-role key (same style as agent/store.py)."""
import os

import httpx


class RiskZoneStore:
    def __init__(self, supabase_url=None, service_key=None, client=None):
        self.base = (supabase_url or os.environ["SUPABASE_URL"]).rstrip("/") + "/rest/v1"
        key = service_key or os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        self.h = {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.http = client or httpx.Client(timeout=20)

    def upsert(self, rows, chunk=100):
        """Insert or update by source_ref. Needs the plain unique index on risk_zones(source_ref)."""
        for i in range(0, len(rows), chunk):
            r = self.http.post(f"{self.base}/risk_zones", params={"on_conflict": "source_ref"},
                               headers={**self.h, "Prefer": "resolution=merge-duplicates,return=minimal"},
                               json=rows[i:i + chunk])
            r.raise_for_status()

    def active_refs(self, prefix):
        r = self.http.get(f"{self.base}/risk_zones", headers=self.h,
                          params={"source_ref": f"like.{prefix}*", "status": "eq.active", "select": "source_ref"})
        r.raise_for_status()
        return [x["source_ref"] for x in r.json()]

    def expire(self, refs):
        if not refs:
            return
        quoted = ",".join('"' + x.replace('"', '') + '"' for x in refs)
        r = self.http.patch(f"{self.base}/risk_zones", params={"source_ref": f"in.({quoted})"},
                            headers={**self.h, "Prefer": "return=minimal"}, json={"status": "expired"})
        r.raise_for_status()

    def zones_in_bbox(self, min_lat, min_lon, max_lat, max_lon, now_iso, gdacs_fresh_after_iso=None):
        """Active, unexpired zones whose bounding box overlaps the box. Callers pad the box by the
        largest buffer they care about, since radius_m is applied after this pre-filter.

        gdacs_fresh_after_iso is the dead-man switch for ingested disaster zones: a GDACS row that the
        hourly sync has not touched since then is ignored, so a stopped sync cannot leave a stale
        forecast alive for days. (Each upsert refreshes updated_at via the table trigger.)"""
        conds = [f"or(expires_at.is.null,expires_at.gt.{now_iso})"]
        if gdacs_fresh_after_iso:
            conds.append(f"or(source.neq.GDACS,updated_at.gt.{gdacs_fresh_after_iso})")
        r = self.http.get(f"{self.base}/risk_zones", headers=self.h, params=[
            ("status", "eq.active"), ("min_lat", f"lte.{max_lat}"), ("max_lat", f"gte.{min_lat}"),
            ("min_lon", f"lte.{max_lon}"), ("max_lon", f"gte.{min_lon}"),
            ("and", "(" + ",".join(conds) + ")"), ("select", "*")])
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------------------------------
import math
import time
from datetime import datetime, timezone

from .zones import zone_from_risk_row, zone_from_unsafe_row

TILE_QUERY_PAD_DEG = 5.5       # > 500 km, the largest buffer risk_zones allows, plus half a tile
UNSAFE_PAD_DEG = 1.0
GDACS_MAX_AGE_S = 12 * 3600    # a GDACS zone not refreshed by the hourly sync for this long is ignored
STALE_TILE_OK_S = 15 * 60      # if Supabase is briefly unreachable, keep using a tile fetched this recently


def _iso(epoch):
    return None if epoch is None else datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _epoch(iso):
    if not iso:
        return None
    return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).timestamp()


class SupabaseGeofenceDb(RiskZoneStore):
    """Everything GeofenceMonitor needs from Supabase, behind plain methods."""

    def __init__(self, *a, clock=time.time, tile_ttl_s=60.0, **kw):
        super().__init__(*a, **kw)
        self.clock, self.tile_ttl_s, self._tiles = clock, tile_ttl_s, {}

    # ---- zones (cached per 1 degree tile: tourists ping every few seconds, hazards change hourly)
    def zones_near(self, lat, lon):
        tile = (math.floor(lat), math.floor(lon))
        now = self.clock()
        hit = self._tiles.get(tile)
        if hit and now - hit[0] < self.tile_ttl_s:
            return hit[1]
        try:
            zones = self._fetch_zones(tile, now)
        except Exception:
            if hit and now - hit[0] < STALE_TILE_OK_S:
                return hit[1]
            raise
        self._tiles[tile] = (now, zones)
        return zones

    def _fetch_zones(self, tile, now):
        clat, clon = tile[0] + 0.5, tile[1] + 0.5
        now_iso = _iso(now)
        zones = []
        for r in self.zones_in_bbox(clat - TILE_QUERY_PAD_DEG, clon - TILE_QUERY_PAD_DEG,
                                    clat + TILE_QUERY_PAD_DEG, clon + TILE_QUERY_PAD_DEG, now_iso,
                                    gdacs_fresh_after_iso=_iso(now - GDACS_MAX_AGE_S)):
            try:
                zones.append(zone_from_risk_row(r))
            except (KeyError, TypeError, ValueError):
                continue
        resp = self.http.get(f"{self.base}/unsafe_zones", headers=self.h, params=[
            ("select", "id,category,severity,description,lat,lon,radius_m,status,created_at,expires_at"),
            ("status", "in.(verified,pending)"),
            ("lat", f"gte.{clat - UNSAFE_PAD_DEG}"), ("lat", f"lte.{clat + UNSAFE_PAD_DEG}"),
            ("lon", f"gte.{clon - UNSAFE_PAD_DEG}"), ("lon", f"lte.{clon + UNSAFE_PAD_DEG}")])
        resp.raise_for_status()
        for r in resp.json():
            try:
                z = zone_from_unsafe_row(r)
            except (KeyError, TypeError, ValueError):
                continue
            if z is not None:
                zones.append(z)
        return zones

    # ---- per (user, zone-group) alert state
    def load_states(self, user_id):
        r = self.http.get(f"{self.base}/geofence_state", headers=self.h,
                          params={"user_id": f"eq.{user_id}", "select": "*"})
        r.raise_for_status()
        return {x["zone_key"]: {
            "state": x["state"], "clear_streak": x.get("clear_streak") or 0,
            "first_seen_at": _epoch(x.get("first_seen_at")), "last_approach_at": _epoch(x.get("last_approach_at")),
            "last_entry_at": _epoch(x.get("last_entry_at")), "last_alert_id": x.get("last_alert_id"),
            "last_level": x.get("last_level")} for x in r.json()}

    def save_state(self, user_id, zone_key, row):
        body = {"user_id": user_id, "zone_key": zone_key, "state": row["state"], "clear_streak": row["clear_streak"],
                "first_seen_at": _iso(row.get("first_seen_at")), "last_approach_at": _iso(row.get("last_approach_at")),
                "last_entry_at": _iso(row.get("last_entry_at")), "last_alert_id": row.get("last_alert_id"),
                "last_level": row.get("last_level"), "updated_at": _iso(self.clock())}
        r = self.http.post(f"{self.base}/geofence_state", params={"on_conflict": "user_id,zone_key"},
                           headers={**self.h, "Prefer": "resolution=merge-duplicates,return=minimal"}, json=body)
        r.raise_for_status()

    # ---- alerts
    def insert_alert(self, alert):
        """Returns the new alert id, or None when this exact event was already recorded (webhook retry)."""
        r = self.http.post(f"{self.base}/alerts", params={"on_conflict": "event_id"},
                           headers={**self.h, "Prefer": "resolution=ignore-duplicates,return=representation"}, json=alert)
        r.raise_for_status()
        rows = r.json()
        return rows[0]["id"] if rows else None

    def resolve_alert(self, alert_id):
        r = self.http.patch(f"{self.base}/alerts", params={"id": f"eq.{alert_id}", "status": "eq.open"},
                            headers={**self.h, "Prefer": "return=minimal"},
                            json={"status": "resolved", "resolved_at": _iso(self.clock())})
        r.raise_for_status()
