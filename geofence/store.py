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
